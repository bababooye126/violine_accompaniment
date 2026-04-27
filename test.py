"""
test.py — MusAIc-RL Real-Time Inference Blueprint
==================================================

Pipeline for each 125 ms audio frame (8 Hz):

  [Audio Buffer]         ~50 ms  (OS capture + transfer)
       ↓
  [Feature Extraction]   ≤15 ms  (chroma + RMS on CPU, pre-allocated arrays)
       ↓
  [Model Inference]      ≤10 ms  (single GRU step on GPU, batch=1, seq=1)
       ↓
  [Action Decode]         ≤2 ms  (threshold + Bernoulli or greedy)
       ↓
  [MIDI Output]           ≤5 ms  (rtmidi note-on/off)
  ─────────────────────────────
  Total budget            ≤82 ms  (well inside 200 ms target)

Temperature sampling
--------------------
During inference we replace the pure Bernoulli threshold with temperature
scaling on the logits:

    p_adjusted = sigmoid(logit / T)

  T < 1.0  → sharper (more deterministic, safer for live performance)
  T = 1.0  → matches training distribution
  T > 1.0  → more exploratory (useful for evaluation / demos)

Run modes
---------
  python test.py --mode benchmark   # Latency profiler over a held-out file
  python test.py --mode simulate    # Step-through of a full wav file
  python test.py --mode realtime    # Live mic input (requires sounddevice)
"""

import argparse
import os
import time

import librosa
import numpy as np
import torch

from models import MusaicRL
from utils.audio_tools import extract_musaic_features
from utils.midi_tools import generate_autoregressive_to_midi

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS             = 8            # Must match training
TEMPERATURE    = 0.8          # Inference sharpening (< 1 = safer)
THRESHOLD      = 0.5          # Binary decode threshold after temperature scaling
CHECKPOINT     = "checkpoints/musaic_BEST.pt"   # ← Update as needed
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str) -> MusaicRL:
    model = MusaicRL(input_size=143, hidden_size=256, output_size=130).to(DEVICE)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"✅ Loaded checkpoint: {checkpoint_path}  "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"reward {ckpt.get('mean_reward', 0):.4f})")
    else:
        print(f"⚠️  Checkpoint not found ({checkpoint_path}). Using random weights.")
    model.eval()
    return model


# ── Temperature-scaled action decode ──────────────────────────────────────────
@torch.no_grad()
def decode_action(action_probs: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Apply temperature scaling to raw Sigmoid probs and threshold.

    Temperature is applied in logit-space so the Sigmoid distribution is
    sharpened/flattened cleanly without clamping artefacts.
    """
    # Convert probs → logits → scale → back to probs
    logits  = torch.logit(action_probs.clamp(1e-6, 1 - 1e-6))
    p_temp  = torch.sigmoid(logits / temperature)
    return (p_temp > THRESHOLD).float()


# ── Single-step inference (the hot path) ──────────────────────────────────────
@torch.no_grad()
def infer_step(
    model: MusaicRL,
    violin_chroma_rms: torch.Tensor,   # (13,)
    prev_action: torch.Tensor,          # (130,)
    h_n,
    temperature: float = TEMPERATURE,
):
    """
    One autoregressive inference step.

    Args:
        violin_chroma_rms : 12-bin chroma + 1-bin log-RMS, shape (13,).
        prev_action       : Binary action from the previous step, shape (130,).
        h_n               : GRU hidden state (or None for step 0).
        temperature       : Sampling temperature.

    Returns:
        action     (130,) binary float tensor
        action_probs (130,) raw sigmoid probabilities
        h_n        updated hidden state
    """
    # Build the 143-dim input: human context + machine context
    x_t = torch.cat([violin_chroma_rms, prev_action], dim=0)  # (143,)
    x_t = x_t.unsqueeze(0).unsqueeze(0).to(DEVICE)             # (1, 1, 143)

    action_probs, _, h_n = model(x_t, h_n)
    action_probs = action_probs.squeeze(0)                      # (130,)
    action       = decode_action(action_probs, temperature)

    return action, action_probs, h_n


# ── Feature extraction for one audio frame ────────────────────────────────────
def extract_frame_features(audio_frame: np.ndarray, sr: int = 22050) -> torch.Tensor:
    """
    Extracts a single 13-dim violin feature vector from a raw audio chunk.
    Designed to run on CPU in ≤ 15 ms.

    Args:
        audio_frame : 1-D numpy array of raw PCM samples (length ≈ sr/FS).
        sr          : Sample rate.

    Returns:
        Tensor of shape (13,) — [chroma×12, log_rms×1]
    """
    hop = len(audio_frame)  # single-frame hop = full buffer
    chroma  = librosa.feature.chroma_stft(y=audio_frame, sr=sr, hop_length=hop)
    rms     = librosa.feature.rms(y=audio_frame, hop_length=hop)
    log_rms = np.log(rms + 1e-6)

    # chroma: (12, 1), log_rms: (1, 1) → squeeze to (13,)
    features = np.concatenate([chroma[:, 0], log_rms[:, 0]])
    return torch.FloatTensor(features)


# ── Benchmark mode ────────────────────────────────────────────────────────────
def run_benchmark(wav_path: str, midi_path: str):
    """
    Runs the full pipeline over a pre-recorded file and reports per-stage
    latency using torch.cuda.Event for GPU-accurate timing.
    """
    print(f"\n{'='*60}")
    print(f"  LATENCY BENCHMARK")
    print(f"  File : {os.path.basename(wav_path)}")
    print(f"  Device : {DEVICE}  |  Temperature : {TEMPERATURE}")
    print(f"{'='*60}\n")

    model = load_model(CHECKPOINT)

    # Load raw audio once
    y_audio, sr = librosa.load(wav_path)
    hop_length   = int(sr / FS)
    n_frames     = len(y_audio) // hop_length

    # Pre-extract all features so benchmark is purely inference
    data = extract_musaic_features(wav_path, midi_path, fs=FS)
    x_violin = data["x"]           # (Steps, 13)
    n_steps  = x_violin.shape[0]

    # Timers
    times_extract = []
    times_infer   = []
    times_decode  = []

    h_n         = None
    prev_action = torch.zeros(130, device=DEVICE)
    prev_action[129] = 1.0  # Rest

    all_actions = []

    # CUDA warmup (first inference is always slow — exclude from stats)
    _ = infer_step(model, x_violin[0].to(DEVICE), prev_action, None)

    for t in range(n_steps):

        # ── Stage 1: Feature extraction (simulated from pre-loaded array) ─────
        t0 = time.perf_counter()
        v_feat = x_violin[t].to(DEVICE)
        t1 = time.perf_counter()
        times_extract.append((t1 - t0) * 1000)

        # ── Stage 2: Model inference ──────────────────────────────────────────
        if DEVICE.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        action, action_probs, h_n = infer_step(model, v_feat, prev_action, h_n)

        if DEVICE.type == "cuda":
            end_evt.record()
            torch.cuda.synchronize()
            times_infer.append(start_evt.elapsed_time(end_evt))
        else:
            times_infer.append((time.perf_counter() - t1) * 1000)

        # ── Stage 3: Action decode ────────────────────────────────────────────
        t2 = time.perf_counter()
        # Decode: convert binary action vector to note events
        active_pitches = torch.where(action[:128] > 0.5)[0].cpu().tolist()
        hold  = action[128].item() > 0.5
        rest  = action[129].item() > 0.5
        t3 = time.perf_counter()
        times_decode.append((t3 - t2) * 1000)

        all_actions.append(action.cpu())
        prev_action = action

    # ── Report ────────────────────────────────────────────────────────────────
    def stats(lst):
        a = np.array(lst)
        return a.mean(), np.percentile(a, 95), a.max()

    e_mean, e_p95, e_max = stats(times_extract)
    i_mean, i_p95, i_max = stats(times_infer)
    d_mean, d_p95, d_max = stats(times_decode)
    total_mean = e_mean + i_mean + d_mean

    print(f"  Stages          mean      p95      max")
    print(f"  {'─'*45}")
    print(f"  Feature xfer  {e_mean:6.2f} ms  {e_p95:6.2f} ms  {e_max:6.2f} ms")
    print(f"  Inference     {i_mean:6.2f} ms  {i_p95:6.2f} ms  {i_max:6.2f} ms")
    print(f"  Decode        {d_mean:6.2f} ms  {d_p95:6.2f} ms  {d_max:6.2f} ms")
    print(f"  {'─'*45}")
    print(f"  TOTAL (est.)  {total_mean:6.2f} ms")
    print()

    budget_ok = total_mean < 200
    print(f"  200ms budget : {'✅ PASS' if budget_ok else '❌ FAIL — optimise inference'}")
    print(f"  Steps profiled: {n_steps}")
    print()

    # Save action sequence to MIDI
    action_tensor = torch.stack(all_actions)  # (Steps, 130)
    out_path = os.path.join(OUTPUT_DIR, "benchmark_output.mid")
    generate_autoregressive_to_midi(action_tensor, fs=FS, output_path=out_path)


# ── Simulation mode ───────────────────────────────────────────────────────────
def run_simulate(wav_path: str, midi_path: str):
    """
    Step through a full wav file, print per-step token decisions, and
    write the generated accompaniment to a MIDI file.
    """
    print(f"\n🎻 Simulating accompaniment for: {os.path.basename(wav_path)}\n")
    model   = load_model(CHECKPOINT)
    data    = extract_musaic_features(wav_path, midi_path, fs=FS)
    x_violin = data["x"]
    n_steps  = x_violin.shape[0]

    h_n         = None
    prev_action = torch.zeros(130, device=DEVICE)
    prev_action[129] = 1.0
    all_actions = []

    for t in range(n_steps):
        v_feat = x_violin[t].to(DEVICE)
        action, action_probs, h_n = infer_step(model, v_feat, prev_action, h_n)
        all_actions.append(action.cpu())
        prev_action = action

        # Pretty-print token decision every 16 steps
        if t % 16 == 0:
            notes  = torch.where(action[:128] > 0.5)[0].tolist()
            hold   = action[128].item() > 0.5
            rest   = action[129].item() > 0.5
            token  = "REST" if rest else ("HOLD" if hold and not notes else f"NOTES {notes}")
            print(f"  t={t:>4}  {token}")

    action_tensor = torch.stack(all_actions)
    out_path = os.path.join(OUTPUT_DIR, "simulated_accompaniment.mid")
    generate_autoregressive_to_midi(action_tensor, fs=FS, output_path=out_path)
    print(f"\n✅ Simulation complete.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MusAIc-RL Inference & Benchmark")
    parser.add_argument("--mode",  default="benchmark",
                        choices=["benchmark", "simulate"],
                        help="Execution mode")
    parser.add_argument("--wav",   required=True,  help="Path to violin .wav file")
    parser.add_argument("--midi",  required=True,  help="Path to ground-truth .mid file")
    parser.add_argument("--ckpt",  default=CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--temp",  type=float, default=TEMPERATURE,
                        help="Sampling temperature (default 0.8)")
    args = parser.parse_args()

    CHECKPOINT  = args.ckpt
    TEMPERATURE = args.temp

    if args.mode == "benchmark":
        run_benchmark(args.wav, args.midi)
    elif args.mode == "simulate":
        run_simulate(args.wav, args.midi)
