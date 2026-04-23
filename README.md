# MusAIc-RL: Violin Accompaniment Generator

An autoregressive reinforcement learning system that generates expressive piano accompaniments for solo violin in real time.

---

## Overview

MusAIc-RL frames piano accompaniment as a sequential decision-making problem. At each time step the model observes the current violin audio, its own previous output, and selects the next piano action (a chord, a hold, or a rest). A multi-component reward signal steers the model toward accompaniments that are harmonically coherent, physically smooth, and perceptually consonant.

---

## Architecture

### Model — `models/musaic_rl.py`

`MusaicRL` is a bidirectional GRU actor-critic network.

- **Backbone** — bidirectional GRU (input 143-dim → hidden 256-dim per direction).
- **Actor head** — two-layer MLP with Sigmoid output producing a 130-bin probability vector (128 pitch bins + hold + rest).
- **Critic head** — two-layer MLP producing a scalar value estimate.

Input dimension breakdown: 13 violin features (chroma 12 + log-RMS 1) concatenated with 130 machine-context features (the previous time step's own output), totalling 143 bins.

### Reward Function

Four components are combined at each step:

| Component | Function | Description |
|-----------|----------|-------------|
| `reward_intra` | melodic smoothness | Rewards small intervallic jumps (≤ semitone 12), octave leaps, and fifth leaps; penalises large random jumps |
| `reward_inter` | chroma agreement | Fraction of active piano pitches whose chroma class matches the dominant chroma of the violin STFT peak |
| `reward_temporal` | token logic | Rewards valid hold/sustain; penalises holding silence or sounding notes while the rest token is active |
| roughness penalty | psychoacoustic dissonance | Sethares–Plomp-Levelt kernel computed between the violin's peak frequency and each active piano pitch |

Total reward: `r_intra + r_inter + r_temporal − 0.5 × r_rough`

### Roughness Kernel — `utils/rewards.py`

Implements the Sethares (1993) parametrisation of the Plomp-Levelt roughness curve. For two frequencies `f1` and `f2` with amplitudes `a1` and `a2`:

```
s    = 0.24 / (0.0207 × f_min + 18.96)
R(Δf) = min(a1, a2) × [exp(−3.5 × s × Δf) − exp(−5.75 × s × Δf)]
```

The kernel operates on scalar floats for efficiency and is called per active piano pitch.

---

## Data Pipeline

### Feature Extraction — `utils/audio_tools.py`

`extract_musaic_features(wav, midi)` produces:

- **x** — (Steps, 13) chroma + log-RMS tensor.
- **x_stft** — (Steps, 1025) magnitude STFT for reward computation.
- **y** — (Steps, 128) raw piano roll.

`extract_musaic_features_autoregressive(wav, midi)` extends this for training:

1. Encodes the piano roll into the 130-bin vocabulary (see below).
2. Creates a **machine context** by shifting the target one step forward (teacher forcing): the model sees what it "played" at `t−1`.
3. Returns `x_combined` (Steps, 143), `p_targets` (Steps, 130), and the STFT tensor.

### 130-bin MIDI Encoding — `utils/midi_tools.py`

`encode_roll(roll, fs)` converts a standard (128, Steps) piano roll into a (Steps, 130) matrix with the following semantics:

- Bins 0–127 — onset of that MIDI pitch in this frame.
- Bin 128 — hold: at least one note from the previous frame is continuing.
- Bin 129 — rest: no notes are active.

---

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   └── musaic_rl.py          # GRU actor-critic + reward methods
├── utils/
│   ├── __init__.py
│   ├── audio_tools.py        # Librosa feature extraction + autoregressive pipeline
│   ├── midi_tools.py         # 130-bin piano roll encoder / MIDI generator
│   └── rewards.py            # Plomp-Levelt kernel + intra/inter reward functions
├── train.py                  # Training entry point
└── test.py                   # Evaluation entry point
```

---

## Getting Started

### Requirements

Install dependencies (Python 3.9+ recommended):

```bash
pip install torch librosa pretty_midi numpy
```

### Data

Place paired `.wav` / `.midi` files in a `Data/` directory (excluded from version control via `.gitignore`). The training script expects a `paired_paths` list of `(wav_path, midi_path)` tuples.

### Feature Extraction

```python
from utils.audio_tools import extract_musaic_features_autoregressive

x_combined, p_targets, stft = extract_musaic_features_autoregressive(wav_path, midi_path)
# x_combined: (Steps, 143)
# p_targets:  (Steps, 130)
# stft:       (Steps, 1025)
```

### Training

```bash
python train.py
```

The script iterates over all paired paths, extracts autoregressive features, concatenates them into master tensors, and prints shape diagnostics on completion.

---

## Key Design Decisions

**Autoregressive input** — feeding the model's own previous output as part of the input enables it to reason about musical continuity without an explicit sequence memory beyond the GRU hidden state.

**Hold / rest tokens** — distinguishing an onset (bin 0–127), a sustained note (bin 128), and silence (bin 129) lets the model learn legato phrasing and natural phrase endings rather than re-triggering every active note each frame.

**Scalar roughness kernel** — using Python `math.exp` instead of `torch` operations for the per-pitch roughness loop keeps the reward fast and avoids unnecessary GPU round-trips for a scalar accumulator.

**Bidirectional GRU** — using bidirectional encoding over the input window gives the actor a richer context representation before committing to an action, at the cost of a slight latency window during inference.

---

## Output Format

The actor outputs a 130-dimensional sigmoid vector. At inference time, threshold at 0.5 to decode active events:

- Active bins in 0–127 → note onsets to trigger.
- Bin 128 active → sustain/hold previous notes.
- Bin 129 active → rest / silence.

Use `utils/midi_tools.generate_autoregressive_to_midi` to convert a sequence of these action vectors back to a MIDI file.