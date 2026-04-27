"""
train.py — MusAIc-RL A2C Training Loop v3
==========================================
Key fixes over v2:
  1. Reward standardization via running mean/std (not hard clamping)
  2. Critic warmup phase (first N_WARMUP epochs, actor frozen)
  3. GAE (lambda=0.95) instead of TD(0) — lower variance advantages
  4. Separate optimizers for actor and critic with different LRs
  5. Removed reward hard-clamp — replaced with z-score normalization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from models import MusaicRL
from utils.audio_tools import extract_musaic_features_autoregressive, get_validated_pairs

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS          = 50
LR_ACTOR        = 1e-4
LR_CRITIC       = 5e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
ENTROPY_BETA    = 0.005
CRITIC_WEIGHT   = 0.5
ROLLOUT_LEN     = 32
GRAD_CLIP       = 0.5
N_WARMUP        = 3           # Epochs where ONLY the critic trains
CHECKPOINT_EVERY= 10
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR     = "Data/Dataset/Synthesized Piano-Violin Duet"
CHECKPOINT_DIR  = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Running reward statistics (Welford online algorithm) ───────────────────────
class RunningStats:
    def __init__(self, eps=1e-8):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0; self.eps = eps

    def update(self, x):
        x = float(x); self.n += 1
        delta = x - self.mean; self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self):
        return max((self.M2 / (self.n - 1)) ** 0.5, self.eps) if self.n >= 2 else 1.0

    def normalize(self, x):
        return (float(x) - self.mean) / self.std

reward_stats = RunningStats()

class RollingMean:
    def __init__(self): self.total = 0.0; self.n = 0
    def update(self, v): self.total += float(v); self.n += 1
    def mean(self): return self.total / self.n if self.n else 0.0

# ── Data extraction ────────────────────────────────────────────────────────────
print("=" * 60)
print("  MusAIc-RL — A2C Training  v3")
print("=" * 60)
print(f"Device : {DEVICE}  |  Warmup epochs: {N_WARMUP}")

paired_paths = get_validated_pairs(DATASET_DIR)
print(f"\n🎻 Extracting features for {len(paired_paths)} pairs …\n")

song_data = []
for wav, midi in paired_paths:
    try:
        x, y, stft = extract_musaic_features_autoregressive(wav, midi)
        song_data.append((x.to(DEVICE), y.to(DEVICE), stft.to(DEVICE)))
    except Exception as e:
        print(f"  ⚠️  Skipping {os.path.basename(wav)}: {e}")

print(f"\n✅ {len(song_data)} songs loaded onto {DEVICE}\n")

# ── Model & separate optimizers ────────────────────────────────────────────────
model = MusaicRL(input_size=143, hidden_size=256, output_size=130).to(DEVICE)

actor_params  = list(model.backbone.parameters()) + list(model.actor.parameters())
critic_params = list(model.critic.parameters())

opt_actor  = optim.Adam(actor_params,  lr=LR_ACTOR)
opt_critic = optim.Adam(critic_params, lr=LR_CRITIC)

sched_actor  = optim.lr_scheduler.CosineAnnealingLR(opt_actor,  T_max=max(1, EPOCHS - N_WARMUP))
sched_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=EPOCHS)

print(f"Parameters — total: {sum(p.numel() for p in model.parameters()):,}  "
      f"actor: {sum(p.numel() for p in actor_params):,}  "
      f"critic: {sum(p.numel() for p in critic_params):,}\n")

# ── GAE ────────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, next_value):
    T = len(rewards)
    advantages = torch.zeros(T, device=DEVICE)
    gae = 0.0
    for t in reversed(range(T)):
        nv    = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv.detach() - values[t].detach()
        gae   = delta + GAMMA * GAE_LAMBDA * gae
        advantages[t] = gae
    returns = advantages + values.detach()
    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    warmup = epoch <= N_WARMUP

    # Freeze/unfreeze actor
    for p in actor_params:
        p.requires_grad_(not warmup)

    m_raw = RollingMean(); m_rough = RollingMean()
    m_al  = RollingMean(); m_cl    = RollingMean(); m_ent = RollingMean()

    for song_idx in torch.randperm(len(song_data)).tolist():
        x_song, _, stft_song = song_data[song_idx]
        T_song = x_song.shape[0]
        h_n = None

        buf_lp=[]; buf_val=[]; buf_rew=[]; buf_ent=[]
        prev_action = torch.zeros(130, device=DEVICE)
        prev_action[129] = 1.0

        for t in range(T_song):
            x_t = x_song[t].unsqueeze(0).unsqueeze(0)
            action_probs, value, h_n = model(x_t, h_n)
            action_probs = action_probs.squeeze(0)
            value        = value.squeeze()

            dist     = torch.distributions.Bernoulli(probs=action_probs)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum()
            entropy  = dist.entropy().sum()

            with torch.no_grad():
                raw_r = model.get_reward(stft_song[t], action_probs, prev_action)
                rough = model.calculate_total_roughness(stft_song[t], action_probs[:128])

            m_raw.update(raw_r.item())
            m_rough.update(rough.item())
            reward_stats.update(raw_r.item())
            norm_r = torch.tensor(reward_stats.normalize(raw_r.item()),
                                  dtype=torch.float32, device=DEVICE)

            buf_lp.append(log_prob); buf_val.append(value)
            buf_rew.append(norm_r);  buf_ent.append(entropy)
            prev_action = action.detach()

            is_last  = (t == T_song - 1)
            if len(buf_lp) == ROLLOUT_LEN or is_last:
                if is_last:
                    nv = torch.tensor(0.0, device=DEVICE)
                else:
                    with torch.no_grad():
                        _, nv, _ = model(x_song[t+1].unsqueeze(0).unsqueeze(0), h_n)
                        nv = nv.squeeze()

                lp  = torch.stack(buf_lp)
                vs  = torch.stack(buf_val)
                rws = torch.stack(buf_rew)
                ent = torch.stack(buf_ent)

                adv, ret = compute_gae(rws, vs, nv)
                a_loss = -(lp * adv.detach()).mean()
                c_loss = nn.functional.mse_loss(vs, ret.detach())
                e_loss = -ent.mean()

                # Critic always updates
                opt_critic.zero_grad()
                (CRITIC_WEIGHT * c_loss).backward(retain_graph=not warmup)
                nn.utils.clip_grad_norm_(critic_params, GRAD_CLIP)
                opt_critic.step()

                # Actor only after warmup
                if not warmup:
                    opt_actor.zero_grad()
                    (a_loss + ENTROPY_BETA * e_loss).backward()
                    nn.utils.clip_grad_norm_(actor_params, GRAD_CLIP)
                    opt_actor.step()

                m_al.update(a_loss.item())
                m_cl.update(c_loss.item())
                m_ent.update(-e_loss.item())

                if h_n is not None:
                    h_n = h_n.detach()
                buf_lp.clear(); buf_val.clear(); buf_rew.clear(); buf_ent.clear()

        torch.cuda.empty_cache()

    sched_critic.step()
    if not warmup:
        sched_actor.step()

    tag = "  ← critic warmup" if warmup else ""
    print(
        f"Epoch [{epoch:>3}/{EPOCHS}]  "
        f"RawRew: {m_raw.mean():+.4f}  "
        f"Roughness: {m_rough.mean():.4f}  "
        f"Actor L: {m_al.mean():.5f}  "
        f"Critic L: {m_cl.mean():.4f}  "
        f"Entropy: {m_ent.mean():.4f}{tag}"
    )

    if epoch % CHECKPOINT_EVERY == 0:
        path = os.path.join(CHECKPOINT_DIR, f"musaic_rl_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "opt_actor": opt_actor.state_dict(), "opt_critic": opt_critic.state_dict(),
            "reward_stats": {"mean": reward_stats.mean, "M2": reward_stats.M2, "n": reward_stats.n},
            "mean_raw_reward": m_raw.mean(), "mean_roughness": m_rough.mean(),
        }, path)
        print(f"  💾 Checkpoint → {path}")

print("\n✅ Training complete.")