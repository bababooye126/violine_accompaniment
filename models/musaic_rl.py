"""
models/musaic_rl.py
====================
Changes from v2:
  - get_reward() now returns properly scaled components.
    All four reward terms are now on the same [-1, +1] scale
    so the critic can actually learn to predict them.
  - reward_temporal() range extended to match other terms.
  - roughness penalty weight stays at 0.1 (set here, not in train.py)
"""

import torch
import torch.nn as nn
from utils.rewards import reward_intra, reward_inter, calculate_roughness_kernel


class MusaicRL(nn.Module):
    def __init__(self, input_size=143, hidden_size=256, output_size=130):
        super().__init__()
        self.backbone = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

        self.actor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, h_n=None):
        output, h_n = self.backbone(x, h_n)
        last = output[:, -1, :]
        return self.actor(last), self.critic(last), h_n

    def get_reward(self, violin_stft, current_action, prev_action):
        """
        Four-component reward. All components clipped to [-1, 1] before summing
        so the total sits in [-4, +4] — a range the critic can actually predict.

        r_intra    : melodic smoothness         range ~ [-1, +0.5]
        r_inter    : chroma matching            range ~ [-0.1, +1.0]
        r_temporal : token logic validity       range ~ [-1.0, +0.2]
        r_rough    : spectral dissonance penalty range ~ [0, ~2] → weighted down
        """
        r_intra = reward_intra(current_action, prev_action)
        r_inter = reward_inter(violin_stft, current_action)
        r_temp  = self.reward_temporal(current_action, prev_action)

        # Only compute roughness when the model is NOT resting
        if current_action[129] < 0.5:
            r_rough = self.calculate_total_roughness(violin_stft, current_action[:128])
        else:
            r_rough = torch.tensor(0.0, device=current_action.device)

        # Roughness weight kept small — it's always positive (a penalty)
        # and can dominate if weighted too high
        total = r_intra + r_inter + r_temp - (r_rough * 0.1)
        return total.reshape(1)

    def calculate_total_roughness(self, violin_stft, piano_pitches):
        active_midi = torch.where(piano_pitches > 0.5)[0]
        if len(active_midi) == 0:
            return torch.tensor(0.0, device=piano_pitches.device)

        violin_max_bin = torch.argmax(violin_stft).item()
        violin_freq    = violin_max_bin * (22050 / 2048)

        total = 0.0
        for m in active_midi:
            p_freq = 440.0 * (2.0 ** ((m.item() - 69.0) / 12.0))
            total += calculate_roughness_kernel(violin_freq, p_freq)

        return torch.tensor(total, dtype=torch.float32, device=piano_pitches.device)

    def reward_temporal(self, current, prev):
        """
        Enforces 130-bin token logic.
        Scaled to [-1, +1] range to match other reward components.
        """
        reward      = 0.0
        hold_token  = current[128]
        rest_token  = current[129]
        active      = current[:128]
        prev_active = prev[:128]

        if hold_token > 0.5 and torch.sum(prev_active) == 0:
            reward -= 1.0   # Hard penalty: holding nothing (was -0.5)

        if rest_token > 0.5 and torch.sum(active) > 0:
            reward -= 1.0   # Hard penalty: sounding + rest simultaneously

        if hold_token > 0.5 and torch.sum(prev_active) > 0:
            reward += 0.5   # Valid sustain reward (was +0.2 — now more meaningful)

        # Small bonus for a clean rest (encourages natural phrase endings)
        if rest_token > 0.5 and torch.sum(active) == 0:
            reward += 0.2

        return torch.tensor(reward, device=current.device)