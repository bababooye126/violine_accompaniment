
import torch
import torch.nn as nn
from utils.rewards import reward_intra, reward_inter, calculate_roughness_kernel


class MusaicRL(nn.Module):
    def __init__(self, input_size=143, hidden_size=256, output_size=130):
        super(MusaicRL, self).__init__()
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
        last_step_hidden = output[:, -1, :] 
        return self.actor(last_step_hidden), self.critic(last_step_hidden), h_n

    def get_reward(self, violin_stft, current_action, prev_action):
        """
        Calculates the 4-part ensemble reward.
        """
    
        r_intra = reward_intra(current_action, prev_action)
        r_inter = reward_inter(violin_stft, current_action)
        
        if current_action[129] < 0.5: 
            r_rough = self.calculate_total_roughness(violin_stft, current_action[:128])
        else:
            r_rough = torch.tensor(0.0, device=current_action.device)
            
        r_temp = self.reward_temporal(current_action, prev_action)
        
        # Combine: Penalize roughness with a tuning weight (e.g., 0.5)
        total_reward = r_intra + r_inter + r_temp - (r_rough * 0.5) 
        return total_reward.reshape(1)

    def calculate_total_roughness(self, violin_stft, piano_pitches):
        # Find active MIDI notes
        active_midi = torch.where(piano_pitches > 0.5)[0]
        if len(active_midi) == 0:
            return torch.tensor(0.0, device=piano_pitches.device)
            
        violin_max_bin = torch.argmax(violin_stft).item()
        violin_freq = violin_max_bin * (22050 / 2048)
        
        total_roughness = 0.0
        
        # Loop through active notes and apply your kernel
        for m in active_midi:
            # Convert MIDI number to Hz
            p_freq = 440.0 * (2.0 ** ((m.item() - 69.0) / 12.0))
            
            # Call your provided scalar function
            total_roughness += calculate_roughness_kernel(violin_freq, p_freq)
            
        return torch.tensor(total_roughness, dtype=torch.float32, device=piano_pitches.device)

    def reward_temporal(self, current, prev):
        reward = 0.0
        hold_token = current[128]
        rest_token = current[129]
        active_pitches = current[:128]
        prev_pitches = prev[:128]
        
        if hold_token > 0.5 and torch.sum(prev_pitches) == 0:
            reward -= 0.5 # Penalize holding nothing
        if rest_token > 0.5 and torch.sum(active_pitches) > 0:
            reward -= 1.0 # Penalize playing while resting
        if hold_token > 0.5 and torch.sum(prev_pitches) > 0:
            reward += 0.2 # Reward a valid sustain
            
        return torch.tensor(reward, device=current.device)