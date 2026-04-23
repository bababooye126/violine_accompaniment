import math
import torch

def calculate_roughness_kernel(f1, f2, a1=1.0, a2=1.0):
    """
    Parametrization of Plomp-Levelt curve (Sethares model).
    Works with both Tensors and Floats.
    """
    # Convert to float if they are tensors
    f1 = float(f1)
    f2 = float(f2)
    
    f_min = min(f1, f2)
    f_max = max(f1, f2)
    
    # Parameters from Sethares (1993)
    s1, s2 = 0.0207, 18.96
    b1, b2 = 3.5, 5.75
    
    # Calculate s (Scaling factor for critical bandwidth)
    s = 0.24 / (s1 * f_min + s2)
    
    f_diff = f_max - f_min
    # Use Python's built-in min for raw floats
    amp_weight = min(a1, a2) 
    
    # Use math.exp for scalar speed and stability
    roughness = amp_weight * (math.exp(-b1 * s * f_diff) - math.exp(-b2 * s * f_diff))
    
    return roughness

def reward_intra(current_p, prev_p):
    c_p = current_p.reshape(-1)
    p_p = prev_p.reshape(-1)
    
    curr_idx = torch.where(c_p > 0.5)[0]
    prev_idx = torch.where(p_p > 0.5)[0]
    
    if curr_idx.numel() == 0 or prev_idx.numel() == 0:
        return torch.tensor(-0.1, device=c_p.device)
        
    jump = torch.abs(curr_idx.float().mean() - prev_idx.float().mean())
    if jump <= 12:
        reward = 0.5 
    elif jump % 12 == 0:
        reward = 0.3 
    elif jump % 7 == 0:
        reward = 0.1  
    else:
        reward = -0.1 * (jump - 12) 
        
    return torch.as_tensor(reward, device=c_p.device)

def reward_inter(violin_stft, piano_midi):
  
    v_s = violin_stft.reshape(-1)
    p_m = piano_midi.reshape(-1)
  
    v_peak_chroma = torch.argmax(v_s) % 12
    p_notes_chroma = torch.where(p_m > 0.5)[0] % 12
    
    if p_notes_chroma.numel() == 0:
        return torch.tensor(-0.1, device=p_m.device)
    
    matches = (p_notes_chroma == v_peak_chroma).sum().float()
    return (matches / p_notes_chroma.numel()).to(p_m.device)


'''
def reward_spectral(violin_stft, piano_midi):
    v_s = violin_stft.reshape(-1)
    p_m = piano_midi.reshape(-1)
    
    v_freq = torch.argmax(v_s).float() * (22050.0 / 1024.0)
    p_indices = torch.where(p_m > 0.5)[0]
    
    if p_indices.numel() == 0:
        return torch.tensor(-0.05, device=p_m.device)
        
    dissonance = 0.0
    for p in p_indices:
        p_freq = 440.0 * (2.0 ** ((p.float() - 69.0) / 12.0))
        # This now uses the new math-based kernel
        dissonance += calculate_roughness_kernel(v_freq, p_freq)
        
    return torch.tensor(-dissonance, device=p_m.device, dtype=torch.float32)'''