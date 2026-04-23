import numpy as np
import torch

def encode_roll(roll, fs=8):
    """
    Converts a (128, Steps) piano roll into a (Steps, 130) matrix.
    Bins 0-127: Pitch Start (Velocity/On)
    Bin 128: Hold (Continue previous note)
    Bin 129: Rest (Silence)
    """
    steps = roll.shape[1]
    encoded = np.zeros((steps, 130))
    
    prev_active = set()
    
    for t in range(steps):
        current_active = set(np.where(roll[:, t] > 0)[0])
        
        if not current_active:
            encoded[t, 129] = 1  # Rest
        else:
            # Check for new notes (Onsets)
            new_notes = current_active - prev_active
            for note in new_notes:
                encoded[t, note] = 1
            
            # If there are notes continuing from previous step
            if current_active & prev_active:
                encoded[t, 128] = 1 # Hold
                
        prev_active = current_active
            
    return torch.FloatTensor(encoded)

