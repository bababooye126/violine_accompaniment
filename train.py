import torch
from utils.audio_tools import extract_musaic_features_autoregressive,get_validated_pairs

all_x = []      
all_y = []    
all_stft = []   

paired_paths = get_validated_pairs("Data/Dataset/Synthesized Piano-Violin Duet")
print(f"🚀 Starting Autoregressive Feature Extraction for the pairs...")

for wav, midi in paired_paths:
    try:
        # 1. Use the new function and unpack the tuple directly
        x_combined, p_targets, stft = extract_musaic_features_autoregressive(wav, midi)
        
        # 2. Append directly to master lists
        all_x.append(x_combined)      
        all_y.append(p_targets)      
        all_stft.append(stft)
        
    except Exception as e:
        import os # Ensure os is imported for basename
        print(f"❌ Error processing {os.path.basename(wav)}: {e}")

# Concatenate all songs into unified master tensors
final_x = torch.cat(all_x, dim=0)    
final_y = torch.cat(all_y, dim=0)    
final_stft = torch.cat(all_stft, dim=0)

print("-" * 30)
print(f"✅ AUTOREGRESSIVE EXTRACTION COMPLETE")
# The shapes printed here will help you verify the autoregressive logic worked
print(f"Final Input Tensor (X):  {final_x.shape}  (Human(13) + Machine(130) = 143 bins)")
print(f"Final Target Tensor (Y): {final_y.shape} (128 Pitches + Hold + Rest = 130 bins)")
print(f"Final Spectral Tensor:   {final_stft.shape} (For Roughness Kernel)")