
import librosa
import pretty_midi
import numpy as np
import torch
from utils.midi_tools import encode_roll


def extract_musaic_features(wav_path, midi_path, fs=8):

    y_audio, sr = librosa.load(wav_path)
    hop_length = int(sr / fs)
    
    chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr, hop_length=hop_length)
    
    rms = librosa.feature.rms(y=y_audio, hop_length=hop_length)
    log_rms = np.log(rms + 1e-6) 
    
    stft = np.abs(librosa.stft(y_audio, hop_length=hop_length))

    pm = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = pm.get_piano_roll(fs=fs)
    
    min_len = min(chroma.shape[1], piano_roll.shape[1], stft.shape[1])
    
    x_combined = np.vstack([chroma[:, :min_len], log_rms[:, :min_len]]).T
    
    return {
        "x": torch.FloatTensor(x_combined),          # (Steps, 13)
        "x_stft": torch.FloatTensor(stft[:, :min_len].T), # (Steps, 1025)
        "y": torch.FloatTensor(piano_roll[:, :min_len].T)  # (Steps, 128)
    }
    
    
def extract_musaic_features_autoregressive(wav_path, midi_path, fs=8):
    # Get standard audio features (Chroma + RMS)
    raw_data = extract_musaic_features(wav_path, midi_path, fs)
    
    # Get the 130-bin ground truth
    pm = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = pm.get_piano_roll(fs=fs)
    p_encoded = encode_roll(piano_roll, fs)
    
    # Align lengths
    min_len = min(raw_data['x'].shape[0], p_encoded.shape[0])
    v_features = raw_data['x'][:min_len] # (Steps, 13)
    p_targets = p_encoded[:min_len]      # (Steps, 130)
    
    # Create Machine Context: Shift targets by 1 (What did I play last?)
    machine_context = torch.roll(p_targets, shifts=1, dims=0)
    machine_context[0, :] = 0
    machine_context[0, 129] = 1 # Start with a Rest
    
    # Final Input: Human(13) + Machine(130) = 143
    x_combined = torch.cat([v_features, machine_context], dim=-1)
    
    return x_combined, p_targets, raw_data['x_stft'][:min_len]