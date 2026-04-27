"""
utils/midi_tools.py
====================
130-bin piano roll encoder and autoregressive MIDI decoder.
"""

import numpy as np
import torch
import pretty_midi


def encode_roll(roll, fs=8):
    """
    Converts a (128, Steps) piano roll into a (Steps, 130) matrix.

    Bin semantics:
        0–127 : Onset — this MIDI pitch starts in this frame
        128   : Hold  — at least one note from the previous frame continues
        129   : Rest  — complete silence
    """
    steps = roll.shape[1]
    encoded = np.zeros((steps, 130))
    prev_active = set()

    for t in range(steps):
        current_active = set(np.where(roll[:, t] > 0)[0])

        if not current_active:
            encoded[t, 129] = 1  # Rest
        else:
            new_notes = current_active - prev_active
            for note in new_notes:
                encoded[t, note] = 1
            if current_active & prev_active:
                encoded[t, 128] = 1  # Hold

        prev_active = current_active

    return torch.FloatTensor(encoded)


# Alias used in utils/__init__.py
encode_roll_130 = encode_roll


def generate_autoregressive_to_midi(
    action_sequence,
    fs=8,
    program=0,
    velocity=80,
    output_path="outputs/generated.mid",
):
    """
    Converts a (Steps, 130) action tensor / numpy array produced by the
    autoregressive model back into a playable MIDI file.

    Args:
        action_sequence : (Steps, 130) float tensor or ndarray.
                          Values are thresholded at 0.5.
        fs              : Frame rate used during feature extraction (default 8 Hz).
        program         : General MIDI program number (0 = Acoustic Grand Piano).
        velocity        : Fixed MIDI velocity for all note-ons.
        output_path     : Path to write the .mid file.

    Returns:
        pretty_midi.PrettyMIDI object (also written to output_path).

    Decoding rules
    --------------
    Bin 129 active  → silence; end any open note.
    Bin 128 active  → sustain; keep all currently open notes ringing.
    Bins 0–127 active → onset; end any open note on that pitch, then open a new one.
    If bin 128 is active alongside new onsets, the sustained notes keep ringing
    and the new pitches are added (chord extension).
    """
    if isinstance(action_sequence, torch.Tensor):
        actions = (action_sequence.detach().cpu().numpy() > 0.5).astype(np.int32)
    else:
        actions = (np.array(action_sequence) > 0.5).astype(np.int32)

    frame_duration = 1.0 / fs          # seconds per frame
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name="AI Piano")

    # Track open notes: pitch → start_time
    open_notes: dict[int, float] = {}

    def close_note(pitch, end_time):
        if pitch in open_notes:
            start = open_notes.pop(pitch)
            if end_time > start:
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start,
                        end=end_time,
                    )
                )

    for t, frame in enumerate(actions):
        t_start = t * frame_duration
        t_end   = (t + 1) * frame_duration

        rest_active  = bool(frame[129])
        hold_active  = bool(frame[128])
        onset_pitches = [p for p in range(128) if frame[p]]

        if rest_active:
            # Close all open notes
            for pitch in list(open_notes.keys()):
                close_note(pitch, t_start)
            continue

        if hold_active and not onset_pitches:
            # Pure sustain: let open notes ring, don't open new ones
            continue

        # Process onsets (new note-ons)
        for pitch in onset_pitches:
            # If this pitch was already ringing, close it and re-trigger
            close_note(pitch, t_start)
            open_notes[pitch] = t_start

    # Close any notes still open at the end of the sequence
    final_time = len(actions) * frame_duration
    for pitch in list(open_notes.keys()):
        close_note(pitch, final_time)

    pm.instruments.append(instrument)
    pm.write(output_path)
    print(f"  🎹 MIDI written → {output_path}  ({len(instrument.notes)} notes)")
    return pm
