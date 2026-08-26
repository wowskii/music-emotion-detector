import os
import numpy as np
import librosa as lb
from midiutil import MIDIFile
from utilities import *

HOP_LENGTH = 512


def analyze_song(path, key=('A', 'major'), key_bias=True, keep_top=3):
    """
    Return the harmonic representation plus the decoded chord sequence.

    Intermediate stage:
    - chroma: array with shape (12, T), one pitch class vector per time frame
    - chord_labels: list of length T, one chord label per frame
    - times: list of frame timestamps in seconds, one per frame
    """
    y, sr = lb.load(path)
    y_harm, _ = lb.effects.hpss(y)

    chroma = lb.feature.chroma_cens(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    chroma = process_chroma(chroma, keep_top=keep_top)

    trans = lb.sequence.transition_loop(84, 0.5)
    key_bias_vec = key_bias_vector(*key)

    probs = np.exp(weights.dot(chroma))
    if key_bias:
        probs *= key_bias_vec[:, None]
    probs /= probs.sum(axis=0, keepdims=True)

    path_indices = lb.sequence.viterbi_discriminative(probs, trans)
    chord_labels = [labels[i] for i in path_indices]
    times = lb.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    return y, sr, times, chord_labels, chroma


def group_chord_labels_into_events(chord_labels, times, min_duration=0.15):
    """
    Convert the per-frame chord list into a clearer intermediate form:
    a list of chord events, each with start/end time and the chord label.

    Example:
        [
            {'label': 'C:maj', 'start': 0.0, 'end': 1.4},
            {'label': 'G:maj', 'start': 1.4, 'end': 2.8},
        ]

    This is the stage between frame-wise labels and final MIDI note events.
    """
    if len(chord_labels) == 0:
        return []

    events = []
    current_label = chord_labels[0]
    current_start = times[0]

    for i in range(1, len(chord_labels)):
        label = chord_labels[i]
        t = times[i]

        if label == current_label:
            continue

        duration = t - current_start
        if duration >= min_duration:
            events.append({
                'label': current_label,
                'start': current_start,
                'end': t,
            })

        current_label = label
        current_start = t

    final_duration = times[-1] - current_start
    if final_duration >= min_duration:
        events.append({
            'label': current_label,
            'start': current_start,
            'end': times[-1],
        })

    return events


def chord_label_to_midi_notes(label):
    """Map a chord label like 'C:maj' to the MIDI note numbers for its notes."""
    if label == 'N':
        return []

    root, chord_type = label.split(':')
    root_index = NOTE_NAMES.index(root)
    intervals = CHORD_INTERVALS.get(chord_type, [])
    root_midi = 60 + root_index

    return [root_midi + interval for interval in intervals]


def export_chords_to_midi(
    audio_path,
    output_path,
    key=('A', 'major'),
    key_bias=True,
    keep_top=3,
    min_duration=0.15,
    bpm=120,
    velocity=80,
    channel=0,
    beats_per_bar=4,
):
    """
    Analyze an audio file, quantize the resulting chord events to a bar grid,
    convert the Viterbi chord labels into note events, and save a MIDI file.

    Returns a dictionary with the clear intermediate stages:
    {
        'frame_labels': [...],   # one label per audio frame
        'events': [
            {'label': 'C:maj', 'start': 0.0, 'end': 2.0},
            ...
        ],
        'midi_path': 'output.mid'
    }
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _, sr, times, chord_labels, _ = analyze_song(
        audio_path,
        key=key,
        key_bias=key_bias,
        keep_top=keep_top,
    )

    events = group_chord_labels_into_events(chord_labels, times, min_duration=min_duration)
    # events = quantize_chord_events(events, bpm=bpm, beats_per_bar=beats_per_bar)

    midi = MIDIFile(1)
    midi.addTrackName(0, 0, 'Chord transcription')
    midi.addTempo(0, 0, bpm)

    for event in events:
        note_numbers = chord_label_to_midi_notes(event['label'])
        if not note_numbers:
            continue

        start_beats = event['start'] * bpm / 60.0
        end_beats = event['end'] * bpm / 60.0
        dur_beats = max(end_beats - start_beats, 1.0 / 32.0)

        for note_num in note_numbers:
            midi.addNote(
                track=0,
                channel=channel,
                pitch=note_num,
                time=start_beats,
                duration=dur_beats,
                volume=velocity,
            )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    return {
        'frame_labels': chord_labels,
        'events': events,
        'midi_path': output_path,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export a chord transcription MIDI from an audio file.')
    parser.add_argument('audio_path', help='Path to the input audio file')
    parser.add_argument('output_path', help='Output MIDI filename')
    parser.add_argument('--key', nargs=2, default=('A', 'major'), metavar=('ROOT', 'MODE'))
    parser.add_argument('--no-key-bias', action='store_true')
    parser.add_argument('--keep-top', type=int, default=3)
    parser.add_argument('--min-duration', type=float, default=0.15)
    parser.add_argument('--bpm', type=float, default=120.0)
    args = parser.parse_args()

    key_bias = not args.no_key_bias
    export_chords_to_midi(
        args.audio_path,
        args.output_path,
        key=(args.key[0], args.key[1]),
        key_bias=key_bias,
        keep_top=args.keep_top,
        min_duration=args.min_duration,
        bpm=args.bpm,
    )
    print(f"Wrote MIDI to {args.output_path}")