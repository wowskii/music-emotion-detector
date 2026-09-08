import os
import numpy as np
import librosa as lb
from midiutil import MIDIFile
from utilities import *

HOP_LENGTH = 512


def process_audio_file(audio_path: str, bpm=None, perc_path=None, key=(None, None), use_quantize=True, separated=False):
    """Exctract the chords from any audio file and convert them to midi
    This function's goal is to handle all the different use cases.
    
    Keyword arguments:
    argument -- description
    Return: a midi file
    """

    if not separated :
        pathnumber = 0
        if not os.path.exists(f"data/separated/output{pathnumber}"):
            os.makedirs(f"data/separated/output{pathnumber}")
        separated_path = f"data/separated/output{pathnumber}"
        print("Separating song stems. This may take a few minutes.")
        separate_song_stems(audio_path, f"data/separated/output{pathnumber}")
        perc_path = os.path.join(separated_path, "drums.wav")
        chords_path = os.path.join(separated_path, "other.wav")
        #we keep separated and perc_path for the case where the audio has no percussion (separated=True and perc_path==None)
    else:
        chords_path = audio_path

    beat_times, _, detected_bpm = get_beat_info(
        perc_path if perc_path is not None else audio_path,
        bpm=bpm,
    )
    if bpm is None:
        bpm = detected_bpm
        print(f"Found BPM: {bpm}")


    chroma, times = get_chromagram(chords_path)
    print("Got chromagram")
    trans = lb.sequence.transition_loop(84, 0.5)

    #detect chords keyless
    probs = np.exp(weights.dot(chroma))

    if key != (None, None): #detect chords with key
        key_bias_vec = key_bias_vector(*key)
        probs *= key_bias_vec[:, None]

    probs /= probs.sum(axis=0, keepdims=True)

    #viterbi's chosen chord sequence: an array of indices into the labels list, one per frame
    print("Running Viterbi discriminative...")
    # print(probs)
    # print(trans)
    path_indices = lb.sequence.viterbi_discriminative(probs, trans)
    chord_labels = [labels[i] for i in path_indices]

    events = group_chord_labels_into_events(chord_labels, times)
    print(events)

    if use_quantize:
        print("Quantizing...")
        # Quantize against detected beat timestamps, not chromagram frame times.
        # The resulting start_index/end_index values are exact MIDI beat indices.
        events = quantize_chord_events(events, beat_times)
        print(events)

    return generate_midi(events, bpm)



def generate_midi(events, bpm):

    midi = MIDIFile(1)
    midi.addTrackName(0, 0, 'Chord transcription')
    midi.addTempo(0, 0, bpm)

    for event in events:
        note_numbers = chord_label_to_midi_notes(event['label'])
        if not note_numbers:
            continue

        if 'start_index' not in event or 'end_index' not in event:
            # Unquantized event times are seconds; MIDIUtil expects beats.
            start_beats = float(event['start']) * float(bpm) / 60.0
            end_beats = float(event['end']) * float(bpm) / 60.0
        else:
            start_beats = float(event['start_index'])
            end_beats = float(event['end_index'])
        print(f"start_beats: {start_beats}, end_beats: {end_beats}")
        dur_beats = max(end_beats - start_beats, 1.0 / 32.0)

        for note_num in note_numbers:
            midi.addNote(
                track=0,
                channel=0,
                pitch=note_num,
                time=start_beats,
                duration=dur_beats,
                volume=80,
            )

    output_path = "output.mid"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    print("success")
    return output_path
    

    


def get_chromagram(path):
    y, sr = lb.load(path)

    chroma = lb.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH)
    chroma = process_chroma(chroma)

    times = lb.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    return chroma, times





def get_beat_info(perc_path, time_signature=(4, 4), bpm=None):
    y, sr = lb.load(perc_path)
    onset_env = lb.onset.onset_strength(y=y, sr=sr)
    if bpm is None:
        bpm, beat_times = lb.beat.beat_track(onset_envelope=onset_env, sr=sr, units='time')
    else:
        _, beat_times = lb.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=bpm, units='time')
    beats_per_bar = time_signature[0]
    bars = []
    for i in range(0, len(beat_times), beats_per_bar):
        new_bar = {'start': beat_times[i], 
                   'end': beat_times[i + beats_per_bar - 1] if i + beats_per_bar - 1 < len(beat_times) else beat_times[-1]}
        bars.append(new_bar)

    bpm = float(np.asarray(bpm).reshape(-1)[0])
    return beat_times, bars, bpm


def quantize_chord_events(events, beat_times):
    """Quantizes every chord event to the nearest beat. """
    beat_times_search_start_index=0
    for event in events:
        start = event['start']
        end = event['end']
        for i in range(beat_times_search_start_index, len(beat_times), 1):
            if beat_times[i] > start:
                if beat_times[i] - start > start - beat_times[i-1]:
                    new_start = (beat_times[i-1], i-1)
                else:
                    new_start = (beat_times[i], i)
                beat_times_search_start_index = i
                break
        event['start'], event['start_index'] = new_start
        # Quantize the end time as well
        for i in range(beat_times_search_start_index, len(beat_times), 1):
            if beat_times[i] > end:
                if beat_times[i] - end > end - beat_times[i-1]:
                    new_end = (beat_times[i-1], i-1)    
                else:
                    new_end = (beat_times[i], i)
                beat_times_search_start_index = i
                break
        event['end'], event['end_index'] = new_end
    return events




def group_chord_labels_into_events(chord_labels = list, times = list, min_duration=0.15):
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



# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description='Export a chord transcription MIDI from an audio file.')
#     parser.add_argument('audio_path', help='Path to the input audio file')
#     parser.add_argument('output_path', help='Output MIDI filename')
#     parser.add_argument('--key', nargs=2, default=('A', 'major'), metavar=('ROOT', 'MODE'))
#     parser.add_argument('--no-key-bias', action='store_true')
#     parser.add_argument('--keep-top', type=int, default=3)
#     parser.add_argument('--min-duration', type=float, default=0.15)
#     parser.add_argument('--bpm', type=float, default=120.0)
#     args = parser.parse_args()

#     key_bias = not args.no_key_bias
#     export_chords_to_midi(
#         args.audio_path,
#         args.output_path,
#         key=(args.key[0], args.key[1]),
#         key_bias=key_bias,
#         keep_top=args.keep_top,
#         min_duration=args.min_duration,
#         bpm=args.bpm,
#     )
#     print(f"Wrote MIDI to {args.output_path}")



# CODE BIN

# def analyze_song(path, perc_path, key=('A', 'major'), time_signature=(4, 4), bpm=None, key_bias=True, keep_top=3):
#     """
#     Return the harmonic representation plus the decoded chord sequence.

#     Intermediate stage:
#     - chroma: array with shape (12, T), one pitch class vector per time frame
#     - chord_labels: list of length T, one chord label per frame
#     - times: list of frame timestamps in seconds, one per frame
#     """
#     y, sr = lb.load(path)
#     y_harm, _ = lb.effects.hpss(y)

#     chroma = lb.feature.chroma_cens(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
#     chroma = process_chroma(chroma, keep_top=keep_top)

#     trans = lb.sequence.transition_loop(84, 0.5)
#     key_bias_vec = key_bias_vector(*key)

#     probs = np.exp(weights.dot(chroma))
#     if key_bias:
#         probs *= key_bias_vec[:, None]
#     probs /= probs.sum(axis=0, keepdims=True)
#     #viterbi's chosen chord sequence: an array of indices into the labels list, one per frame
#     path_indices = lb.sequence.viterbi_discriminative(probs, trans)
#     # print(path_indices)
#     chord_labels = [labels[i] for i in path_indices]
#     times = lb.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LENGTH)
#     beat_times, bars, bpm = get_beat_info(perc_path, time_signature=time_signature, bpm=bpm)

#     return y, sr, times, chord_labels, chroma, beat_times, bars, bpm

# def export_chords_to_midi(
#     audio_path,
#     output_path,
#     perc_path,
#     key=('A', 'major'),
#     key_bias=True,
#     keep_top=3,
#     min_duration=0.15,
#     bpm=120,
#     velocity=80,
#     channel=0,
#     beats_per_bar=4,
# ):
#     """
#     Analyze an audio file, quantize the resulting chord events to a bar grid,
#     convert the Viterbi chord labels into note events, and save a MIDI file.

#     Returns a dictionary with the clear intermediate stages:
#     {
#         'frame_labels': [...],   # one label per audio frame
#         'events': [
#             {'label': 'C:maj', 'start': 0.0, 'end': 2.0},
#             ...
#         ],
#         'midi_path': 'output.mid'
#     }
#     """
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio file not found: {audio_path}")

#     _, sr, times, chord_labels, _, beat_times, _, _ = analyze_song(
#         audio_path,
#         perc_path,
#         key=key,
#         key_bias=key_bias,
#         keep_top=keep_top,
#     )

#     events = group_chord_labels_into_events(chord_labels, times, min_duration=min_duration)
#     events = quantize_chord_events(events, beat_times)

#     midi = MIDIFile(1)
#     midi.addTrackName(0, 0, 'Chord transcription')
#     midi.addTempo(0, 0, bpm)

#     for event in events:
#         note_numbers = chord_label_to_midi_notes(event['label'])
#         if not note_numbers:
#             continue

#         start_beats = event['start_index']
#         end_beats = event['end_index']
#         dur_beats = max(end_beats - start_beats, 1.0 / 32.0)

#         for note_num in note_numbers:
#             midi.addNote(
#                 track=0,
#                 channel=channel,
#                 pitch=note_num,
#                 time=start_beats,
#                 duration=dur_beats,
#                 volume=velocity,
#             )

#     os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
#     with open(output_path, 'wb') as f:
#         midi.writeFile(f)

#     return {
#         'frame_labels': chord_labels,
#         'events': events,
#         'midi_path': output_path,
#     }


# export_chords_to_midi('data/separated/neverender/other.wav', 'output.mid', perc_path='data/separated/neverender/drums.wav', key=('F#', 'minor'), key_bias=True, keep_top=3, min_duration=0.15, bpm=120, velocity=80, channel=0, beats_per_bar=4)
