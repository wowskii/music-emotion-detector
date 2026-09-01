import numpy as np
import librosa as lb


# Normalise templates by dividing by the number of active notes, so that the probabilities are more balanced
maj_template  = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0]) / 3.0
maj7_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,1]) / 4.0
min_template  = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0]) / 3.0
min7_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,1,0]) / 4.0
normal7_template    = np.array([1,0,0, 0,1,0, 0,1,0, 0,1,0]) / 4.0
min9_template = np.array([1,1,0, 1,0,0, 0,1,0, 0,1,0]) / 5.0
dim_template = np.array([1,0,0, 1,0,0, 1,0,0, 0,0,0]) / 3.0
N_template    = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 12.0

weights = np.zeros((84, 12), dtype=float)
labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
          'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
          'C:maj7', 'C#:maj7', 'D:maj7', 'D#:maj7', 'E:maj7', 'F:maj7',
          'F#:maj7', 'G:maj7', 'G#:maj7', 'A:maj7', 'A#:maj7', 'B:maj7',
          'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
          'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
          'C:min7', 'C#:min7', 'D:min7', 'D#:min7', 'E:min7', 'F:min7',
          'F#:min7', 'G:min7', 'G#:min7', 'A:min7', 'A#:min7', 'B:min7',
          'C:normal7', 'C#:normal7', 'D:normal7', 'D#:normal7', 'E:normal7', 'F:normal7',
          'F#:normal7', 'G:normal7', 'G#:normal7', 'A:normal7', 'A#:normal7', 'B:normal7',
          'C:min9', 'C#:min9', 'D:min9', 'D#:min9', 'E:min9', 'F:min9',
          'F#:min9', 'G:min9', 'G#:min9', 'A:min9', 'A#:min9', 'B:min9',
          'C:dim', 'C#:dim', 'D:dim', 'D#:dim', 'E:dim', 'F:dim',
          'F#:dim', 'G:dim', 'G#:dim', 'A:dim', 'A#:dim', 'B:dim',
          'N']
print(len(labels))
for c in range(12):
    weights[c, :] = np.roll(maj_template, c) # c:maj
    weights[c + 12, :] = np.roll(maj7_template, c)  # c:maj7
    weights[c + 24, :] = np.roll(min_template, c)  # c:min
    weights[c + 36, :] = np.roll(min7_template, c)  # c:min7
    weights[c + 48, :] = np.roll(normal7_template, c)  # c:normal7
    weights[c + 60, :] = np.roll(min9_template, c)  # c:min9
    weights[c + 72, :] = np.roll(dim_template, c)  # c:dim
weights[83, :] = N_template


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_SCALE = [0,2,4,5,7,9,11]
MINOR_SCALE = [0,2,3,5,7,8,10]
HARMONIC_MINOR_SCALE = [0,2,3,5,7,8,11]


MAJOR_CHORDS = [':maj', ':min', ':min', ':maj', ':maj', ':min', ':dim']
NATURAL_MINOR_CHORDS = [':min', ':dim', ':maj', ':min', ':min', ':maj', ':maj']
HARMONIC_MINOR_CHORDS = [':min', ':dim', ':maj', ':min', ':maj', ':maj', ':dim']

def allowed_chords(key, mode):
    """Returns a boolean mask for the given key, where True indicates that the chord is in the key."""
    key_index = NOTE_NAMES.index(key)
    allowed_chords = []

    match mode:
        case 'major':
            current_scale = MAJOR_SCALE
            current_chords = MAJOR_CHORDS
        case 'minor':
            current_scale = MINOR_SCALE
            current_chords = NATURAL_MINOR_CHORDS
        case 'harmonic_minor':
            current_scale = HARMONIC_MINOR_SCALE
            current_chords = HARMONIC_MINOR_CHORDS
    
    for i in range(7):
        chord = NOTE_NAMES[(current_scale[i] + key_index) % 12] + current_chords[i]
        allowed_chords.append(chord)

    print(f"Allowed chords for {key} {mode}: {allowed_chords}")
    return allowed_chords


def key_bias_vector(key, mode, in_key_boost=1.0, out_key_penalty=1e-3):
    """Returns a vector of length 84, where each element corresponds to a chord. Chords in the key are boosted, while chords out of the key are penalized."""
    allowed = allowed_chords(key, mode)
    bias_vector = np.full(84, out_key_penalty)  # Start with all chords penalized
    for i, lbl in enumerate(labels[:-1]):  # Exclude the 'N' chord
        base = lbl.split(':')[0] + ':' + lbl.split(':')[1].replace('7', '').replace('9', '')
        #print(base)
        if base in allowed:
            bias_vector[i] = in_key_boost
    bias_vector[-1] = in_key_boost  # Ensure the 'N' chord is always allowed
    return bias_vector

def process_chroma(chromagram, keep_top=0):

    #thinking of phasing out this "keep_top" thing, let's see if i need it

    chroma_filter = np.minimum(chromagram,
                           lb.decompose.nn_filter(chromagram,
                                                       aggregate=np.median,
                                                       metric='cosine'))
    if keep_top == 0:
        return chroma_filter
    chroma_smooth = np.zeros_like(chroma_filter)
    top3_idx = np.argpartition(chroma_filter, -keep_top, axis=0)[-keep_top:, :]
    cols = np.arange(chroma_filter.shape[1])
    chroma_smooth[top3_idx, cols] = chroma_filter[top3_idx, cols]

    return chroma_smooth


def separate_song_stems(path, output_dir='data/separated', model_name='htdemucs', device=None):
    """Separates a song into stems using Demucs."""
    import demucs
    import demucs.separate
    import demucs.pretrained

    separator = demucs.api.Separator(model_name)

    # Separate the audio file into stems
    origin, separated = separator.separate_audio_file(path)


    for stem, source in separated.items():
        demucs.api.save_audio(source, f"{output_dir}/{stem}.wav", samplerate=separator.samplerate)

    return separated

# separate_song_stems('data/neverender.mp3', output_dir='data/separated/neverender', model_name='htdemucs')


# Intervals (in semitones from the root) used to derive the notes shown for each chord type
CHORD_INTERVALS = {
    'maj': [0, 4, 7],
    'maj7': [0, 4, 7, 11],
    'min': [0, 3, 7],
    'min7': [0, 3, 7, 10],
    'normal7': [0, 4, 7, 10],
    'min9': [0, 3, 7, 10],
    'dim': [0, 3, 6],
}


def chord_notes(label):
    """Returns the list of note names (3-4 notes) that make up a chord label, e.g. 'C:maj' -> ['C', 'E', 'G']."""
    if label == 'N':
        return []
    root, chord_type = label.split(':')
    root_index = NOTE_NAMES.index(root)
    intervals = CHORD_INTERVALS.get(chord_type, [])
    return [NOTE_NAMES[(root_index + i) % 12] for i in intervals]