import numpy as np
import librosa as lb


# Normalise templates by dividing by the number of active notes, so that the probabilities are more balanced
maj_template  = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0]) / 3.0
maj7_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,1]) / 4.0
min_template  = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0]) / 3.0
min7_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,1,0]) / 4.0
normal7_template    = np.array([1,0,0, 0,1,0, 0,1,0, 0,1,0]) / 4.0
min9_template = np.array([1,1,0, 1,0,0, 0,1,0, 0,1,0]) / 5.0
N_template    = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 12.0

weights = np.zeros((73, 12), dtype=float)
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
          'N']
print(len(labels))
for c in range(12):
    weights[c, :] = np.roll(maj_template, c) # c:maj
    weights[c + 12, :] = np.roll(maj7_template, c)  # c:maj7
    weights[c + 24, :] = np.roll(min_template, c)  # c:min
    weights[c + 36, :] = np.roll(min7_template, c)  # c:min7
    weights[c + 48, :] = np.roll(normal7_template, c)  # c:normal7
    weights[c + 60, :] = np.roll(min9_template, c)  # c:min9
weights[72, :] = N_template