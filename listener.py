import numpy as np
import librosa as lb
import sounddevice as sd

# --- CONFIGURATION ---
SR = 22050        # Sample Rate
WINDOW_SIZE = 1.0 # Seconds of audio to analyze (the "memory")
HOP_SIZE = 0.2    # How often to update the chord (5 times per second)
DEVICE_ID = None  # Change this if you have multiple mics

# Convert seconds to samples
window_samples = int(SR * WINDOW_SIZE)
hop_samples = int(SR * HOP_SIZE)

# LOGIC
labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
          'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min', 'N']

weights = np.zeros((25, 12))
maj_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0])
min_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0])
for c in range(12):
    weights[c, :] = np.roll(maj_template, c)
    weights[c + 12, :] = np.roll(min_template, c)
weights[24, :] = np.ones(12) / 4.
trans = lb.sequence.transition_loop(25, 0.9)

# This holds the sliding window of audio
audio_buffer = np.zeros(window_samples)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    # Shift buffer and append new data
    new_data = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data

def get_chord_from_buffer(audio):
    # 1. Get Chroma
    chroma = lb.feature.chroma_cqt(y=audio, sr=SR)
    
    # 2. Match Templates
    probs = np.exp(weights.dot(chroma))
    probs /= probs.sum(axis=0, keepdims=True)
    
    # 3. Viterbi (using the last state of the path)
    chords_vit = lb.sequence.viterbi_discriminative(probs, trans)
    return labels[chords_vit[-1]]

# --- MAIN LOOP ---
print("Starting live chord detection... Press Ctrl+C to stop.")
try:
    with sd.InputStream(samplerate=SR, channels=1, callback=audio_callback, blocksize=hop_samples, device=DEVICE_ID):
        while True:
            # Analyze the current buffer
            current_chord = get_chord_from_buffer(audio_buffer)
            
            # Print with \r to stay on one line
            print(f"Current Chord: {current_chord:10}", end='\r')
            
            # Small sleep to prevent CPU spiking
            sd.sleep(int(HOP_SIZE * 1000))
except KeyboardInterrupt:
    print("\nStopped.")