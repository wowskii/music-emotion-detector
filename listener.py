import numpy as np
import librosa as lb
import sounddevice as sd
import tkinter as tk
from threading import Thread

# --- 1. CONFIGURATION & LOGIC ---
SR = 22050
WINDOW_SIZE = 1.0  # 1 second of memory
DEVICE_ID = None

# Normalise templates by dividing by the number of active notes, so that the probabilities are more balanced
maj_template  = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0]) / 3.0
maj7_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,1]) / 4.0
min_template  = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0]) / 3.0
min7_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,1,0]) / 4.0
N_template    = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 12.0

weights = np.zeros((49, 12), dtype=float)
labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
          'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
          'C:maj7', 'C#:maj7', 'D:maj7', 'D#:maj7', 'E:maj7', 'F:maj7',
          'F#:maj7', 'G:maj7', 'G#:maj7', 'A:maj7', 'A#:maj7', 'B:maj7',
          'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
          'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
          'C:min7', 'C#:min7', 'D:min7', 'D#:min7', 'E:min7', 'F:min7',
          'F#:min7', 'G:min7', 'G#:min7', 'A:min7', 'A#:min7', 'B:min7',
          'N']
print(len(labels))
for c in range(12):
    weights[c, :] = np.roll(maj_template, c) # c:maj
    weights[c + 12, :] = np.roll(maj7_template, c)  # c:maj7
    weights[c + 24, :] = np.roll(min_template, c)  # c:min
    weights[c + 36, :] = np.roll(min7_template, c)  # c:min7
weights[48, :] = N_template
trans = lb.sequence.transition_loop(49, 0.9)

# Global buffer for the sliding window
audio_buffer = np.zeros(int(SR * WINDOW_SIZE))

# --- 2. GUI SETUP ---
root = tk.Tk()
root.title("Real-Time Chord Detector")
root.geometry("600x400")
root.configure(bg='black')

# Large label for the chord name
chord_label = tk.Label(root, text="WAITING...", font=("Helvetica", 120, "bold"), fg="#00FF00", bg="black")
chord_label.pack(expand=True)

def update_ui(chord_name):
    """Safely update the UI text from the audio thread."""
    chord_label.config(text=chord_name)

# --- 3. AUDIO PROCESSING ---
def audio_callback(indata, frames, time, status):
    global audio_buffer
    new_data = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data

def processing_loop():
    """Continuously analyzes the buffer and updates the UI."""
    while True:
        # Skip processing if the audio is essentially silent (Noise Gate)
        if np.max(np.abs(audio_buffer)) < 0.01:
            root.after(0, update_ui, "...")
            sd.sleep(100)
            continue

        # We only want 'y_harm' (pitched sounds), ignoring 'y_perc' (drums/noise)
        y_harm = lb.effects.hpss(audio_buffer)[0]

        # CENS is smoothed and normalized, making it much better for chord ID than raw CQT.
        chroma = lb.feature.chroma_cens(y=y_harm, sr=SR)

        probs = np.exp(weights.dot(chroma))
        probs /= probs.sum(axis=0, keepdims=True)
        path = lb.sequence.viterbi_discriminative(probs, trans)
        
        current_chord = labels[path[-1]]
        
        # Strip the ':maj' part if you want it to look cleaner
        display_name = current_chord.replace(':maj7', 'maj7').replace(':maj', '').replace(':min', 'm').replace('N', '...')
        
        root.after(0, update_ui, display_name)
        sd.sleep(100) # Analyze every 100ms

# --- 4. START EVERYTHING ---
# Run the audio analysis in a separate thread so the GUI doesn't freeze
thread = Thread(target=processing_loop, daemon=True)
thread.start()

with sd.InputStream(samplerate=SR, channels=1, callback=audio_callback, device=DEVICE_ID):
    print("UI is running. Play your instrument!")
    root.mainloop()