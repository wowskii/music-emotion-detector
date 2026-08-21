import numpy as np
import librosa as lb
import sounddevice as sd
import tkinter as tk
from threading import Thread
from utilities import *

# --- CONFIGURATION & LOGIC ---
SR = 22050
WINDOW_SIZE = 2.0  # seconds of memory
DEVICE_ID = 1

print(sd.query_devices())

trans = lb.sequence.transition_loop(84, 0.9)

# Global buffer for the sliding window
audio_buffer = np.zeros(int(SR * WINDOW_SIZE))

# --- GUI SETUP ---
root = tk.Tk()
root.title("Real-Time Chord Detector")
root.geometry("600x400")
root.configure(bg='black')

# Large label for the chord name
chord_label = tk.Label(root, text="WAITING...", font=("Helvetica", 120, "bold"), fg="#00FF00", bg="black")
chord_label.pack(expand=True)


current_key = ('A', 'major')

def update_ui(chord_name):
    """Safely update the UI text from the audio thread."""
    chord_label.config(text=chord_name)

# --- AUDIO PROCESSING ---
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
        y_harm, y_perc = lb.effects.hpss(audio_buffer)

        # CENS is smoothed and normalized, making it much better for chord ID than raw CQT.
        chroma = lb.feature.chroma_cens(y=y_harm, sr=SR)

        #print(trans)
        key_bias = key_bias_vector(*current_key)

        probs = np.exp(weights.dot(chroma))
        probs *= key_bias[:, None]
        probs /= probs.sum(axis=0, keepdims=True)
        path = lb.sequence.viterbi_discriminative(probs, trans)
        
        current_chord = labels[path[-1]]
        
        # Strip the ':maj' part if you want it to look cleaner
        display_name = current_chord.replace(':maj7', 'maj7').replace(':maj', '').replace(':min', 'm').replace('N', '...').replace(':normal7', '7')
        
        root.after(0, update_ui, display_name)
        sd.sleep(100) # Analyze every 100ms

# --- START EVERYTHING ---
# Run the audio analysis in a separate thread so the GUI doesn't freeze
thread = Thread(target=processing_loop, daemon=True)
thread.start()

with sd.InputStream(samplerate=SR, channels=1, callback=audio_callback, device=DEVICE_ID):
    print("UI is running. Play your instrument!")
    root.mainloop()