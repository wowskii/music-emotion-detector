import time
import numpy as np
import librosa as lb
import sounddevice as sd
import tkinter as tk
from utilities import *

HOP_LENGTH = 512



def analyze_song(path, key=('A', 'major'), key_bias=True, keep_top=3):
    """Runs the same chroma extraction + Viterbi chord decoding used elsewhere on a full audio file."""
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


def play_audio_with_chords(path, key=('A', 'major'), key_bias=True, keep_top=3):
    y, sr, times, chord_labels, chroma = analyze_song(path, key=key, key_bias=key_bias, keep_top=keep_top)

    root = tk.Tk()
    root.title("Chord Preview")
    root.geometry("600x550")
    root.configure(bg='black')

    chord_label = tk.Label(root, text="...", font=("Helvetica", 120, "bold"), fg="#00FF00", bg="black")
    chord_label.pack(expand=True)

    notes_label = tk.Label(root, text="", font=("Helvetica", 32), fg="#00AAFF", bg="black")
    notes_label.pack(expand=True)

    CHROMA_WIDTH, CHROMA_HEIGHT = 560, 150
    chroma_canvas = tk.Canvas(root, width=CHROMA_WIDTH, height=CHROMA_HEIGHT, bg='black', highlightthickness=0)
    chroma_canvas.pack(pady=10)
    bar_width = CHROMA_WIDTH / 12

    def display_chord(display_name, note_names):
        chord_label.config(text=display_name)
        notes_label.config(text=' - '.join(note_names))

    def display_chroma(bin_values):
        chroma_canvas.delete('all')
        for i, value in enumerate(bin_values):
            bar_height = value * CHROMA_HEIGHT
            x0 = i * bar_width + 2
            x1 = (i + 1) * bar_width - 2
            y0 = CHROMA_HEIGHT - bar_height
            y1 = CHROMA_HEIGHT
            chroma_canvas.create_rectangle(x0, y0, x1, y1, fill="#FF8800", outline="")
            chroma_canvas.create_text((x0 + x1) / 2, CHROMA_HEIGHT - 10, text=NOTE_NAMES[i], fill='white')

    def update_loop(start_time):
        elapsed = time.time() - start_time
        idx = np.searchsorted(times, elapsed, side='right') - 1

        if idx >= len(chord_labels):
            display_chord("DONE", [])
            return

        idx = max(idx, 0)
        raw_label = chord_labels[idx]
        display_name = raw_label.replace(':maj7', 'maj7').replace(':maj', '').replace(':min', 'm').replace('N', '...').replace(':normal7', '7').replace(':min9', 'm9').replace(':dim', 'dim')
        display_chord(display_name, chord_notes(raw_label))
        display_chroma(chroma[:, idx])

        root.after(100, update_loop, start_time)

    sd.play(y, sr)
    root.after(0, update_loop, time.time())
    root.mainloop()


if __name__ == "__main__":
    # play_audio_with_chords('data/separated/toramoyo_ft/other.wav')
    play_audio_with_chords('data/separated/neverender/other.wav', key=('F#', 'minor'))