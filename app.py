import gradio as gr

from midi import process_audio_file
from utilities import NOTE_NAMES

KEY_MODES = ["major", "minor", "harmonic_minor"]


def run(audio_file, drums_file, bpm, key_root, key_mode, quantize, split_audio):
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")

    key = (key_root, key_mode) if key_root and key_mode else (None, None)
    bpm_value = float(bpm) if bpm else None

    midi_path = process_audio_file(
        audio_file,
        bpm=bpm_value,
        perc_path=drums_file,
        key=key,
        use_quantize=quantize,
        separated=not split_audio,
    )

    return midi_path


with gr.Blocks(title="Chord to MIDI Converter") as demo:
    gr.Markdown("# Chord to MIDI Converter")
    gr.Markdown(
        "Upload a song and get back a MIDI file with the detected chord progression."
    )

    with gr.Row():
        audio_input = gr.Audio(
            label="Audio file", type="filepath", sources=["upload"]
        )
        drums_input = gr.Audio(
            label="Drums track (optional)",
            type="filepath",
            sources=["upload"],
        )
    gr.Markdown(
        "If you don't have a separate drums track, enable **Split audio** below "
        "and it will be extracted automatically. May take a few minutes to split."
    )

    with gr.Row():
        bpm_input = gr.Number(
            label="BPM", precision=2, value=None,
            info="Leave empty to auto-detect the tempo.",
        )
        key_root_input = gr.Dropdown(
            choices=NOTE_NAMES, label="Key", value=None,
        )
        key_mode_input = gr.Dropdown(
            choices=KEY_MODES, label="Mode", value=None,
        )

    with gr.Row():
        quantize_input = gr.Checkbox(
            label="Quantize chords to the beat", value=True,
        )
        split_audio_input = gr.Checkbox(
            label="Split audio (separate stems automatically - slower)",
            value=False,
        )

    convert_button = gr.Button("Convert to MIDI", variant="primary")
    output_file = gr.File(label="Result MIDI file")

    convert_button.click(
        fn=run,
        inputs=[
            audio_input,
            drums_input,
            bpm_input,
            key_root_input,
            key_mode_input,
            quantize_input,
            split_audio_input,
        ],
        outputs=output_file,
    )


if __name__ == "__main__":
    demo.launch()
