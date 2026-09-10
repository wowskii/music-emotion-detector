import base64
import json
import mimetypes

import gradio as gr

from midi import process_audio_file
from utilities import NOTE_NAMES

KEY_MODES = ["major", "minor", "harmonic_minor"]


def build_chord_player_html(audio_path, events):
    """Embed the audio as a data URI and sync a chord label to its playback time."""
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    with open(audio_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode("utf-8")

    events_json = json.dumps([
        {"label": e["label"], "start": float(e["start"]), "end": float(e["end"])}
        for e in events
    ])

    return f"""
    <div style="display:flex; flex-direction:column; gap:10px; align-items:center;">
        <audio id="chord-audio-player" controls style="width:100%;" src="data:{mime};base64,{b64_audio}"></audio>
        <div id="chord-label-display" style="font-size:2.5em; font-weight:bold; min-height:1.4em;">–</div>
    </div>
    <script>
    (function() {{
        const events = {events_json};
        const audio = document.getElementById('chord-audio-player');
        const label = document.getElementById('chord-label-display');
        if (!audio || !label) return;
        audio.ontimeupdate = function() {{
            const t = audio.currentTime;
            let current = '–';
            for (const e of events) {{
                if (t >= e.start && t < e.end) {{ current = e.label; break; }}
            }}
            label.innerText = current;
        }};
    }})();
    </script>
    """


def run(audio_file, drums_file, bpm, key_root, key_mode, quantize, split_audio):
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")

    key = (key_root, key_mode) if key_root and key_mode else (None, None)
    bpm_value = float(bpm) if bpm else None

    midi_path, events = process_audio_file(
        audio_file,
        bpm=bpm_value,
        perc_path=drums_file,
        key=key,
        use_quantize=quantize,
        separated=not split_audio,
    )

    player_html = build_chord_player_html(audio_file, events)

    return midi_path, player_html


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
    chord_player = gr.HTML(label="Play along with detected chords")

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
        outputs=[output_file, chord_player],
    )


if __name__ == "__main__":
    demo.launch()
