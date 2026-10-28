import argparse

from midi import process_audio_file
from utilities import NOTE_NAMES

KEY_MODES = ["major", "minor", "harmonic_minor"]


def main():
    p = argparse.ArgumentParser(description="Convert an audio file's chords to MIDI.")
    p.add_argument("audio_file")
    p.add_argument("--drums", default=None, help="optional drums track")
    p.add_argument("--bpm", type=float, default=None, help="default: auto-detect")
    p.add_argument("--key", choices=NOTE_NAMES, default=None)
    p.add_argument("--mode", choices=KEY_MODES, default=None)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--split-audio", action="store_true", help="separate stems (slower)")
    args = p.parse_args()

    key = (args.key, args.mode) if args.key and args.mode else (None, None)
    midi_path, events = process_audio_file(
        args.audio_file,
        bpm=args.bpm,
        perc_path=args.drums,
        key=key,
        use_quantize=not args.no_quantize,
        separated=not args.split_audio,
    )
    print(f"MIDI written to: {midi_path}")
    for event in events:
        print(event)


if __name__ == "__main__":
    main()
