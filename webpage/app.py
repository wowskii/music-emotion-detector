import os
import sys
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# midi.py / utilities.py live at the repo root, one level up from this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from midi import process_audio_file  # noqa: E402

app = FastAPI(title="Chord to MIDI Converter")

VALID_KEY_MODES = {"major", "minor", "harmonic_minor"}


def _save_upload(upload: UploadFile, dest_dir: str) -> str:
    filename = os.path.basename(upload.filename or "upload")
    dest_path = os.path.join(dest_dir, filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest_path


@app.post("/api/process")
async def process(
    audio: UploadFile = File(...),
    drums: UploadFile | None = File(None),
    bpm: str | None = Form(None),
    key_root: str = Form("None"),
    key_mode: str = Form("major"),
    quantize: bool = Form(False),
    split_audio: bool = Form(False),
):
    if key_mode not in VALID_KEY_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid key_mode: {key_mode}")

    bpm_value = None
    if bpm not in (None, ""):
        try:
            bpm_value = float(bpm)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid bpm: {bpm}")

    key = (None, None) if key_root in (None, "", "None") else (key_root, key_mode)

    # Isolate each request's files so concurrent uploads never clobber each other.
    work_dir = os.path.join(tempfile.gettempdir(), "chord-to-midi", uuid.uuid4().hex)
    os.makedirs(work_dir, exist_ok=True)

    try:
        audio_path = _save_upload(audio, work_dir)
        perc_path = _save_upload(drums, work_dir) if drums is not None else None

        if perc_path is not None:
            separated = True
        elif split_audio:
            separated = False
            perc_path = None
        else:
            separated = True
            perc_path = None

        midi_path = process_audio_file(
            audio_path,
            bpm=bpm_value,
            perc_path=perc_path,
            key=key,
            use_quantize=quantize,
            separated=separated,
        )

        download_name = os.path.splitext(os.path.basename(audio.filename or "output"))[0] + ".mid"
        return FileResponse(
            os.path.join(REPO_ROOT, midi_path),
            media_type="audio/midi",
            filename=download_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
