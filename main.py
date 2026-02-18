from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uuid
import shutil
from pathlib import Path
from dotenv import load_dotenv

from models.data_models import (
    TranscriptSegment, ProcessedTranscript, SpeakerAssignment,
    CategoryUpdate, AnalysisResult
)
from services.audio_extractor import prepare_audio_for_processing
from services.transcriber import get_transcriber
from services.diarizer import get_diarizer, align_transcript_with_speakers
from services.metrics_calculator import calculate_metrics
from services.categorizer import categorize_coach_utterances
from services.report_generator import (
    generate_analysis_report, save_analysis_result,
    load_analysis_result, export_to_csv
)

# Load environment variables
load_dotenv()

app = FastAPI(title="Zoom Recording Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Configuration
# =============================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SESSION_DIR = DATA_DIR / "sessions"

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", "base")
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.8"))
AUTO_DELETE = os.getenv("AUTO_DELETE_FILES", "true").lower() == "true"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = BASE_DIR.parent / "frontend"

if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"

    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")

    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


# =============================
# Pages
# =============================

@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Zoom Recording Analysis API", "status": "running"}


@app.get("/assign.html")
async def assign_page():
    assign_path = FRONTEND_DIR / "assign.html"
    if assign_path.exists():
        return FileResponse(assign_path)
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/report.html")
async def report_page():
    report_path = FRONTEND_DIR / "report.html"
    if report_path.exists():
        return FileResponse(report_path)
    raise HTTPException(status_code=404, detail="Page not found")


# =============================
# Upload
# =============================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {'.mp4', '.m4a', '.mp3', '.wav', '.avi', '.mov'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    session_id = str(uuid.uuid4())

    upload_path = UPLOAD_DIR / session_id / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "message": "File uploaded successfully"
    }


# =============================
# Process
# =============================

@app.post("/api/process/{session_id}")
async def process_audio(session_id: str, language: str = "ja"):

    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    overall_start = time.time()
    session_dir = UPLOAD_DIR / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    files = list(session_dir.glob("*"))
    if not files:
        raise HTTPException(status_code=404, detail="No file found in session")

    input_file = files[0]

    try:
        audio_path = prepare_audio_for_processing(
            str(input_file),
            str(session_dir)
        )

        if not HF_TOKEN or HF_TOKEN == "your_token_here":
            raise HTTPException(
                status_code=500,
                detail="HuggingFace token not configured. Please set HUGGINGFACE_TOKEN in .env file"
            )

        def run_transcription():
            transcriber = get_transcriber(WHISPER_MODEL)
            return transcriber.get_segments(audio_path, language)

        def run_diarization():
            diarizer = get_diarizer(HF_TOKEN)
            return diarizer.diarize(audio_path, num_speakers=2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_event_loop()
            whisper_future = loop.run_in_executor(executor, run_transcription)
            diarization_future = loop.run_in_executor(executor, run_diarization)

            whisper_segments, diarization_segments = await asyncio.gather(
                whisper_future,
                diarization_future
            )

        aligned_segments = align_transcript_with_speakers(
            whisper_segments,
            diarization_segments
        )

        total_duration = max(seg["end_time"] for seg in aligned_segments) if aligned_segments else 0

        return {
            "session_id": session_id,
            "segments": aligned_segments,
            "total_duration": total_duration,
            "processing_time": time.time() - overall_start,
            "message": "Processing complete"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 🔥 互換ルート（これが今回の核心）
# =============================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), language: str = Form("ja")):
    upload_result = await upload_file(file)
    session_id = upload_result["session_id"]
    return await process_audio(session_id=session_id, language=language)


# =============================
# Health
# =============================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_model": WHISPER_MODEL,
        "hf_token_configured": bool(HF_TOKEN and HF_TOKEN != "your_token_here")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
