from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SESSION_DIR = DATA_DIR / "sessions"

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", "base")
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.8"))
AUTO_DELETE = os.getenv("AUTO_DELETE_FILES", "true").lower() == "true"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Mount frontend static files
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    # Mount CSS and JS directories
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the main upload page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Zoom Recording Analysis API", "status": "running"}


@app.get("/assign.html")
async def assign_page():
    """Serve the speaker assignment page."""
    assign_path = FRONTEND_DIR / "assign.html"
    if assign_path.exists():
        return FileResponse(assign_path)
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/report.html")
async def report_page():
    """Serve the report page."""
    report_path = FRONTEND_DIR / "report.html"
    if report_path.exists():
        return FileResponse(report_path)
    raise HTTPException(status_code=404, detail="Page not found")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload audio/video file for processing.
    
    Returns:
        session_id: Unique session identifier
        filename: Original filename
    """
    # Validate file type
    allowed_extensions = {'.mp4', '.m4a', '.mp3', '.wav', '.avi', '.mov'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / session_id / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "session_id": session_id,
        "filename": file.filename,
        "message": "File uploaded successfully"
    }


@app.post("/api/process/{session_id}")
async def process_audio(session_id: str, language: str = "ja"):
    """
    Process uploaded file with parallel execution: extract audio, transcribe, and diarize.
    
    Args:
        session_id: Session identifier
        language: Language code (ja or en)
        
    Returns:
        segments: Transcription segments with speaker labels
        total_duration: Total audio duration
        processing_time: Time taken for processing
    """
    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    overall_start = time.time()
    session_dir = UPLOAD_DIR / session_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find uploaded file
    files = list(session_dir.glob("*"))
    if not files:
        raise HTTPException(status_code=404, detail="No file found in session")
    
    input_file = files[0]
    
    try:
        print(f"\n{'='*60}")
        print(f"PROCESSING SESSION: {session_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract/prepare audio (optimized)
        step1_start = time.time()
        audio_path = prepare_audio_for_processing(
            str(input_file),
            str(session_dir)
        )
        step1_time = time.time() - step1_start
        print(f"[STEP 1] Audio preprocessing: {step1_time:.2f}s\n")
        
        # Check HF token
        if not HF_TOKEN or HF_TOKEN == "your_token_here":
            raise HTTPException(
                status_code=500,
                detail="HuggingFace token not configured. Please set HUGGINGFACE_TOKEN in .env file"
            )
        
        # Step 2 & 3: Parallel execution of transcription and diarization
        step2_start = time.time()
        
        def run_transcription():
            transcriber = get_transcriber(WHISPER_MODEL)
            return transcriber.get_segments(audio_path, language)
        
        def run_diarization():
            diarizer = get_diarizer(HF_TOKEN)
            return diarizer.diarize(audio_path, num_speakers=2)
        
        # Run both in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_event_loop()
            whisper_future = loop.run_in_executor(executor, run_transcription)
            diarization_future = loop.run_in_executor(executor, run_diarization)
            
            # Wait for both to complete
            whisper_segments, diarization_segments = await asyncio.gather(
                whisper_future,
                diarization_future
            )
        
        step2_time = time.time() - step2_start
        print(f"[STEP 2&3] Parallel transcription + diarization: {step2_time:.2f}s\n")
        
        # Step 4: Align transcription with speakers
        step4_start = time.time()
        aligned_segments = align_transcript_with_speakers(
            whisper_segments,
            diarization_segments
        )
        step4_time = time.time() - step4_start
        print(f"[STEP 4] Alignment: {step4_time:.2f}s\n")
        
        # Calculate total duration
        total_duration = max(seg["end_time"] for seg in aligned_segments) if aligned_segments else 0
        
        # Save intermediate result
        intermediate_data = {
            "segments": aligned_segments,
            "total_duration": total_duration
        }
        
        import json
        with open(session_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
        
        overall_time = time.time() - overall_start
        
        print(f"{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"Total time: {overall_time:.2f}s")
        print(f"Audio duration: {total_duration:.1f}s")
        print(f"Real-time factor: {total_duration/overall_time:.1f}x")
        print(f"{'='*60}\n")
        
        return {
            "session_id": session_id,
            "segments": aligned_segments,
            "total_duration": total_duration,
            "processing_time": overall_time,
            "message": "Processing complete"
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assign-speakers/{session_id}")
async def assign_speakers(session_id: str, assignment: SpeakerAssignment):
    """
    Assign speaker roles and generate full analysis report.
    
    Args:
        session_id: Session identifier
        assignment: Speaker role assignments
        
    Returns:
        Complete analysis result
    """
    session_dir = UPLOAD_DIR / session_id
    transcript_file = session_dir / "transcript.json"
    
    if not transcript_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Please process the file first."
        )
    
    try:
        # Load transcript
        import json
        with open(transcript_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create ProcessedTranscript
        segments = [TranscriptSegment(**seg) for seg in data["segments"]]
        transcript = ProcessedTranscript(
            segments=segments,
            speaker_assignment=assignment.speaker_assignment,
            total_duration=data["total_duration"]
        )
        
        # Calculate metrics
        metrics = calculate_metrics(transcript, SILENCE_THRESHOLD)
        
        # Get coach segments
        coach_segments = [
            s for s in segments
            if assignment.speaker_assignment.get(s.speaker) == "coach"
        ]
        
        # Categorize coach utterances
        coach_utterances = categorize_coach_utterances(coach_segments)
        
        # Generate report
        result = generate_analysis_report(
            session_id,
            transcript,
            metrics,
            coach_utterances
        )
        
        # Save result
        save_analysis_result(result, str(SESSION_DIR))
        
        # Auto-delete uploaded files if configured
        if AUTO_DELETE:
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                print(f"Warning: Failed to delete uploaded files: {e}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Retrieve saved analysis result.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Analysis result
    """
    try:
        result = load_analysis_result(session_id, str(SESSION_DIR))
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/update-category/{session_id}")
async def update_category(session_id: str, update: CategoryUpdate):
    """
    Update category for a specific utterance and recalculate metrics.
    
    Args:
        session_id: Session identifier
        update: Category update data
        
    Returns:
        Updated analysis result
    """
    try:
        # Load existing result
        result = load_analysis_result(session_id, str(SESSION_DIR))
        
        # Find and update utterance
        updated = False
        for utterance in result.all_coach_utterances:
            if utterance.id == update.utterance_id:
                utterance.category = update.new_category
                utterance.manually_edited = True
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Utterance not found")
        
        # Recalculate coach quality metrics
        from services.report_generator import calculate_coach_quality
        result.coach_quality = calculate_coach_quality(
            result.all_coach_utterances,
            result.metrics.student_speaking_time
        )
        
        # Save updated result
        save_analysis_result(result, str(SESSION_DIR))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/csv/{session_id}")
async def export_csv(session_id: str):
    """
    Export analysis result as CSV.
    
    Args:
        session_id: Session identifier
        
    Returns:
        CSV file
    """
    try:
        result = load_analysis_result(session_id, str(SESSION_DIR))
        csv_content = export_to_csv(result)
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=analysis_{session_id}.csv"
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "whisper_model": WHISPER_MODEL,
        "hf_token_configured": bool(HF_TOKEN and HF_TOKEN != "your_token_here")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), language: str = Form("ja")):
    """
    フロント互換: /analyze に投げられたら
    /api/upload → /api/process を内部で順番に呼んで返す
    """
    upload_result = await upload_file(file)
    session_id = upload_result["session_id"]
    return await process_audio(session_id=session_id, language=language)
