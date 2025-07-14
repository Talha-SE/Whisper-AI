from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import os
from typing import Optional
from pydantic import BaseModel

app = FastAPI(
    title="Whisper AI Speech-to-Text API",
    description="API for transcribing audio files using OpenAI's Whisper model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load faster-whisper model once at startup (base size)
model = WhisperModel("base", device="cpu", compute_type="int8")

class TranscriptionResponse(BaseModel):
    success: bool
    transcription: Optional[str]
    error: Optional[str]

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text
    
    - **file**: Audio file to transcribe (WAV, MP3 supported)
    """
    try:
                # Save uploaded bytes to temp file
        temp_path = "temp_audio"
        with open(temp_path, "wb") as tmp:
            tmp.write(await file.read())

        # Transcribe using faster-whisper
        segments, _ = model.transcribe(temp_path)
        transcription = " ".join([seg.text for seg in segments])

        os.remove(temp_path)
        return {"success": True, "transcription": transcription}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/")
async def root():
    """
    API information endpoint
    """
    return {
        "message": "Whisper AI Speech-to-Text API",
        "endpoints": {
            "transcribe": "POST /transcribe - Upload audio file for transcription",
            "health": "GET /health - Service health check"
        },
        "documentation": "Available at /docs or /redoc"
    }
