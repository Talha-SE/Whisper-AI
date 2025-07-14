from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
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

# Load the model once at startup
model = whisper.load_model("base")

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
        # Save the uploaded file temporarily
        temp_file = "temp_audio"
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe the audio
        result = model.transcribe(temp_file)
        
        # Clean up
        os.remove(temp_file)
        
        return {"success": True, "transcription": result["text"]}
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
