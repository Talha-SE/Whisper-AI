# Whisper AI Speech-to-Text API

This project provides a FastAPI endpoint for transcribing audio files using OpenAI's Whisper model (base version).

## Setup

1. Install requirements: `pip install -r requirements.txt`
2. Run the server: `uvicorn main:app --host 0.0.0.0 --port 8000`

## API Endpoints

- POST `/transcribe` - Upload an audio file (WAV, MP3, etc.) and get transcription
- GET `/health` - Health check endpoint

## Deployment to Render.com

1. Create a new Web Service on Render
2. Connect your GitHub/GitLab repository
3. Set the following environment variables (if needed):
   - `PYTHON_VERSION`: 3.10
4. Set the start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Deploy!
