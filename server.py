import os
import time
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
import edge_tts
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)
app = FastAPI(title="Pixel Companion Server")

# Voice mapping per detected language
VOICE_MAP = {
    "ar": "ar-TN-HediNeural",
    "fr": "fr-FR-HenriNeural",
    "en": "en-US-GuyNeural",
}
DEFAULT_VOICE = VOICE_MAP["en"]

# Folder to store temporary audio files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in VOICE_MAP else "en"
    except Exception:
        return "en"


def transcribe_audio(file_path):
    """Send an audio file to Groq's Whisper endpoint and return transcribed text."""
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    return transcription


def ask_pixel(user_message):
    """Send a message to the LLM and return the text response."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Pixel, a friendly AI companion. "
                    "You understand Arabic, French, and English, and can reply "
                    "in whichever language (or mix) the user uses. "
                    "Keep replies short and conversational, like a companion device would speak."
                )
            },
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content


async def generate_speech(text, output_file, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


@app.get("/")
def root():
    """Simple health check."""
    return {"status": "Pixel server is running"}


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    """
    Accepts an audio file from the user.
    Returns the transcribed text, the reply text, and a reply audio file.
    """
    timestamp = int(time.time() * 1000)

    # 1. Save uploaded audio temporarily
    input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.wav")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2. Transcribe
    transcribed_text = transcribe_audio(input_path)

    # 3. Ask the LLM
    reply_text = ask_pixel(transcribed_text)

    # 4. Detect language and generate speech
    lang = detect_language(reply_text)
    voice = VOICE_MAP.get(lang, DEFAULT_VOICE)
    output_path = os.path.join(TEMP_DIR, f"reply_{timestamp}.mp3")
    await generate_speech(reply_text, output_path, voice)

    # 5. Clean up the input file (no longer needed)
    try:
        os.remove(input_path)
    except OSError:
        pass

    # 6. Return both text info and a link to fetch the audio
    return JSONResponse({
        "transcribed_text": transcribed_text,
        "reply_text": reply_text,
        "detected_language": lang,
        "audio_url": f"/audio/{os.path.basename(output_path)}"
    })


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve a generated reply audio file."""
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)
