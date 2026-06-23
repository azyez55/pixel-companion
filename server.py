import os
import time
import json
import sqlite3
import asyncio
from contextlib import contextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
import edge_tts
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# ---------- Setup ----------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your environment variables.")

client = Groq(api_key=api_key)
app = FastAPI(title="Pixel Companion Server")

VOICE_MAP = {
    "ar": "ar-TN-HediNeural",
    "fr": "fr-FR-HenriNeural",
    "en": "en-US-GuyNeural",
}
DEFAULT_VOICE = VOICE_MAP["en"]

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

DB_PATH = "pixel_memory.db"
MAX_HISTORY_TURNS = 10  # how many past exchanges to feed back to the LLM for context

SYSTEM_PROMPT = """You are Pixel, a small AI companion device with a warm, curious, slightly playful personality.
You understand Arabic, French, and English, and reply in whichever language (or mix) the user uses.
Keep replies short (1-3 sentences) and conversational, like a companion speaking out loud, not writing an essay.
You remember past parts of the conversation and refer back to them naturally when relevant.

After your reply, on a new line, output exactly one emotion tag in this format:
[emotion: happy|sad|thinking|curious|neutral|excited]

Choose the emotion that matches the tone of your reply."""


# ---------- Database ----------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT,
                timestamp REAL NOT NULL
            )
        """)


init_db()


def save_message(role, content, emotion=None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversation (role, content, emotion, timestamp) VALUES (?, ?, ?, ?)",
            (role, content, emotion, time.time())
        )


def get_recent_history(limit=MAX_HISTORY_TURNS):
    """Return the last `limit` messages (user + assistant) in chronological order."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversation ORDER BY id DESC LIMIT ?",
            (limit * 2,)  # *2 because each turn has a user + assistant message
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ---------- Helpers ----------
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in VOICE_MAP else "en"
    except Exception:
        return "en"


def parse_reply(raw_reply):
    """Split the LLM's reply into (clean_text, emotion). Falls back to 'neutral' if no tag found."""
    emotion = "neutral"
    text = raw_reply.strip()

    if "[emotion:" in text:
        try:
            main_part, tag_part = text.rsplit("[emotion:", 1)
            emotion = tag_part.replace("]", "").strip().lower()
            text = main_part.strip()
        except Exception:
            pass

    valid_emotions = {"happy", "sad", "thinking", "curious", "neutral", "excited"}
    if emotion not in valid_emotions:
        emotion = "neutral"

    return text, emotion


def transcribe_audio(file_path):
    """Send an audio file to Groq's Whisper endpoint. Raises on failure."""
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    if not transcription or not transcription.strip():
        raise ValueError("Transcription returned empty text — audio may be silent or unclear.")
    return transcription.strip()


def ask_pixel(user_message):
    """Send the user message + recent history to the LLM. Raises on failure."""
    history = get_recent_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        timeout=20,
    )
    raw_reply = response.choices[0].message.content
    if not raw_reply or not raw_reply.strip():
        raise ValueError("LLM returned an empty response.")
    return raw_reply


async def generate_speech(text, output_file, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "Pixel server is running"}


@app.get("/history")
def history():
    """Debug endpoint: view recent conversation history."""
    return {"history": get_recent_history(limit=20)}


@app.delete("/history")
def clear_history():
    """Debug endpoint: wipe conversation memory."""
    with get_db() as conn:
        conn.execute("DELETE FROM conversation")
    return {"status": "history cleared"}


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    timestamp = int(time.time() * 1000)
    input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.wav")
    output_path = None

    try:
        # 1. Save uploaded audio
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse({"error": "No audio data received."}, status_code=400)

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # 2. Transcribe
        try:
            transcribed_text = transcribe_audio(input_path)
        except Exception as e:
            return JSONResponse({"error": f"Transcription failed: {str(e)}"}, status_code=502)

        # 3. Ask the LLM (with memory)
        try:
            raw_reply = ask_pixel(transcribed_text)
        except Exception as e:
            return JSONResponse({"error": f"LLM request failed: {str(e)}"}, status_code=502)

        reply_text, emotion = parse_reply(raw_reply)

        # 4. Save both turns to memory
        save_message("user", transcribed_text)
        save_message("assistant", reply_text, emotion)

        # 5. Generate speech
        try:
            lang = detect_language(reply_text)
            voice = VOICE_MAP.get(lang, DEFAULT_VOICE)
            output_path = os.path.join(TEMP_DIR, f"reply_{timestamp}.mp3")
            await generate_speech(reply_text, output_path, voice)
            audio_url = f"/audio/{os.path.basename(output_path)}"
        except Exception as e:
            # TTS failure shouldn't kill the whole response — text still works
            audio_url = None
            print(f"TTS generation failed: {e}")

        return JSONResponse({
            "transcribed_text": transcribed_text,
            "reply_text": reply_text,
            "emotion": emotion,
            "detected_language": lang if 'lang' in locals() else "en",
            "audio_url": audio_url
        })

    except Exception as e:
        return JSONResponse({"error": f"Unexpected server error: {str(e)}"}, status_code=500)

    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)
