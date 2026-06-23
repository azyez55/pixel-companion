import os
import time
import asyncio
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
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
database_url = os.getenv("DATABASE_URL")

if not api_key:
    raise ValueError("GROQ_API_KEY not found.")
if not database_url:
    raise ValueError("DATABASE_URL not found.")

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

MAX_HISTORY_TURNS = 10

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
    conn = psycopg2.connect(database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id BIGSERIAL PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT,
                    timestamp FLOAT NOT NULL
                )
            """)


init_db()


def save_message(role, content, emotion=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation (role, content, emotion, timestamp) VALUES (%s, %s, %s, %s)",
                (role, content, emotion, time.time())
            )


def get_recent_history(limit=MAX_HISTORY_TURNS):
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content FROM conversation ORDER BY id DESC LIMIT %s",
                (limit * 2,)
            )
            rows = cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_all_history(limit=50):
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, emotion, timestamp FROM conversation ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]


# ---------- Helpers ----------
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in VOICE_MAP else "en"
    except Exception:
        return "en"


def parse_reply(raw_reply):
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
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    if not transcription or not transcription.strip():
        raise ValueError("Transcription returned empty — audio may be silent or unclear.")
    return transcription.strip()


def ask_pixel(user_message):
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
    return {"history": get_all_history()}


@app.delete("/history")
def clear_history():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversation")
    return {"status": "history cleared"}


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    """Voice endpoint — accepts an audio file."""
    timestamp = int(time.time() * 1000)
    input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.wav")
    lang = "en"

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse({"error": "No audio data received."}, status_code=400)

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        try:
            transcribed_text = transcribe_audio(input_path)
        except Exception as e:
            return JSONResponse({"error": f"Transcription failed: {str(e)}"}, status_code=502)

        try:
            raw_reply = ask_pixel(transcribed_text)
        except Exception as e:
            return JSONResponse({"error": f"LLM request failed: {str(e)}"}, status_code=502)

        reply_text, emotion = parse_reply(raw_reply)
        save_message("user", transcribed_text)
        save_message("assistant", reply_text, emotion)

        audio_url = None
        try:
            lang = detect_language(reply_text)
            voice = VOICE_MAP.get(lang, DEFAULT_VOICE)
            output_path = os.path.join(TEMP_DIR, f"reply_{timestamp}.mp3")
            await generate_speech(reply_text, output_path, voice)
            audio_url = f"/audio/{os.path.basename(output_path)}"
        except Exception as e:
            print(f"TTS failed: {e}")

        return JSONResponse({
            "transcribed_text": transcribed_text,
            "reply_text": reply_text,
            "emotion": emotion,
            "detected_language": lang,
            "audio_url": audio_url
        })

    except Exception as e:
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)

    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


@app.post("/chat/text")
async def chat_text(body: dict):
    """Text endpoint — accepts JSON with a 'message' field. No audio needed."""
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "No message provided."}, status_code=400)

    lang = "en"

    try:
        raw_reply = ask_pixel(user_message)
    except Exception as e:
        return JSONResponse({"error": f"LLM request failed: {str(e)}"}, status_code=502)

    reply_text, emotion = parse_reply(raw_reply)
    save_message("user", user_message)
    save_message("assistant", reply_text, emotion)

    audio_url = None
    timestamp = int(time.time() * 1000)
    try:
        lang = detect_language(reply_text)
        voice = VOICE_MAP.get(lang, DEFAULT_VOICE)
        output_path = os.path.join(TEMP_DIR, f"reply_{timestamp}.mp3")
        await generate_speech(reply_text, output_path, voice)
        audio_url = f"/audio/{os.path.basename(output_path)}"
    except Exception as e:
        print(f"TTS failed: {e}")

    return JSONResponse({
        "user_message": user_message,
        "reply_text": reply_text,
        "emotion": emotion,
        "detected_language": lang,
        "audio_url": audio_url
    })


@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)
