import os
import time
import asyncio
from contextlib import contextmanager
from datetime import datetime
import pytz

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
SUMMARY_EVERY_N_TURNS = 10  # summarize after every 10 user messages
TIMEZONE = "Africa/Tunis"


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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    id BIGSERIAL PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at FLOAT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotion_stats (
                    id BIGSERIAL PRIMARY KEY,
                    emotion TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    last_seen FLOAT NOT NULL,
                    UNIQUE(emotion)
                )
            """)
            # Ensure all emotion rows exist
            for emotion in ["happy", "sad", "thinking", "curious", "neutral", "excited"]:
                cur.execute("""
                    INSERT INTO emotion_stats (emotion, count, last_seen)
                    VALUES (%s, 0, 0)
                    ON CONFLICT (emotion) DO NOTHING
                """, (emotion,))


init_db()


# ---------- Memory helpers ----------
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


def get_user_message_count():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM conversation WHERE role = 'user'")
            return cur.fetchone()[0]


# ---------- User profile (long-term memory) ----------
def get_user_profile():
    """Get the latest user profile summary, or None if none exists yet."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT summary FROM user_profile ORDER BY updated_at DESC LIMIT 1"
            )
            row = cur.fetchone()
    return row["summary"] if row else None


def save_user_profile(summary):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_profile (summary, updated_at) VALUES (%s, %s)",
                (summary, time.time())
            )


def maybe_update_profile():
    """Every SUMMARY_EVERY_N_TURNS user messages, regenerate the user profile summary."""
    count = get_user_message_count()
    if count > 0 and count % SUMMARY_EVERY_N_TURNS == 0:
        history = get_all_history(limit=SUMMARY_EVERY_N_TURNS * 2)
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )
        emotion_summary = get_emotion_summary()

        prompt = f"""Based on this conversation history, write a short factual summary (5-10 bullet points max) of what you know about the user.
Include: their name if mentioned, interests, personality traits, mood patterns, topics they care about, and anything personal they've shared.
Be specific and concise. This will be used as context for future conversations.

Emotion pattern so far: {emotion_summary}

Conversation:
{history_text}

Write only the bullet points, nothing else."""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                timeout=20,
            )
            summary = response.choices[0].message.content.strip()
            save_user_profile(summary)
            print(f"User profile updated after {count} messages.")
        except Exception as e:
            print(f"Profile update failed: {e}")


# ---------- Emotion tracking ----------
def update_emotion_stats(emotion):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO emotion_stats (emotion, count, last_seen)
                VALUES (%s, 1, %s)
                ON CONFLICT (emotion)
                DO UPDATE SET count = emotion_stats.count + 1, last_seen = %s
            """, (emotion, time.time(), time.time()))


def get_emotion_stats():
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT emotion, count, last_seen FROM emotion_stats ORDER BY count DESC")
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_emotion_summary():
    """Return a short human-readable mood summary for injection into the system prompt."""
    stats = get_emotion_stats()
    total = sum(r["count"] for r in stats)
    if total == 0:
        return "No emotion data yet."

    top = [r for r in stats if r["count"] > 0][:3]
    parts = [f"{r['emotion']} ({r['count']} times)" for r in top]
    return f"User's most common moods: {', '.join(parts)}."


# ---------- Time awareness ----------
def get_time_context():
    """Return a natural language time context string for injection into the system prompt."""
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    hour = now.hour

    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"

    return (
        f"Current date and time: {now.strftime('%A, %B %d %Y')} at {now.strftime('%H:%M')} "
        f"({period} in Tunis, Tunisia)."
    )


# ---------- System prompt builder ----------
def build_system_prompt():
    """Dynamically build the system prompt with current time, user profile, mood context, and active character."""
    time_context = get_time_context()
    mood_context = get_emotion_summary()
    profile = get_user_profile()
    active_char_id = get_active_character()
    character = CHARACTERS.get(active_char_id, CHARACTERS["pixel"])

    profile_section = (
        f"\n\nWhat you know about the user:\n{profile}"
        if profile
        else "\n\nYou don't know much about the user yet — learn from this conversation."
    )

    mood_section = f"\n\nMood pattern: {mood_context}"

    return f"""{character['personality']}
You understand Arabic, French, and English, and reply in whichever language (or mix) the user uses.
Keep replies short (1-3 sentences) and conversational, like a companion speaking out loud, not writing an essay.
You remember past parts of the conversation and refer back to them naturally when relevant.
You are aware of the time and can reference it naturally when appropriate.

{time_context}{profile_section}{mood_section}

After your reply, on a new line, output exactly one emotion tag in this format:
[emotion: happy|sad|thinking|curious|neutral|excited]

Choose the emotion that best matches the tone of your reply."""


# ---------- Core helpers ----------
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
    system_prompt = build_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]
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


def process_after_reply(reply_text, emotion):
    """Run post-reply tasks: update emotion stats and maybe update user profile."""
    update_emotion_stats(emotion)
    maybe_update_profile()


# ---------- Shared response builder ----------
async def build_response(user_message, transcribed_text=None):
    """Core logic shared between voice and text endpoints."""
    timestamp = int(time.time() * 1000)
    lang = "en"

    try:
        raw_reply = ask_pixel(user_message)
    except Exception as e:
        return JSONResponse({"error": f"LLM request failed: {str(e)}"}, status_code=502)

    reply_text, emotion = parse_reply(raw_reply)
    save_message("user", user_message)
    save_message("assistant", reply_text, emotion)
    process_after_reply(reply_text, emotion)

    audio_url = None
    try:
        lang = detect_language(reply_text)
        voice = VOICE_MAP.get(lang, DEFAULT_VOICE)
        output_path = os.path.join(TEMP_DIR, f"reply_{timestamp}.mp3")
        await generate_speech(reply_text, output_path, voice)
        audio_url = f"/audio/{os.path.basename(output_path)}"
    except Exception as e:
        print(f"TTS failed: {e}")

    result = {
        "reply_text": reply_text,
        "emotion": emotion,
        "detected_language": lang,
        "audio_url": audio_url
    }
    if transcribed_text:
        result["transcribed_text"] = transcribed_text

    return JSONResponse(result)


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


@app.get("/profile")
def profile():
    """View the current user profile summary."""
    p = get_user_profile()
    return {"profile": p or "No profile generated yet. Chat more with Pixel!"}


@app.get("/mood")
def mood():
    """View emotion stats and current mood summary."""
    return {
        "stats": get_emotion_stats(),
        "summary": get_emotion_summary()
    }


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    """Voice endpoint — accepts an audio file."""
    timestamp = int(time.time() * 1000)
    input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.wav")

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

        return await build_response(transcribed_text, transcribed_text=transcribed_text)

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
    """Text endpoint — accepts JSON with a 'message' field."""
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "No message provided."}, status_code=400)
    return await build_response(user_message)


@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)


# ── Serve dashboard ──
from fastapi.responses import HTMLResponse

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the companion dashboard UI."""
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ── Character system ──
CHARACTERS = {
    "pixel": {
        "name": "Pixel",
        "description": "Your default companion. Warm, curious, slightly playful.",
        "color": "#7c6af7",
        "personality": "You are Pixel, a warm, curious, slightly playful AI companion.",
    },
    "nova": {
        "name": "Nova",
        "description": "A calm, analytical scientist. Precise and thoughtful.",
        "color": "#38bdf8",
        "personality": "You are Nova, a calm and analytical AI with a scientific mindset. You speak precisely and thoughtfully, often drawing on logic and curiosity about the world.",
    },
    "blaze": {
        "name": "Blaze",
        "description": "Energetic and hype. Always fired up.",
        "color": "#f97316",
        "personality": "You are Blaze, an energetic and enthusiastic AI companion. You speak with high energy, use casual language, and always hype up the person you're talking to.",
    },
    "sage": {
        "name": "Sage",
        "description": "Wise and poetic. Speaks in a calm, deep way.",
        "color": "#4ade80",
        "personality": "You are Sage, a wise and poetic AI companion. You speak calmly and deeply, often using metaphors and thoughtful observations about life.",
    },
    "glitch": {
        "name": "Glitch",
        "description": "A chaotic, funny, unpredictable trickster.",
        "color": "#f472b6",
        "personality": "You are Glitch, a chaotic and funny AI companion. You're unpredictable, love jokes and wordplay, and sometimes 'glitch' mid-sentence for comedic effect.",
    },
}

# Store active character in memory (persists until server restart, then resets to pixel)
# For full persistence, it's saved to the DB below
def get_active_character():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cur.execute("SELECT value FROM settings WHERE key = 'active_character'")
            row = cur.fetchone()
    return row[0] if row else "pixel"


def set_active_character(character_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cur.execute("""
                INSERT INTO settings (key, value) VALUES ('active_character', %s)
                ON CONFLICT (key) DO UPDATE SET value = %s
            """, (character_id, character_id))


@app.get("/characters")
def list_characters():
    """List all available characters."""
    active = get_active_character()
    return {
        "active": active,
        "characters": {k: {**v, "active": k == active} for k, v in CHARACTERS.items()}
    }


@app.post("/characters/{character_id}")
def switch_character(character_id: str):
    """Switch to a different character."""
    if character_id not in CHARACTERS:
        return JSONResponse({"error": f"Character '{character_id}' not found."}, status_code=404)
    set_active_character(character_id)
    char = CHARACTERS[character_id]
    return {"status": "switched", "character": char["name"], "message": f"Pixel is now {char['name']}!"}
