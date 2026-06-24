import os
import time
import asyncio
from contextlib import contextmanager
from datetime import datetime
import pytz
import httpx

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from groq import Groq
import edge_tts
from langdetect import detect, DetectorFactory
from apscheduler.schedulers.asyncio import AsyncIOScheduler

DetectorFactory.seed = 0

# ---------- Setup ----------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
database_url = os.getenv("DATABASE_URL")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
weather_city = os.getenv("WEATHER_CITY", "Tunis")

if not api_key:
    raise ValueError("GROQ_API_KEY not found.")
if not database_url:
    raise ValueError("DATABASE_URL not found.")

client = Groq(api_key=api_key)
app = FastAPI(title="Pixel Companion Server")
scheduler = AsyncIOScheduler()

VOICE_MAP = {
    "ar": "ar-TN-HediNeural",
    "fr": "fr-FR-HenriNeural",
    "en": "en-US-GuyNeural",
}
DEFAULT_VOICE = VOICE_MAP["en"]
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)
MAX_HISTORY_TURNS = 10
SUMMARY_EVERY_N_TURNS = 10
TIMEZONE = "Africa/Tunis"

CHARACTERS = {
    "pixel": {
        "name": "Pixel", "description": "Your default companion. Warm, curious, slightly playful.",
        "color": "#7c6af7",
        "personality": "You are Pixel, a warm, curious, slightly playful AI companion.",
    },
    "nova": {
        "name": "Nova", "description": "A calm, analytical scientist. Precise and thoughtful.",
        "color": "#38bdf8",
        "personality": "You are Nova, a calm and analytical AI with a scientific mindset. You speak precisely and thoughtfully, often drawing on logic and curiosity about the world.",
    },
    "blaze": {
        "name": "Blaze", "description": "Energetic and hype. Always fired up.",
        "color": "#f97316",
        "personality": "You are Blaze, an energetic and enthusiastic AI companion. You speak with high energy, use casual language, and always hype up the person you're talking to.",
    },
    "sage": {
        "name": "Sage", "description": "Wise and poetic. Speaks in a calm, deep way.",
        "color": "#4ade80",
        "personality": "You are Sage, a wise and poetic AI companion. You speak calmly and deeply, often using metaphors and thoughtful observations about life.",
    },
    "glitch": {
        "name": "Glitch", "description": "A chaotic, funny, unpredictable trickster.",
        "color": "#f472b6",
        "personality": "You are Glitch, a chaotic and funny AI companion. You're unpredictable, love jokes and wordplay, and sometimes 'glitch' mid-sentence for comedic effect.",
    },
}


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
            cur.execute("""CREATE TABLE IF NOT EXISTS conversation (
                id BIGSERIAL PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
                emotion TEXT, timestamp FLOAT NOT NULL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS user_profile (
                id BIGSERIAL PRIMARY KEY, summary TEXT NOT NULL, updated_at FLOAT NOT NULL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS emotion_stats (
                id BIGSERIAL PRIMARY KEY, emotion TEXT NOT NULL UNIQUE,
                count INTEGER DEFAULT 1, last_seen FLOAT NOT NULL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS reminders (
                id BIGSERIAL PRIMARY KEY, content TEXT NOT NULL,
                due_at FLOAT, created_at FLOAT NOT NULL, done BOOLEAN DEFAULT FALSE)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS proactive_messages (
                id BIGSERIAL PRIMARY KEY, content TEXT NOT NULL,
                created_at FLOAT NOT NULL, read BOOLEAN DEFAULT FALSE)""")
            for emotion in ["happy", "sad", "thinking", "curious", "neutral", "excited"]:
                cur.execute("""INSERT INTO emotion_stats (emotion, count, last_seen)
                    VALUES (%s, 0, 0) ON CONFLICT (emotion) DO NOTHING""", (emotion,))


init_db()


# ---------- Memory ----------
def save_message(role, content, emotion=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation (role, content, emotion, timestamp) VALUES (%s, %s, %s, %s)",
                (role, content, emotion, time.time()))


def get_recent_history(limit=MAX_HISTORY_TURNS):
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT role, content FROM conversation ORDER BY id DESC LIMIT %s", (limit * 2,))
            rows = cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_all_history(limit=50):
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, emotion, timestamp FROM conversation ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]


def get_user_message_count():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM conversation WHERE role = 'user'")
            return cur.fetchone()[0]


# ---------- User profile ----------
def get_user_profile():
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT summary FROM user_profile ORDER BY updated_at DESC LIMIT 1")
            row = cur.fetchone()
    return row["summary"] if row else None


def save_user_profile(summary):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO user_profile (summary, updated_at) VALUES (%s, %s)", (summary, time.time()))


def maybe_update_profile():
    count = get_user_message_count()
    if count > 0 and count % SUMMARY_EVERY_N_TURNS == 0:
        history = get_all_history(limit=SUMMARY_EVERY_N_TURNS * 2)
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        emotion_summary = get_emotion_summary()
        prompt = f"""Based on this conversation, write 5-10 bullet points summarizing what you know about the user.
Include name, interests, personality, mood patterns, topics they care about.
Emotion pattern: {emotion_summary}
Conversation:\n{history_text}\nWrite only the bullet points."""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}], timeout=20)
            save_user_profile(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Profile update failed: {e}")


# ---------- Emotion ----------
def update_emotion_stats(emotion):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO emotion_stats (emotion, count, last_seen) VALUES (%s, 1, %s)
                ON CONFLICT (emotion) DO UPDATE SET count = emotion_stats.count + 1, last_seen = %s""",
                (emotion, time.time(), time.time()))


def get_emotion_stats():
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT emotion, count, last_seen FROM emotion_stats ORDER BY count DESC")
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_emotion_summary():
    stats = get_emotion_stats()
    total = sum(r["count"] for r in stats)
    if total == 0:
        return "No emotion data yet."
    top = [r for r in stats if r["count"] > 0][:3]
    return f"User's most common moods: {', '.join(f'{r[\"emotion\"]} ({r[\"count\"]} times)' for r in top)}."


# ---------- Settings / Character ----------
def get_setting(key, default=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cur.fetchone()
    return row[0] if row else default


def set_setting(key, value):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO settings (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = %s""", (key, value, value))


def get_active_character():
    return get_setting("active_character", "pixel")


def set_active_character(character_id):
    set_setting("active_character", character_id)


# ---------- Reminders ----------
def save_reminder(content, due_at=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reminders (content, due_at, created_at) VALUES (%s, %s, %s)",
                (content, due_at, time.time()))


def get_pending_reminders():
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM reminders WHERE done = FALSE ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def mark_reminder_done(reminder_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE reminders SET done = TRUE WHERE id = %s", (reminder_id,))


def get_all_reminders():
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM reminders ORDER BY created_at DESC LIMIT 20")
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------- Proactive messages ----------
def save_proactive_message(content):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO proactive_messages (content, created_at) VALUES (%s, %s)",
                (content, time.time()))


def get_unread_proactive():
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM proactive_messages WHERE read = FALSE ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def mark_proactive_read(msg_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE proactive_messages SET read = TRUE WHERE id = %s", (msg_id,))


# ---------- Skills ----------
async def get_weather():
    """Fetch current weather for configured city."""
    if not weather_api_key:
        return "Weather API key not configured."
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            res = await http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": weather_city, "appid": weather_api_key, "units": "metric"})
            if res.status_code != 200:
                return f"Could not fetch weather (status {res.status_code})."
            data = res.json()
            temp = data["main"]["temp"]
            feels = data["main"]["feels_like"]
            desc = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            return f"{weather_city}: {desc}, {temp:.1f}°C (feels like {feels:.1f}°C), humidity {humidity}%"
    except Exception as e:
        return f"Weather fetch failed: {e}"


async def search_web(query):
    """Search DuckDuckGo instant answers — free, no API key needed."""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            res = await http.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
            data = res.json()
            abstract = data.get("AbstractText", "")
            answer = data.get("Answer", "")
            result = answer or abstract
            if not result:
                # Fall back to related topics
                topics = data.get("RelatedTopics", [])
                snippets = [t.get("Text", "") for t in topics[:3] if isinstance(t, dict) and t.get("Text")]
                result = " | ".join(snippets) if snippets else "No results found."
            return result[:500]
    except Exception as e:
        return f"Search failed: {e}"


def get_current_time():
    """Return current time and date in Tunis."""
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    return now.strftime("%A, %B %d %Y — %H:%M")


# ---------- Time context ----------
def get_time_context():
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    hour = now.hour
    if 5 <= hour < 12: period = "morning"
    elif 12 <= hour < 17: period = "afternoon"
    elif 17 <= hour < 21: period = "evening"
    else: period = "night"
    return (f"Current date and time: {now.strftime('%A, %B %d %Y')} at {now.strftime('%H:%M')} "
            f"({period} in Tunis, Tunisia).")


# ---------- System prompt ----------
def build_system_prompt(extra_context=""):
    time_context = get_time_context()
    mood_context = get_emotion_summary()
    profile = get_user_profile()
    active_char_id = get_active_character()
    character = CHARACTERS.get(active_char_id, CHARACTERS["pixel"])
    reminders = get_pending_reminders()

    profile_section = (f"\n\nWhat you know about the user:\n{profile}"
        if profile else "\n\nYou don't know much about the user yet — learn from this conversation.")
    mood_section = f"\n\nMood pattern: {mood_context}"
    reminder_section = ""
    if reminders:
        reminder_list = "\n".join(f"- {r['content']}" for r in reminders[:5])
        reminder_section = f"\n\nPending reminders the user has set:\n{reminder_list}"
    extra_section = f"\n\nReal-time information:\n{extra_context}" if extra_context else ""

    return f"""{character['personality']}
You understand Arabic, French, and English, and reply in whichever language (or mix) the user uses.
Keep replies short (1-3 sentences) and conversational, like a companion speaking out loud.
You remember past conversations and refer back to them naturally when relevant.
You are aware of the time and reference it naturally when appropriate.

You have the following abilities — use them when relevant:
- You know the current weather (provided in real-time context when fetched)
- You can set reminders for the user by saying "I'll remind you about [X]" — the system will detect and save it
- You can answer factual questions using web search results (provided when relevant)
- You always know the current time and date

{time_context}{profile_section}{mood_section}{reminder_section}{extra_section}

After your reply, on a new line, output exactly one emotion tag:
[emotion: happy|sad|thinking|curious|neutral|excited]

Also, if the user asked you to remind them of something, add on a new line:
[reminder: the thing to remember]

Choose the emotion that best matches the tone of your reply."""


# ---------- Reply parser ----------
def parse_reply(raw_reply):
    emotion = "neutral"
    reminder = None
    text = raw_reply.strip()

    if "[reminder:" in text:
        try:
            parts = text.split("[reminder:")
            reminder = parts[-1].replace("]", "").strip()
            text = parts[0].strip()
        except Exception:
            pass

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

    return text, emotion, reminder


# ---------- Core helpers ----------
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in VOICE_MAP else "en"
    except Exception:
        return "en"


def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3", response_format="text")
    if not transcription or not transcription.strip():
        raise ValueError("Transcription returned empty.")
    return transcription.strip()


async def ask_pixel(user_message):
    """Ask the LLM, injecting real-time context (weather/search) when relevant."""
    lower = user_message.lower()
    extra_context = ""

    # Inject weather if asked
    if any(w in lower for w in ["weather", "température", "météo", "طقس", "حرارة", "hot", "cold", "rain"]):
        weather = await get_weather()
        extra_context += f"Current weather: {weather}\n"

    # Inject web search if asked a factual question
    if any(w in lower for w in ["what is", "who is", "when did", "how does", "define", "explain",
                                  "search", "look up", "find", "ما هو", "من هو", "كيف", "qu'est"]):
        results = await search_web(user_message)
        extra_context += f"Web search results for '{user_message}':\n{results}\n"

    # Always inject current time if asked
    if any(w in lower for w in ["time", "date", "day", "today", "وقت", "تاريخ", "heure", "aujourd"]):
        extra_context += f"Exact current time: {get_current_time()}\n"

    history = get_recent_history()
    system_prompt = build_system_prompt(extra_context)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, timeout=20)
    raw_reply = response.choices[0].message.content
    if not raw_reply or not raw_reply.strip():
        raise ValueError("LLM returned empty response.")
    return raw_reply


async def generate_speech(text, output_file, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def process_after_reply(reply_text, emotion, reminder):
    update_emotion_stats(emotion)
    if reminder:
        save_reminder(reminder)
        print(f"Reminder saved: {reminder}")
    maybe_update_profile()


# ---------- Shared response builder ----------
async def build_response(user_message, transcribed_text=None):
    timestamp = int(time.time() * 1000)
    lang = "en"
    try:
        raw_reply = await ask_pixel(user_message)
    except Exception as e:
        return JSONResponse({"error": f"LLM request failed: {str(e)}"}, status_code=502)

    reply_text, emotion, reminder = parse_reply(raw_reply)
    save_message("user", user_message)
    save_message("assistant", reply_text, emotion)
    process_after_reply(reply_text, emotion, reminder)

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
        "audio_url": audio_url,
        "reminder_saved": reminder or None
    }
    if transcribed_text:
        result["transcribed_text"] = transcribed_text
    return JSONResponse(result)


# ---------- Proactive scheduler ----------
async def morning_greeting():
    """Runs every day at 8am Tunis time."""
    profile = get_user_profile()
    weather = await get_weather()
    mood = get_emotion_summary()
    char_id = get_active_character()
    char = CHARACTERS.get(char_id, CHARACTERS["pixel"])

    prompt = f"""You are {char['name']}, a companion AI. Generate a warm, short morning greeting (2-3 sentences).
Current weather in Tunis: {weather}
What you know about the user: {profile or 'Not much yet.'}
Recent mood pattern: {mood}
Make it feel personal and reference the weather naturally. Add [emotion: happy] at the end."""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}], timeout=15)
        msg = res.choices[0].message.content.strip()
        msg = msg.replace("[emotion: happy]", "").strip()
        save_proactive_message(msg)
        print(f"Morning greeting saved: {msg[:60]}...")
    except Exception as e:
        print(f"Morning greeting failed: {e}")


async def evening_checkin():
    """Runs every day at 8pm Tunis time."""
    mood = get_emotion_summary()
    reminders = get_pending_reminders()
    char_id = get_active_character()
    char = CHARACTERS.get(char_id, CHARACTERS["pixel"])

    reminder_text = ""
    if reminders:
        reminder_text = f"Pending reminders: {', '.join(r['content'] for r in reminders[:3])}"

    prompt = f"""You are {char['name']}, a companion AI. Generate a short, warm evening check-in message (2-3 sentences).
Recent mood pattern: {mood}
{reminder_text}
If there are reminders, casually mention them. Add [emotion: curious] at the end."""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}], timeout=15)
        msg = res.choices[0].message.content.strip()
        msg = msg.replace("[emotion: curious]", "").strip()
        save_proactive_message(msg)
        print(f"Evening check-in saved: {msg[:60]}...")
    except Exception as e:
        print(f"Evening check-in failed: {e}")


async def reminder_checker():
    """Runs every hour — nudges about reminders set more than 1 hour ago."""
    reminders = get_pending_reminders()
    if not reminders:
        return
    one_hour_ago = time.time() - 3600
    old = [r for r in reminders if r["created_at"] < one_hour_ago]
    if not old:
        return
    reminder_list = ", ".join(r["content"] for r in old[:3])
    save_proactive_message(f"Hey, just a nudge — you wanted to remember: {reminder_list}")


# ---------- App lifespan ----------
@app.on_event("startup")
async def startup():
    tz = pytz.timezone(TIMEZONE)
    scheduler.add_job(morning_greeting, "cron", hour=8, minute=0, timezone=tz)
    scheduler.add_job(evening_checkin, "cron", hour=20, minute=0, timezone=tz)
    scheduler.add_job(reminder_checker, "interval", hours=1)
    scheduler.start()
    print("Scheduler started.")


@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()


# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "Pixel server is running"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


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
    p = get_user_profile()
    return {"profile": p or "No profile generated yet."}


@app.get("/mood")
def mood():
    return {"stats": get_emotion_stats(), "summary": get_emotion_summary()}


@app.get("/weather")
async def weather():
    return {"weather": await get_weather()}


@app.get("/search")
async def search(q: str):
    return {"query": q, "results": await search_web(q)}


@app.get("/reminders")
def reminders():
    return {"reminders": get_all_reminders()}


@app.delete("/reminders/{reminder_id}")
def complete_reminder(reminder_id: int):
    mark_reminder_done(reminder_id)
    return {"status": "done"}


@app.get("/notifications")
def notifications():
    """Get unread proactive messages from Pixel."""
    msgs = get_unread_proactive()
    for m in msgs:
        mark_proactive_read(m["id"])
    return {"messages": msgs}


@app.get("/characters")
def list_characters():
    active = get_active_character()
    return {"active": active, "characters": {k: {**v, "active": k == active} for k, v in CHARACTERS.items()}}


@app.post("/characters/{character_id}")
def switch_character(character_id: str):
    if character_id not in CHARACTERS:
        return JSONResponse({"error": f"Character '{character_id}' not found."}, status_code=404)
    set_active_character(character_id)
    char = CHARACTERS[character_id]
    return {"status": "switched", "character": char["name"]}


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
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
