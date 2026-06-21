import os
import asyncio
from dotenv import load_dotenv
from groq import Groq
import sounddevice as sd
from scipy.io.wavfile import write
import time
import edge_tts
from playsound import playsound
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # makes langdetect deterministic

# Load the API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)

SAMPLE_RATE = 16000  # Whisper likes 16kHz
RECORD_FILE = "recording.wav"
RECORD_DURATION = 5  # seconds

# Edge-TTS voices per language — picked for natural pronunciation in each
VOICE_MAP = {
    "ar": "ar-TN-HediNeural",     # Tunisian Arabic
    "fr": "fr-FR-HenriNeural",    # French
    "en": "en-US-GuyNeural",      # English
}
DEFAULT_VOICE = VOICE_MAP["en"]


def detect_language(text):
    """Detect the dominant language of a text string. Falls back to 'en' on failure."""
    try:
        lang = detect(text)
        # langdetect returns codes like 'ar', 'fr', 'en' — map anything unrecognized to English
        if lang in VOICE_MAP:
            return lang
        return "en"
    except Exception:
        return "en"

def record_audio(duration=RECORD_DURATION):
    """Record from the default microphone for `duration` seconds and save as WAV."""
    print(f"🎙️  Recording for {duration} seconds... speak now.")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    write(RECORD_FILE, SAMPLE_RATE, audio)
    return RECORD_FILE


def transcribe_audio(file_path):
    """Send an audio file to Groq's Whisper endpoint and return the transcribed text."""
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
    """Convert text to speech using Edge-TTS and save as an audio file."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def play_audio(file_path):
    """Play an audio file through the default speakers."""
    playsound(file_path)


def speak(text):
    """Detect language, pick the matching voice, generate and play speech."""
    lang = detect_language(text)
    voice = VOICE_MAP.get(lang, DEFAULT_VOICE)
    print(f"(speaking in: {lang} → {voice})")

    output_file = f"reply_{int(time.time() * 1000)}.mp3"
    asyncio.run(generate_speech(text, output_file, voice))
    play_audio(output_file)
    try:
        os.remove(output_file)
    except OSError:
        pass  # ignore if Windows hasn't released the file lock yet


if __name__ == "__main__":
    # Clean up any leftover reply audio files from previous runs
    for f in os.listdir("."):
        if f.startswith("reply_") and f.endswith(".mp3"):
            try:
                os.remove(f)
            except OSError:
                pass

    print("Pixel is ready.")
    print("Press '+' then ENTER to talk. Type a message directly to type. Type 'quit' to exit.\n")

    while True:
        command = input("[ + to talk | type a message | 'quit' to exit ]: ").strip()

        if command.lower() == "quit":
            break

        elif command == "+":
            audio_path = record_audio()
            print("Transcribing...")
            transcribed_text = transcribe_audio(audio_path)
            print(f"You said: {transcribed_text}")

            reply = ask_pixel(transcribed_text)
            print(f"Pixel: {reply}\n")
            speak(reply)

        elif command == "":
            continue

        else:
            reply = ask_pixel(command)
            print(f"Pixel: {reply}\n")
            speak(reply)
