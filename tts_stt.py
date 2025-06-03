# tts_stt.py
import os
import base64
import tempfile
import speech_recognition as sr
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import simpleaudio as sa
import sounddevice as sd
import numpy as np
import soundfile as sf
import io

load_dotenv()

# Ses ayarları
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = 'int16'
THRESHOLD = 0.03  # Ses algılama eşiği

VOICE_ID = os.getenv("VOICE_ID")  # Deniz
MODEL_NAME = os.getenv("MODEL_ID")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(api_key=ELEVEN_API_KEY)

def record_and_transcribe():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000  # Gürültü seviyesi ayarı
    recognizer.pause_threshold = 0.8   # Konuşma arası bekleme süresi
    
    with sr.Microphone() as source:
        print("🎙️ Mikrofon ayarlanıyor...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("🎙️ Dinleniyor...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Gelişmiş ses işleme
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            if np.max(np.abs(audio_data)) < 1000:  # Çok düşük ses seviyesi kontrolü
                print("⚠️ Ses seviyesi çok düşük")
                return None
                
            text = recognizer.recognize_google(audio, language="tr-TR")
            print(f"🗣️ Algılanan: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️ Zaman aşımı: Ses algılanmadı")
            return None
        except sr.UnknownValueError:
            print("❌ Anlaşılamadı")
            return None
        except Exception as e:
            print(f"⚠️ Hata: {str(e)}")
            return None



def text_to_speech(text):
    if not text:
        return

    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_NAME,
            text=text
        )

        # BytesIO'ya yaz
        with io.BytesIO() as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)
            f.seek(0)

            # soundfile ile oku (wav gibi düşün)
            data, samplerate = sf.read(f)
            
            # Çal
            sd.play(data, samplerate=samplerate)
            sd.wait()

    except Exception as e:
        print(f"Ses oynatma hatası: {str(e)}")
        raise

