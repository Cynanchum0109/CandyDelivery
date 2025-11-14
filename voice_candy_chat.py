#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import tempfile
import threading

def ensure_dependencies():
    missing = []
    try:
        import speech_recognition  # noqa: F401
    except ImportError:
        missing.append("speechrecognition")
    try:
        import pyttsx3  # noqa: F401
    except ImportError:
        missing.append("pyttsx3")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper")

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Install with:")
        print("  pip install speechrecognition pyttsx3 faster-whisper")
        sys.exit(1)

ensure_dependencies()

import speech_recognition as sr
import pyttsx3
from faster_whisper import WhisperModel

from candy_delivery_chat import CandyDeliveryChat
from face_server import FaceServer   # ← ★ ONLY using WebSocket face


class VoiceCandyChat:
    def __init__(self):
        # ------------------------------------------------------------ #
        #            WEBSOCKET FACE SERVER (replaces FaceDisplay)       #
        # ------------------------------------------------------------ #
        self.face = FaceServer()    # start WebSocket server

        self.chat = CandyDeliveryChat()

        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.6
        self.recognizer.phrase_threshold = 0.1
        self.recognizer.non_speaking_duration = 0.3
        self.microphone = sr.Microphone()

        # Text-to-speech
        self.engine = pyttsx3.init()

        # Events
        self._speech_playing = threading.Event()
        self._exit_event = threading.Event()
        self._can_listen = threading.Event()
        self._can_listen.set()

        # Exit listener
        self._exit_listener = threading.Thread(target=self._monitor_exit_key, daemon=True)
        self._exit_listener.start()

        # Whisper setup
        model_name = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "auto")
        compute_type = os.getenv("WHISPER_COMPUTE", "float16")

        try:
            self.whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            print("Retrying Whisper with float32...")
            self.whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type="float32",
            )

        self.configure_tts()

    # ------------------------------------------------------------------ #
    #                        TTS CONFIGURATION                           #
    # ------------------------------------------------------------------ #

    def configure_tts(self):
        rate = self.engine.getProperty("rate")
        self.engine.setProperty("rate", int(rate * 1.0))
        self.engine.setProperty("volume", 0.9)

        try:
            for voice in self.engine.getProperty("voices"):
                if "en" in voice.id.lower():
                    self.engine.setProperty("voice", voice.id)
                    break
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #                              SPEAK                                 #
    # ------------------------------------------------------------------ #

    def speak(self, text: str):
        print(f"Robot: {text}")

        self._can_listen.clear()
        self._speech_playing.set()

        # ★ NEW: WebSocket 表情切换
        self.face.send("speaking")

        self.engine.say(text)
        self.engine.runAndWait()

        # Reset state
        self._speech_playing.clear()
        self.face.send("idle")

        time.sleep(0.5)
        self._can_listen.set()

    # ------------------------------------------------------------------ #
    #                           WHISPER STT                              #
    # ------------------------------------------------------------------ #

    def _transcribe_with_whisper(self, audio: sr.AudioData) -> str:
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            # ★ NEW: 表情 → thinking
            self.face.send("thinking")

            segments, _ = self.whisper_model.transcribe(
                tmp_path,
                beam_size=1,
                vad_filter=True,
                language="en",
                task="transcribe",
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()

        finally:
            os.remove(tmp_path)

        return transcript

    # ------------------------------------------------------------------ #
    #                              LISTEN                                #
    # ------------------------------------------------------------------ #

    def listen(self) -> str:
        if self._exit_event.is_set():
            return "__EXIT__"

        self._can_listen.wait()

        # ★ NEW: 表情 → listening
        self.face.send("listening")

        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source)

        try:
            text = self._transcribe_with_whisper(audio)
            if text:
                print("You (voice):", text)
                return text
        except Exception as err:
            print("Error:", err)

        return ""

    # ------------------------------------------------------------------ #
    #                        EXIT MONITOR                                #
    # ------------------------------------------------------------------ #

    def _monitor_exit_key(self):
        while True:
            user_input = input().strip().lower()
            if user_input == "q":
                self._exit_event.set()
                try:
                    self.engine.stop()
                except:
                    pass
                break

    # ------------------------------------------------------------------ #
    #                             RUN LOOP                               #
    # ------------------------------------------------------------------ #

    def run(self):
        print("=" * 60)
        print("Candy Delivery Robot Voice Chat")
        print("Speak or type your responses. Type 'quit' to exit.")
        print("=" * 60)

        greeting = self.chat.start_greeting()
        self.speak(greeting)

        while True:
            user_text = self.listen()
            if not user_text:
                continue

            if user_text in {"quit", "exit", "__EXIT__"}:
                self.speak("See you soon! I'm off to deliver more candy!")
                return

            response = self.chat.get_response(user_text)
            self.speak(response)


def main():
    VoiceCandyChat().run()


if __name__ == "__main__":
    main()
