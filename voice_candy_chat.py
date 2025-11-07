#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice interface for the candy delivery chat bot.
- Captures speech from the default microphone, saves to a temporary WAV, and
  transcribes locally with Whisper (via faster-whisper).
- Falls back to typed input when speech recognition fails.
- Uses pyttsx3 for text-to-speech so the robot can speak responses aloud.
"""

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
        print("Missing dependencies: {}".format(", ".join(missing)))
        print("Install inside the virtual environment:")
        print("  pip install speechrecognition pyttsx3 faster-whisper")
        sys.exit(1)


ensure_dependencies()

import speech_recognition as sr
import pyttsx3
from faster_whisper import WhisperModel

from candy_delivery_chat import CandyDeliveryChat


class VoiceCandyChat:
    def __init__(self):
        self.chat = CandyDeliveryChat()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.6
        self.recognizer.phrase_threshold = 0.1
        self.recognizer.non_speaking_duration = 0.3
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self._exit_event = threading.Event()
        self._exit_listener = threading.Thread(target=self._monitor_exit_key, daemon=True)
        self._exit_listener.start()
        model_name = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "auto")
        initial_compute = os.getenv("WHISPER_COMPUTE", "float16")
        compute_type = initial_compute
        self.whisper_model = None
        try:
            self.whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception as err:
            if compute_type != "float32":
                print(f"Failed to load Whisper model with compute '{compute_type}': {err}")
                print("Retrying with compute type 'float32'...")
                try:
                    compute_type = "float32"
                    self.whisper_model = WhisperModel(
                        model_name,
                        device=device,
                        compute_type=compute_type,
                    )
                except Exception as err2:
                    print(f"Failed to load Whisper model '{model_name}': {err2}")
                    print("Tip: download models ahead of time or choose a smaller model (e.g. 'base').")
                    sys.exit(1)
            else:
                print(f"Failed to load Whisper model '{model_name}': {err}")
                print("Tip: download models ahead of time or choose a smaller model (e.g. 'base').")
                sys.exit(1)
        self.configure_tts()

    def configure_tts(self):
        # Moderate speaking speed for clarity
        rate = self.engine.getProperty("rate")
        self.engine.setProperty("rate", int(rate * 0.9))
        self.engine.setProperty("volume", 0.85)
        try:
            for voice in self.engine.getProperty("voices"):
                if "en" in voice.id.lower() or "samantha" in voice.name.lower() or "alex" in voice.name.lower():
                    self.engine.setProperty("voice", voice.id)
                    break
        except Exception:
            pass

    def speak(self, text: str):
        print(f"Robot: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def _transcribe_with_whisper(self, audio: sr.AudioData) -> str:
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
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

    def listen(self) -> str:
        if self._exit_event.is_set():
            return "__EXIT__"
        # Automated voice capture: robot listens immediately after speaking.
        # Debug helper (keep for future use):
        # print("Press Enter, then speak. Stop talking to finish.")
        # user_input = input("(Leave blank to use voice, or type text): ")
        # if user_input.strip():
        #     return user_input.strip()

        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=None)

        try:
            text = self._transcribe_with_whisper(audio)
            if text:
                print(f"You (voice): {text}")
                return text
            print("Whisper could not transcribe audio. Please repeat or type.")
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again or type your message.")
        except Exception as err:
            print(f"Whisper transcription error: {err}")
        return ""

    def _monitor_exit_key(self):
        while True:
            try:
                user_input = input()
            except EOFError:
                break
            if user_input.strip().lower() == "q":
                self._exit_event.set()
                break

    def run(self):
        print("=" * 60)
        print("Candy Delivery Robot Voice Chat")
        print("Speak or type your responses. Type 'quit' to exit.")
        print("Tip: press 'q' + Enter at any time to end the session with a goodbye.")
        print("=" * 60)
        print()

        greeting = self.chat.start_greeting()
        self.speak(greeting)

        try:
            while True:
                user_text = ""
                while not user_text:
                    user_text = self.listen()
                if user_text == "__EXIT__":
                    self.speak("Fantastic! I'm glad we met. Time to find the next candy friend. See you soon!")
                    return

                if user_text.lower() in {"quit", "exit", "q"}:
                    print("Exiting voice chat...")
                    break

                response = self.chat.get_response(user_text)
                self.speak(response)
                time.sleep(0.3)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye!")


def main():
    VoiceCandyChat().run()


if __name__ == "__main__":
    main()
