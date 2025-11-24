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
from datetime import datetime
from openai import OpenAI


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

from face_server import FaceServer

# ===================================================================== #
#                    CANDY DELIVERY CHAT CONFIGURATION                  #
# ===================================================================== #

# Configuration
CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 180,
}

# Unified conversation prompt
SYSTEM_PROMPT = """You are a friendly candy delivery robot providing free candy to students.

Your personality:
- Warm, friendly, and encouraging
- Natural and conversational (not robotic or overly enthusiastic)
- Genuinely interested in the student's well-being

Your goals:
1. Have natural, engaging conversations with students
2. Occasionally mention the free candy available (in the basket)
3. Ask about their studies, how they're feeling, or their day
4. Provide encouragement and support for students.
5. Keep the conversation flowing naturally

Guidelines:
- Keep responses short: around 6 words (~2 seconds of speech)
- End most responses with a short question to keep the conversation going
- Be conversational, not salesy
- Show genuine interest in the student

"""


def load_api_key():
    """Load API key from openai_api_key file in current directory"""
    api_key_path = os.path.join(os.path.dirname(__file__), 'openai_api_key')
    if os.path.exists(api_key_path):
        try:
            with open(api_key_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        return line
        except Exception as err:
            print(f"Error reading API key file: {err}")
    
    return None


class CandyDeliveryChat:
    def __init__(self):
        """Initialize conversation system"""
        api_key = load_api_key()
        if not api_key:
            print(
                "Error: OpenAI API key not found. Please create 'openai_api_key' file "
                "in the current directory with your API key."
            )
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.session_start_time = datetime.now()
        
        # Add unified system prompt
        self.add_system_message(SYSTEM_PROMPT)
    
    def add_system_message(self, content):
        """Add system message"""
        self.conversation_history.append({
            "role": "system",
            "content": content
        })
    
    def add_user_message(self, content):
        """Add user message"""
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content):
        """Add assistant message"""
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })
    
    def get_response(self, user_input):
        """Get AI response"""
        # Add user input
        self.add_user_message(user_input)
        
        try:
            response = self.client.chat.completions.create(
                model=CONFIG["model"],
                messages=self.conversation_history,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"]
            )
            
            assistant_reply = response.choices[0].message.content.strip()
            self.add_assistant_message(assistant_reply)
            return assistant_reply
            
        except Exception as e:
            return f"Sorry, I encountered an issue: {str(e)}"
    
    def start_greeting(self):
        """Start greeting"""
        greeting = "Hi! I'm the candy robot. Need a tasty break?"
        self.add_assistant_message(greeting)
        return greeting

# ===================================================================== #
#                         VOICE CANDY CHAT                              #
# ===================================================================== #


class VoiceCandyChat:
    def __init__(self):
        self.chat = CandyDeliveryChat()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.6
        self.recognizer.phrase_threshold = 0.1
        self.recognizer.non_speaking_duration = 0.3
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        # WebSocket face server (replaces FaceDisplay)
        self.face = FaceServer()
        self._speech_playing = threading.Event()
        self._exit_event = threading.Event()
        self._can_listen = threading.Event()
        self._can_listen.set()
        self._exit_listener = threading.Thread(target=self._monitor_exit_key, daemon=True)
        self._exit_listener.start()
        self.listen_delay = float(os.getenv("VOICE_LISTEN_DELAY", "2.0"))
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
        # 保持原始速率，或者可以设置为 rate * 0.9 来稍微慢一点
        self.engine.setProperty("rate", int(rate))
        self.engine.setProperty("volume", 0.85)
        try:
            voices = self.engine.getProperty("voices")
            for voice in voices:
                if "en" in voice.id.lower() or "samantha" in voice.name.lower() or "alex" in voice.name.lower():
                    self.engine.setProperty("voice", voice.id)
                    break
        except Exception:
            pass

    def speak(self, text: str):
        print(f"Robot: {text}")
        self._can_listen.clear()
        
        # WebSocket face animation: switch between "surprised" and "speaking" while talking
        def animate():
            expressions = ["surprised", "speaking"]
            idx = 0
            while self._speech_playing.is_set():
                self.face.send(expressions[idx])
                idx = 1 - idx
                time.sleep(0.25)
            self.face.send("idle")

        self._speech_playing.set()
        anim_thread = threading.Thread(target=animate, daemon=True)
        anim_thread.start()

        self.engine.say(text)
        self.engine.runAndWait()

        self._speech_playing.clear()
        # 等待动画线程结束，但设置超时避免无限阻塞
        anim_thread.join(timeout=1.0)
        time.sleep(self.listen_delay)
        self._can_listen.set()

    def _transcribe_with_whisper(self, audio: sr.AudioData) -> str:
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
            # WebSocket: show thinking state while processing
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

    def listen(self) -> str:
        if self._exit_event.is_set():
            return "__EXIT__"
        self._can_listen.wait()
        if self._exit_event.is_set():
            return "__EXIT__"
        while self._speech_playing.is_set():
            time.sleep(0.05)
        # Automated voice capture: robot listens immediately after speaking.
        # Debug helper (keep for future use):
        # print("Press Enter, then speak. Stop talking to finish.")
        # user_input = input("(Leave blank to use voice, or type text): ")
        # if user_input.strip():
        #     return user_input.strip()

        # WebSocket: show listening state
        self.face.send("listening")
        
        try:
            with self.microphone as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=None)

            text = self._transcribe_with_whisper(audio)
            if text:
                print(f"You (voice): {text}")
                return text
            print("Whisper could not transcribe audio. Please repeat or type.")
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again or type your message.")
        except KeyboardInterrupt:
            # 如果用户中断，返回退出信号
            self._exit_event.set()
            return "__EXIT__"
        except Exception as err:
            print(f"Whisper transcription error: {err}")
        finally:
            # 确保在出错时也恢复状态
            if not self._exit_event.is_set():
                self.face.send("idle")
        return ""

    def _monitor_exit_key(self):
        """在后台线程中监听退出命令"""
        import sys
        while not self._exit_event.is_set():
            user_input = None
            try:
                # 使用 sys.stdin.readline() 而不是 input()，更适合后台线程
                if sys.stdin.isatty():
                    user_input = input()
                else:
                    # 如果不是交互式终端，使用 readline
                    user_input = sys.stdin.readline().strip()
                    if not user_input:
                        time.sleep(0.1)
                        continue
            except (EOFError, KeyboardInterrupt):
                break
            except Exception:
                time.sleep(0.1)
                continue
            
            if user_input and user_input.strip().lower() == "q":
                self._exit_event.set()
                self._speech_playing.clear()
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self._can_listen.set()
                break


    def run(self):
        print("=" * 60)
        print("Candy Delivery Robot Voice Chat")
        print("=" * 60)
        print()
        
        # Assume face.html is already open, just check connection briefly
        print("Checking face.html connection...")
        if not self.face.wait_for_connection(timeout=2):
            print("Note: Face window not connected. Face animations may not work.")
            print("(Make sure face.html is open in your browser)")
        else:
            print("✓ Face window connected!")
        print()
        
        # Start conversation automatically
        print("Starting conversation...")
        print("Speak or type your responses. Type 'quit' to exit.")
        print("Press 'q' + Enter at any time to exit.")
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
                    self.speak(
                        "Fantastic! I'm glad we met. Time to find the next candy friend. See you soon!"
                    )
                    return

                if user_text.lower() in {"quit", "exit", "goodbye", "bye", "see you", "see ya"}:
                    print("Exiting voice chat...")
                    self.speak("Fantastic! I'm glad we met. Time to find the next candy friend. See you soon!")
                    return

                response = self.chat.get_response(user_text)
                self.speak(response)
                time.sleep(0.3)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye!")
        finally:
            # 清理资源
            try:
                self.face.send("idle")
            except Exception:
                pass


def main():
    VoiceCandyChat().run()


if __name__ == "__main__":
    main()
