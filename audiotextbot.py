import sys
import os
import requests
import json
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit,
    QHBoxLayout, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QCloseEvent

# 1. Hugging Face ASR Model
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

# --- Configuration ---
OPENROUTER_API_KEY = ""
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-chat"

# Audio Recording Parameters
SAMPLERATE = 16000
CHUNK_SIZE = 1024

# --- Thread for Audio Recording and ASR Processing ---
class AudioRecorderThread(QThread):
    transcription_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, processor, model):
        super().__init__()
        self.processor = processor
        self.model = model
        self.recording = False
        self.frames = []
        self.stream = None

    def run(self):
        self.recording = True
        try:
            with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32',
                                callback=self._audio_callback) as self.stream:
                print("Recording started...")
                while self.recording:
                    sd.sleep(100)
        except Exception as e:
            self.error_signal.emit(f"Audio recording error: {e}")
            self.recording = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.frames.append(indata.copy())
            if len(self.frames) * CHUNK_SIZE >= SAMPLERATE * 10:
                self.process_audio_chunk()

    def process_audio_chunk(self):
        if not self.frames:
            return

        audio_data = np.concatenate(self.frames, axis=0).flatten()
        self.frames = []

        try:
            input_features = self.processor(audio_data, sampling_rate=SAMPLERATE, return_tensors="pt").input_features
            predicted_ids = self.model.generate(input_features,language='el')
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"ASR raw transcription: {transcription}")
            self.transcription_ready.emit(transcription)

        except Exception as e:
            self.error_signal.emit(f"ASR processing error: {e}")

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Recording stopped.")


# --- Thread for OpenRouter API Calls ---
class OpenRouterAPIThread(QThread):
    api_response_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, api_key, api_url, model_name):
        super().__init__()
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.text_to_send = ""

    def run(self):
        if not self.text_to_send:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": self.text_to_send}
            ]
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if data and "choices" in data and len(data["choices"]) > 0:
                self.api_response_ready.emit(data["choices"][0]["message"]["content"])
            else:
                self.error_signal.emit("No valid response from DeepSeek API.")
        except requests.exceptions.RequestException as e:
            self.error_signal.emit(f"OpenRouter API error: {e}")
        finally:
            self.text_to_send = ""


# --- PyQt5 GUI with Dark Theme ---
class AudioToTextBot(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_dark_theme()
        self.init_ui()
        self.load_models()
        self.recorder_thread = None
        self.api_thread = None

    def setup_dark_theme(self):
        # Dark color palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(dark_palette)
        
        # Additional style tweaks
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dcdcdc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                font-weight: bold;
                color: #61dafb;
                font-size: 12pt;
            }
            QTextEdit {
                background-color: #252526;
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                padding: 8px;
                font-size: 11pt;
                selection-background-color: #0078d7;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 120px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #1c97ea;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #a0a0a0;
            }
            QFrame {
                background-color: #2d2d30;
                border-radius: 8px;
            }
        """)

    def init_ui(self):
        self.setWindowTitle('Audio-to-Text Assistant')
        self.setGeometry(100, 100, 800, 700)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Status panel
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(15, 15, 15, 15)
        
        self.status_label = QLabel("Ready to record. Click Start Recording to begin.")
        self.status_label.setStyleSheet("font-size: 12pt;")
        status_layout.addWidget(self.status_label)
        
        # Button panel
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # Transcription panel
        trans_frame = QFrame()
        trans_frame.setObjectName("transFrame")
        trans_layout = QVBoxLayout(trans_frame)
        trans_layout.setContentsMargins(15, 15, 15, 15)
        
        self.transcription_label = QLabel("Transcription:")
        trans_layout.addWidget(self.transcription_label)
        
        self.transcription_display = QTextEdit()
        self.transcription_display.setReadOnly(True)
        self.transcription_display.setMinimumHeight(150)
        trans_layout.addWidget(self.transcription_display)

        # Response panel
        resp_frame = QFrame()
        resp_frame.setObjectName("respFrame")
        resp_layout = QVBoxLayout(resp_frame)
        resp_layout.setContentsMargins(15, 15, 15, 15)
        
        self.response_label = QLabel("Bot Response:")
        resp_layout.addWidget(self.response_label)
        
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        self.response_display.setMinimumHeight(200)
        resp_layout.addWidget(self.response_display)

        # Assemble main layout
        main_layout.addWidget(status_frame)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(trans_frame, 1)
        main_layout.addWidget(resp_frame, 2)

        self.setLayout(main_layout)

    def load_models(self):
        self.status_label.setText("Loading ASR model... This might take a moment.")
        QApplication.processEvents()
        try:
            self.processor = AutoProcessor.from_pretrained("AqeelShafy7/AudioSangraha-Audio_to_Text")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained("AqeelShafy7/AudioSangraha-Audio_to_Text")
            self.status_label.setText("ASR model loaded. Ready to record.")
        except Exception as e:
            self.status_label.setText(f"Error loading ASR model: {e}")
            print(f"Error loading ASR model: {e}")

    def start_recording(self):
        if not self.processor or not self.model:
            self.status_label.setText("ASR model not loaded yet. Please wait or restart.")
            return

        self.transcription_display.clear()
        self.response_display.clear()
        self.status_label.setText("Recording... Speak now!")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.recorder_thread = AudioRecorderThread(self.processor, self.model)
        self.recorder_thread.transcription_ready.connect(self.handle_transcription)
        self.recorder_thread.error_signal.connect(self.handle_error)
        self.recorder_thread.start()

    def stop_recording(self):
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait()
        self.status_label.setText("Recording stopped. Processing final transcription...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def handle_transcription(self, text):
        current_text = self.transcription_display.toPlainText()
        self.transcription_display.setText(current_text + text + " ")
        self.process_for_deepseek(text)

    def process_for_deepseek(self, transcribed_text):
        if transcribed_text.strip():
            self.api_thread = OpenRouterAPIThread(OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL)
            self.api_thread.text_to_send = transcribed_text
            self.api_thread.api_response_ready.connect(self.handle_deepseek_response)
            self.api_thread.error_signal.connect(self.handle_error)
            self.api_thread.start()

    def handle_deepseek_response(self, response_text):
        self.response_display.setText(response_text)

    def handle_error(self, message):
        self.status_label.setText(f"Error: {message}")
        print(f"Error: {message}")

    def closeEvent(self, a0):
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait()
        if self.api_thread and self.api_thread.isRunning():
            self.api_thread.wait()
        a0.accept()

# --- Main Application Entry Point ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    bot = AudioToTextBot()
    bot.show()
    sys.exit(app.exec_())