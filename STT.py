import streamlit as st
import pyaudio
import wave
import numpy as np
import threading
import os
import time
from faster_whisper import WhisperModel

class LiveAudioTranscriber:
    def __init__(self, model_size="medium", device="auto"):
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.recording = False
        self.frames = []
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize Whisper model
        self.model = WhisperModel(model_size, device=device)
        
        # Temporary file path
        self.temp_file = "temp_recording.wav"
    
    def start_recording(self):
        """Start recording audio from microphone"""
        self.recording = True
        self.frames = []
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
    
    def _record(self):
        """Record audio in a separate thread"""
        while self.recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
    
    def stop_recording(self):
        """Stop recording and transcribe the audio"""
        if not self.recording:
            return "Not currently recording."
        
        self.recording = False
        self.record_thread.join()
        
        # Close and terminate audio stream
        self.stream.stop_stream()
        self.stream.close()
        
        # Save recording to temporary file
        self._save_audio()
        
        # Transcribe the audio
        result = self._transcribe()
        
        # Clean up temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        
        return result
    
    def _save_audio(self):
        """Save recorded audio to a WAV file"""
        wf = wave.open(self.temp_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
    
    def _transcribe(self):
        """Transcribe the recorded audio using Faster Whisper"""
        segments, info = self.model.transcribe(self.temp_file, beam_size=5)
        
        # Collect transcription text
        text = " ".join([segment.text for segment in segments])
        return text
    
    def close(self):
        """Clean up resources"""
        self.audio.terminate()

# Create Streamlit app
st.title("Audio Transcription App")

# Create a text input field
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

text_input = st.text_input("Text Input", value=st.session_state.transcribed_text)

# Initialize transcriber in session state if not already there
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = LiveAudioTranscriber(model_size="medium")
    st.session_state.recording = False

col1, col2 = st.columns(2)

# Function to start recording
def start_recording():
    st.session_state.recording = True
    st.session_state.transcriber.start_recording()
    st.rerun()

# Function to stop recording
def stop_recording():
    st.session_state.recording = False
    transcribed_text = st.session_state.transcriber.stop_recording()
    st.session_state.transcribed_text = transcribed_text
    st.rerun()

# Create a button to start/stop recording
with col1:
    if not st.session_state.recording:
        st.button("🎤 Record Audio", on_click=start_recording)
    else:
        st.button("⏹️ Stop Recording", on_click=stop_recording)

# Display recording status
with col2:
    if st.session_state.recording:
        st.info("Recording... Click 'Stop Recording' when finished.")
    else:
        st.info("Click 'Record Audio' to start recording.")

# Handle app close
def on_close():
    if 'transcriber' in st.session_state:
        st.session_state.transcriber.close()

# Register the on_close function to be called when the app is closed
try:
    # This is a workaround as Streamlit doesn't have a built-in on_close event
    st.cache_resource.clear()
except:
    pass