import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import os
import tempfile
import subprocess
import wave
import imageio_ffmpeg

_ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

def convert_to_wav_bytes(input_bytes):
    process = subprocess.Popen(
        [_ffmpeg_path, "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = process.communicate(input=input_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[-500:]}")
    return out
import numpy as np
import matplotlib.pyplot as plt
import requests
import speech_recognition as sr
import google.generativeai as genai

st.set_page_config(page_title="Speak & See", page_icon="🎙️")

st.title("Speak & See")


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-3.5-flash-lite")

if "transcription" not in st.session_state:
    st.session_state.transcription = ""

if "edit_box" not in st.session_state:
    st.session_state.edit_box = ""

st.caption("idle")
st.caption("canvas")

audio = mic_recorder(
    start_prompt="Start speaking",
    stop_prompt="Stop speaking",
    key="recorder",
)

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if audio and audio["id"] != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio["id"]

    st.audio(audio["bytes"])

    wav_bytes = convert_to_wav_bytes(audio["bytes"])

    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(samples, linewidth=0.5)
    ax.axis("off")
    st.pyplot(fig)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        with st.spinner("Transcribing..."):
            text = recognizer.recognize_google(audio_data, language="en-IN")

        if text:
            st.session_state.transcription = text
            st.session_state.edit_box = text
            st.rerun()
        else:
            st.warning("Could not understand the audio. Try speaking clearly.")

    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Try speaking clearly.")
    except Exception as e:
        st.error(f"Speech recognition error: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


st.subheader("live captions")
st.caption("Your words will appear here as you speak.")

st.text_area(
    "edit before fixing (correct any misheard words here)",
    key="edit_box",
    height=140,
)

st.caption(
    "Fix grammar mistakes uses whatever is in this box, not the raw captions above."
)


def fix_grammar(text: str) -> str:
    """Fix grammar using Google Gemini (free tier) for much higher accuracy."""

    cleaned = text.strip()
    if not cleaned:
        return cleaned

    prompt = (
        "Fix all grammar, spelling, tense, and word-choice mistakes in the "
        "following text. Keep the meaning and tone the same. Return ONLY "
        "the corrected text, with no explanation, no preamble, and no "
        "quotation marks.\n\n"
        f"Text: {cleaned}"
    )

    response = gemini_model.generate_content(prompt)
    return response.text.strip()


if st.button("Fix grammar mistakes"):
    text_to_fix = st.session_state.edit_box.strip()

    if not text_to_fix:
        st.warning("Please record or enter some text first.")
    else:
        try:
            corrected = fix_grammar(text_to_fix)
            st.subheader("corrected result")
            st.write(corrected)
        except Exception as e:
            st.error(f"Grammar correction error: {e}")

st.subheader("type text to test (no mic needed)")

typed_text = st.text_area(
    "Type text here",
    height=120,
    key="typed_text",
)

if st.button("Fix grammar (typed text)"):
    text_to_fix = typed_text.strip()

    if not text_to_fix:
        st.warning("Please type some text first.")
    else:
        try:
            corrected = fix_grammar(text_to_fix)
            st.subheader("corrected result")
            st.write(corrected)
        except Exception as e:
            st.error(f"Grammar correction error: {e}")
