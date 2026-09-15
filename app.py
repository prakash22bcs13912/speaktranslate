import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import os
import tempfile
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import requests
import whisper

st.set_page_config(page_title="Speak & See", page_icon="🎙️")

st.title("Speak & See")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")


whisper_model = load_whisper_model()

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

    audio_segment = AudioSegment.from_file(io.BytesIO(audio["bytes"]))

    samples = np.array(audio_segment.get_array_of_samples())

    if audio_segment.channels == 2:
        samples = samples[::2]

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(samples, linewidth=0.5)
    ax.axis("off")
    st.pyplot(fig)

    tmp_path = None
    try:
        whisper_ready_audio = audio_segment.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            whisper_ready_audio.export(tmp.name, format="wav")
            tmp_path = tmp.name

        with st.spinner("Transcribing with Whisper..."):
            result = whisper_model.transcribe(
                tmp_path,
                language="en",
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            text = result["text"].strip()

        if text:
            st.session_state.transcription = text
            st.session_state.edit_box = text
            st.rerun()
        else:
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
    """Fix grammar using LanguageTool's free public API (no cost, no API key needed)."""

    def clean_and_correct(input_text: str) -> str:
        cleaned = input_text.strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            if cleaned[-1] not in ".!?":
                cleaned += "."

        response = requests.post(
            "https://api.languagetool.org/v2/check",
            data={
                "text": cleaned,
                "language": "en-US",
                "level": "picky",
            },
            timeout=30,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])

        corrected = cleaned
        for match in sorted(matches, key=lambda m: m["offset"], reverse=True):
            if match["replacements"]:
                start = match["offset"]
                end = start + match["length"]
                replacement = match["replacements"][0]["value"]
                corrected = corrected[:start] + replacement + corrected[end:]

        return corrected

    corrected = clean_and_correct(text)

    original_words = text.split()
    fixed_words = corrected.split()
    for i, word in enumerate(fixed_words):
        bare = word.strip(".,!?")
        if bare.isupper() and len(bare) > 1:
            was_upper_originally = any(
                w.strip(".,!?").isupper() and w.strip(".,!?") == bare
                for w in original_words
            )
            if not was_upper_originally:
                fixed_words[i] = word.capitalize()

    return " ".join(fixed_words)


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
