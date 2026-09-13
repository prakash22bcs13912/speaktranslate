import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

st.title("Speak & See")

st.write("Click below to record your voice:")

audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key="recorder")

if audio:
    st.audio(audio["bytes"])

    # Convert recorded audio to a format we can read
    audio_segment = AudioSegment.from_file(io.BytesIO(audio["bytes"]))

    # Get raw sample data for the waveform
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples[::2]  # use one channel if stereo

    # Plot the waveform
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(samples, color="#1DB954", linewidth=0.5)
    ax.axis("off")
    st.pyplot(fig)

    if st.button("Show Subtitles"):
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            st.subheader("Subtitles:")
            st.write(text)
        except sr.UnknownValueError:
            st.write("Could not understand the audio. Try speaking clearly.")
        except sr.RequestError as e:
            st.write(f"Could not reach Google's service: {e}")