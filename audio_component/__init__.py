import streamlit as st



def audio_player(audio_bytes: bytes, key=None):
    st.audio(audio_bytes, format="audio/wav")
