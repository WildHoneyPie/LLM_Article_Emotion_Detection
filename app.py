import streamlit as st
import numpy as np
from transformers import pipeline
from datetime import datetime
import json
import os

from audio_component import audio_player

# Set page config
st.set_page_config(
    page_title="EchoDiary - Feel Your Days in Music",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state for playback control
if "play_audio" not in st.session_state:
    st.session_state.play_audio = False
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

try:
    # Open the local audio file in binary read mode
    with open("music/110_Bm_KlessPad_01_TL.wav", "rb") as audio_file:
        audio_bytes = audio_file.read()

        audio_player(audio_bytes=audio_bytes, key="local_player")

except FileNotFoundError:
    st.error(f"File not found: ''")
    st.info(
        "Please make sure the audio file exists in the same directory as app.py"
    )

# Initialize session state
if 'journal_entries' not in st.session_state:
    st.session_state.journal_entries = []

# Initialize models
@st.cache_resource
def load_models():
    # Text sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_analyzer

# Function to save entries
def save_entries():
    with open('journal_entries.json', 'w') as f:
        json.dump(st.session_state.journal_entries, f)

# Function to load entries
def load_entries():
    if os.path.exists('journal_entries.json'):
        with open('journal_entries.json', 'r') as f:
            st.session_state.journal_entries = json.load(f)

# Load existing entries
load_entries()

# Initialize models
sentiment_analyzer = load_models()

# Title and description
st.title("🎵 EchoDiary - Feel Your Days in Music")
st.markdown("""
    Capture your emotions through text and transform them into musical experiences.
""")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["New Entry", "View Entries", "About"])

if page == "New Entry":
    st.header("Create New Journal Entry")
    
    # Create columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Journal entry form
        with st.form("journal_entry_form"):
            title = st.text_input("Title")
            content = st.text_area("Write your thoughts here...", height=200)
            submitted = st.form_submit_button("Save Entry")
            
            if submitted and content:
                # Analyze text sentiment
                sentiment = sentiment_analyzer(content)[0]
                
                entry = {
                    "title": title or "Untitled",
                    "content": content,
                    "sentiment": sentiment['label'],
                    "sentiment_score": sentiment['score'],
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.journal_entries.append(entry)
                save_entries()
                st.success("Entry saved successfully! ✨")

elif page == "View Entries":
    st.header("Your Journal Entries")
    
    if not st.session_state.journal_entries:
        st.info("No entries yet. Start writing in your journal!")
    else:
        # Sort entries by date (newest first)
        sorted_entries = sorted(
            st.session_state.journal_entries,
            key=lambda x: x.get("date", ""),
            reverse=True
        )
        
        for entry in sorted_entries:
            with st.expander(f"{entry.get('date', 'No date')} - {entry.get('title', 'Untitled')}"):
                st.write(entry.get('content', 'No content'))
                
                # Display sentiment
                if entry.get('sentiment'):
                    st.write("Sentiment:", entry['sentiment'])
                
                # Add delete button
                if st.button(f"Delete Entry", key=entry.get('date', 'no_date')):
                    st.session_state.journal_entries.remove(entry)
                    save_entries()
                    st.rerun()

else:  # About page
    st.header("About EchoDiary")
    st.write("""
    EchoDiary is an AI-powered journaling experience that captures your emotions through:
    
    - 📝 Written text analysis
    - 🎼 AI-generated musical representations
    
    Your entries are transformed into a unique musical experience, helping you feel your memories through sound.
    """)
    
    st.info("Start your emotional journaling journey today! ✨") 