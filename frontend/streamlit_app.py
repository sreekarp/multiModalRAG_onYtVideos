# frontend/app.py
import streamlit as st
import requests
import time

# -----------------------
# CONFIG
# -----------------------
# adjust if you keep cookies somewhere else
import os

# -----------------------
# CONFIG
# -----------------------
# adjust if FastAPI backend runs on a different host/port
# Default to localhost for local dev, but override in Docker
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api/v1")

PROCESS_URL = f"{BACKEND_URL}/process"
QUERY_URL = f"{BACKEND_URL}/query"

# -----------------------
# STREAMLIT PAGE SETUP
# -----------------------
st.set_page_config(page_title="YouTube RAG", layout="centered")
st.title("🎥 YouTube RAG — Ingest + Ask")

st.markdown("""
Paste a YouTube link below.  
The backend will download the video/audio, extract keyframes, caption them, generate embeddings,  
and build a RAG store.

**Note:** You need a Groq API Key for transcription and answering.
""")

# -----------------------
# SESSION STATE
# -----------------------
if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "chat" not in st.session_state:
    st.session_state.chat = []     # list of {question, answer}


# -----------------------
# SIDEBAR - API KEY
# -----------------------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if not api_key:
        st.warning("Please enter your Groq API Key to proceed.")


# -----------------------
# URL INPUT (Triggers Pipeline)
# -----------------------
st.subheader("Step 1 — Ingest YouTube URL")

youtube_url = st.text_input(
    "Enter YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

if st.button("Run Pipeline"):
    if not api_key:
        st.error("❌ API Key is required.")
    elif not youtube_url.strip():
        st.warning("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Running ingestion pipeline... this may take several minutes depending on video length."):
            try:
                payload = {
                    "url": youtube_url,
                    "api_key": api_key
                }
                resp = requests.post(PROCESS_URL, json=payload, timeout=1800)
                resp.raise_for_status()
                data = resp.json()

                st.session_state.ingested = True
                st.session_state.pipeline_result = data

                st.success("Pipeline completed successfully!")
                st.json(st.session_state.pipeline_result)

            except Exception as e:
                st.session_state.ingested = False
                st.error(f"❌ Pipeline failed: {e}")


# -----------------------
# ASK QUESTIONS
# -----------------------
st.markdown("---")
st.subheader("Step 2 — Ask a question about the video")

question = st.text_input(
    "Ask your question",
    placeholder="What is the video about?",
)

if st.button("Ask"):
    if not api_key:
        st.error("❌ API Key is required.")
    elif not st.session_state.ingested:
        st.warning("Please ingest a YouTube URL first.")
    elif not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Generating answer..."):
            try:
                payload = {
                    "query": question,
                    "api_key": api_key
                }
                resp = requests.post(QUERY_URL, json=payload, timeout=120)
                resp.raise_for_status()

                ans = resp.json()

                entry = {
                    "question": question,
                    "answer": ans.get("answer")
                }
                st.session_state.chat.append(entry)

            except Exception as e:
                st.error(f"❌ Error getting answer: {e}")


# -----------------------
# CHAT HISTORY
# -----------------------
if st.session_state.chat:
    st.markdown("---")
    st.subheader("Chat History")

    for turn in reversed(st.session_state.chat):
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(f"**A:** {turn['answer']}")
        st.markdown("---")
