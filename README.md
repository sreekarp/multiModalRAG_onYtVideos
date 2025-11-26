# YouTube RAG Application

This project is a Retrieval-Augmented Generation (RAG) application for YouTube videos. It allows you to ingest YouTube videos, process them (transcription, keyframe extraction, captioning), and ask questions about the content.

## Architecture

- **Backend**: FastAPI application.
- **Database**: Qdrant vector database (runs in Docker).
- **Frontend**: Streamlit application.

## Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose.
- Python 3.10+ installed locally.
- A [Groq API Key](https://console.groq.com/).

## How to Start

### 1. Start Application (Recommended)

Run the following steps in the root directory to build and start the entire application (Backend, Frontend, Database):
1. Run a qdrant server from docker 
- **Qdrant**: Available at `http://localhost:6333`.

2.  **Backend**:
    ```bash
    cd backend
    pip install -r requirements.txt
    uvicorn app.main:app --reload
    ```

3.  **Frontend**:
    ```bash
    pip install streamlit requests
    streamlit run frontend/streamlit_app.py
    ```


## Usage

1.  Open the Streamlit app in your browser.
2.  Enter your **Groq API Key** in the sidebar.
3.  Paste a **YouTube URL** in the input field and click "Run Pipeline".
    - *Note: This process involves downloading, transcoding, and processing, which may take a few minutes.*
4.  Once ingestion is complete, type your question in the "Ask a question" box and click "Ask".
