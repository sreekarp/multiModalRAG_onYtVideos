from fastapi import FastAPI
from .api import router as api_router
from dotenv import load_dotenv
import os

# Load environment variables from .env file (if it exists)
# This is for local development. In production, env vars are set differently.
load_dotenv()

app = FastAPI(
    title="YouTube RAG API",
    description="Process YouTube videos and query them with RAG.",
    version="0.1.0"
)

# Include the API routes
app.include_router(api_router, prefix="/api/v1")
from fastapi import FastAPI
from .api import router as api_router
from dotenv import load_dotenv
import os

# Load environment variables from .env file (if it exists)
# This is for local development. In production, env vars are set differently.
load_dotenv()

app = FastAPI(
    title="YouTube RAG API",
    description="Process YouTube videos and query them with RAG.",
    version="0.1.0"
)

# Include the API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the YouTube RAG API. Go to /docs for API documentation."}

# --- Environment Variable Check ---
# GROQ_API_KEY is now passed at runtime per request.
# HF_TOKEN is not needed as we use local models.

print(f"--- Qdrant Client connecting to: {os.environ.get('QDRANT_HOST', 'localhost')}:{os.environ.get('QDRANT_PORT', 6333)} ---")