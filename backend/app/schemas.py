from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class ProcessRequest(BaseModel):
    """Request model to process a new YouTube URL."""
    url: HttpUrl
    api_key: str

class ProcessResponse(BaseModel):
    """Response model after successful processing."""
    video_file: str
    audio_file: str
    keyframes: int
    segments: int
    message: str = "Pipeline completed successfully"

class QueryRequest(BaseModel):
    """Request model for asking a question."""
    query: str
    api_key: str

class QueryResponse(BaseModel):
    """Response model containing the RAG answer."""
    answer: str