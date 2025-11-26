from fastapi import APIRouter, HTTPException, Depends
from .schemas import ProcessRequest, ProcessResponse, QueryRequest, QueryResponse
from .services.pipeline import YouTubePipeline
# Import the new RAGSystem class
from .services.rag_retriever import RAGSystem
from .services.qdrant_db import QdrantDB
import os
from pathlib import Path

router = APIRouter()

# --- Dependencies ---
# These create a 'singleton' instance for the app lifecycle
# Update to use the new RAGSystem class
rag_retriever_instance = RAGSystem()
qdrant_db_instance = QdrantDB()

def get_retriever():
    return rag_retriever_instance

def get_db():
    return qdrant_db_instance


# --- API Endpoints ---

@router.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest, db: QdrantDB = Depends(get_db)):
    """
    Endpoint to process a new YouTube video.
    This will download, transcribe, caption, and embed the video content.
    It clears the existing database before processing.
    """
    try:
        # Clear the database before processing a new video
        print("--- Clearing database for new video ---")
        db.clear_collection()
        
        # Initialize and run the pipeline
        # Pass the API key from the request
        pipeline = YouTubePipeline(url=str(request.url), api_key=request.api_key)
        result = pipeline.run() # This is a blocking call
        
        return ProcessResponse(
            video_file=result["video_file"],
            audio_file=result["audio_file"],
            keyframes=result["keyframes"],
            segments=result["segments"]
        )
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@router.post("/query", response_model=QueryResponse)
async def query_video(
    request: QueryRequest,
    # Update the type hint to use the new class name
    retriever: RAGSystem = Depends(get_retriever)
):
    """
    Endpoint to ask a question about the processed video.
    Uses the RAG system to generate an answer.
    """
    try:
        # Pass the API key from the request
        answer = retriever.query(request.query, api_key=request.api_key)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@router.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}