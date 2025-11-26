# test_pipeline.py

from app.services.pipeline import YouTubePipeline
from app.services.qdrant_db import QdrantDB
import webbrowser
import os

def run_pipeline():
    url = "https://www.youtube.com/watch?v=2MdQWo9fHZs"
    api_key = input("Enter Groq API Key: ").strip()

    if not url:
        print("❌ No URL provided.")
        return
        
    if not api_key:
        print("❌ No API Key provided.")
        return

    print("\n🚀 Running YouTube pipeline...")
    # Pass api_key to the pipeline
    pipeline = YouTubePipeline(url, api_key=api_key)
    result = pipeline.run()

    print("\n✅ Pipeline finished!")
    print(result)

    # Verify DB contents
    db = QdrantDB()
    print("\n🔍 Checking Qdrant collection...")
    print(f"📦 Total stored vectors: {db.count()}")

    # Auto-open Qdrant UI (only if Docker Qdrant server is running)
    try:
        print("🌐 Attempting to open Qdrant Dashboard...")
        webbrowser.open("http://localhost:6333/dashboard")
    except:
        print("⚠️ Could not open dashboard. If using Docker, open:")
        print("http://localhost:6333/dashboard")

    print("\n🎉 Pipeline test complete!\n")


if __name__ == "__main__":
    run_pipeline()
