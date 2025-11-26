# test_query.py

from services.qdrant_db import QdrantDB
from services.model import Embedder

def query_qdrant():
    db = QdrantDB()
    embedder = Embedder()

    query = input("\n🔍 Enter your search query: ").strip()

    if not query:
        print("❌ Empty query.")
        return

    # Convert NL → embedding
    vec = embedder.embed_batch([query])[0]

    print("\n🔎 Searching Qdrant...")
    results = db.query(vec, top_k=5)

    print(f"\n📦 Found {len(results)} matching results:\n")

    for r in results:
        meta = r.payload
        print("──────────────────────────")
        print("Score      :", r.score)
        print("Type       :", meta.get("type"))
        print("Timestamp  :", meta.get("timestamp"))
        print("Text       :", meta.get("text"))
        if meta.get("type") == "frame":
            print("Keyframe   :", meta.get("filename"))

    print("\n🎉 Query completed!\n")


if __name__ == "__main__":
    query_qdrant()
