import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, HnswConfigDiff
from qdrant_client.http.models import Filter, FieldCondition, Range
import uuid
from typing import List, Dict, Any, Optional

# --- Qdrant Server Configuration ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

class QdrantDB:
    """
    A service class to interact with a local QdrantDB instance.
    This uses a file-based (on-disk) storage.
    """
    
    def __init__(self, collection_name: str = "youtube_content", vector_size: int = 384):
        # Determine paths
        self.services_dir = Path(__file__).resolve().parent
        self.project_root = self.services_dir.parent.parent  # backend/
        self.db_path = self.project_root / "data" / "db"
        
        # Ensure db_path exists (This is no longer needed for the DB, but might be for other local data)
        # self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.vector_size = vector_size # all-MiniLM-L6-v2 produces 384-dim vectors
        
        print(f"🛢️ Initializing QdrantDB client for server at: {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Initialize client for on-disk storage
        # self.client = QdrantClient(path=str(self.db_path))
        
        # --- Initialize Client with Fallback ---
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
            # Simple check to verify connection
            self.client.get_collections()
            print(f"✅ Connected to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            print(f"⚠️ Could not connect to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}. Error: {e}")
            print(f"⚠️ Falling back to local disk storage at: {self.db_path}")
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.db_path))
        
        self.create_collection()

    def create_collection(self):
        """Creates the collection if it doesn't already exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    # Set a small HNSW config for local testing
                    # This improves search speed on larger datasets
                    hnsw_config=HnswConfigDiff(
                        payload_m=16,
                        m=0, # Set to 0 for small datasets, qdrant will manage
                    )
                )
                
                # Create payload indexes for filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="timestamp",
                    field_type="integer",
                    wait=True
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="type",
                    field_type="keyword",
                    wait=True
                )
                print("✅ Collection and payload indexes created.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"❌ Error creating/checking collection: {e}")

    def add_embeddings(self, embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Adds embeddings, documents, and metadatas to the Qdrant collection.
        """
        if not embeddings:
            print("⚠️ No embeddings to add.")
            return

        points = []
        for i, emb in enumerate(embeddings):
            point_id = str(uuid.uuid4())
            payload = metadatas[i]
            payload["document"] = documents[i] # Store the text itself in the payload
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
            )
            
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True # Wait for operation to complete
            )
            print(f"✅ Added {len(points)} points to Qdrant.")
        except Exception as e:
            print(f"❌ Error adding points to Qdrant: {e}")

    def search(self, query_vector: List[float], top_k: int = 5, custom_filter: Optional[Dict[str, Any]] = None):
        """
        Performs a vector search with an optional metadata filter.
        
        Filter format example:
        {
            "must": [
                {"key": "type", "match": {"value": "transcript"}},
                {"key": "timestamp", "range": {"gte": 10, "lte": 30}}
            ]
        }
        """
        
        # Build Qdrant filter
        qdrant_filter = None
        if custom_filter:
            qdrant_filter = Filter(**custom_filter)
            print(f"Searching with filter: {qdrant_filter}")

        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True, # Include metadata
                with_vectors=False # Don't need vectors in result
            )
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error during Qdrant search: {e}")
            return []

    def clear_collection(self):
        """Deletes and recreates the collection, effectively clearing it."""
        print(f"Clearing collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)
        self.create_collection()

