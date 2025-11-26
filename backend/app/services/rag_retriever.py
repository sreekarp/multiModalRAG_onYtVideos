import os
import re
from typing import List, Dict, Any, Optional
from .qdrant_db import QdrantDB
from .model import Embedder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from qdrant_client.http.models import Filter, FieldCondition, Range

# --- RAG System Class ---

class RAGSystem:
    """
    Encapsulates the entire RAG logic, from services to chain.
    """
    
    def __init__(self):
        print("Initializing RAG services...")
        
        # 1. Initialize services
        self.db = QdrantDB()
        self.embedder = Embedder() # This is now the local embedder
        
        # 2. Define prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant for answering questions about a YouTube video.
Use the following context from the video to answer the question.
The context contains transcripts and keyframe captions, sorted by time.
If the answer is not in the context, say so.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        )
        print("✅ RAG services initialized (LLM waits for runtime key).")


    def _build_chain(self, api_key: str):
        """Builds the LangChain Expression Language (LCEL) chain with the provided API key."""
        
        if not api_key:
            raise ValueError("API Key is required to build the RAG chain.")

        llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

        # This sub-chain is responsible for fetching and formatting context
        # This sub-chain is responsible for fetching and formatting context
        from langchain_core.runnables import RunnableLambda
        
        context_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: self._parse_timestamp_query(x['query'])) # -> (cleaned_query, ts)
            | RunnableLambda(lambda x: (x[0], self._get_timestamp_filter(x[1]) if x[1] else None)) # -> (cleaned_query, filter)
            | RunnableLambda(lambda x: self._retriever(x[0], x[1])) # -> [docs]
            | RunnableLambda(self._find_related_transcripts) # -> [docs_with_related]
            | RunnableLambda(self._format_context) # -> "formatted_context_string"
        )

        # This is the main chain
        return (
            RunnableParallel(
                # Extract the 'query' string from the input dict
                question=lambda x: x['query'],
                context=context_chain
            )
            | self.prompt
            | llm
            | StrOutputParser()
        )

    # --- Public Query Method ---

    def query(self, query: str, api_key: str) -> str:
        """
        Main entry point to query the RAG system.
        """
        if not api_key:
            return "Error: API Key is required."

        print(f"RAG Query: {query}")
        
        try:
            # Build chain dynamically with the provided key
            rag_chain = self._build_chain(api_key)
            
            # The input to the chain is a dictionary
            result = rag_chain.invoke({"query": query})
            print(f"RAG Answer: {result}")
            return result
        except Exception as e:
            print(f"❌ Error during RAG chain invocation: {e}")
            return f"An error occurred while processing the query: {e}"

    # --- Internal "Private" Methods ---

    def _parse_timestamp_query(self, query: str) -> (str, Optional[int]):
        """
        Parses a query for a timestamp (e.g., "1:30", "12:05").
        Returns the cleaned query and the timestamp in seconds (if found).
        """
        match = re.search(r"(\d{1,2}):(\d{2})", query)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            total_seconds = (minutes * 60) + seconds
            
            cleaned_query = re.sub(r"(\d{1,2}):(\d{2})", "", query).strip()
            print(f"Found timestamp: {match.group(0)}, {total_seconds}s")
            return cleaned_query, total_seconds
            
        return query, None

    def _get_timestamp_filter(self, timestamp_in_seconds: int, window: int = 15) -> Dict[str, Any]:
        """
        Creates a Qdrant filter for a timestamp range.
        """
        return {
            "must": [
                FieldCondition(
                    key="timestamp",
                    range=Range(
                        gte=max(0, timestamp_in_seconds - window),
                        lte=timestamp_in_seconds + window
                    )
                )
            ]
        }

    def _retriever(self, query: str, timestamp_filter: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves documents from QdrantDB based on query and optional filter.
        """
        query_vector_list = self.embedder.embed_batch([query])
        if not query_vector_list:
             print("❌ Embedding failed, returning no results.")
             return []
        query_vector = query_vector_list[0]

        search_results = self.db.search(
            query_vector=query_vector,
            top_k=top_k,
            custom_filter=timestamp_filter
        )
        
        docs = []
        for result in search_results:
            doc = result.payload
            doc["score"] = result.score
            docs.append(doc)
            
        return docs

    def _find_related_transcripts(self, results: List[Dict[str, Any]], window: int = 10) -> List[Dict[str, Any]]:
        """
        Implements the logic: "if we retrieve a keyframe, also include transcript data around that timestamp."
        """
        final_results = list(results)
        keyframe_timestamps_processed = set()
        
        for result in results:
            if result.get("type") == "frame":
                ts = result.get("timestamp", 0)
                if ts in keyframe_timestamps_processed:
                    continue
                keyframe_timestamps_processed.add(ts)

                print(f"Found keyframe. Fetching transcripts around {ts}s...")
                
                transcript_filter = {
                    "must": [
                        FieldCondition(key="type", match={"value": "transcript"}),
                        FieldCondition(key="timestamp", range=Range(
                            gte=max(0, ts - window),
                            lte=ts + window
                        ))
                    ]
                }
                
                dummy_vector = [0.0] * self.db.vector_size
                
                related_transcripts = self.db.client.search(
                    collection_name=self.db.collection_name,
                    query_vector=dummy_vector,
                    query_filter=Filter(**transcript_filter),
                    limit=10
                )
                
                for trans_result in related_transcripts:
                    final_results.append(trans_result.payload)
                    
        seen_docs = set()
        deduplicated_results = []
        for res in final_results:
            doc_content = res.get("document")
            if doc_content and doc_content not in seen_docs:
                deduplicated_results.append(res)
                seen_docs.add(doc_content)
                
        return deduplicated_results

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Formats the retrieved documents into a string for the LLM prompt.
        """
        if not docs:
            return "No relevant context found."
            
        context = ""
        sorted_docs = sorted(docs, key=lambda x: x.get("timestamp", 0))
        
        for doc in sorted_docs:
            # --- Point 4 (Timestamp) ---
            # ts is already an int from the pipeline, so divmod is safe.
            ts = doc.get('timestamp', 0)
            minutes, seconds = divmod(ts, 60)
            time_str = f"{minutes:02}:{seconds:02}"
            
            if doc.get("type") == "frame":
                context += f"[Keyframe at {time_str}] Caption: {doc.get('document')}\n"
            elif doc.get("type") == "transcript":
                context += f"[Transcript at {time_str}] Text: {doc.get('document')}\n"
        
        return context.strip()
