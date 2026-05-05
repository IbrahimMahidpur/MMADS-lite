"""
Agent Memory — ChromaDB-backed persistent memory.
Stores analysis steps, hypotheses, results across agent runs.
Uses Ollama nomic-embed-text for local embeddings.
"""
import logging
from typing import Optional
from datetime import datetime

from multimodal_ds.config import CHROMA_DIR, EMBED_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Persistent memory for the agent system using ChromaDB.
    Agents can store and retrieve context across steps.
    """

    def __init__(self, collection_name: str = "agent_memory"):
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"[Memory] ChromaDB initialized at {CHROMA_DIR}")
        except ImportError:
            logger.warning("[Memory] chromadb not installed — memory will be in-memory only")
            self._collection = None

    def store(self, content: str, metadata: dict = None, doc_id: str = None) -> str:
        """Store a memory entry with optional metadata."""
        import uuid
        entry_id = doc_id or str(uuid.uuid4())

        meta = {
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        # Convert all metadata values to strings (ChromaDB requirement)
        meta = {k: str(v) for k, v in meta.items()}

        if self._collection:
            try:
                embedding = self._get_embedding(content)
                self._collection.upsert(
                    ids=[entry_id],
                    documents=[content],
                    embeddings=[embedding] if embedding else None,
                    metadatas=[meta]
                )
            except Exception as e:
                logger.warning(f"[Memory] Store failed: {e}")

        return entry_id

    def retrieve(self, query: str, n_results: int = 5, where: dict = None) -> list[dict]:
        """Retrieve relevant memories by semantic similarity."""
        if not self._collection:
            return []

        try:
            embedding = self._get_embedding(query)
            kwargs = {"n_results": min(n_results, self._collection.count() or 1)}
            if embedding:
                kwargs["query_embeddings"] = [embedding]
            else:
                kwargs["query_texts"] = [query]
            if where:
                if len(where) > 1:
                    kwargs["where"] = {"$and": [{k: v} for k, v in where.items()]}
                else:
                    kwargs["where"] = where

            results = self._collection.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]

        except Exception as e:
            logger.warning(f"[Memory] Retrieve failed: {e}")
            return []

    def store_analysis_step(self, step_name: str, result: str, session_id: str = "default"):
        """Convenience method to store an analysis step result."""
        return self.store(
            content=f"[Step: {step_name}]\n{result}",
            metadata={"step": step_name, "session_id": session_id, "type": "analysis_step"}
        )

    def get_session_history(self, session_id: str) -> list[dict]:
        """Get all stored steps for a session."""
        return self.retrieve(
            query="analysis step result",
            n_results=20,
            where={"session_id": session_id}
        )

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding using Ollama nomic-embed-text."""
        try:
            import httpx
            model_name = EMBED_MODEL.replace("ollama/", "")
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model_name, "prompt": text[:2000]},
                timeout=30,
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except Exception as e:
            logger.debug(f"[Memory] Embedding failed: {e}")
        return None

    def count(self) -> int:
        """Return number of stored memories."""
        if self._collection:
            try:
                return self._collection.count()
            except Exception:
                pass
        return 0
