"""ChromaDB persistence layer for semantic chunks."""

import chromadb
from chromadb.config import Settings
from pathlib import Path

import config

CHROMA_PERSIST_DIR = config.CHROMA_PERSIST_DIR
COLLECTION_NAME = config.CHROMA_COLLECTION_NAME


class ChromaStore:
    """Thin wrapper around ChromaDB for chunk storage and retrieval."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [
            {
                "doc_id": c.get("doc_id", ""),
                "source": c.get("source", ""),
                "file_name": c.get("file_name", ""),
                "file_hash": c.get("file_hash", ""),
                "page_no": c.get("page_no") or -1,
                "chunk_index": c.get("chunk_index", 0),
                "sentence_count": c.get("sentence_count", 0),
                "char_count": c.get("char_count", 0),
                "threshold_used": c.get("threshold_used", 0.0),
                "chunk_type": c.get("chunk_type", "text"),
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results = []
        for i, doc in enumerate(raw["documents"][0]):
            meta = raw["metadatas"][0][i]
            results.append({
                "text": doc,
                "chunk_id": raw["ids"][0][i],
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "file_name": meta.get("file_name"),
                "page_no": meta.get("page_no"),
                "chunk_index": meta.get("chunk_index"),
                "sentence_count": meta.get("sentence_count"),
                "distance": raw["distances"][0][i],
            })
        return results

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": {"$eq": doc_id}})

    def count(self) -> int:
        return self.collection.count()
