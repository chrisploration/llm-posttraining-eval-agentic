from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import chromadb
import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_default_model = None


def default_embed_fn(texts: Sequence[str]) -> np.ndarray:
    """Embed texts with a small local sentence-transformers model (CPU)."""
    global _default_model
    if _default_model is None:
        from sentence_transformers import SentenceTransformer
        _default_model = SentenceTransformer(_DEFAULT_MODEL_NAME, device="cpu")
    return np.asarray(_default_model.encode(list(texts), normalize_embeddings=True))


class Retriever:
    """Top-k retrieval over a fixed corpus using Chroma + an injected embedding function."""

    def __init__(
        self,
        corpus: Sequence[dict[str, str]],
        *,
        embed_fn: EmbedFn,
        persist_directory: str | None = "data/chroma_rag",
        collection_name: str = "rag_corpus"
    ) -> None:
        self.corpus = list(corpus)
        self._embed_fn = embed_fn
        self._by_id = {d["id"]: d for d in self.corpus}

        client = chromadb.PersistentClient(path=persist_directory) if persist_directory else chromadb.EphemeralClient()
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if self._collection.count() != len(self.corpus):
            existing_ids = self._collection.get()["ids"]
            if existing_ids:
                self._collection.delete(ids=existing_ids)

            embeddings = embed_fn([d["text"] for d in self.corpus])
            self._collection.add(
                ids=[d["id"] for d in self.corpus],
                documents=[d["text"] for d in self.corpus],
                embeddings=[vec.tolist() for vec in embeddings]
            )

    def retrieve(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._embed_fn([query])[0]
        results = self._collection.query(query_embeddings=[q_vec.tolist()], n_results=top_k)

        out: list[dict[str, Any]] = []
        for doc_id, doc_text, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
            entry = dict(self._by_id.get(doc_id, {"id": doc_id, "text": doc_text}))
            # cosine space configured, so this is a similarity
            entry["score"] = 1.0 - dist
            out.append(entry)
        return out