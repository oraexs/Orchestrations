"""Helper utilities for document ingestion and semantic search."""

from __future__ import annotations

import importlib
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import PyPDF2

def read_file(file_path: str) -> str:
    """Read supported file types and return raw text."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = ""
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        return raw_text

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported file format: {file_path}")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


@dataclass
class VectorRecord:
    document_text: str
    embedding_vector: List[float]
    metadata: Dict[str, Any]


class QdrantVectorDB:
    """Local persistent Qdrant-backed vector store."""

    def __init__(
        self,
        collection_name: str = "docs",
        storage_path: str = "./data/qdrant",
        vector_size: int = 384,
        client: Any | None = None,
    ) -> None:
        self.collection_name = collection_name

        qdrant_mod = importlib.import_module("qdrant_client")
        models_mod = importlib.import_module("qdrant_client.models")

        self._point_struct_cls = getattr(models_mod, "PointStruct")
        vector_params_cls = getattr(models_mod, "VectorParams")
        distance_enum = getattr(models_mod, "Distance")

        client_cls = getattr(qdrant_mod, "QdrantClient")
        self._owns_client = client is None
        self._client = client if client is not None else client_cls(path=storage_path)

        existing_names = {
            col.name for col in self._client.get_collections().collections
        }
        if self.collection_name not in existing_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params_cls(
                    size=vector_size,
                    distance=distance_enum.COSINE,
                ),
            )

    def store(self, document_text: str, embedding_vector: List[float], metadata: Dict[str, Any]) -> None:
        point_id = metadata.get("point_id") or str(uuid.uuid4())
        payload = {
            "document_text": document_text,
            "metadata": metadata,
        }

        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                self._point_struct_cls(
                    id=point_id,
                    vector=embedding_vector,
                    payload=payload,
                )
            ],
        )

    def search(self, embedding_vector: List[float], limit: int) -> List[tuple[float, VectorRecord]]:
        # qdrant-client >= 1.10 uses query_points; fall back to legacy search.
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=embedding_vector,
                limit=limit,
            )
            hits = response.points
        else:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=embedding_vector,
                limit=limit,
            )

        # Keep prompt history in metadata so callers can reuse workspace prompts.
        results: List[tuple[float, VectorRecord]] = []
        for hit in hits:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            request_prompt = metadata.get("request", "")
            existing_prompts = metadata.get("search_prompts", [])
            prompts = [p for p in existing_prompts if isinstance(p, str)] if isinstance(existing_prompts, list) else []
            if isinstance(request_prompt, str) and request_prompt and request_prompt not in prompts:
                prompts.append(request_prompt)
            metadata["search_prompts"] = prompts

            results.append(
                (
                    float(getattr(hit, "score", 0.0)),
                    VectorRecord(
                        document_text=str(payload.get("document_text", "")),
                        embedding_vector=[],
                        metadata=metadata,
                    ),
                )
            )

        return results

    def close(self) -> None:
        """Close Qdrant client explicitly to avoid shutdown-time warnings."""
        if not self._owns_client:
            return
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


class SimpleEmbeddingModel:
    """Deterministic lightweight embedding model based on token hashing (fallback only)."""

    def __init__(self, dimension: int = 500) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            idx = hash(token) % self.dimension
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector

        return [v / norm for v in vector]

    def embed_query(self, text: str) -> List[float]:
        """For SimpleEmbeddingModel, query and passage embedding are identical."""
        return self.embed(text)


class FastEmbedEmbeddingModel:
    """Semantic embedding model backed by fastembed (BAAI/bge-small-en-v1.5).

    Uses asymmetric retrieval:
      - embed()       → passage/document embedding (no prefix)
      - embed_query() → query embedding (BGE instruction prefix applied internally)
    Output dimension: 384, stable across restarts, fully offline.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        fastembed_mod = importlib.import_module("fastembed")
        text_embedding_cls = getattr(fastembed_mod, "TextEmbedding")
        self._model = text_embedding_cls(model_name=model_name)
        self.dimension = 384

    def embed(self, text: str) -> List[float]:
        """Embed a document/passage chunk."""
        result = list(self._model.embed([text]))
        return result[0].tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query using BGE's asymmetric query prefix."""
        result = list(self._model.query_embed([text]))
        return result[0].tolist()


def similarity_search(embedding: List[float], vector_db: QdrantVectorDB, limit: int) -> List[tuple[float, VectorRecord]]:
    """Return top-k records from Qdrant sorted by descending similarity score."""
    return vector_db.search(embedding_vector=embedding, limit=limit)
