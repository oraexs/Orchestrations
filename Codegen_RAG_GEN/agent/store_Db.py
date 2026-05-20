"""Database ingestion/search utilities with LangGraph-ready node functions."""

from __future__ import annotations

import importlib
import json
import uuid
from typing import Any, Dict, List, TypedDict

try:
    from agent.Db_helper import (
        FastEmbedEmbeddingModel,
        QdrantVectorDB,
        SimpleEmbeddingModel,
        read_file,
        similarity_search,
        split_text,
    )
except ImportError:
    from Db_helper import (
        FastEmbedEmbeddingModel,
        QdrantVectorDB,
        SimpleEmbeddingModel,
        read_file,
        similarity_search,
        split_text,
    )


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Use fastembed (BAAI/bge-small-en-v1.5) for semantic retrieval.
# Falls back to the lightweight hash-based model if fastembed is not installed.
try:
    embedding_model: FastEmbedEmbeddingModel | SimpleEmbeddingModel = FastEmbedEmbeddingModel()
except Exception:
    embedding_model = SimpleEmbeddingModel()

qdrant_mod = importlib.import_module("qdrant_client")
qdrant_client_cls = getattr(qdrant_mod, "QdrantClient")
# Use path relative to this file's location (agent/data/qdrant) for consistency.
# This way, Qdrant data is scoped to the agent module regardless of cwd.
qdrant_data_path = str(Path(__file__).parent / "data" / "qdrant")
shared_qdrant_client = qdrant_client_cls(path=qdrant_data_path)

local_vector_db = QdrantVectorDB(
    vector_size=embedding_model.dimension,
    client=shared_qdrant_client,
)

workspace_vector_db = QdrantVectorDB(
    collection_name="workspace_structures",
    vector_size=embedding_model.dimension,
    client=shared_qdrant_client,
)


def ingest_document(file_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> str:
    """Read, chunk, embed, and store a document."""
    raw_text = read_file(file_path)
    chunks = split_text(raw_text, chunk_size, chunk_overlap)

    for chunk_index, chunk in enumerate(chunks):
        vector = embedding_model.embed(chunk)
        local_vector_db.store(
            document_text=chunk,
            embedding_vector=vector,
            metadata={
                "source": file_path,
                "chunk_index": chunk_index,
                "point_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}:{chunk_index}")),
            },
        )

    return f"Document ingested successfully: {len(chunks)} chunks"


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k matching chunks for a query.

    Uses embed_query() so that BGE's asymmetric query prefix is applied,
    improving retrieval quality over using the same embedding path as documents.
    """
    query_vector = embedding_model.embed_query(query)
    matches = similarity_search(query_vector, local_vector_db, top_k)

    return [
        {
            "score": score,
            "document_text": record.document_text,
            "metadata": record.metadata,
        }
        for score, record in matches
    ]


def store_workspace_structure(structure: Dict[str, Any], request_text: str = "") -> str:
    """Persist workspace structure in a dedicated Qdrant collection."""
    structure_text = json.dumps(structure, sort_keys=True)
    embedding_input = (
        f"workspace_request: {request_text}\nworkspace_structure: {structure_text}"
        if request_text
        else f"workspace_structure: {structure_text}"
    )
    structure_vector = embedding_model.embed(embedding_input)

    point_seed = request_text or f"workspace:{structure_text}"
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_seed))

    workspace_vector_db.store(
        document_text=structure_text,
        embedding_vector=structure_vector,
        metadata={
            "type": "workspace_structure",
            "request": request_text,
            "point_id": point_id,
        },
    )
    return point_id


def search_workspace_structures(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve previously stored workspace structures by semantic similarity."""
    query_vector = embedding_model.embed_query(query)
    matches = similarity_search(query_vector, workspace_vector_db, top_k)

    def _decode_structure(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    
    return [
        {
            "score": score,
            "structure": _decode_structure(record.document_text),
            "prompt": (
                record.metadata.get("request", "")
                if isinstance(record.metadata.get("request", ""), str)
                else ""
            ),
            "metadata": record.metadata,
        }
        for score, record in matches
    ]


def retrieve_workspace_structure_by_prompt(user_prompt: str) -> tuple[Dict[str, Any], str]:
    """Return the best-matching stored workspace structure for a user prompt."""
    hits = search_workspace_structures(query=user_prompt, top_k=1)
    if not hits:
        return {}, ""
    structure = hits[0].get("structure", {})
    return structure if isinstance(structure, dict) else {}, hits[0].get("prompt", "") if isinstance(hits[0].get("prompt", ""), str) else ""


def close_vector_db() -> None:
    """Close vector DB client explicitly to prevent interpreter-shutdown warnings."""
    local_vector_db.close()
    workspace_vector_db.close()
    close_fn = getattr(shared_qdrant_client, "close", None)
    if callable(close_fn):
        close_fn()


class GraphState(TypedDict, total=False):
    file_path: str
    query: str
    top_k: int
    ingest_status: str
    results: List[Dict[str, Any]]


def ingest_document_node(state: GraphState) -> GraphState:
    """LangGraph node to ingest a document from state."""
    file_path = state["file_path"]
    return {"ingest_status": ingest_document(file_path)}


def semantic_search_node(state: GraphState) -> GraphState:
    """LangGraph node to run semantic search from state."""
    query = state["query"]
    top_k = state.get("top_k", 5)
    return {"results": semantic_search(query, top_k)}


def build_graph():
    """Build and compile the ingestion -> search orchestration graph."""
    graph_mod = importlib.import_module("langgraph.graph")
    StateGraph = getattr(graph_mod, "StateGraph")
    START = getattr(graph_mod, "START")
    END = getattr(graph_mod, "END")

    workflow = StateGraph(GraphState)
    workflow.add_node("ingest_document", ingest_document_node)
    workflow.add_node("semantic_search", semantic_search_node)

    workflow.add_edge(START, "ingest_document")
    workflow.add_edge("ingest_document", "semantic_search")
    workflow.add_edge("semantic_search", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    output = app.invoke(
        {
            "file_path": "sample.pdf",
            "query": "What is the main topic of the document?",
            "top_k": 5,
        }
    )

    print(output.get("ingest_status", ""))
    for result in output.get("results", []):
        metadata = result["metadata"]
        print(f"Source: {metadata['source']}, Chunk Index: {metadata['chunk_index']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document Text: {result['document_text']}")
        print("------")
