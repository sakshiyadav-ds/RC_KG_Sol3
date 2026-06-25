"""
embeddings.py
─────────────────────────────────────────────────────────────────────────
Embedding model options for the Royal Caribbean GraphRAG pipeline.
Supports three backends selectable at runtime:

  1. "openai"   — text-embedding-3-small  (1536 dims, API, best quality)
  2. "minilm"   — all-MiniLM-L6-v2        (384  dims, local, no API key)
  3. "mpnet"    — all-mpnet-base-v2        (768  dims, local, stronger local)

The active backend is set via set_backend(name) before the first embed call,
or via the EMBEDDING_BACKEND env var ("openai" | "minilm" | "mpnet").

Neo4j vector indexes are dimension-specific — if you switch backends after
data is already ingested, clear the graph and re-ingest.
"""

import os
import numpy as np

print("[embeddings.py] Module loading...")

# ── Backend registry ──────────────────────────────────────────────────────
BACKENDS = {
    "openai": {
        "dim":   1536,
        "model": "text-embedding-3-small",
        "label": "OpenAI text-embedding-3-small (1536d) — best quality, needs API key",
    },
    "mpnet": {
        "dim":   768,
        "model": "all-mpnet-base-v2",
        "label": "all-mpnet-base-v2 (768d) — strong local model, no API key",
    },
    "minilm": {
        "dim":   384,
        "model": "all-MiniLM-L6-v2",
        "label": "all-MiniLM-L6-v2 (384d) — fast local model, no API key",
    },
}

_backend_name = os.environ.get("EMBEDDING_BACKEND", "minilm")
_st_model      = None   # sentence-transformers model instance (lazy)
_openai_client = None   # openai client instance (lazy)


def set_backend(name: str, openai_api_key: str = ""):
    """
    Switch the active embedding backend.  Must be called before the first
    embed_text() / embed_batch() call (or after clear_model() to force reload).

    Args:
        name:           "openai" | "mpnet" | "minilm"
        openai_api_key: required when name == "openai"
    """
    global _backend_name, _st_model, _openai_client
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(BACKENDS)}")
    _backend_name  = name
    _st_model      = None   # reset so lazy-loader picks up new model
    _openai_client = None
    if name == "openai":
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise ValueError("openai_api_key required for the 'openai' backend.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
    print(f"[embeddings] Backend set to: {name}  ({BACKENDS[name]['label']})")


def get_backend_name() -> str:
    return _backend_name


def get_embedding_dim() -> int:
    return BACKENDS[_backend_name]["dim"]


# Keep a module-level alias so neo4j_manager.py can `from embeddings import EMBEDDING_DIM`
# This is evaluated at import time — call set_backend() BEFORE importing neo4j_manager
# if you need a non-default dimension.
EMBEDDING_DIM = BACKENDS[_backend_name]["dim"]


import streamlit as st

@st.cache_resource
def load_model(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _get_st_model():
    model_name = BACKENDS[_backend_name]["model"]
    return load_model(model_name)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Call set_backend('openai', api_key=...) first.")
        _openai_client = OpenAI(api_key=api_key)
        print("[embeddings] ✓ OpenAI client initialised.")
    return _openai_client


# ── Public API ────────────────────────────────────────────────────────────

def embed_text(text: str) -> list:
    """Embed a single string. Returns list[float] of length get_embedding_dim()."""
    if _backend_name == "openai":
        client = _get_openai_client()
        resp = client.embeddings.create(
            model=BACKENDS["openai"]["model"],
            input=text,
        )
        return resp.data[0].embedding
    else:
        model = _get_st_model()
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()


def embed_batch(texts: list) -> list:
    """Embed a list of strings. Returns list[list[float]]."""
    if not texts:
        return []
    if _backend_name == "openai":
        client = _get_openai_client()
        print(f"[embeddings] OpenAI batch embed: {len(texts)} texts...")
        # OpenAI supports up to 2048 inputs per call; chunk for safety
        results = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            resp  = client.embeddings.create(model=BACKENDS["openai"]["model"], input=batch)
            results.extend([d.embedding for d in resp.data])
        print(f"[embeddings] ✓ OpenAI batch complete. {len(results)} vectors.")
        return results
    else:
        model = _get_st_model()
        print(f"[embeddings] Batch embed {len(texts)} texts ({_backend_name})...")
        vecs = model.encode(texts, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        print(f"[embeddings] ✓ Batch complete. shape={vecs.shape}")
        return vecs.tolist()


def cosine_sim(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
