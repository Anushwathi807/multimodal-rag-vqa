import faiss
import numpy as np
import json
import os

EMBEDDING_DIMENSION = 384  # matches all-MiniLM-L6-v2's output size

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata_store.json"

def _load_or_create_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatIP(EMBEDDING_DIMENSION)

def _load_or_create_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return []

# Load existing data at startup if present, otherwise start fresh —
# this runs once, at import time, same timing as our other startup-loaded resources.
index = _load_or_create_index()
metadata_store = _load_or_create_metadata()

def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Scale a vector to length 1, so Inner Product search behaves like cosine similarity.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def save():
    """Persist the current index and metadata to disk."""
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata_store, f)

def add_chunks(chunks: list[dict]):
    """
    Add a list of embedded chunks to the FAISS index, then persist to disk immediately.
    Each chunk dict must already have an 'embedding' field (from embed_chunks()).
    """
    vectors = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
    normalized_vectors = np.array([normalize(v) for v in vectors], dtype="float32")

    index.add(normalized_vectors)

    for chunk in chunks:
        metadata_store.append({
            "text": chunk["text"],
            "source": chunk["source"]
        })

    save()

def get_index_size() -> int:
    """How many vectors are currently stored."""
    return index.ntotal

MIN_SIMILARITY_THRESHOLD = 0.15

def search(query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Find the top_k chunks most similar to the given query vector,
    discarding any below MIN_SIMILARITY_THRESHOLD as not genuinely relevant.
    Returns a list of chunk dicts (text, source) with a similarity score, ranked best-first.
    """
    query_array = np.array([query_vector], dtype="float32")
    normalized_query = normalize(query_array[0])
    normalized_query = np.array([normalized_query], dtype="float32")

    scores, indices = index.search(normalized_query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < MIN_SIMILARITY_THRESHOLD:
            continue
        chunk = metadata_store[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "similarity_score": float(score)
        })

    return results
