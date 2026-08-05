from sentence_transformers import SentenceTransformer

# Loaded once at import time, not per-call — same reasoning as EasyOCR's Reader:
# loading the model is slow, so we pay that cost once, not on every request.
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str) -> list[float]:
    """
    Convert a single piece of text into its embedding vector.
    Returns a list of floats (length 384 for this model).
    """
    embedding = model.encode(text)
    return embedding.tolist()

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to each chunk dict, in place of returning
    a separate parallel list — keeps each chunk's text, source, and
    vector bundled together as one unit.
    """
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])
    return chunks
