CHUNK_SIZE = 5   # number of lines per chunk
CHUNK_OVERLAP = 1  # number of lines shared between consecutive chunks

def chunk_lines(lines: list[str], source_label: str) -> list[dict]:
    """
    Group lines into overlapping chunks for embedding/retrieval.
    Each chunk keeps a reference back to its source (e.g. page number or filename).
    Returns a list of dicts: [{"text": ..., "source": ...}, ...]
    """
    if not lines:
        return []

    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        chunk_lines_group = lines[start:end]
        chunk_text = "\n".join(chunk_lines_group)

        chunks.append({
            "text": chunk_text,
            "source": source_label
        })

        if end == len(lines):
            break

        start = end - CHUNK_OVERLAP

    return chunks
