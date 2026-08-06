from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingestion.pdf_handler import pdf_to_images
from app.ingestion.preprocess import preprocess_image
from app.ingestion.ocr import extract_text
from app.ingestion.layout import process_ocr_results
from app.ingestion.chunking import chunk_lines
from app.embeddings import embed_chunks
from app.vector_store import add_chunks, get_index_size, search
from app.embeddings import embed_text

app = FastAPI()

ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.get("/health")
def health_check():
    return {"status": "ok"}

def process_single_image(image_bytes: bytes, source_label: str) -> list[dict]:
    """Run the full per-page pipeline: preprocess -> OCR -> layout -> chunk."""
    processed = preprocess_image(image_bytes)
    ocr_results = extract_text(processed)
    lines = process_ocr_results(ocr_results)
    return chunk_lines(lines, source_label)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPG, PNG, PDF."
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents)} bytes. Max allowed: {MAX_FILE_SIZE} bytes."
        )

    if file.content_type == "application/pdf":
        page_images = pdf_to_images(contents)
        all_chunks = []
        for page_number, page_bytes in enumerate(page_images, start=1):
            source_label = f"{file.filename} - page {page_number}"
            page_chunks = process_single_image(page_bytes, source_label)
            all_chunks.extend(page_chunks)

        all_chunks = embed_chunks(all_chunks)
        add_chunks(all_chunks)
        return {
            "filename": file.filename,
            "type": "pdf",
            "num_pages": len(page_images),
            "num_chunks_added": len(all_chunks),
            "total_vectors_in_index": get_index_size()
        }

    source_label = file.filename
    chunks = process_single_image(contents, source_label)
    chunks = embed_chunks(chunks)
    add_chunks(chunks)
    return {
        "filename": file.filename,
        "type": "image",
        "num_chunks_added": len(chunks),
        "total_vectors_in_index": get_index_size()
    }

@app.post("/ask")
async def ask_question(question: str):
    query_vector = embed_text(question)
    results = search(query_vector, top_k=3)
    return {
        "question": question,
        "results": results
    }
