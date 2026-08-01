from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingestion.pdf_handler import pdf_to_images
from app.ingestion.preprocess import preprocess_image
from PIL import Image
import io

app = FastAPI()

ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.get("/health")
def health_check():
    return {"status": "ok"}

def get_dimensions(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return img.size

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
        first_page = page_images[0]
        original_size = get_dimensions(first_page)
        processed = preprocess_image(first_page)
        processed_size = get_dimensions(processed)
        return {
            "filename": file.filename,
            "type": "pdf",
            "num_pages": len(page_images),
            "first_page_original_size": original_size,
            "first_page_processed_size": processed_size
        }

    original_size = get_dimensions(contents)
    processed = preprocess_image(contents)
    processed_size = get_dimensions(processed)
    return {
        "filename": file.filename,
        "type": "image",
        "original_size": original_size,
        "processed_size": processed_size
    }