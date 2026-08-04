import easyocr
from PIL import Image
import io
import numpy as np

# Reader is created once, not per-request — loading the model is slow,
# so we want to pay that cost only once when the app starts.
reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_bytes: bytes) -> list[dict]:
    """
    Run OCR on an image and return extracted text with position and confidence.
    Returns a list of dicts: [{"text": ..., "confidence": ..., "bbox": ...}, ...]
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)

    results = reader.readtext(image_array)

    extracted = []
    for bbox, text, confidence in results:
        extracted.append({
            "text": text,
            "confidence": round(float(confidence), 3),
            "bbox": [[float(x), float(y)] for x, y in bbox]
        })

    return extracted
