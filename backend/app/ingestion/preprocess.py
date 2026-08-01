from PIL import Image
import io

MAX_DIMENSION = 1600

def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Normalize an image for downstream OCR/embedding use:
    - Convert to RGB (consistent format)
    - Resize so the longest side is at most MAX_DIMENSION
    Returns processed image as PNG bytes.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")

    width, height = image.size
    longest_side = max(width, height)

    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height))

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
