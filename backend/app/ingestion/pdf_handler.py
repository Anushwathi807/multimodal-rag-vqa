import fitz

def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """
    Convert each page of a PDF into a PNG image.
    Returns a list of image bytes, one per page.
    """
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        pixmap = page.get_pixmap()
        image_bytes = pixmap.tobytes("png")
        page_images.append(image_bytes)

    pdf_document.close()
    return page_images
