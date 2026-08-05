CONFIDENCE_THRESHOLD = 0.5
Y_TOLERANCE = 22  # pixels — how close two fragments' vertical positions must be to count as "same line"

def filter_low_confidence(ocr_results: list[dict]) -> list[dict]:
    """
    Remove OCR fragments below our confidence threshold —
    these are likely garbled/incorrect reads, not real text.
    """
    return [r for r in ocr_results if r["confidence"] >= CONFIDENCE_THRESHOLD]

def get_vertical_center(bbox: list[list[float]]) -> float:
    """Average y-coordinate of a bounding box's four corners."""
    y_values = [point[1] for point in bbox]
    return sum(y_values) / len(y_values)

def group_into_lines(ocr_results: list[dict]) -> list[str]:
    """
    Group OCR fragments into lines based on vertical proximity,
    then order each line left-to-right by x-position.
    Returns a list of reconstructed line strings.
    """
    if not ocr_results:
        return []

    # Sort all fragments top-to-bottom first
    sorted_results = sorted(ocr_results, key=lambda r: get_vertical_center(r["bbox"]))

    lines = []
    current_line = [sorted_results[0]]
    current_y = get_vertical_center(sorted_results[0]["bbox"])

    for fragment in sorted_results[1:]:
        fragment_y = get_vertical_center(fragment["bbox"])
        if abs(fragment_y - current_y) <= Y_TOLERANCE:
            current_line.append(fragment)
        else:
            lines.append(current_line)
            current_line = [fragment]
            current_y = fragment_y
    lines.append(current_line)

    # Within each line, order left-to-right by x-position, then join into one string
    line_strings = []
    for line in lines:
        line_sorted = sorted(line, key=lambda r: (get_vertical_center(r["bbox"]), r["bbox"][0][0]))
        text = " ".join(r["text"] for r in line_sorted)
        line_strings.append(text)

    return line_strings

def process_ocr_results(ocr_results: list[dict]) -> list[str]:
    """
    Full layout pipeline: filter low-confidence noise, then group into lines.
    """
    filtered = filter_low_confidence(ocr_results)
    return group_into_lines(filtered)
