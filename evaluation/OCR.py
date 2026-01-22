
import os
os.environ['path'] += r';C:\Program Files\UniConvertor-2.0rc5\dlls'
import cairosvg
from PIL import Image
import pytesseract
import xml.etree.ElementTree as ET
import re

def svg_to_png(svg_path, png_path, dpi=600):

    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        dpi=dpi
    )
    
def perform_ocr(image_path, lang="eng"):

    image = Image.open(image_path)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def character_error_rate(reference, hypothesis):
    reference = reference.replace("\n", "").strip()
    hypothesis = hypothesis.replace("\n", "").strip()

    if len(reference) == 0:
        return 0.0

    return  len(reference) / len(hypothesis)

def evaluate_svg_ocr(svg_path, reference_text, tmp_png="temp.png"):
    svg_to_png(svg_path, tmp_png)

    ocr_text = perform_ocr(tmp_png)

    cer = character_error_rate(reference_text, ocr_text)

    return {
        "ocr_text": ocr_text,
        "cer": cer
    }

def _local_name(tag: str) -> str:
    """Strip namespace: '{ns}tag' -> 'tag'."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def iter_text_content(elem: ET.Element) -> str:
    parts = []
    if elem.text:
        parts.append(elem.text)

    for child in list(elem):
        # Child's own text
        if child.text:
            parts.append(child.text)
        # Recurse (covers nested tspans)
        parts.append(iter_text_content(child))
        # Tail text after child element
        if child.tail:
            parts.append(child.tail)

    return "".join(parts)

def extract_reference_text_from_svg(svg_path: str) -> str:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    collected = []
    for elem in root.iter():
        if _local_name(elem.tag) == "textPath":
            text = iter_text_content(elem)
            if text:
                collected.append(text)

    # Join blocks with spaces (or '\n' if you prefer)
    raw = " ".join(collected)

    # Normalize whitespace: collapse multiple spaces/newlines
    normalized = re.sub(r"\s+", " ", raw).strip()
    return normalized
    
if __name__ == "__main__":
    svg_file = "text_output.svg"
    
    reference_text = extract_reference_text_from_svg(svg_file)

    result = evaluate_svg_ocr(svg_file, reference_text)

    print("Recognized text:")
    print(result["ocr_text"])

    print(f"Character Error Rate (CER): {result['cer']:.4f}")
