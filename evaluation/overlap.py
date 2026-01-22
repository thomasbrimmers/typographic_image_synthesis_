import xml.etree.ElementTree as ET
import copy
import os
os.environ['path'] += r';C:\Program Files\UniConvertor-2.0rc5\dlls'
import cairosvg
from io import BytesIO
from PIL import Image
import numpy as np

def get_text_elements(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def local(tag):
        return tag.split("}")[-1]

    text_elements = []
    for elem in root.iter():
        if local(elem.tag) in ("text", "textPath"):
            text_elements.append(elem)

    return tree, root, text_elements

def render_colored_text_layers(svg_path, dpi=300):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def local(tag):
        return tag.split("}")[-1]

    text_elems = []
    for elem in root.iter():
        if local(elem.tag) in ("textPath"):
            text_elems.append(elem)

    # Assign unique grayscale values
    for i, elem in enumerate(text_elems):
        shade = 10 + (i % 240)  # avoid white
        style = elem.attrib.get("style", "")
        style += f";fill:rgb({shade},{shade},{shade})"
        elem.attrib["style"] = style

    png = cairosvg.svg2png(
        bytestring=ET.tostring(root),
        dpi=dpi
    )

    img = Image.open(BytesIO(png)).convert("L")
    return np.array(img)

def render_full_svg_mask(svg_path, dpi=300):
    png_bytes = cairosvg.svg2png(url=svg_path, dpi=dpi)
    img = Image.open(BytesIO(png_bytes)).convert("L")
    arr = np.array(img)

    # Text = 1, Hintergrund = 0
    text_mask = (arr < 250).astype(np.uint8)
    return text_mask

def render_single_text_mask(tree, root, target_elem, dpi=300):
    # Deep copy SVG
    tree_copy = copy.deepcopy(tree)
    root_copy = tree_copy.getroot()

    def local(tag):
        return tag.split("}")[-1]

    # Remove all other text elements
    for elem in list(root_copy.iter()):
        if local(elem.tag) in ("textPath") and elem is not target_elem:
            parent = elem.getparent() if hasattr(elem, "getparent") else None

    # Alternative: brute-force remove all text, then reinsert one
    for parent in root_copy.iter():
        for child in list(parent):
            if local(child.tag) in ("textPath"):
                parent.remove(child)

    root_copy.append(copy.deepcopy(target_elem))

    # Render to PNG (in-memory)
    png_bytes = cairosvg.svg2png(bytestring=ET.tostring(root_copy), dpi=dpi)

    img = Image.open(BytesIO(png_bytes)).convert("L")
    arr = np.array(img)

    # Binary mask: text = 1, background = 0
    mask = (arr < 250).astype(np.uint8)
    return mask

def overlap_and_coverage_fast(svg_path, dpi=300):
    img = render_colored_text_layers(svg_path, dpi=dpi)

    # text pixels
    text_mask = img < 250

    # how many unique text values per pixel
    occupancy = np.zeros_like(img, dtype=np.uint8)
    occupancy[text_mask] = 1

    # overlap = pixels where multiple text layers overlap
    # approximation: darker-than-single-layer threshold
    overlap_pixels = np.sum(img < 240)

    text_pixels = np.sum(text_mask)
    total_pixels = img.size

    return {
        "overlap_ratio": overlap_pixels / text_pixels if text_pixels else 0.0,
        "coverage_ratio": text_pixels / total_pixels,
        "overlap_pixels": overlap_pixels,
        "text_pixels": text_pixels,
        "total_pixels": total_pixels
    }

def compute_overlap_and_coverage(svg_path, dpi=300):
    tree, root, text_elems = get_text_elements(svg_path)

    if len(text_elems) == 0:
        return {
            "overlap_ratio": 0.0,
            "coverage_ratio": 0.0,
            "overlap_pixels": 0,
            "text_pixels": 0,
            "total_pixels": 0
        }

    # --- Overlap ---
    masks = []
    tl = len(text_elems)
    for elem in text_elems:
        mask = render_single_text_mask(tree, root, elem, dpi=dpi)
        masks.append(mask)
        print("Rendered mask for text element {}/{}".format(len(masks), tl))

    stack = np.stack(masks, axis=0)
    occupancy = stack.sum(axis=0)

    text_pixels = np.sum(occupancy >= 1)
    overlap_pixels = np.sum(occupancy > 1)

    overlap_ratio = overlap_pixels / text_pixels if text_pixels > 0 else 0.0

    # --- Coverage ---
    full_text_mask = render_full_svg_mask(svg_path, dpi=dpi)
    total_pixels = full_text_mask.size
    coverage_ratio = np.sum(full_text_mask) / total_pixels

    return {
        "overlap_ratio": overlap_ratio,
        "coverage_ratio": coverage_ratio,
        "overlap_pixels": overlap_pixels,
        "text_pixels": text_pixels,
        "total_pixels": total_pixels
    }

if __name__ == "__main__":
    svg_file = "text_output.svg"
    result = overlap_and_coverage_fast(svg_file, dpi=600)
    print(f"Overlap Ratio: {result['overlap_ratio']:.4f} ({result['overlap_pixels']} overlapping pixels out of {result['text_pixels']} text pixels)")
    print(f"Coverage Ratio: {result['coverage_ratio']:.4f}")