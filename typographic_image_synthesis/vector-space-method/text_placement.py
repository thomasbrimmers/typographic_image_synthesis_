#!/usr/bin/env python3
"""
text_placement.py
------------------
Render text along precomputed streamlines to create a masked SVG that reveals an underlying image,
then rasterize to PNG.

Requirements:
    - Python 3
    - numpy
    - opencv-python (cv2)
    - svgwrite

Usage:
    Place `image.png` and `streamlines.npy` in the same directory as this script, then run:
        python3 text_placement.py

Expected outputs:
    - text_output.svg
    - text_output.png
"""

import os
import sys
import math
import numpy as np
import cv2
import svgwrite
from typing import List, Tuple
from scipy.spatial import cKDTree

# -----------------------------
# Configurable Parameters
# -----------------------------
FONT_SIZE = 10                # in px
FONT_FAMILY = "Arial"
TEXT_COLOR = "white"          # Mask color; white reveals, black conceals
SPACING = 8                   # Not strictly needed for single-path text; kept for future multi-line use
STROKE_WIDTH = 3              # Optional: set >0 to draw the streamline path (debug/preview)
TEXT_SOURCE_PATH = "text_source.txt"  # Path to text source file

# File paths
IMAGE_PATH = "img.png"
STREAMLINES_PATH = "streamlines.npy"
SVG_OUT = "text_output.svg"
PNG_OUT = "text_output.png"
MIN_FONT = 3
MAX_FONT = 25
K_NEIGHBOURS = 160
CHUNK_SIZE = 4  # number of vertices per chunk, tweak as needed

# Minimum path length (in px) to consider placing text (very short paths can cause rendering artifacts)
MIN_PATH_LENGTH = 10.0


def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """Load image with OpenCV (BGR) and return (img, width, height)."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'. Ensure the file exists.")
    height, width = img.shape[:2]
    return img, width, height


def load_streamlines(npy_path: str) -> List[List[Tuple[float, float]]]:
    """
    Load streamlines: expected format is a numpy array / object array where each entry
    is a list/array of (x, y) coordinate pairs in image pixel space.
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Streamlines file not found at '{npy_path}'.")
    data = np.load(npy_path, allow_pickle=True)
    streamlines = []
    for entry in data:
        pts = np.asarray(entry, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            # Try to coerce or skip malformed entries
            try:
                pts = np.array([[p[0], p[1]] for p in entry], dtype=float)
            except Exception:
                continue
        streamlines.append(pts.tolist())
    if len(streamlines) == 0:
        raise ValueError("No valid streamlines loaded from the provided .npy file.")
    return streamlines


def path_length(points: List[Tuple[float, float]]) -> float:
    """Compute polyline length."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        total += math.hypot(dx, dy)
    return total


def points_to_svg_path(points: List[Tuple[float, float]]) -> str:
    """Convert a list of (x, y) points into an SVG path string 'M x0,y0 L x1,y1 ...'"""
    if not points:
        return ""
    # Round to 3 decimals for smaller SVG output
    coords = [f"{round(p[0],3)},{round(p[1],3)}" for p in points]
    return "M " + " L ".join(coords)


def repeated_text_generator(source: str):
    """
    Generator that yields characters from `source` indefinitely, cycling as needed.
    Ensures there's always text to fill any path lengths.
    """
    if not source:
        source = " "
    i = 0
    n = len(source)
    while True:
        yield source[i % n]
        i += 1


def consume_chars(gen, n_chars: int) -> str:
    """Consume n_chars from a character generator."""
    return "".join(next(gen) for _ in range(max(n_chars, 0)))


def estimate_chars_for_path(length_px: float, font_size_px: float) -> int:
    """
    Estimate the number of characters needed to approximately fill a path of given length.
    Heuristic: average character width ~ 0.6 * font_size (for typical Latin text).
    """
    if length_px <= 0 or font_size_px <= 0:
        return 0
    avg_char_w = 0.5 * font_size_px
    # Add a small cushion so textPath tends to cover the full length without running short.
    n = int(math.ceil(length_px / max(avg_char_w, 1e-6)))
    return max(n, 0)

def distance_profile_for_streamline(
    idx,
    streamline_points,
    all_points,
    all_ids,
    tree,
    k=20,
):
    """
    For each point on a given streamline, return a *one-sided* distance
    to the nearest point belonging to a different streamline, measured
    along the normal pointing towards the text side.

    Parameters
    ----------
    idx : int
        Streamline id of the current streamline.
    streamline_points : array-like, shape (n_i, 2)
        Points of the current streamline.
    all_points : array-like, shape (N, 2)
        All points used to build the KD-tree (global point cloud).
    all_ids : array-like, shape (N,)
        Streamline id for each point in all_points.
    tree : cKDTree
        KD-tree built on all_points.
    k : int, optional
        Number of neighbors to query per point (default 20).

    Returns
    -------
    distances : ndarray, shape (n_i,)
        One-sided distance for each point along the streamline.
    """

    pts = np.asarray(streamline_points, dtype=float)
    all_points = np.asarray(all_points, dtype=float)
    n_pts = len(pts)

    # --- 1) Compute tangents and normals for each point on this streamline ---
    tangents = np.zeros_like(pts)

    for i in range(n_pts):
        if n_pts == 1:
            diff = np.array([1.0, 0.0])
        elif i == 0:
            diff = pts[1] - pts[0]
        elif i == n_pts - 1 or i == n_pts:
            diff = pts[-1] - pts[-2]
        else:
            diff = pts[i + 1] - pts[i - 1]

        norm = np.linalg.norm(diff)
        if norm == 0.0:
            tangents[i] = np.array([1.0, 0.0])  # arbitrary fallback
        else:
            tangents[i] = diff / norm

    # normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    # # else:  # "right"
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)

    # --- 2) Query neighbors for each point on this streamline ---
    dists, neighbors = tree.query(pts, k=k)

    # Ensure 2D arrays
    if k == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    # --- 3) One-sided nearest distance along the normal, skipping same-streamline points ---
    min_d = np.empty(n_pts, dtype=float)

    for i in range(n_pts):
        p = pts[i]
        n = normals[i]
        best = np.inf

        for j in range(k):
            nb_idx = neighbors[i, j]
            if nb_idx == -1 or nb_idx >= len(all_points):
                continue

            # Skip same-streamline points
            if all_ids[nb_idx] == idx:
                continue

            q = all_points[nb_idx]
            v = q - p

            # Projection onto the normal (one-sided)
            proj = np.dot(v, n)

            # Only consider neighbors on the text side
            if proj <= 0.0:
                continue

            dist = np.linalg.norm(v)
            # Use the projected distance as our measure of available space
            if dist < best:
                best = dist

        if not np.isfinite(best):
            best = 0.0  # or some fallback

        min_d[i] = best

    return min_d

def map_distance_to_font_size(d, d_min, d_max):
    # normalize
    if d_max <= d_min:
        t = 0.0
    else:
        t = (d - d_min) / (d_max - d_min)
    t = max(0.0, min(1.0, t))
    return MIN_FONT + t * (MAX_FONT - MIN_FONT)


def main():
    # Load inputs
    img, width, height = load_image(IMAGE_PATH)
    streamlines = load_streamlines(STREAMLINES_PATH)

    # Prepare SVG
    dwg = svgwrite.Drawing(SVG_OUT, size=(width, height), profile='full')
    # Embed the base image (for reference; final visibility is through mask)
    # Some viewers render masked images better if the base image exists in the scene graph.
    base_img = dwg.image(IMAGE_PATH, insert=(0, 0), size=(width, height))
    dwg.add(base_img)

    # Definitions: paths + mask
    defs = dwg.defs
    dwg.add(defs)

    # Create a mask to reveal the image wherever text appears (white = reveal, black = hide).
    mask = dwg.mask(id="textMask")
    defs.add(mask)

    # Optional: Background for mask is black (fully conceals). If not specified, default may be 'transparent' (treated as black).
    mask_bg = dwg.rect(insert=(0, 0), size=(width, height), fill="black")
    mask.add(mask_bg)

    # Group that will hold all text following paths
    mask_group = dwg.g(id="textMaskGroup")
    
    # Read text File
    with open(TEXT_SOURCE_PATH, "r", encoding="utf-8") as f:
        text_source = f.read().replace(" ", "").replace("\n", "")

    # Text generator
    char_gen = repeated_text_generator(text_source)
    
    all_points = []
    all_ids = []  # which streamline index each point belongs to

    for idx, pts in enumerate(streamlines):
        for p in pts:
            all_points.append(p)
            all_ids.append(idx)
            
    tree = cKDTree(all_points)

    all_points = np.asarray(all_points, dtype=float)
    all_ids = np.asarray(all_ids, dtype=int)

    debug_group = dwg.g()
    debug_group.add(dwg.image(IMAGE_PATH, insert=(0, 0), size=(width, height)))
    
    # Add each streamline path and corresponding textPath
    for idx, pts in enumerate(streamlines):
        if pts is None or len(pts) < 2:
            continue

        # Clean invalid points
        clean_pts = []
        for x, y in pts:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            # Clamp to image bounds to avoid extremely out-of-bounds artifacts
            cx = float(min(max(x, 0.0), width))
            cy = float(min(max(y, 0.0), height))
            clean_pts.append((cx, cy))

        if len(clean_pts) < 2:
            continue

        length = path_length(clean_pts)
        if length < MIN_PATH_LENGTH:
            continue
        
        dist_profile = distance_profile_for_streamline(idx, pts, all_points, all_ids, tree, k=K_NEIGHBOURS)

        d_min = float(dist_profile.min())
        d_max = float(dist_profile.max())

       
        num_pts = len(clean_pts)

        for start in range(0, num_pts - 1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_pts)
            sub_pts = clean_pts[max(start-1,0):min(end+1,num_pts-1)]
            if len(sub_pts) < 2:
                continue

            # average distance in this chunk
            avg_d = float(np.mean(dist_profile[start:end]))
            local_font_size = map_distance_to_font_size(avg_d, d_min, d_max)

            # create a path for this chunk
            chunk_id = f"path_{idx}_chunk_{start}"
            chunk_d = points_to_svg_path(sub_pts)
            chunk_path = dwg.path(d=chunk_d, id=chunk_id, fill="none")
            defs.add(chunk_path)

            # estimate chars for this chunk
            chunk_len = path_length(sub_pts)
            n_chars = estimate_chars_for_path(chunk_len, local_font_size)
            text_string = consume_chars(char_gen, n_chars)
            
            #print(text_string)
            #print(f"Streamline {idx} chunk {start}-{end}: length={chunk_len:.1f}px, avg_dist={avg_d:.1f}px, font_size={local_font_size:.1f}px, chars={n_chars}")
            text_el = dwg.text(
                "",
                font_size=local_font_size,
                font_family=FONT_FAMILY,
                fill=TEXT_COLOR,
                font_weight="bold"
            )
            text_el.add(dwg.textPath(f"#{chunk_id}", text_string, method='align', textLength=chunk_len, lengthAdjust="spacingAndGlyphs"))
            debug_group.add(text_el)

    # Add the group of all text to the mask
    #mask.add(mask_group)

    # DEBUG: Just show the text directly on top of the image (no masking)
    # Comment out mask usage


    # # Draw visible text (e.g., black text over image)
    # for idx, pts in enumerate(streamlines):
    #     if pts is None or len(pts) < 2:
    #         continue

    #     clean_pts = [(min(max(float(x), 0), width), min(max(float(y), 0), height)) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
    #     if len(clean_pts) < 2:
    #         continue

    #     length = path_length(clean_pts)
    #     if length < MIN_PATH_LENGTH:
    #         pass #continue

    #     path_id = f"path_{idx}"
    #     text_string = TEXT_SOURCE[:int(length)]  # shorter preview text
    #     text_el = dwg.text(
    #         "",
    #         font_size=FONT_SIZE,
    #         font_family=FONT_FAMILY,
    #         fill="black"
    #     )
    #     tp = dwg.textPath(f"#{path_id}", text_string)
    #     text_el.add(tp)
    #     debug_group.add(text_el)

    dwg.add(debug_group)

    # Save SVG
    dwg.save()
    print(f"Saved SVG to {SVG_OUT}")
    
    # Postprocess SVG to replace xlink:href with href for modern renderers
    with open(SVG_OUT, "r", encoding="utf-8") as f:
        svg_text = f.read()

    svg_text = svg_text.replace("xlink:href", "href")
    # Optionally remove the old xmlns:xlink declaration
    svg_text = svg_text.replace('xmlns:xlink="http://www.w3.org/1999/xlink"', "")

    with open(SVG_OUT, "w", encoding="utf-8") as f:
        f.write(svg_text)
    print("Patched SVG for href compatibility.")


if __name__ == "__main__":
    main()

