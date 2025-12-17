"""
streamline_tracing.py

Streamline tracing over a normalized 2D direction field for segmented regions.
Produces smooth, spaced flow lines suitable for guiding later text/pattern placement.

Dependencies:
  - numpy
  - opencv-python (cv2)
  - scikit-image (skimage)
  - matplotlib (optional, not required here)

Inputs (defaults placed next to this script):
  - vector_field.npy      (H, W, 2) float32, ideally unit vectors per pixel
  - segmentation.png/.npy (optional) region labels; if absent, a single region is used

Outputs:
  - streamlines.npy            list of (N_i, 2) float32 arrays (x, y)
  - streamlines_preview.png    visualization overlay

Design notes
------------
* Seeds are sampled on a jittered grid per region, away from borders.
* Each seed traces a streamline using RK4 integration with bilinear interpolation
  of the direction field; tracing occurs forward and backward then concatenated.
* Spacing enforced via a cover mask + distance transform. Seeds or paths too
  close to existing lines are skipped/terminated.
* Optional curvature check prunes overly tight curves.

"""
from __future__ import annotations

import os
import math
from typing import List, Tuple, Optional, Iterable

import numpy as np
import cv2
from skimage import io as skio
from skimage.morphology import binary_erosion, disk



# ==========================
# I/O
# ==========================

def load_field_and_seg(field_path: str = "vector_field.npy", seg_path: Optional[str] = "segmentation.png") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not os.path.exists(field_path):
        raise FileNotFoundError(f"Direction field not found: {field_path}")
    field = np.load(field_path)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("vector_field.npy must have shape (H, W, 2)")
    field = field.astype(np.float32)

    seg = None
    if seg_path and os.path.exists(seg_path):
        if seg_path.lower().endswith('.npy'):
            seg = np.load(seg_path)
        else:
            arr = skio.imread(seg_path)
            if arr.ndim > 2:
                arr = arr[..., 0]
            seg = arr
        seg = seg.astype(np.int32)
        if seg.shape[:2] != field.shape[:2]:
            raise ValueError(f"Segmentation shape {seg.shape} does not match field {field.shape[:2]}")
    return field, seg


# ==========================
# Field smoothing
# ==========================

def pre_smooth_field(field: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Smooth vector field components with Gaussian blur and re-normalize."""
    vx = cv2.GaussianBlur(field[..., 0], (0, 0), sigma)
    vy = cv2.GaussianBlur(field[..., 1], (0, 0), sigma)
    mag = np.sqrt(vx ** 2 + vy ** 2) + 1e-6
    field[..., 0] = vx / mag
    field[..., 1] = vy / mag
    return field


# ==========================
# Sampling utilities
# ==========================

def sample_seed_points(seg: Optional[np.ndarray], spacing: int, jitter: float = 0.3, border_margin: int = 2) -> List[Tuple[int, int, float, float]]:
    if seg is None:
        raise ValueError("Segmentation map required for seed sampling.")
    h, w = seg.shape
    labels = list(np.unique(seg))
    masks = {lab: (seg == lab) for lab in labels}

    seeds: List[Tuple[int, int, float, float]] = []
    rng = np.random.default_rng(42)

    for lab in labels:
        mask = masks[lab]
        if border_margin > 0:
            se = disk(border_margin)
            eroded = binary_erosion(mask, se)
        else:
            eroded = mask
        if not np.any(eroded):
            continue

        ys = np.arange(spacing // 2, h, spacing)
        xs = np.arange(spacing // 2, w, spacing)
        for y0 in ys:
            for x0 in xs:
                jy = int(round((rng.random() * 2 - 1) * jitter * spacing))
                jx = int(round((rng.random() * 2 - 1) * jitter * spacing))
                y = np.clip(y0 + jy, 0, h - 1)
                x = np.clip(x0 + jx, 0, w - 1)
                if eroded[y, x]:
                    seeds.append((lab, int(y), int(x), 1.0))
    return seeds

# Helper to access global H,W from a dummy cache (set in main)
_FIELD_HW: Tuple[int, int] = (0, 0)

def field_hw() -> Tuple[int, int]:
    return _FIELD_HW


# ==========================
# Field interpolation
# ==========================

def interpolate_vector(field: np.ndarray, x: float, y: float) -> Tuple[float, float, float]:
    """Bilinear interpolate vector at subpixel (x,y). Returns (vx, vy, mag)."""
    h, w, _ = field.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0, 0.0, 0.0
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0

    # Corners
    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]

    # Bilinear
    v0 = v00 * (1 - dx) + v10 * dx
    v1 = v01 * (1 - dx) + v11 * dx
    v = v0 * (1 - dy) + v1 * dy

    vx, vy = float(v[0]), float(v[1])
    mag = math.hypot(vx, vy)
    return vx, vy, mag

# ==========================
# Tracing
# ==========================

def rk4_step(field: np.ndarray, x: float, y: float, hstep: float) -> Tuple[float, float]:
    """One RK4 integration step for dx/ds = v(x), assuming v is normalized."""
    def v_at(xp, yp):
        vx, vy, m = interpolate_vector(field, xp, yp)
        if m > 1e-6:
            return vx / m, vy / m
        return 0.0, 0.0

    k1x, k1y = v_at(x, y)
    k2x, k2y = v_at(x + 0.5 * hstep * k1x, y + 0.5 * hstep * k1y)
    k3x, k3y = v_at(x + 0.5 * hstep * k2x, y + 0.5 * hstep * k2y)
    k4x, k4y = v_at(x + hstep * k3x, y + hstep * k3y)

    dx = (hstep / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    dy = (hstep / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    return x + dx, y + dy


def trace_single_streamline(
    seed_xy: Tuple[float, float],
    field: np.ndarray,
    seg: Optional[np.ndarray],
    region_label: Optional[int],
    step_size: float,
    max_length: int,
    cover_dist: np.ndarray,
    min_spacing: float,
    loop_epsilon: float = 2.0,
    min_mag: float = 1e-3,
) -> Optional[np.ndarray]:
    """Trace one streamline from a seed, forward and backward, with spacing checks.

    cover_dist: distance map of existing coverage; tracing terminates if below min_spacing*0.5.
    Returns points as (N,2) float32 array in (x,y) order, or None if rejected.
    """
    h, w, _ = field.shape

    def in_region(x, y) -> bool:
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        if seg is None or region_label is None:
            return True
        return seg[int(math.floor(y)), int(math.floor(x))] == region_label

    def adv_dir(x, y, sign: float) -> Tuple[float, float, float]:
        vx, vy, m = interpolate_vector(field, x, y)
        vx *= sign
        vy *= sign
        if m > 1e-6:
            return vx / m, vy / m, m
        return 0.0, 0.0, m

    def propagate(x0, y0, sign: float) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        x, y = x0, y0
        visited_check_every = 5
        for i in range(max_length):
            # Spacing termination
            if cover_dist[int(math.floor(y)), int(math.floor(x))] < (min_spacing * 0.5):
                break #dont break just move?
            vx, vy, m = interpolate_vector(field, x, y)
            if m < min_mag:
                break
            # RK4 step in signed direction
            x_next, y_next = rk4_step(field if sign > 0 else -field, x, y, step_size)
            if not in_region(x_next, y_next):
                pass #change back to break
            pts.append((x_next, y_next))
            x, y = x_next, y_next
            # Simple loop/backtrack detection
            if i % visited_check_every == 0 and len(pts) > 8:
                px, py = pts[-1]
                for j in range(len(pts) - 8):
                    qx, qy = pts[j]
                    if (px - qx) ** 2 + (py - qy) ** 2 < loop_epsilon ** 2:
                        return pts
        return pts

    sx, sy = seed_xy
    if not in_region(sx, sy):
        return None

    # Check initial spacing
    if cover_dist[int(round(sy)), int(round(sx))] < min_spacing:
        return None

    fwd = propagate(sx, sy, +1.0)
    bwd = propagate(sx, sy, -1.0)

    if len(fwd) + len(bwd) < 6:
        return None

    bwd_rev = list(reversed(bwd))
    pts = bwd_rev + [(sx, sy)] + fwd

    #flip streamline if right to left or bottom up
    if len(pts) > 1:
        if pts[0][0] > pts[-1][0] or pts[0][1] > pts[-1][1]:
            pts = list(reversed(pts))
    
    arr = np.array(pts, dtype=np.float32)
    return arr


# ==========================
# Spacing & curvature
# ==========================

def update_coverage_and_distance(cover_mask: np.ndarray, polyline_xy: np.ndarray, thickness: int) -> None:
    """Draw polyline on cover_mask (in-place) and update distance transform region sparsely.
    For simplicity we recompute full distance transform; for large images consider ROI updates.
    """
    pts_i = np.round(polyline_xy[:, [0, 1]]).astype(np.int32)
    pts_i[:, 0] = np.clip(pts_i[:, 0], 0, cover_mask.shape[1] - 1)
    pts_i[:, 1] = np.clip(pts_i[:, 1], 0, cover_mask.shape[0] - 1)
    for i in range(len(pts_i) - 1):
        p1 = (int(pts_i[i, 0]), int(pts_i[i, 1]))
        p2 = (int(pts_i[i + 1, 0]), int(pts_i[i + 1, 1]))
        cv2.line(cover_mask, p1, p2, color=255, thickness=thickness, lineType=cv2.LINE_AA)


def compute_distance_from_coverage(cover_mask: np.ndarray) -> np.ndarray:
    inv = (cover_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    return dist


def compute_curvature(streamline: np.ndarray) -> np.ndarray:
    """Discrete curvature κ along a polyline. Returns array of shape (N,) with NaN at ends."""
    xy = streamline.astype(np.float64)
    x = xy[:, 0]
    y = xy[:, 1]
    # First and second derivatives via central differences
    xp = np.gradient(x)
    yp = np.gradient(y)
    xpp = np.gradient(xp)
    ypp = np.gradient(yp)
    denom = (xp * xp + yp * yp) ** 1.5 + 1e-8
    kappa = np.abs(xp * ypp - yp * xpp) / denom
    return kappa.astype(np.float32)


# ==========================
# Visualization
# ==========================

def visualize_streamlines(
    streamlines: List[np.ndarray],
    seg: Optional[np.ndarray],
    out_path: str,
    bg: Optional[np.ndarray] = None,
    color_by: str = "region",
) -> None:
    if seg is not None:
        h, w = seg.shape
        # Color map by region label
        seg_norm = (seg.astype(np.float32) - seg.min())
        if seg_norm.max() > 0:
            seg_norm /= seg_norm.max()
        seg_rgb = cv2.applyColorMap((seg_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas = seg_rgb
    else:
        if bg is not None:
            canvas = bg.copy()
        else:
            # neutral background
            h, w = streamlines[0].shape[0] if streamlines else 512, 512
            canvas = np.full((h, w, 3), 240, np.uint8)

    # Draw streamlines
    for sl in streamlines:
        if len(sl) < 2:
            continue
        # choose color by length or region random
        col = (0, 0, 0)
        if color_by == "length":
            L = max(2, len(sl))
            v = int(255 * (min(L, 400) / 400.0))
            col = (0, v, 255 - v)
        else:
            # region coloring approximated by first point's seg label
            if seg is not None:
                x0, y0 = int(math.floor(sl[0, 0])), int(math.floor(sl[0, 1]))
                lab = int(seg[y0, x0])
                rng = np.random.default_rng(lab)
                col = tuple(int(c) for c in rng.integers(64, 255, size=3))
            else:
                col = (10, 10, 10)
        pts = np.round(sl).astype(np.int32)
        for i in range(len(pts) - 1):
            p1 = (int(pts[i, 0]), int(pts[i, 1]))
            p2 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
            cv2.line(canvas, p1, p2, color=col, thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


# ==========================
# Main orchestration
# ==========================

def trace_streamlines(
    field: np.ndarray,
    seeds: Iterable[Tuple[int, int, float, float]],
    seg: Optional[np.ndarray],
    step_size: float = 1.0,
    max_length: int = 1000,
    min_spacing: int = 4,
    curvature_thresh: Optional[float] = 0.25,
    thickness_for_spacing: Optional[int] = None,
) -> List[np.ndarray]:
    """High-level streamline tracing with spacing and curvature pruning."""
    global _FIELD_HW
    h, w, _ = field.shape
    _FIELD_HW = (h, w)

    cover_mask = np.zeros((h, w), np.uint8)
    if thickness_for_spacing is None:
        thickness_for_spacing = max(1, int(round(min_spacing)))

    streamlines: List[np.ndarray] = []
    cover_dist = compute_distance_from_coverage(cover_mask)

    for lab, sy, sx, _wt in seeds:
        # Skip seeds too close to current coverage
        if cover_dist[sy, sx] < min_spacing:
            continue
        sl = trace_single_streamline(
            (float(sx), float(sy)), field, seg, lab if seg is not None else None,
            step_size=step_size, max_length=max_length, cover_dist=cover_dist, min_spacing=min_spacing,
        )
        if sl is None or len(sl) < 8:
            continue
        # Curvature check
        if curvature_thresh is not None and curvature_thresh > 0:
            kappa = compute_curvature(sl)
            print(kappa)
            if np.nanmax(kappa) > curvature_thresh:
                print(f"Streamline at seed ({sx},{sy}) rejected due to high curvature {np.nanmax(kappa):.3f}")
                continue

        # Accept streamline
        streamlines.append(sl.astype(np.float32))
        update_coverage_and_distance(cover_mask, sl, thickness=thickness_for_spacing)
        cover_dist = compute_distance_from_coverage(cover_mask)

    return streamlines


def main():
    field, seg = load_field_and_seg("vector_field_2rosy.npy", "segmentation.png")
    h, w, _ = field.shape


    # Smooth the field for more coherent text-block flow
    field = pre_smooth_field(field, sigma=2.5)


    # Seed sampling (denser grid)
    seeds = sample_seed_points(seg, spacing=7, jitter=0.35, border_margin=2)


    # Trace with adjusted parameters for smoother, denser lines
    streamlines = trace_streamlines(
        field, seeds, seg,
        step_size=1.0, # larger step smooths small curls
        max_length=1200, # allow longer flow lines
        min_spacing=5, # closer lines for text-like density
        curvature_thresh=30,
    )


    np.save("streamlines.npy", np.array(streamlines, dtype=object))
    visualize_streamlines(streamlines, seg, "streamlines_preview.png")
    print(f"Saved {len(streamlines)} streamlines to streamlines.npy and preview to streamlines_preview.png")




if __name__ == "__main__":
    main()
