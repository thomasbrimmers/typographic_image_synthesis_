"""
vector_room_based_method_2rosy.py

2-RoSy (orientation-only) vector field on a raster grid with relaxed boundary
alignment, inspired by Maharik et al., "Digital Micrography" (SIGGRAPH 2011).

Differences from the original prototype:
- Field is represented as an angle theta in [0, pi) (2-RoSy symmetry: v and -v
  are equivalent). All smoothing and blending are done in circular space.
- Boundary relaxation decides if the field near a contour should align with the
  boundary tangent or be perpendicular to it, based on curvature and simple
  spatial coherence.

Dependencies:
  - numpy
  - opencv-python (cv2)
  - scikit-image (skimage)
  - matplotlib (for visualization)

Inputs (next to this script by default):
  - img.png            (RGB or grayscale raster image)
  - segmentation.png   (uint labels) — or .npy, .tif supported via skimage

Outputs:
  - vector_field_2rosy.npy            (H, W, 2) float32 of unit vectors
  - vector_field_2rosy_preview.png    (visualization)
  - boundary_alignment_labels.png     (optional visualization; 0=tangent,1=perp)

Usage:
  python vector_room_based_method_2rosy.py
"""
from __future__ import annotations

import os
import math
from typing import Tuple, Optional

import numpy as np
import cv2
from skimage import io as skio
from skimage import img_as_float32
from skimage.util import img_as_ubyte


# ==========================
# Utility helpers
# ==========================

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB to single-channel float32 in [0,1]."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    gray = img_as_float32(gray)
    return gray


def _normalize_vectors(vx: np.ndarray, vy: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.sqrt(vx * vx + vy * vy)
    mag = np.maximum(mag, eps)
    return vx / mag, vy / mag


def _masked_gaussian_blur(arr: np.ndarray, mask: np.ndarray, ksize: int = 0, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur with masked normalized convolution to avoid region bleeding."""
    mask_f = mask.astype(np.float32)
    arr_ = arr * mask_f
    blurred = cv2.GaussianBlur(arr_, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    norm = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    norm = np.maximum(norm, 1e-6)
    out = blurred / norm
    out[mask == 0] = 0.0
    return out


def _distance_weight(d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian falloff weight from boundary distance (1 near boundary, ->0 inside)."""
    return np.exp(-(d.astype(np.float32) ** 2) / (2.0 * (sigma ** 2)))


# ==========================
# Angle (2-RoSy) utilities
# ==========================

def angle_normalize_pi(theta: np.ndarray) -> np.ndarray:
    """Normalize angles to [0, pi)."""
    return np.mod(theta, 2* np.pi)


def vectors_to_angle_mod_pi(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    theta = np.arctan2(ty, tx)
    # collapse 2*pi to pi periodicity (2-RoSy): map theta and theta+pi
    return angle_normalize_pi(theta)


def angle_to_vector(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def circular_blend_angles(theta_a: np.ndarray, theta_b: np.ndarray, w_b: np.ndarray) -> np.ndarray:
    """Blend two angle fields with weight w_b in [0,1] using 2-RoSy averaging."""
    ca, sa = np.cos(theta_a), np.sin(theta_a)
    cb, sb = np.cos(theta_b), np.sin(theta_b)
    cx = (1.0 - w_b) * ca + w_b * cb
    cy = (1.0 - w_b) * sa + w_b * sb
    return angle_normalize_pi(np.arctan2(cy, cx))


def smooth_angles_within_regions(seg: np.ndarray, theta: np.ndarray, iters: int = 2, sigma: float = 1.5) -> np.ndarray:
    """Iteratively smooth an angle field by blurring cos/sin under region masks."""
    labels = np.unique(seg)
    masks = {lab: (seg == lab) for lab in labels}
    th = theta.copy()
    for _ in range(max(1, iters)):
        c = np.cos(th).astype(np.float32)
        s = np.sin(th).astype(np.float32)
        c_blur = np.zeros_like(c)
        s_blur = np.zeros_like(s)
        for lab, m in masks.items():
            if not np.any(m):
                continue
            c_blur[m] = _masked_gaussian_blur(c, m, ksize=0, sigma=sigma)[m]
            s_blur[m] = _masked_gaussian_blur(s, m, ksize=0, sigma=sigma)[m]
        th = angle_normalize_pi(np.arctan2(s_blur, c_blur))
    return th


# ==========================
# Core stages (2-RoSy)
# ==========================

def compute_image_gradient_tangent_angle(gray: np.ndarray, method: str = "sobel", ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tangent direction from image gradients; return (theta, grad_mag, (gx,gy))."""
    
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (11, 11))
    gray = cv2.blur(gray, (7, 7))
    gray = cv2.blur(gray, (5, 5))
    gray = cv2.GaussianBlur(gray, (7, 7), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
    #gray = cv2.bilateralFilter(gray, 5, 75, 75, borderType=cv2.BORDER_REPLICATE)
    #gray = cv2.bilateralFilter(gray, 5, 75, 75, borderType=cv2.BORDER_REPLICATE)
    cv2.imshow("gray_blurred", gray)
    
    if method == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    # tangent is perpendicular to gradient
    gx = cv2.blur(gx, (11, 11))
    gy = cv2.blur(gy, (11, 11))
    gx = cv2.blur(gx, (11, 11))
    gx = cv2.blur(gx, (11, 11))
    gy = cv2.blur(gy, (11, 11))
    gx = cv2.blur(gx, (11, 11))
    gy = cv2.blur(gy, (11, 11))
    gx = cv2.blur(gx, (11, 11))
    gy = cv2.blur(gy, (11, 11))
    gx = cv2.blur(gx, (11, 11))
    gy = cv2.blur(gy, (11, 11))
    gy = cv2.blur(gy, (11, 11)) #try opencv example i saw online that had clean lenna
    
    gx = cv2.GaussianBlur(gx, (7, 7), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.GaussianBlur(gy, (7, 7), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
    tx = -gy
    ty = gx
    cv2.imshow("gx", gx)
    cv2.imshow("gy", gy)
    tx, ty = _normalize_vectors(tx, ty)
    cv2.imshow("tx", tx)
    cv2.imshow("ty", ty)
    theta = vectors_to_angle_mod_pi(tx, ty)

    grad_mag = np.sqrt(gx * gx + gy * gy)
    cv2.imshow("theta", theta / (2 * np.pi))
    return theta, grad_mag, (gx, gy)


def boundary_geometry_for_region(region_mask: np.ndarray, dist_sigma: float = 6.0, boundary_band: float = 3.0):
    """Compute distance, inward normals, tangent/normal angles, weights and a boundary band mask."""
    reg_u8 = (region_mask.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(reg_u8, distanceType=cv2.DIST_L2, maskSize=3)

    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny = _normalize_vectors(gx, gy)  # inward normal

    tx_b = -ny
    ty_b = nx
    tx_b, ty_b = _normalize_vectors(tx_b, ty_b)

    theta_t = vectors_to_angle_mod_pi(tx_b, ty_b)
    theta_n = angle_normalize_pi(theta_t + np.pi * 0.5)

    w_b = _distance_weight(dist, sigma=dist_sigma)
    band = dist <= boundary_band

    # curvature estimate: divergence of normalized normals
    dnx_dx = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3)
    dny_dy = cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3)
    curvature = np.abs(dnx_dx + dny_dy)

    return {
        "dist": dist,
        "nx": nx,
        "ny": ny,
        "theta_t": theta_t,
        "theta_n": theta_n,
        "w_b": w_b,
        "band": band & region_mask,
        "curvature": curvature,
    }


def decide_boundary_alignment(curvature_map: np.ndarray, grad_mag: np.ndarray, coherence_passes: int = 2) -> np.ndarray:
    """
    For each boundary pixel, decide between tangent (0) or perpendicular (1).
    Heuristics:
      - Higher curvature -> prefer perpendicular to avoid folding.
      - Strong image structure (grad) -> slightly prefer tangent.
      - Spatial coherence via few 3x3 majority passes.
    Returns binary mask (uint8) of same shape, where 1 = perpendicular, 0 = tangent.
    """
    # Normalize maps to [0,1] robustly using percentiles
    cur = curvature_map.astype(np.float32)
    gm = grad_mag.astype(np.float32)

    def norm_by_pct(a):
        lo, hi = np.percentile(a[a > 0], (10, 90)) if np.any(a > 0) else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        b = (a - lo) / (hi - lo)
        return np.clip(b, 0.0, 1.0)

    cur_n = norm_by_pct(cur)
    gm_n = norm_by_pct(gm)

    # Score: positive means perpendicular preferred
    score = cur_n - 0.35 * gm_n

    # Initial label by threshold around 0.5 (tunable). Using 0 for tangent, 1 for perp
    labels = (score > 0.5).astype(np.uint8)

    # Spatial coherence: majority filter (3x3) a few passes
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(0, coherence_passes)):
        # Count neighbors voting for 1 (perp)
        neigh_sum = cv2.filter2D(labels.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        # If majority (>4) neighbors are 1 -> set 1, if <4 -> set 0, else keep
        set_perp = neigh_sum > 4.5
        set_tang = neigh_sum < 4.5
        labels[set_perp] = 1
        labels[set_tang] = 0

    return labels


def blend_field_with_boundary_relaxation(
    seg: np.ndarray,
    theta_grad: np.ndarray,
    grad_mag: np.ndarray,
    boundary_dist_sigma: float = 6.0,
    boundary_band: float = 3.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """For each region, compute boundary geometry + labels and blend orientations.

    Returns the blended angle field (theta) and an optional label map where
    0=tangent (blue), 1=perpendicular (red) on boundary pixels (else 255).
    """
    h, w = seg.shape
    theta_out = theta_grad.copy()
    label_vis = np.full((h, w), 255, np.uint8)
    
    all_mask = np.zeros_like(seg, bool)
    blended_sum_c = np.cos(theta_out)
    blended_sum_s = np.sin(theta_out)
    weight_accum = np.ones_like(theta_out)

    labels = np.unique(seg)
    for lab in labels:
        region = (seg == lab)
        if not np.any(region):
            continue
        geom = boundary_geometry_for_region(region, dist_sigma=boundary_dist_sigma, boundary_band=boundary_band)

        # boundary alignment labels only for band
        band = geom["band"]
        if not np.any(band):
            continue
        labels_bp_full = decide_boundary_alignment(geom["curvature"], grad_mag)
        labels_bp = labels_bp_full[band]

        # create a full-size label map for visualization
        label_vis[band] = labels_bp * 1  # 0 or 1

        # target angles on band: choose tangent or normal
        theta_target = geom["theta_t"].copy()
        theta_target[band] = np.where(labels_bp == 0, geom["theta_t"][band], geom["theta_n"][band])

        # blend with weight that decays from boundary inward
        w = geom['w_b'] * region.astype(np.float32)
        blended_sum_c += np.cos(theta_target)*w
        blended_sum_s += np.sin(theta_target)*w
        weight_accum += w
        all_mask |= region
        
    theta_out = angle_normalize_pi(np.arctan2(blended_sum_s/weight_accum, blended_sum_c/weight_accum))
    theta_out[~all_mask] = theta_grad[~all_mask]
    return theta_out,label_vis


def suppress_singularities_angles(theta: np.ndarray, window: int = 5, var_thresh_deg: float = 75.0) -> np.ndarray:
    """Detect high-variance zones and replace with local circular mean (2-RoSy)."""
    c = np.cos(theta)
    s = np.sin(theta)

    k = window | 1
    kernel = np.ones((k, k), np.float32)
    n = cv2.filter2D(np.ones_like(c), -1, kernel, borderType=cv2.BORDER_REFLECT)

    mc = cv2.filter2D(c, -1, kernel, borderType=cv2.BORDER_REFLECT) / n
    ms = cv2.filter2D(s, -1, kernel, borderType=cv2.BORDER_REFLECT) / n

    R = np.sqrt(mc**2 + ms**2)
    circ_std = np.sqrt(-2.0 * np.log(np.clip(R, 1e-6, 1.0)))
    thr = np.deg2rad(var_thresh_deg) / math.sqrt(2.0)

    unstable = circ_std > thr
    theta_out = theta.copy()
    theta_out[unstable] = np.arctan2(ms[unstable], mc[unstable])
    return angle_normalize_pi(theta_out)


# ==========================
# I/O and visualization
# ==========================

def load_inputs(image_path: str, seg_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow("img_bgr", img_bgr)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = _ensure_gray(img_bgr)
    cv2.imshow("gray", gray)

    if seg_path.lower().endswith('.npy'):
        seg = np.load(seg_path)
    else:
        seg_img = skio.imread(seg_path)
        if seg_img.ndim > 2:
            seg_img = seg_img[..., 0]
        seg = seg_img.astype(np.int32)

    if seg.shape != gray.shape:
        raise ValueError(f"Segmentation shape {seg.shape} does not match image {gray.shape}")

    return img_bgr, gray, seg


def orientation_colormap(theta: np.ndarray) -> np.ndarray:
    """Map theta in [0,pi) to HSV wheel (OpenCV range)."""
    h = (theta / np.pi) * 179.0  # 0..179
    hsv = np.zeros((*theta.shape, 3), np.float32)
    hsv[..., 0] = h.astype(np.float32)
    hsv[..., 1] = 255.0
    hsv[..., 2] = 255.0
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def colorize_alignment_labels(labels: np.ndarray) -> np.ndarray:
    """0=tangent (blue), 1=perpendicular (red), 255=unset (black)."""
    out = np.zeros((labels.shape[0], labels.shape[1], 3), np.uint8)
    # blue for tangent
    out[labels == 0] = (255, 0, 0)
    # red for perpendicular
    out[labels == 1] = (0, 0, 255)
    return out


def visualize_quiver_overlay(
    img_bgr: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    step: int = 12,
    alpha: float = 0.65,
    color=(0, 255, 255),   # BGR
    thickness: int = 1,
    tip_length: float = 0.3,
    max_len: float | None = None,  # pixels, optional clamp
) -> np.ndarray:
    """
    Draw a quiver-style overlay using OpenCV only.

    img_bgr: HxWx3 uint8
    tx, ty: HxW float (or int) vectors (x and y components in pixels)
    step: sampling stride
    alpha: blending factor for overlay
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be HxWx3 BGR image")
    if tx.shape != ty.shape:
        raise ValueError("tx and ty must have the same shape")
    h, w = tx.shape
    if img_bgr.shape[0] != h or img_bgr.shape[1] != w:
        raise ValueError("img_bgr spatial size must match tx/ty")

    # Overlay to draw on
    overlay = img_bgr.copy()

    # Sample grid points
    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)

    # Compute a reasonable scale so arrows are visible:
    # use a robust magnitude statistic from sampled vectors.
    U = tx[np.ix_(ys, xs)].astype(np.float32)
    V = ty[np.ix_(ys, xs)].astype(np.float32)
    mag = np.sqrt(U * U + V * V).ravel()

    # If the field is mostly zeros, just return blended copy (no arrows)
    nonzero = mag[mag > 1e-6]
    if nonzero.size == 0:
        return img_bgr.copy()

    # Target arrow length (in pixels) for a "typical" vector
    # (tweakable, but works well for optical-flow-like fields)
    typical = np.percentile(nonzero, 90)  # robust "large-ish" magnitude
    target_len = 0.8 * step               # arrows roughly comparable to grid spacing
    scale = target_len / (typical + 1e-6)

    # Optional clamp for max arrow length
    if max_len is None:
        max_len = 2.0 * step

    # Draw arrows
    for y in ys:
        for x in xs:
            u = float(tx[y, x]) * scale
            v = float(ty[y, x]) * scale

            # Clamp length to avoid giant arrows
            L = (u * u + v * v) ** 0.5
            if L < 1e-3:
                continue
            if L > max_len:
                s = max_len / L
                u *= s
                v *= s

            x2 = int(round(x + u))
            y2 = int(round(y + v))
            cv2.arrowedLine(
                overlay,
                (x, y),
                (x2, y2),
                color,
                thickness=thickness,
                tipLength=tip_length,
                line_type=cv2.LINE_AA,
            )

    # Blend overlay with original
    out = cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0.0)
    return out


# ==========================
# Main pipeline
# ==========================

def compute_2rosy_field(
    img_gray: np.ndarray,
    seg: np.ndarray,
    grad_method: str = "sobel",
    grad_ksize: int = 3,
    boundary_dist_sigma: float = 6.0,
    boundary_band: float = 3.0,
    smooth_iters: int = 2,
    smooth_sigma: float = 1.5,
    do_singularity_suppress: bool = True,
    singularity_window: int = 5,
    singularity_var_deg: float = 75.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (theta_final, label_vis) where theta is in [0,pi)."""
    
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REPLICATE)
    # 1) interior cue: gradient tangent (angle)
    theta_g, grad_mag, gradients = compute_image_gradient_tangent_angle(img_gray, method=grad_method, ksize=grad_ksize)

    # 2) boundary relaxation and blending
    theta_b, label_vis = blend_field_with_boundary_relaxation(
        seg,
        theta_grad=theta_g,
        grad_mag=grad_mag,
        boundary_dist_sigma=boundary_dist_sigma,
        boundary_band=boundary_band,
    )

    # 3) smoothing by circular averaging within regions
    theta_s = smooth_angles_within_regions(seg, theta_b, iters=smooth_iters, sigma=smooth_sigma)

    # 4) optional singularity suppression (angle version)
    if do_singularity_suppress:
        theta_s = suppress_singularities_angles(theta_s, window=singularity_window, var_thresh_deg=singularity_var_deg)

    theta_s = angle_normalize_pi(theta_s)
    return theta_g, label_vis, gradients


def main(
    image_path: str = "img.png",
    seg_path: str = "segmentation.png",
    out_npy: str = "vector_field_2rosy.npy",
    out_preview: str = "vector_field_2rosy_preview.png",
    out_labels: str = "boundary_alignment_labels.png",
):
    img_bgr, gray, seg = load_inputs(image_path, seg_path)    
    
    theta, label_vis, gradients = compute_2rosy_field(
        gray,
        seg,
        grad_method="sobel",
        grad_ksize=5,
        boundary_dist_sigma=8.0,
        boundary_band=3.0,
        smooth_iters=2,
        smooth_sigma=1.5,
        do_singularity_suppress=True,
        singularity_window=5,
        singularity_var_deg=75.0,
    )

    # Save vector map (H,W,2) using 2-RoSy orientation (cos,sin)
    gx, gy = gradients
    tx = -gy
    ty = gx
    cv2.imshow("tx2", tx)
    cv2.imshow("ty2", ty)
    vec = np.stack([tx, ty], axis=-1).astype(np.float32)
    np.save(out_npy, vec)
    cv2.imshow("theta2", theta / np.pi)
    # Visualization: orientation color + quiver
    orient_bgr = orientation_colormap(theta)
    color_overlay = cv2.addWeighted(img_bgr, 0.0, orient_bgr, 0.5, 0.0)

    quiver_overlay = visualize_quiver_overlay(img_bgr, tx, ty, step=max(8, min(img_bgr.shape[:2]) // 80), alpha=0.7)

    preview = np.vstack([color_overlay, quiver_overlay])
    cv2.imwrite(out_preview, preview)
    
    cv2.waitKey(0)

    if label_vis is not None:
        cv2.imwrite(out_labels, colorize_alignment_labels(label_vis))

    print(f"Saved 2-RoSy field to {out_npy} and preview to {out_preview}")
    if label_vis is not None:
        print(f"Saved boundary alignment labels to {out_labels}")


if __name__ == "__main__":
    main()
