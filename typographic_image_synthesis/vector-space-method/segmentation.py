"""Module providing a function printing python version."""
import cv2
import numpy as np
#from sklearn.cluster import KMeans
import skimage.segmentation as seg
from scipy import ndimage as ndi
from skimage import color, graph
from skimage.morphology import disk, closing
from skimage.measure import label, regionprops
import math


HIGH_THRESH = 60
LOW_THRESH = 30
K_INT = 70
GAUSSIAN_BLUR_SIZE = (5, 5)
MORPH_KERNEL_SIZE = (3, 3)       
MORPH_ITER = 2                   
DILATE_ITER = 3                  
DIST_TRANSFORM_THRESH = 0.4      
BOUNDARY_COLOR = [255, 0, 0]     

def hessian_2nd_derivatives(image, sigma):
    """
    Compute second derivatives of image after Gaussian smoothing.
    Returns Ixx, Ixy, Iyy.
    """
    # Gaussian smoothing already included in gaussian_filter via order args
    Ixx = ndi.gaussian_filter(image, sigma=sigma, order=(2,0))
    Iyy = ndi.gaussian_filter(image, sigma=sigma, order=(0,2))
    Ixy = ndi.gaussian_filter(image, sigma=sigma, order=(1,1))
    return Ixx, Ixy, Iyy

def hessian_eig(Ixx, Ixy, Iyy):
    """
    For each pixel compute eigenvalues and eigenvectors of 2x2 Hessian [[Ixx, Ixy],[Ixy, Iyy]]
    Returns lambda1, lambda2 (lambda1 >= lambda2), and the corresponding eigenvector for lambda1 (vx, vy).
    """
    # compute trace and determinant
    tr = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    # eigenvalues analytic solution: (tr +/- sqrt(tr^2 - 4 det))/2
    disc = np.clip(tr*tr - 4*det, 0, None)
    sqrt_disc = np.sqrt(disc)
    l1 = 0.5 * (tr + sqrt_disc)
    l2 = 0.5 * (tr - sqrt_disc)
    # eigenvector for l1: (Ixx - l1, Ixy)  or (Ixy, Iyy - l1) choose stable
    vx = Ixy
    vy = l1 - Ixx
    # when both vx and vy are zero (e.g., flat) fall back to (1,0)
    mag = np.hypot(vx, vy)
    zero_mask = (mag == 0)
    vx[zero_mask] = 1.0
    vy[zero_mask] = 0.0
    mag = np.hypot(vx, vy)
    vx /= mag
    vy /= mag
    return l1, l2, vx, vy

def eigenvector_support(vx, vy):
    """
    Compute average absolute dot product of each pixel's major eigenvector with its 8 neighbors.
    Returns support map in [0,1].
    """
    # pad and compute dot products with neighbors using shifts
    support = np.zeros_like(vx)
    # neighbors offsets
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    count = len(offsets)
    for dy, dx in offsets:
        vx_n = ndi.shift(vx, shift=(dy,dx), order=0, mode='nearest')
        vy_n = ndi.shift(vy, shift=(dy,dx), order=0, mode='nearest')
        dot = np.abs(vx * vx_n + vy * vy_n)  # absolute dot product
        support += dot
    support /= count
    return support

def hysteresis_eigenvector_threshold(Pnorm, vx, vy, high_thresh=0.04, support_threshold=0.9):
    """
    Perform eigenvector-flow guided hysteresis thresholding.
    Pnorm assumed normalized in [0,1].
    Returns binary ridge mask (True for ridge pixels kept).
    """
    high = high_thresh
    support_map = eigenvector_support(vx, vy)
    # low-to-high ratio per pixel
    low_ratio = np.where(support_map >= support_threshold, 0.2, 0.7)
    low = high * low_ratio
    # seeds above high threshold
    seeds = Pnorm >= high
    # candidate mask above low threshold
    candidates = Pnorm >= low
    # label connected components of candidates and keep those that contain at least one seed
    labeled, n = label(candidates, return_num=True)
    keep = np.zeros_like(labeled, dtype=bool)
    for comp in range(1, n+1):
        comp_mask = (labeled == comp)
        if np.any(seeds & comp_mask):
            keep[comp_mask] = True
    return keep

def enhanced_watershed_regions(
    image,
    sigma=2,
    closing_radius=3,
    high_thresh=0.04,
    support_threshold=0.9,
    min_area=30,
):
    """
    Principal Curvature-Based Enhanced Watershed Region Detection (single-scale).

    Parameters
    ----------
    image : 2D np.ndarray
        Grayscale image (float32 or float64, range [0,1]).
    sigma : float
        Gaussian smoothing scale for Hessian derivatives.
    closing_radius : int
        Radius of the morphological closing disk (default: 5).
    high_thresh : float
        High threshold for eigenvector hysteresis (relative to normalized curvature).
    support_threshold : float
        Eigenvector direction consistency threshold (0-1).
    min_area : int
        Minimum pixel area for accepted watershed basins.

    Returns
    -------
    ellipses : list of tuples
        List of detected region ellipses, each as:
        (cx, cy, major_axis, minor_axis, angle_deg)
    ridge_mask : np.ndarray[bool]
        Binary ridge map (True = ridge pixels).
    labeled_basins : np.ndarray[int]
        Integer-labeled segmentation of basins (each region has a unique label).
    """
    #Normalize image
    image = (image - image.min()) / (image.max() - image.min() + 1e-12)
    # --- 1. Hessian second derivatives ---
    Ixx = ndi.gaussian_filter(image, sigma=sigma, order=(2, 0))
    Iyy = ndi.gaussian_filter(image, sigma=sigma, order=(0, 2))
    Ixy = ndi.gaussian_filter(image, sigma=sigma, order=(1, 1))

    # --- 2. Eigen decomposition (Hessian) ---
    tr = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    disc = np.clip(tr * tr - 4 * det, 0, None)
    sqrt_disc = np.sqrt(disc)
    l1 = 0.5 * (tr + sqrt_disc)
    l2 = 0.5 * (tr - sqrt_disc)
    vx, vy = Ixy, l1 - Ixx
    mag = np.hypot(vx, vy)
    vx[mag == 0], vy[mag == 0] = 1.0, 0.0
    mag = np.hypot(vx, vy)
    vx = vx/mag
    vy = vy/mag

    # --- 3. Principal curvature image ---
    P = np.maximum(l1, 0.0)
    Pnorm = P / (P.max() + 1e-12)
    print("P range:", P.min(), P.max(), " P mean:", P.mean())
    # --- 4. Morphological closing ---
    Pclosed = closing(Pnorm, disk(closing_radius))
    # --- 5. Eigenvector-flow hysteresis thresholding ---
    support = np.zeros_like(vx)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in offsets:
        vx_n = ndi.shift(vx, shift=(dy, dx), order=0, mode="nearest")
        vy_n = ndi.shift(vy, shift=(dy, dx), order=0, mode="nearest")
        support += np.abs(vx * vx_n + vy * vy_n)
    support /= len(offsets)

    cv2.imshow('Support Map', (support * 255).astype(np.uint8))
    low_ratio = np.where(support >= support_threshold, 0.2, 0.7)
    low = high_thresh * low_ratio
    seeds = Pclosed >= high_thresh
    candidates = Pclosed >= low
    labeled, n = label(candidates, return_num=True)
    ridge_mask = np.zeros_like(labeled, dtype=bool)
    for i in range(1, n + 1):
        comp = labeled == i
        if np.any(seeds & comp):
            ridge_mask[comp] = True

    # --- 6. Watershed basins (inverse of ridge mask) ---
    basins = ~ridge_mask

    labeled_basins, n_basins = label(basins, return_num=True)

    # --- 7. Fit ellipses (via PCA) to filtered basins ---
    def fit_ellipse(coords):
        if coords.shape[0] < 5:
            return None
        mean = coords.mean(axis=0)
        cov = np.cov(coords - mean, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        axes = 4.0 * np.sqrt(np.clip(evals, 1e-12, None))
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        return (mean[1], mean[0], axes[0], axes[1], angle)

    ellipses = []
    for region in regionprops(labeled_basins):
        if region.area >= min_area:
            ellipse = fit_ellipse(region.coords)
            if ellipse is not None:
                ellipses.append(ellipse)

    return ellipses, ridge_mask, labeled_basins

def adaptivecanny(image):
    """Apply adaptive Canny edge detection."""
    blur = cv2.GaussianBlur(image, (5, 5), 2.5)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    mean = cv2.blur(image.astype(np.float32), (18, 18))
    mean_sq = cv2.blur(image.astype(np.float32)**2, (18, 18))
    variance = mean_sq - mean**2
    std = np.sqrt(np.maximum(variance, 1e-6))

    adaptive_threshold = 1.5 * (std + 1)
    # Strong edges: above HIGH_THRESH
    strong_edges = (grad > HIGH_THRESH).astype(np.uint8) * 255
    cv2.imshow('Strong Edges Before Thinning', strong_edges)
    # strong_edges = cv2.ximgproc.thinning(strong_edges)
    # cv2.imshow('Strong Edges After Thinning', strong_edges)
    
    # Weak edges: above adaptive threshold but below HIGH_THRESH
    weak_edges = ((grad > adaptive_threshold)).astype(np.uint8) * 255
    cv2.imshow('Weak Edges Before Thinning', weak_edges)
    # weak_edges = cv2.ximgproc.thinning(weak_edges)
    # cv2.imshow('Weak Edges After Thinning', weak_edges)

    # Combine using hysteresis: keep weak edges connected to strong edges
    edges = np.zeros_like(strong_edges, dtype=np.uint8)
    edges[strong_edges] = 255

    # Use OpenCV's connectedComponents to find connected weak edges
    num_labels, labels = cv2.connectedComponents(weak_edges, connectivity=8)
    for label in range(1, num_labels):
        mask = (labels == label)
        # If any pixel in this component touches a strong edge, keep the whole component
        if np.any(strong_edges & mask):
            edges[mask] = 255

    # Optional: Thinning
    cv2.imshow('Edges Before Thinning', edges)
    edges = cv2.ximgproc.thinning(edges)
    return edges


def canny_with_fixed_threshold(image):
    """Apply Canny edge detection with fixed thresholds."""
    blur = cv2.GaussianBlur(image, (5, 5), 2.5)
    edges = cv2.Canny(blur, LOW_THRESH, HIGH_THRESH)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return edges

def get_enclosed_regions(edge_img):
    """Find regions enclosed by edges using contours."""
    edge_bin = (edge_img > 0).astype(np.uint8)
    # Invert edge map: 1 for non-edge, 0 for edge
    non_edge = (edge_bin == 0).astype(np.uint8)
    # Label connected non-edge regions
    num_labels, labels = cv2.connectedComponents(non_edge, connectivity=4)
    # labels: 0 is background, 1...num_labels-1 are regions
    return num_labels - 1, labels

def kmeans_segmentation(image, k=4):
    """Segment image using K-means clustering."""
    h, w = image.shape
    image = cv2.GaussianBlur(image, (5, 5), 2.5)
    # Create feature array: [x, y, intensity]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    features = np.stack([X.ravel(), Y.ravel(), image.ravel()], axis=1).astype(np.float32)
    # Optionally scale features (e.g., intensity vs. position)
    features[:, :2] /= max(h, w)  # Normalize spatial coordinates
    features[:, 2] /= 255.0       # Normalize intensity

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    clustered = labels.reshape((h, w))
    return clustered.astype(np.uint8)

# def kmeans_segmentation_scikit(image, k=4):
#     """Segment image using K-means clustering (scikit-learn)."""
#     h, w = image.shape
#     image = cv2.GaussianBlur(image, (5, 5), 2.5)
#     X, Y = np.meshgrid(np.arange(w), np.arange(h))
#     features = np.stack([X.ravel(), Y.ravel(), image.ravel()], axis=1).astype(np.float32)
#     features[:, :2] /= max(h, w)  # Normalize spatial coordinates
#     features[:, 2] /= 255.0       # Normalize intensity

#     kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
#     labels = kmeans.fit_predict(features)
#     clustered = labels.reshape((h, w))
#     return clustered.astype(np.uint8)

def watershed_segmentation(image):
    """
    Segment an image using the Watershed algorithm.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, 0)

    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)

    #sure_bg = cv2.dilate(opening, kernel, iterations=DILATE_ITER)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform,
                               DIST_TRANSFORM_THRESH * dist_transform.max(),
                               255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # so background is not 0
    markers[unknown == 255] = 0  # mark unknown regions as 0

    img_copy = image.copy()
    markers = cv2.watershed(img_copy, markers)

    img_copy[markers == -1] = BOUNDARY_COLOR

    segmented = color.label2rgb(markers, image, kind='avg')

    return (segmented * 255).astype(np.uint8)


if __name__ == "__main__":
    # Load image from file (change 'img.png' to your image path)
    img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread('img.png', cv2.IMREAD_COLOR)

    # Check if image loaded successfully
    if img is None:
        raise FileNotFoundError('Image file not found.')
    
    #show original image
    cv2.imshow('Original Image', color_img)

    # RGB Sobel
    sobelx = cv2.Sobel(color_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(color_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    cv2.imshow('Sobel Magnitude', cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    inverted_sobel = cv2.normalize(255 - sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Inverted Sobel Magnitude', inverted_sobel)

    adaptive_canny_img = adaptivecanny(img)
    #adaptive_canny_img = cv2.morphologyEx(adaptive_canny_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Optionally, apply erosion to thin edges
    #adaptive_canny_img = cv2.morphologyEx(adaptive_canny_img, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))

    # Display the result
    cv2.imshow('Adaptive Canny Image', adaptive_canny_img)
    canny_img = canny_with_fixed_threshold(img)
    cv2.imshow('Fixed Threshold Canny Image', canny_img)

    num_enclosed_adaptive, enclosed_adaptive = get_enclosed_regions(adaptive_canny_img)
    print(f"Enclosed regions (adaptive): {num_enclosed_adaptive}")

    # Find enclosed regions for fixed-threshold Canny
    num_enclosed_fixed, enclosed_fixed = get_enclosed_regions(canny_img)
    print(f"Enclosed regions (fixed): {num_enclosed_fixed}")

    # Visualize enclosed regions
    adaptive_enclosed_color = color.label2rgb(enclosed_adaptive, bg_label=0, kind='overlay')
    fixed_enclosed_color = cv2.applyColorMap((enclosed_fixed * round(256/num_enclosed_fixed) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('Adaptive Enclosed Regions', adaptive_enclosed_color)
    cv2.imshow('Fixed Enclosed Regions', fixed_enclosed_color)
    
    kmeans_img = kmeans_segmentation(img, k=K_INT)
    kmeans_img_color = cv2.applyColorMap((kmeans_img * round(256/K_INT) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('K-means Segmentation', kmeans_img_color)
    
    # kmeans_img_scikit = kmeans_segmentation_scikit(img, k=K_INT)
    # kmeans_img_scikit_color = cv2.applyColorMap((kmeans_img_scikit * round(256/K_INT) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    # cv2.imshow('K-means Segmentation (scikit)', kmeans_img_scikit_color)

    segments = seg.slic(color_img, n_segments=200, compactness=30, start_label=0)
    segments_color = cv2.applyColorMap((segments * round(256/200) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('SLIC Segmentation', segments_color)

    felzenswalb = seg.felzenszwalb(color_img, scale=100, sigma=0.5, min_size=50)
    felzenswalb_color = cv2.applyColorMap((felzenswalb * round(256/200) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('Felzenszwalb Segmentation', felzenswalb_color)
    
    # Mean shift
    mean_shift = seg.quickshift(color_img, kernel_size=12, max_dist=30, ratio=0.7)
    mean_shift_color = color.label2rgb(mean_shift, color_img, kind='avg')
    cv2.imshow('Mean Shift Segmentation', mean_shift_color)

    inverted_sobel_mean_shift = seg.quickshift(inverted_sobel, kernel_size=12, max_dist=30, ratio=0.7,)
    inverted_sobel_mean_shift_color = color.label2rgb(inverted_sobel_mean_shift, color_img, kind='avg')
    cv2.imshow('Mean Shift inv Segmentation', inverted_sobel_mean_shift_color)
    
    watershed = watershed_segmentation(color_img)
    watershed_coloured = cv2.applyColorMap((cv2.cvtColor(watershed, cv2.COLOR_BGR2GRAY) * round(256/200) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('Watershed Segmentation', watershed_coloured)
    
    ellipses, ridges, watershed_basins = enhanced_watershed_regions(img)
    enhanced_watershed = cv2.applyColorMap((watershed_basins * round(256/np.max(watershed_basins)) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('Enhanced Watershed Segmentation', enhanced_watershed)

    # Graph-based segmentation
    rag = graph.rag_mean_color(color_img, segments, mode='similarity', sigma=20)
    # Number of segments 
    num_segments = len(np.unique(segments))
    print(f"Number of segments (SLIC): {num_segments}")
    if num_segments > 5:
        labels_ncut = graph.cut_normalized(segments, rag, num_cuts=num_segments//5, thresh=0.00000000001)
        labels_ncut_color = cv2.applyColorMap((labels_ncut * round(256/np.max(labels_ncut)) % 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        cv2.imshow('Graph-based Segmentation', labels_ncut.astype(np.uint8)*255)
        cv2.imwrite('segmentation.png', labels_ncut.astype(np.uint8)*255)
    else:
        #cv2.imwrite('segmentation.png', segments.astype(np.uint8)*255)
        cv2.imwrite('segmentation.png', np.zeros_like(segments, dtype=np.uint8))
    cv2.imwrite('segmentation.png', np.zeros_like(img, dtype=np.uint8))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
