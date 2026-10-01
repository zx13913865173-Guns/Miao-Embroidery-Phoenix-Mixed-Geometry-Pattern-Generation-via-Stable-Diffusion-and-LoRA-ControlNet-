"""
Evaluation metrics:
- Length-weighted outer-frame MAE
- Discrete Fréchet distance
- Chamfer distance
- CIEDE2000 color difference
"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab, deltaE_ciede2000


# ---------- Length-weighted MAE ----------

def fit_line_angle(points):
    """Fit a line to points and return its angle in degrees."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.degrees(np.arctan2(vy, vx))


def length_weighted_mae(gen_frame_lines, gt_frame_lines):
    """
    Compute length-weighted MAE over 4 borders.

    Args:
        gen_frame_lines: list of 4 arrays, each Nx2 points (generated)
        gt_frame_lines:  list of 4 arrays, each Nx2 points (ground truth)

    Returns:
        mae_deg: scalar
    """
    assert len(gen_frame_lines) == 4
    assert len(gt_frame_lines) == 4
    lengths = []
    errors = []
    for g, t in zip(gen_frame_lines, gt_frame_lines):
        g = np.asarray(g, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        length = np.linalg.norm(g.max(axis=0) - g.min(axis=0))
        ang_g = fit_line_angle(g)
        ang_t = fit_line_angle(t)
        diff = abs(ang_g - ang_t)
        if diff > 90:
            diff = 180 - diff
        lengths.append(length)
        errors.append(diff)
    lengths = np.array(lengths)
    errors = np.array(errors)
    return float(np.sum(lengths * errors) / np.sum(lengths))


# ---------- Discrete Fréchet distance ----------

def discrete_frechet(P, Q):
    """Classical discrete Fréchet distance between two point sets."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(P[i] - Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(P[0] - Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            d = np.linalg.norm(P[i] - Q[j])
            ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)
    return float(ca[n-1, m-1])


def normalized_frechet(P, Q, img_diag):
    return discrete_frechet(P, Q) / img_diag


# ---------- Chamfer distance ----------

def chamfer_distance(P, Q, img_diag):
    """
    Normalized bidirectional Chamfer distance.
    d = (1/|P| * sum min ||p-q|| + 1/|Q| * sum min ||q-p||) / D
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    D = cdist(P, Q)
    d_pq = D.min(axis=1).mean()
    d_qp = D.min(axis=0).mean()
    return float((d_pq + d_qp) / img_diag)


# ---------- CIEDE2000 ----------

def ciede2000(gen_rgb, gt_rgb):
    """
    Mean CIEDE2000 color difference between two RGB images (0-255).
    """
    gen = np.asarray(gen_rgb, dtype=np.float64) / 255.0
    gt = np.asarray(gt_rgb, dtype=np.float64) / 255.0
    lab_gen = rgb2lab(gen)
    lab_gt = rgb2lab(gt)
    delta = deltaE_ciede2000(lab_gen, lab_gt)
    return float(np.mean(delta))


# ---------- Skeleton extraction ----------

def extract_interior_skeleton(edge_map, outer_mask, eps=1.5):
    """
    Extract interior curve skeleton from a Canny edge map.

    Args:
        edge_map: binary edge image
        outer_mask: binary mask (1=interior)
        eps: Douglas-Peucker epsilon

    Returns:
        ordered point set Nx2
    """
    from skimage.morphology import skeletonize
    edges = (edge_map > 0).astype(np.uint8)
    edges = edges * outer_mask
    skel = skeletonize(edges > 0)
    pts = np.column_stack(np.nonzero(skel))  # (row, col)
    if len(pts) < 2:
        return pts.astype(np.float64)
    # Order points along the curve via nearest-neighbor walk
    ordered = [pts[0]]
    remaining = pts[1:].tolist()
    while remaining:
        last = ordered[-1]
        dists = [np.linalg.norm(last - p) for p in remaining]
        idx = int(np.argmin(dists))
        ordered.append(remaining.pop(idx))
    ordered = np.array(ordered, dtype=np.float32)
    ordered = cv2.approxPolyDP(ordered.reshape(-1, 1, 2), eps, False)
    return ordered.reshape(-1, 2).astype(np.float64)
