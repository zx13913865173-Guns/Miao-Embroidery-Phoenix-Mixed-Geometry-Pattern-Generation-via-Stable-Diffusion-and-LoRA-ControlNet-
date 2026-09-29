"""
Discrete Fréchet distance computation for interior curve fidelity.
This script evaluates the curve similarity between generated interior curves and ground truth curves in Miao embroidery phoenix patterns.
Usage: python compute_frechet.py --generated_dir /outputs/group_f/ --ground_truth /annotations/interior_curves.json
"""
import numpy as np
from scipy.spatial.distance import cdist
import json
import os
import argparse
import cv2

def compute_discrete_frechet(curve_p, curve_q):
    """
    Compute the discrete Fréchet distance between two polygonal curves.
    Implementation follows the dynamic programming algorithm described in:
        Eiter T, Mannila H. Computing discrete Fréchet distance. 1994.
        Fast implementation based on Bringmann et al. (2021).
    Parameters:
        curve_p: Point set of the first curve, shape (N, 2).
        curve_q: Point set of the second curve, shape (M, 2).
    Returns:
        frechet_distance: The discrete Fréchet distance value.
    """
    n, m = len(curve_p), len(curve_q)
    if n == 0 or m == 0:
        return float('inf')
    dist_matrix = cdist(curve_p, curve_q, metric='euclidean')
    dp = np.full((n, m), np.inf)
    dp[0, 0] = dist_matrix[0, 0]
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(min(dp[i-1, j], dp[i-1, j-1], dp[i, j-1]), dist_matrix[i, j])
    return dp[n-1, m-1]

def simplify_curve(curve_points, epsilon=1.5):
    """Simplify a curve using the Douglas-Peucker algorithm."""
    if len(curve_points) < 3:
        return curve_points
    curve_points = np.array(curve_points)
    start = curve_points[0]
    end = curve_points[-1]
    max_dist = 0
    max_index = 0
    for i in range(1, len(curve_points) - 1):
        point = curve_points[i]
        line_vec = end - start
        point_vec = point - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            distance = np.linalg.norm(point_vec)
        else:
            distance = np.abs(np.cross(line_vec, point_vec)) / line_len
        if distance > max_dist:
            max_dist = distance
            max_index = i
    if max_dist > epsilon:
        left = simplify_curve(curve_points[:max_index+1], epsilon)
        right = simplify_curve(curve_points[max_index:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])

def extract_curve_from_image(image, curve_region_mask=None):
    """Extract the interior curve from a generated image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if curve_region_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=curve_region_mask)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
    longest_contour = max(contours, key=cv2.contourArea)
    curve_points = longest_contour.squeeze(1)
    if curve_points.ndim == 1:
        curve_points = curve_points.reshape(-1, 2)
    return curve_points.astype(np.float64)

def main():
    parser = argparse.ArgumentParser(description="Compute Fréchet distance for Miao embroidery phoenix interior curves")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing generated images")
    parser.add_argument("--ground_truth", type=str, required=True, help="JSON file with ground truth curve points")
    parser.add_argument("--epsilon", type=float, default=1.5, help="Douglas-Peucker simplification epsilon")
    parser.add_argument("--output", type=str, default='frechet_results.csv', help="Output CSV file for Fréchet distance results")
    args = parser.parse_args()

    with open(args.ground_truth, 'r') as f:
        gt_data = json.load(f)

    results = []
    for img_name, gt_curve_array in gt_data.items():
        img_path = os.path.join(args.generated_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gen_curve = extract_curve_from_image(image)
        if len(gen_curve) == 0:
            print(f"Warning: No curve found in {img_name}, skipping.")
            continue
        gen_curve_simple = simplify_curve(gen_curve, epsilon=args.epsilon)
        gt_curve = np.array(gt_curve_array)
        gt_curve_simple = simplify_curve(gt_curve, epsilon=args.epsilon)
        frechet_dist = compute_discrete_frechet(gen_curve_simple, gt_curve_simple)
        h, w = image.shape[:2]
        diagonal = np.sqrt(h**2 + w**2)
        normalized_frechet = frechet_dist / diagonal
        results.append({
            'image': img_name,
            'frechet_distance': frechet_dist,
            'normalized_frechet': normalized_frechet
        })
        print(f"{img_name}: Fréchet = {frechet_dist:.4f} (normalized: {normalized_frechet:.4f})")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nMean normalized Fréchet: {df['normalized_frechet'].mean():.4f} ± {df['normalized_frechet'].std():.4f}")

if __name__ == '__main__':
    main()
