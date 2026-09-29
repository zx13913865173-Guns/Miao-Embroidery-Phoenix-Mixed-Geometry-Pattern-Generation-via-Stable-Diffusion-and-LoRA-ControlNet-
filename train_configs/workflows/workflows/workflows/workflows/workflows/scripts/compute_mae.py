"""
Mean Angular Error (MAE) computation for outer-frame line precision.
This script evaluates the angular accuracy of straight-line borders in generated Miao embroidery phoenix patterns.
Usage: python compute_mae.py --generated_dir /outputs/group_f/ --ground_truth /annotations/outer_frame.json
"""
import numpy as np
import cv2
import json
import os
import argparse
from pathlib import Path

def fit_line_ransac(points, n_iterations=1000, inlier_threshold=1.0):
    """Fit a straight line to a set of points using RANSAC."""
    if len(points) < 2:
        return None, None
    best_inliers = []
    best_line = None
    for _ in range(n_iterations):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        if np.linalg.norm(p2 - p1) < 1e-6:
            continue
        direction = p2 - p1
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        distances = np.abs(np.dot(points - p1, normal))
        inliers = points[distances < inlier_threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (p1, direction)
    return best_line, len(best_inliers)

def detect_border_lines(edge_map, min_line_length=100, max_line_gap=20):
    """Detect straight border lines from an edge map using HoughLinesP."""
    lines = cv2.HoughLinesP(edge_map, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None or len(lines) < 4:
        return None
    detected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        detected_lines.append((np.array([x1, y1]), np.array([x2, y2])))
    return detected_lines

def compute_outer_frame_mae(generated_image, ground_truth_endpoints):
    """
    Compute Mean Angular Error (MAE) for outer-frame line precision.
    Parameters:
        generated_image: Input image as numpy array (H, W) or (H, W, C).
        ground_truth_endpoints: List of four (x1, y1, x2, y2) tuples representing the ground truth border lines.
    Returns:
        mae_value: Mean Angular Error in degrees. Lower = higher precision.
    """
    if generated_image.ndim == 3:
        gray = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = generated_image
    edges = cv2.Canny(gray, 50, 150)
    detected_lines = detect_border_lines(edges)
    if detected_lines is None:
        return float("inf")
    angular_errors = []
    for gt_line in ground_truth_endpoints:
        x1_gt, y1_gt, x2_gt, y2_gt = gt_line
        gt_angle = np.degrees(np.arctan2(y2_gt - y1_gt, x2_gt - x1_gt)) % 180
        best_error = float("inf")
        for det_line in detected_lines:
            p1, p2 = det_line
            det_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
            error = abs(det_angle - gt_angle)
            error = min(error, 180 - error)
            best_error = min(best_error, error)
        angular_errors.append(best_error)
    return np.mean(angular_errors)

def main():
    parser = argparse.ArgumentParser(description="Compute MAE for Miao embroidery phoenix outer frames")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing generated images")
    parser.add_argument("--ground_truth", type=str, required=True, help="JSON file with ground truth border endpoints")
    parser.add_argument("--output", type=str, default='mae_results.csv', help="Output CSV file for MAE results")
    args = parser.parse_args()

    with open(args.ground_truth, 'r') as f:
        gt_data = json.load(f)

    results = []
    for img_name, gt_endpoints in gt_data.items():
        img_path = os.path.join(args.generated_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mae = compute_outer_frame_mae(image, gt_endpoints)
        results.append({'image': img_name, 'mae_degrees': mae})
        print(f"{img_name}: MAE = {mae:.3f}°")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nMean MAE: {df['mae_degrees'].mean():.3f}° ± {df['mae_degrees'].std():.3f}°")

if __name__ == '__main__':
    main()
