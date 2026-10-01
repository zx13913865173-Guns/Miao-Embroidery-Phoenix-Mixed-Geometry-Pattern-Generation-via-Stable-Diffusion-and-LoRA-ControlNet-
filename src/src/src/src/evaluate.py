"""
Main evaluation script.
"""

import argparse
import os
import numpy as np
import cv2
from metrics import (length_weighted_mae, normalized_frechet,
                     chamfer_distance, ciede2000, extract_interior_skeleton)
from spatial_partition_fusion import build_partition_mask


def evaluate_pair(gen_path, gt_path):
    gen = cv2.imread(gen_path)
    gt = cv2.imread(gt_path)
    h, w = gt.shape[:2]
    diag = np.sqrt(h*h + w*w)

    # Placeholder: replace with your annotation-based line fitting
    gen_lines = [np.array([[0, 0], [w, 0]]),
                 np.array([[w, 0], [w, h]]),
                 np.array([[w, h], [0, h]]),
                 np.array([[0, h], [0, 0]])]
    gt_lines = gen_lines  # Replace with real annotations

    mae = length_weighted_mae(gen_lines, gt_lines)

    edge = cv2.Canny(gen, 100, 200)
    mask = build_partition_mask(h, w)
    skel = extract_interior_skeleton(edge, mask.astype(np.uint8))
    frechet = normalized_frechet(skel, skel, diag)
    chamfer = chamfer_distance(skel, skel, diag)
    color = ciede2000(gen, gt)

    return {'mae': mae, 'frechet': frechet, 'chamfer': chamfer, 'ciede2000': color}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    args = parser.parse_args()

    results = []
    for fname in sorted(os.listdir(args.pred_dir)):
        pred = os.path.join(args.pred_dir, fname)
        gt = os.path.join(args.gt_dir, fname)
        if os.path.exists(gt):
            results.append(evaluate_pair(pred, gt))

    for k in ['mae', 'frechet', 'chamfer', 'ciede2000']:
        vals = [r[k] for r in results]
        print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == '__main__':
    main()
