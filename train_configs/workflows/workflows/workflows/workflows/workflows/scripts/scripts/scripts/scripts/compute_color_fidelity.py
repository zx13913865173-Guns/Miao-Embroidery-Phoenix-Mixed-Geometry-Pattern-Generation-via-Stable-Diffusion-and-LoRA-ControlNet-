"""
CIEDE2000 color difference computation for color fidelity assessment.
This script computes the color difference between generated and reference images in the CIELAB color space.
Usage: python compute_color_fidelity.py --generated_dir /outputs/group_f/ --reference_dir /dataset/test/
"""
import numpy as np
import cv2
import os
import argparse
from skimage.color import rgb2lab, deltaE_ciede2000

def compute_ciede2000(img1, img2):
    """Compute mean CIEDE2000 color difference between two RGB images."""
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    delta_e = deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)

def main():
    parser = argparse.ArgumentParser(description="Compute CIEDE2000 color fidelity")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing generated images")
    parser.add_argument("--reference_dir", type=str, required=True, help="Directory containing reference images")
    parser.add_argument("--output", type=str, default='color_fidelity.csv', help="Output CSV file")
    args = parser.parse_args()

    gen_files = sorted([f for f in os.listdir(args.generated_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    ref_files = sorted([f for f in os.listdir(args.reference_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(gen_files) != len(ref_files):
        print(f"Warning: Number of generated images ({len(gen_files)}) does not match reference images ({len(ref_files)}).")

    results = []
    for gen_name, ref_name in zip(gen_files, ref_files):
        gen_path = os.path.join(args.generated_dir, gen_name)
        ref_path = os.path.join(args.reference_dir, ref_name)
        gen_img = cv2.imread(gen_path)
        ref_img = cv2.imread(ref_path)
        if gen_img is None or ref_img is None:
            print(f"Warning: Could not read {gen_name} or {ref_name}, skipping.")
            continue
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        # Resize reference to match generated if needed
        if gen_img.shape != ref_img.shape:
            ref_img = cv2.resize(ref_img, (gen_img.shape[1], gen_img.shape[0]))
        delta_e = compute_ciede2000(gen_img, ref_img)
        results.append({'image': gen_name, 'ciede2000': delta_e})
        print(f"{gen_name}: CIEDE2000 = {delta_e:.4f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nMean CIEDE2000: {df['ciede2000'].mean():.4f} ± {df['ciede2000'].std():.4f}")

if __name__ == '__main__':
    main()
