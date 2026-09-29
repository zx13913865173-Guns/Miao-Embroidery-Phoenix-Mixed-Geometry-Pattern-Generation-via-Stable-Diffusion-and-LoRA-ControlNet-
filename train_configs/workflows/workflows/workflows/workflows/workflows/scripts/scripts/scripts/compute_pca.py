"""
Principal Component Analysis for intrinsic dimensionality estimation.
This script analyzes the low-rank structure of Miao embroidery phoenix patterns.
Usage: python compute_pca.py --image_dir /dataset/train/ --n_bootstrap 100
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import os
import argparse

def compute_pca_dimensionality(image_dir, variance_threshold=0.90, n_bootstrap=100, image_size=(128, 128)):
    """
    Compute the intrinsic dimensionality of a dataset via PCA with bootstrapping.
    Parameters:
        image_dir: Path to directory containing training images.
        variance_threshold: Fraction of variance to retain (default 0.90).
        n_bootstrap: Number of bootstrap resamples for confidence interval.
        image_size: Target size for image resizing (height, width).
    Returns:
        mean_n_components: Mean number of PCs required.
        std_n_components: Standard deviation across bootstrap samples.
    """
    images = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                continue
            img = cv2.resize(img, image_size)
            images.append(img.flatten())
    if len(images) == 0:
        raise ValueError(f"No valid images found in {image_dir}")
    X = np.array(images)
    print(f"Loaded {len(images)} images, feature dimension: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components_list = []
    for i in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X_scaled[idx]
        pca = PCA().fit(X_boot)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_comp = np.argmax(cumsum >= variance_threshold) + 1
        n_components_list.append(n_comp)

    mean_n = np.mean(n_components_list)
    std_n = np.std(n_components_list)
    print(f"\nIntrinsic dimensionality (retaining {variance_threshold*100:.0f}% variance):")
    print(f"  Mean: {mean_n:.1f} ± {std_n:.1f} components")
    print(f"  95% CI: [{np.percentile(n_components_list, 2.5):.1f}, {np.percentile(n_components_list, 97.5):.1f}]")
    return mean_n, std_n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PCA intrinsic dimensionality for Miao embroidery patterns")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--variance_threshold", type=float, default=0.90, help="Fraction of variance to retain")
    parser.add_argument("--n_bootstrap", type=int, default=100, help="Number of bootstrap iterations")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="Target image size (height width)")
    args = parser.parse_args()
    compute_pca_dimensionality(
        args.image_dir,
        variance_threshold=args.variance_threshold,
        n_bootstrap=args.n_bootstrap,
        image_size=tuple(args.image_size)
    )
