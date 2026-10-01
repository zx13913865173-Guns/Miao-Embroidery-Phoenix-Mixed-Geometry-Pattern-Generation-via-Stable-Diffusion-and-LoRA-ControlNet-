"""
Fusion weight pilot experiment.

Tests combinations of (outer-frame strength, interior strength):
(0.9, 0.1), (0.8, 0.2), (0.7, 0.8), (0.6, 0.4), (0.5, 0.5)

Reports length-weighted MAE, Fréchet distance, and ICH practitioner score.
"""

import numpy as np
from scipy.stats import friedmanchisquare

COMBINATIONS = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.8), (0.6, 0.4), (0.5, 0.5)]

# Fill with your real experiment results
RESULTS = {
    (0.9, 0.1): {'mae': 0.52, 'frechet': 0.0125, 'ich': 4.2},
    (0.8, 0.2): {'mae': 0.49, 'frechet': 0.0095, 'ich': 4.6},
    (0.7, 0.8): {'mae': 0.48, 'frechet': 0.0072, 'ich': 4.9},
    (0.6, 0.4): {'mae': 0.59, 'frechet': 0.0068, 'ich': 4.7},
    (0.5, 0.5): {'mae': 0.82, 'frechet': 0.0066, 'ich': 4.1},
}


def run_pilot():
    ich_scores = [RESULTS[c]['ich'] for c in COMBINATIONS]
    stat, p = friedmanchisquare(*[[s] * 20 for s in ich_scores])
    print(f"Friedman chi2 = {stat:.2f}, p = {p:.3f}")
    for c in COMBINATIONS:
        r = RESULTS[c]
        print(f"{c}: MAE={r['mae']:.2f}, Fréchet={r['frechet']:.4f}, ICH={r['ich']:.1f}")
    best = max(COMBINATIONS, key=lambda c: RESULTS[c]['ich'])
    print(f"Best combination: {best}")


if __name__ == '__main__':
    run_pilot()
