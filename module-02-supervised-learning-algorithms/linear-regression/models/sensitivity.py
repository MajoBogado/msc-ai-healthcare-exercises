import numpy as np

def compute_sensitivity_groups(feature_names, coefficients, high_q=0.75, low_q=0.25):
    """
    Creates sensitivity groups (high/medium/low) based on coefficient magnitudes.

    Rule:
      - abs(coef) >= 75th percentile -> high
      - abs(coef) <= 25th percentile -> low
      - otherwise -> medium
    """
    abs_coefs = np.abs(coefficients)
    high_threshold = np.quantile(abs_coefs, high_q)
    low_threshold = np.quantile(abs_coefs, low_q)

    groups = {}
    for name, c in zip(feature_names, abs_coefs):
        if c >= high_threshold:
            groups[name] = "high"
        elif c <= low_threshold:
            groups[name] = "low"
        else:
            groups[name] = "medium"

    return groups, low_threshold, high_threshold
