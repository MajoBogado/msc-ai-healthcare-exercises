from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

def _compute_confusion_counts(
    true_labels: pd.Series,
    predicted_labels: np.ndarray,
) -> Tuple[int, int, int, int]:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()
    return int(true_negatives), int(false_positives), int(false_negatives), int(true_positives)

def evaluate_thresholds(
    true_labels: pd.Series,
    predicted_probabilities: np.ndarray,
    thresholds: List[float],
) -> List[ThresholdResult]:
    results: List[ThresholdResult] = []

    for threshold in thresholds:
        predicted_labels = (predicted_probabilities >= threshold).astype(int)

        precision = float(precision_score(true_labels, predicted_labels, zero_division=0))
        recall = float(recall_score(true_labels, predicted_labels, zero_division=0))
        f1 = float(f1_score(true_labels, predicted_labels, zero_division=0))

        tn, fp, fn, tp = _compute_confusion_counts(true_labels, predicted_labels)

        results.append(
            ThresholdResult(
                threshold=float(threshold),
                precision=precision,
                recall=recall,
                f1=f1,
                true_negatives=tn,
                false_positives=fp,
                false_negatives=fn,
                true_positives=tp,
            )
        )
    return results

def select_threshold_by_target_recall(
    threshold_results: List[ThresholdResult],
    target_recall: float,
) -> Optional[ThresholdResult]:
    """
    Returns the threshold result with recall >= target_recall that has the highest precision.
    If none meet the target recall, returns None.
    """
    candidates = [result for result in threshold_results if result.recall >= target_recall]
    if not candidates:
        return None
    return max(candidates, key=lambda result: result.precision)

def select_threshold_by_best_f1(
    threshold_results: List[ThresholdResult],
) -> ThresholdResult:
    return max(threshold_results, key=lambda result: result.f1)

def print_threshold_report(
    threshold_results: List[ThresholdResult],
    highlighted_threshold: Optional[float] = None,
) -> None:

    print("threshold | precision | recall | f1    | TP | FN | FP | TN")
    print("----------------------------------------------------------")

    for result in threshold_results:
        marker = ""
        if highlighted_threshold is not None and abs(result.threshold - highlighted_threshold) < 1e-12:
            marker = "  <— selected"

        print(
            f"{result.threshold:>9.2f} |"
            f" {result.precision:>9.4f} |"
            f" {result.recall:>6.4f} |"
            f" {result.f1:>5.4f} |"
            f" {result.true_positives:>2} |"
            f" {result.false_negatives:>2} |"
            f" {result.false_positives:>2} |"
            f" {result.true_negatives:>2}"
            f"{marker}"
        )

def get_positive_class_probabilities(
    model_pipeline,
    clinical_features_dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Returns P(stroke=1). Works only for classifiers that implement predict_proba.
    """
    if not hasattr(model_pipeline, "predict_proba"):
        raise ValueError("This model does not support predict_proba, cannot do threshold tuning.")

    probabilities = model_pipeline.predict_proba(clinical_features_dataframe)
    return probabilities[:, 1]