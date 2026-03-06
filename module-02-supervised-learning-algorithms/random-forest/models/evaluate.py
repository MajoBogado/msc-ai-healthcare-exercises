from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

def _safe_roc_auc(
    true_labels: pd.Series,
    predicted_probabilities: np.ndarray,
) -> float:
    """
    ROC-AUC requires both classes to be present in y_true.
    If not present (rare in tiny splits), return NaN.
    """
    unique_values = set(int(value) for value in pd.Series(true_labels).unique())
    if unique_values != {0, 1}:
        return float("nan")
    return float(roc_auc_score(true_labels, predicted_probabilities))

def evaluate_binary_classifier(
    model_pipeline,
    clinical_features_dataframe: pd.DataFrame,
    target_stroke_series: pd.Series,
    label: str
) -> Tuple[EvaluationMetrics, Dict[str, int]]:
    predicted_labels = model_pipeline.predict(clinical_features_dataframe)

    if hasattr(model_pipeline, "predict_proba"):
        predicted_probabilities = model_pipeline.predict_proba(clinical_features_dataframe)[:, 1]
    else:
        predicted_probabilities = None

    accuracy = float(accuracy_score(target_stroke_series, predicted_labels))
    precision = float(precision_score(target_stroke_series, predicted_labels, zero_division=0))
    recall = float(recall_score(target_stroke_series, predicted_labels, zero_division=0))
    f1 = float(f1_score(target_stroke_series, predicted_labels, zero_division=0))

    roc_auc = float("nan")
    if predicted_probabilities is not None:
        roc_auc = _safe_roc_auc(target_stroke_series, predicted_probabilities)

    confusion = confusion_matrix(target_stroke_series, predicted_labels, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = confusion.ravel()

    print("metric     | value")
    print("-------------------")
    print(f"accuracy   | {accuracy:.4f}")
    print(f"precision  | {precision:.4f}")
    print(f"recall     | {recall:.4f}")
    print(f"f1         | {f1:.4f}")
    if np.isnan(roc_auc):
        print("roc_auc    | n/a")
    else:
        print(f"roc_auc    | {roc_auc:.4f}")

    print("\nConfusion Matrix (Positive = stroke = 1):")
    print(f"  TN (correct no-stroke) : {true_negatives}")
    print(f"  FP (false alarm)       : {false_positives}")
    print(f"  FN (missed stroke)     : {false_negatives}")
    print(f"  TP (correct stroke)    : {true_positives}")

    print("\nClinical interpretation:")
    print(" - FN (missed stroke) is the most critical error in screening contexts.")
    print(" - Recall = TP / (TP + FN) measures stroke detection rate.")
    print(" - Precision = TP / (TP + FP) measures reliability of stroke predictions.\n")

    metrics = EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
    )

    confusion_counts = {
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }

    return metrics, confusion_counts