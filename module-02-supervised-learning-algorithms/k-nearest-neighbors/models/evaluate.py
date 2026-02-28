from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def compute_classification_metrics(
    true_target_heart_disease: pd.Series,
    predicted_target_heart_disease: pd.Series,
) -> ClassificationMetrics:
    """
    Computes core classification metrics and confusion matrix components.

    Positive class:
      1 = heart disease present

    Confusion matrix layout (sklearn default):
      [[TN, FP],
       [FN, TP]]
    """
    tn, fp, fn, tp = confusion_matrix(
        true_target_heart_disease,
        predicted_target_heart_disease,
        labels=[0, 1],
    ).ravel()

    return ClassificationMetrics(
        accuracy=float(accuracy_score(true_target_heart_disease, predicted_target_heart_disease)),
        precision=float(precision_score(true_target_heart_disease, predicted_target_heart_disease, zero_division=0)),
        recall=float(recall_score(true_target_heart_disease, predicted_target_heart_disease, zero_division=0)),
        f1=float(f1_score(true_target_heart_disease, predicted_target_heart_disease, zero_division=0)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )

def display_evaluation_report(
    *,
    model_name: str,
    metrics: ClassificationMetrics,
) -> None:
    """
    Prints a clean evaluation report without pandas metadata.
    """
    print(f"\n=== Evaluation Report: {model_name} ===")

    print("\nConfusion Matrix (Positive = disease=1):")
    print(f"  TN (correct no-disease): {metrics.true_negatives}")
    print(f"  FP (false alarm):        {metrics.false_positives}")
    print(f"  FN (missed disease):     {metrics.false_negatives}")
    print(f"  TP (correct disease):    {metrics.true_positives}")

    print("\nMetrics:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-score:  {metrics.f1:.4f}")

    print("\nClinical note:")
    print("  - Recall (sensitivity) tells us how many true disease cases we correctly detect.")
    print("  - False Negatives (FN) are especially risky in screening settings.")