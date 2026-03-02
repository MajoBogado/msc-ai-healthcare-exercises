from __future__ import annotations
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def evaluate_classification_model(
    model_pipeline,
    features_df,
    target_series,
    split_label: str,
) -> None:
    """
    Evaluates a classification model and prints clinically interpretable results.

    split_label examples: "TRAIN", "TEST"
    Positive class: stroke = 1
    """
    predictions = model_pipeline.predict(features_df)

    accuracy = accuracy_score(target_series, predictions)
    precision = precision_score(target_series, predictions, zero_division=0)
    recall = recall_score(target_series, predictions, zero_division=0)
    f1 = f1_score(target_series, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(target_series, predictions).ravel()

    print("\n====================")
    print(f"Evaluation — {split_label}")

    print("\nOverall Metrics:")
    print(f" - Accuracy : {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall   : {recall:.4f}")
    print(f" - F1-score : {f1:.4f}")

    print("\nConfusion Matrix (Positive = stroke = 1):")
    print(f"  TN (correct no-stroke) : {tn}")
    print(f"  FP (false alarm)       : {fp}")
    print(f"  FN (missed stroke)     : {fn}")
    print(f"  TP (correct stroke)    : {tp}")

    print("\nClinical interpretation:")
    print(" - FN (missed stroke) is the most critical error in screening contexts.")
    print(" - Recall = TP / (TP + FN) measures stroke detection rate.")
    print(" - Precision = TP / (TP + FP) measures reliability of stroke predictions.")