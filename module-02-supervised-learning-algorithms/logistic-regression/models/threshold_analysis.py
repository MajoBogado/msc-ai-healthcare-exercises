import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def evaluate_with_custom_threshold(
    trained_model,
    clinical_features_test_scaled,
    diagnosis_labels_test,
    diagnosis_categories,
    threshold: float,
):
    """
    Applies a custom threshold on P(benign) (class 1) to generate predictions,
    then prints clinically meaningful metrics + confusion matrix.

    NOTE:
    - This dataset uses: 0=malignant, 1=benign
    - We will report Precision/F1/Recall specifically for MALIGNANT detection (pos_label=0),
      because that’s the clinically critical class.
    """

    # Probabilities for each class:
    # [:,0] = P(malignant), [:,1] = P(benign)
    probability_benign = trained_model.predict_proba(clinical_features_test_scaled)[:, 1]
    probability_malignant = 1.0 - probability_benign

    # Custom decision rule (your current approach):
    # if P(benign) >= threshold => predict benign (1), else malignant (0)
    threshold_predictions = (probability_benign >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(diagnosis_labels_test, threshold_predictions)

    # Clinically focused (malignant detection as "positive" class)
    precision_malignant = precision_score(
        diagnosis_labels_test, threshold_predictions, pos_label=0, zero_division=0
    )
    recall_malignant = recall_score(
        diagnosis_labels_test, threshold_predictions, pos_label=0, zero_division=0
    )
    recall_benign = recall_score(
        diagnosis_labels_test, threshold_predictions, pos_label=1, zero_division=0
    )
    f1_malignant = f1_score(
        diagnosis_labels_test, threshold_predictions, pos_label=0, zero_division=0
    )

    # ROC-AUC should be computed using probability of the clinically important class (malignant)
    roc_auc_malignant = roc_auc_score(
        (diagnosis_labels_test == 0).astype(int),  # malignant as 1 for AUC computation
        probability_malignant
    )

    # Confusion matrix: rows = actual, cols = predicted (order: 0 then 1)
    cm = confusion_matrix(diagnosis_labels_test, threshold_predictions, labels=[0, 1])

    tn = cm[1][1]  # actual benign (1) predicted benign (1)
    fp = cm[1][0]  # actual benign (1) predicted malignant (0)
    fn = cm[0][1]  # actual malignant (0) predicted benign (1)
    tp = cm[0][0]  # actual malignant (0) predicted malignant (0)

    # Specificity (benign correctly identified) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"Evaluation with threshold = {threshold}\n")

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision (Malignant detection): {precision_malignant:.3f}")
    print(f"Recall (Malignant detection / Sensitivity): {recall_malignant:.3f}")
    print(f"Recall (Benign detection): {recall_benign:.3f}")
    print(f"F1 Score (Malignant detection): {f1_malignant:.3f}")
    print(f"ROC-AUC (Malignant): {roc_auc_malignant:.3f}")
    print(f"Specificity (Benign correctly identified): {specificity:.3f}")

    print("\nConfusion Matrix (counts):\n")
    print(f"{diagnosis_categories[0]} predicted as {diagnosis_categories[0]}: {tp}")
    print(f"{diagnosis_categories[0]} predicted as {diagnosis_categories[1]}: {fn}")
    print(f"{diagnosis_categories[1]} predicted as {diagnosis_categories[0]}: {fp}")
    print(f"{diagnosis_categories[1]} predicted as {diagnosis_categories[1]}: {tn}")

    # Return these so we can reuse them in new-patient output
    return {
        "threshold": threshold,
        "sensitivity_malignant": recall_malignant,
        "specificity_benign": specificity,
    }
