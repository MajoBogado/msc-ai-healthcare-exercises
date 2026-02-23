import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_classification_model(
    true_diagnosis: pd.Series,
    predicted_diagnosis: pd.Series
) -> None:
    """
    Evaluates classification performance in a clinically interpretable way.

    Assumes:
    - 1 = malignant (positive class)
    - 0 = benign
    """

    confusion = confusion_matrix(true_diagnosis, predicted_diagnosis)

    true_negative = confusion[0, 0]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    true_positive = confusion[1, 1]

    accuracy = accuracy_score(true_diagnosis, predicted_diagnosis)
    precision = precision_score(true_diagnosis, predicted_diagnosis)
    recall_malignant = recall_score(true_diagnosis, predicted_diagnosis, pos_label=1)
    recall_benign = recall_score(true_diagnosis, predicted_diagnosis, pos_label=0)
    f1 = f1_score(true_diagnosis, predicted_diagnosis)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Malignant detection): {recall_malignant:.3f}")
    print(f"Recall (Benign detection):    {recall_benign:.3f}")
    print(f"F1 Score: {f1:.3f}")

    print("\nConfusion Matrix:")
    print("malignant predicted as malignant:", true_positive)
    print("malignant predicted as benign:", false_negative)
    print("benign predicted as malignant:", false_positive)
    print("benign predicted as benign:", true_negative)

    print("\nImportant: False Negatives (missed cancers):", false_negative)