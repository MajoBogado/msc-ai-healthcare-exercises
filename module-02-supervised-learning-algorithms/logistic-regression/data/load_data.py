from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_dataset(as_frame: bool = True):
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns:
        clinical_features: pd.DataFrame
        diagnosis_labels: pd.Series (0=malignant, 1=benign)
        clinical_feature_names: list[str]
        diagnosis_categories: list[str]
    """
    dataset = load_breast_cancer(as_frame=as_frame)

    if as_frame:
        clinical_features = dataset.data
        diagnosis_labels = dataset.target
        clinical_feature_names = list(dataset.feature_names)
        diagnosis_categories = list(dataset.target_names)
    else:
        clinical_features = pd.DataFrame(
            dataset.data, columns=dataset.feature_names
        )
        diagnosis_labels = pd.Series(dataset.target, name="diagnosis")
        clinical_feature_names = list(dataset.feature_names)
        diagnosis_categories = list(dataset.target_names)

    return (
        clinical_features,
        diagnosis_labels,
        clinical_feature_names,
        diagnosis_categories,
    )
