from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_dataset():
    """
    Loads the Breast Cancer Wisconsin dataset and prepares it for
    clinical classification.

    Important:
    - We redefine the target so that:
        1 = malignant (positive class)
        0 = benign
    This ensures that sensitivity (recall) is calculated for cancer detection.
    """

    breast_cancer_data = load_breast_cancer()

    clinical_features_dataframe = pd.DataFrame(
        breast_cancer_data.data,
        columns=breast_cancer_data.feature_names
    )

    diagnosis_series = pd.Series(
        breast_cancer_data.target,
        name="diagnosis"
    )

    # Flip labels so malignant = 1 (positive class), benign = 0
    diagnosis_series = diagnosis_series.map({0: 1, 1: 0})

    return clinical_features_dataframe, diagnosis_series