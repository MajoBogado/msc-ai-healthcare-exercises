from sklearn.svm import SVC
import pandas as pd


def train_linear_svm(
    standardized_features_train: pd.DataFrame,
    diagnosis_train: pd.Series,
    regularization_parameter_C: float = 1.0
) -> SVC:
    """
    Trains a Linear Support Vector Machine classifier.

    Parameters:
    - C controls the trade-off between margin size and classification error.
      Smaller C → wider margin, more tolerance to misclassification.
      Larger C → narrower margin, less tolerance to misclassification.
    """

    linear_svm_model = SVC(
        kernel="linear",
        C=regularization_parameter_C,
        probability=False  # we focus on classification first
    )

    linear_svm_model.fit(standardized_features_train, diagnosis_train)

    return linear_svm_model


def predict_with_linear_svm(
    trained_linear_svm_model: SVC,
    standardized_features_dataframe: pd.DataFrame
) -> pd.Series:
    """
    Generates predictions using a trained Linear SVM model.
    """

    predicted_labels = trained_linear_svm_model.predict(standardized_features_dataframe)

    predicted_series = pd.Series(
        predicted_labels,
        index=standardized_features_dataframe.index,
        name="predicted_diagnosis"
    )

    return predicted_series
    

def train_weighted_linear_svm(
    standardized_features_train: pd.DataFrame,
    diagnosis_train: pd.Series,
    regularization_parameter_C: float = 1.0,
    malignant_weight: float = 2.0
):
    """
    Trains a Linear SVM with higher penalty for misclassifying malignant cases.
    """

    weighted_linear_svm_model = SVC(
        kernel="linear",
        C=regularization_parameter_C,
        class_weight={1: malignant_weight},
        probability=False
    )

    weighted_linear_svm_model.fit(standardized_features_train, diagnosis_train)

    return weighted_linear_svm_model