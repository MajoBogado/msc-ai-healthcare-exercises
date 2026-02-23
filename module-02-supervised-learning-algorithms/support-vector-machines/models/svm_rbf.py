from sklearn.svm import SVC
import pandas as pd


def train_rbf_svm(
    standardized_features_train: pd.DataFrame,
    diagnosis_train: pd.Series,
    regularization_parameter_C: float = 1.0,
    gamma: str = "scale"
) -> SVC:
    """
    Trains an RBF-kernel SVM classifier.

    Parameters:
    - C controls regularization (same idea as linear SVM)
    - gamma controls how 'local' the influence of each training point is:
        * gamma='scale' is a good default for most cases
    """

    rbf_svm_model = SVC(
        kernel="rbf",
        C=regularization_parameter_C,
        gamma=gamma,
        probability=False
    )

    rbf_svm_model.fit(standardized_features_train, diagnosis_train)

    return rbf_svm_model


def predict_with_rbf_svm(
    trained_rbf_svm_model: SVC,
    standardized_features_dataframe: pd.DataFrame
) -> pd.Series:
    predicted_labels = trained_rbf_svm_model.predict(standardized_features_dataframe)

    predicted_series = pd.Series(
        predicted_labels,
        index=standardized_features_dataframe.index,
        name="predicted_diagnosis"
    )

    return predicted_series