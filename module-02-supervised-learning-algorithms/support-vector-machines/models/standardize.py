import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_standard_scaler(clinical_features_train: pd.DataFrame) -> StandardScaler:
    """
    Fits a StandardScaler using ONLY the training data to avoid data leakage.
    """
    standard_scaler = StandardScaler()
    standard_scaler.fit(clinical_features_train)
    return standard_scaler


def standardize_features(
    clinical_features_dataframe: pd.DataFrame,
    fitted_standard_scaler: StandardScaler
) -> pd.DataFrame:
    """
    Transforms features using an already-fitted scaler and returns a DataFrame
    with the same column names (important for interpretability and debugging).
    """
    standardized_array = fitted_standard_scaler.transform(clinical_features_dataframe)

    standardized_features_dataframe = pd.DataFrame(
        standardized_array,
        columns=clinical_features_dataframe.columns,
        index=clinical_features_dataframe.index
    )

    return standardized_features_dataframe


def print_scaling_preview_for_patient(
    clinical_features_dataframe: pd.DataFrame,
    standardized_features_dataframe: pd.DataFrame,
    patient_index: int,
    number_of_features: int = 5
) -> None:
    """
    Prints a small table to understand the effect of scaling for one patient:
    original feature values vs standardized (z-score) values.

    This is interpretability, not evaluation.
    """
    if patient_index not in clinical_features_dataframe.index:
        raise ValueError(
            f"Patient index {patient_index} was not found in the features dataframe index."
        )

    original_patient_row = clinical_features_dataframe.loc[patient_index]
    scaled_patient_row = standardized_features_dataframe.loc[patient_index]

    print(f"{'Feature':20s} {'Original Value':15s} {'Scaled Value':15s}")

    selected_feature_names = list(clinical_features_dataframe.columns[:number_of_features])

    for feature_name in selected_feature_names:
        original_value = float(original_patient_row[feature_name])
        scaled_value = float(scaled_patient_row[feature_name])
        print(f"{feature_name:20s} {original_value:15.5f} {scaled_value:15.6f}")