from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class StandardizationResult:
    scaled_train_features_df: pd.DataFrame
    scaled_test_features_df: pd.DataFrame
    fitted_scaler: StandardScaler


def standardize_features(
    train_clinical_features_df: pd.DataFrame,
    test_clinical_features_df: pd.DataFrame,
) -> StandardizationResult:
    """
    Fits StandardScaler on training data only,
    then transforms both training and test data.

    Returns scaled DataFrames preserving column names.
    """

    scaler = StandardScaler()

    scaled_train_array = scaler.fit_transform(train_clinical_features_df)
    scaled_test_array = scaler.transform(test_clinical_features_df)

    scaled_train_features_df = pd.DataFrame(
        scaled_train_array,
        columns=train_clinical_features_df.columns,
    )

    scaled_test_features_df = pd.DataFrame(
        scaled_test_array,
        columns=test_clinical_features_df.columns,
    )

    return StandardizationResult(
        scaled_train_features_df=scaled_train_features_df,
        scaled_test_features_df=scaled_test_features_df,
        fitted_scaler=scaler,
    )