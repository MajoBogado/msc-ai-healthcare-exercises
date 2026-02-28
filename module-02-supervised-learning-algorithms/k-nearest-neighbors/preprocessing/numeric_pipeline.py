from dataclasses import dataclass
import pandas as pd
from sklearn.impute import SimpleImputer

from preprocessing.standardize import standardize_features


@dataclass(frozen=True)
class NumericPreprocessingResult:
    scaled_train_features_df: pd.DataFrame
    scaled_test_features_df: pd.DataFrame
    fitted_scaler: object
    fitted_imputer: SimpleImputer


def run_numeric_preprocessing_pipeline(
    train_clinical_features_df: pd.DataFrame,
    test_clinical_features_df: pd.DataFrame,
) -> NumericPreprocessingResult:
    """
    Strategy A:
    - Treat all features as numeric
    - Impute missing values using median (fit on training only)
    - Standardize all features
    """

    # Step 1: Imputation
    imputer = SimpleImputer(strategy="median")

    imputed_train_array = imputer.fit_transform(train_clinical_features_df)
    imputed_test_array = imputer.transform(test_clinical_features_df)

    imputed_train_df = pd.DataFrame(
        imputed_train_array,
        columns=train_clinical_features_df.columns,
    )

    imputed_test_df = pd.DataFrame(
        imputed_test_array,
        columns=test_clinical_features_df.columns,
    )

    # Step 2: Standardization
    standardization_result = standardize_features(
        train_clinical_features_df=imputed_train_df,
        test_clinical_features_df=imputed_test_df,
    )

    return NumericPreprocessingResult(
        scaled_train_features_df=standardization_result.scaled_train_features_df,
        scaled_test_features_df=standardization_result.scaled_test_features_df,
        fitted_scaler=standardization_result.fitted_scaler,
        fitted_imputer=imputer,
    )