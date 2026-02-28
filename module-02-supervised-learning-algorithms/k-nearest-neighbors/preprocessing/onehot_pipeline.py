from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class OneHotPreprocessingResult:
    processed_train_features_df: pd.DataFrame
    processed_test_features_df: pd.DataFrame
    fitted_preprocessor: ColumnTransformer


def run_onehot_preprocessing_pipeline(
    train_clinical_features_df: pd.DataFrame,
    test_clinical_features_df: pd.DataFrame,
) -> OneHotPreprocessingResult:
    """
    Strategy B:
    - Separate continuous vs categorical columns
    - Impute missing values (fit on training only)
    - One-hot encode categorical variables (handle_unknown='ignore')
    - Standardize continuous variables
    - Return processed DataFrames with meaningful column names
    """

    continuous_feature_names = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    categorical_feature_names = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    missing_continuous = [
        name for name in continuous_feature_names
        if name not in train_clinical_features_df.columns
    ]
    missing_categorical = [
        name for name in categorical_feature_names
        if name not in train_clinical_features_df.columns
    ]
    if missing_continuous or missing_categorical:
        raise ValueError(
            "Expected feature columns not found in dataset.\n"
            f"Missing continuous: {missing_continuous}\n"
            f"Missing categorical: {missing_categorical}\n"
            f"Available columns: {list(train_clinical_features_df.columns)}"
        )

    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, continuous_feature_names),
            ("categorical", categorical_pipeline, categorical_feature_names),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    processed_train_array = preprocessor.fit_transform(train_clinical_features_df)
    processed_test_array = preprocessor.transform(test_clinical_features_df)

    output_feature_names = preprocessor.get_feature_names_out()

    processed_train_features_df = pd.DataFrame(
        processed_train_array,
        columns=output_feature_names,
    )

    processed_test_features_df = pd.DataFrame(
        processed_test_array,
        columns=output_feature_names,
    )

    return OneHotPreprocessingResult(
        processed_train_features_df=processed_train_features_df,
        processed_test_features_df=processed_test_features_df,
        fitted_preprocessor=preprocessor,
    )