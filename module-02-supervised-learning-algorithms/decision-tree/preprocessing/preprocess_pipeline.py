from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

@dataclass(frozen=True)
class StrokeFeatureGroups:
    numeric_feature_names: list[str]
    categorical_feature_names: list[str]

def define_stroke_feature_groups(clinical_features_df: pd.DataFrame) -> StrokeFeatureGroups:
    """
    Defines which columns are treated as numeric vs categorical.

    Reason:
    - scikit-learn decision trees require numeric arrays
    - this dataset contains mixed dtypes
    """
    numeric_feature_names = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
    ]

    categorical_feature_names = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    missing_columns = [
        column_name
        for column_name in (numeric_feature_names + categorical_feature_names)
        if column_name not in clinical_features_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Expected columns not found in dataset: {missing_columns}")

    return StrokeFeatureGroups(
        numeric_feature_names=numeric_feature_names,
        categorical_feature_names=categorical_feature_names,
    )

def build_stroke_preprocessor(feature_groups: StrokeFeatureGroups) -> ColumnTransformer:
    """
    Builds a preprocessing transformer for the stroke dataset.

    Outputs:
    - numeric features: imputed (median)
    - categorical features: imputed (most_frequent) + one-hot encoded

    Important:
    - no scaling is used because decision trees do not rely on distance measures.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer_median", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_most_frequent", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_groups.numeric_feature_names),
            ("categorical", categorical_pipeline, feature_groups.categorical_feature_names),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor