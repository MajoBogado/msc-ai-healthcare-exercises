from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

@dataclass(frozen=True)
class StrokeFeatureGroups:
    identifier_feature_names: List[str]
    numeric_feature_names: List[str]
    categorical_feature_names: List[str]

def define_stroke_feature_groups(
    clinical_features_dataframe: pd.DataFrame,
) -> StrokeFeatureGroups:
    """
    Determines which columns are treated as:
      - identifier features (dropped)
      - numeric features
      - categorical features
    """

    all_column_names = list(clinical_features_dataframe.columns)

    identifier_feature_names: List[str] = []
    if "id" in all_column_names:
        identifier_feature_names.append("id")

    numeric_feature_names = (
        clinical_features_dataframe.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    # Remove identifiers from numeric list if present
    numeric_feature_names = [
        column_name
        for column_name in numeric_feature_names
        if column_name not in identifier_feature_names
    ]

    categorical_feature_names = (
        clinical_features_dataframe.select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    categorical_feature_names = [
        column_name
        for column_name in categorical_feature_names
        if column_name not in identifier_feature_names
    ]

    return StrokeFeatureGroups(
        identifier_feature_names=sorted(identifier_feature_names),
        numeric_feature_names=sorted(numeric_feature_names),
        categorical_feature_names=sorted(categorical_feature_names),
    )

def _build_one_hot_encoder() -> OneHotEncoder:
    """
    Compatibility helper:
    - Newer sklearn uses sparse_output
    - Older sklearn uses sparse
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_stroke_preprocessor(
    feature_groups: StrokeFeatureGroups,
) -> ColumnTransformer:
    """
    Builds a reusable preprocessing transformer:
      - Drops identifier columns
      - Numeric: median imputation
      - Categorical: most-frequent imputation + one-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("numeric_imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", _build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("drop_identifiers", "drop", feature_groups.identifier_feature_names),
            ("numeric_features", numeric_pipeline, feature_groups.numeric_feature_names),
            ("categorical_features", categorical_pipeline, feature_groups.categorical_feature_names),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor