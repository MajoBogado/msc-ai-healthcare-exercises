from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


def inspect_preprocessing_step(
    preprocessor: ColumnTransformer,
    train_clinical_features_df: pd.DataFrame,
    test_clinical_features_df: pd.DataFrame,
) -> None:
    """
    Prints evidence that preprocessing is doing what we claim:
    1) Imputation: missing BMI is replaced
    2) Encoding: categorical variables are one-hot encoded
    3) Reusability: a raw "new patient" row can be transformed safely
    """
    
    _print_imputation_evidence(preprocessor, train_clinical_features_df)
    _print_encoding_evidence(preprocessor)
    _print_reusability_evidence(preprocessor, train_clinical_features_df, test_clinical_features_df)


def _print_imputation_evidence(
    preprocessor: ColumnTransformer,
    train_clinical_features_df: pd.DataFrame,
) -> None:
    print("\n[4A] Imputation (Missing values handling)")

    if "bmi" not in train_clinical_features_df.columns:
        print(" - Column 'bmi' not found. Skipping imputation evidence.")
        return

    missing_bmi_before = int(train_clinical_features_df["bmi"].isna().sum())
    print(f" - Missing BMI values BEFORE preprocessing (TRAIN): {missing_bmi_before}")

    # numeric pipeline is the first transformer we defined: ("numeric", numeric_pipeline, numeric_feature_names)
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    numeric_imputer = numeric_pipeline.named_steps["imputer_median"]

    # the numeric columns list is stored in the ColumnTransformer structure
    numeric_columns = list(preprocessor.transformers_[0][2])
    bmi_index_in_numeric = numeric_columns.index("bmi")

    median_values_used = numeric_imputer.statistics_
    bmi_median_used = float(median_values_used[bmi_index_in_numeric])

    print(f" - Median BMI learned from TRAIN (used to fill missing BMI): {bmi_median_used:.4f}")

    transformed_train_matrix = preprocessor.transform(train_clinical_features_df)

    feature_names = preprocessor.get_feature_names_out()
    bmi_feature_positions = np.where(feature_names == "bmi")[0]

    if bmi_feature_positions.size == 0:
        print(" - Could not locate transformed feature named 'bmi'.")
        return

    bmi_column_after = transformed_train_matrix[:, int(bmi_feature_positions[0])]
    missing_bmi_after = int(np.isnan(bmi_column_after).sum())

    print(f" - Missing BMI values AFTER preprocessing (TRAIN):  {missing_bmi_after}")


def _print_encoding_evidence(preprocessor: ColumnTransformer) -> None:
    print("\n[4B] Encoding (Categorical → One-Hot)")

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    one_hot_encoder = categorical_pipeline.named_steps["one_hot_encoder"]

    categorical_columns = list(preprocessor.transformers_[1][2])
    print(" - Categorical columns encoded:")
    for column_name in categorical_columns:
        print(f"   • {column_name}")

    # one-hot categories per original column
    encoded_categories = one_hot_encoder.categories_
    print("\n - One-hot columns created per categorical feature:")
    for column_name, categories_for_column in zip(categorical_columns, encoded_categories):
        print(f"   • {column_name}: {len(categories_for_column)} categories -> {list(categories_for_column)}")

    all_transformed_feature_names = preprocessor.get_feature_names_out()
    one_hot_feature_names = [name for name in all_transformed_feature_names if "_" in name and name not in categorical_columns]
    print(f"\n - Total transformed features: {len(all_transformed_feature_names)}")
    print(f" - One-hot encoded features (sample up to 15):")
    for feature_name in one_hot_feature_names[:15]:
        print(f"   • {feature_name}")


def _print_reusability_evidence(
    preprocessor: ColumnTransformer,
    train_clinical_features_df: pd.DataFrame,
    test_clinical_features_df: pd.DataFrame,
) -> None:
    print("\n[4C] Reusability (Transforming a new raw patient)")

    example_patient_raw = _build_example_patient(train_clinical_features_df)

    example_patient_df = pd.DataFrame([example_patient_raw], columns=train_clinical_features_df.columns)

    print(" - Example raw patient (before preprocessing):")
    for key_name, value in example_patient_raw.items():
        print(f"   • {key_name}: {value}")

    transformed_example_patient = preprocessor.transform(example_patient_df)

    print("\n - New patient transformed successfully.")
    print(f" - Transformed new patient shape: {transformed_example_patient.shape}")
    print(" - Note: this uses the SAME preprocessing rules learned from TRAIN data.")


def _build_example_patient(train_clinical_features_df: pd.DataFrame) -> dict[str, Any]:
    """
    Creates a realistic example patient using:
    - valid categories seen in TRAIN data (so encoding is deterministic)
    - bmi intentionally missing to demonstrate imputation
    """
    def pick_mode(column_name: str) -> Any:
        return train_clinical_features_df[column_name].mode(dropna=True).iloc[0]

    return {
        "gender": pick_mode("gender"),
        "age": float(train_clinical_features_df["age"].median()),
        "hypertension": int(train_clinical_features_df["hypertension"].mode().iloc[0]),
        "heart_disease": int(train_clinical_features_df["heart_disease"].mode().iloc[0]),
        "ever_married": pick_mode("ever_married"),
        "work_type": pick_mode("work_type"),
        "Residence_type": pick_mode("Residence_type"),
        "avg_glucose_level": float(train_clinical_features_df["avg_glucose_level"].median()),
        "bmi": np.nan,  # intentionally missing
        "smoking_status": pick_mode("smoking_status"),
    }