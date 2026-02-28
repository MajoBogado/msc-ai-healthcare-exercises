from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class HeartDiseaseDataset:
    clinical_features_df: pd.DataFrame
    target_heart_disease: pd.Series


def load_heart_disease_cleveland_dataset() -> HeartDiseaseDataset:
    """
    Loads the UCI Heart Disease (Cleveland) dataset via OpenML.

    Original target column: 'num'
        0  -> no heart disease
        1-4 -> heart disease present

    We convert to binary:
        0 -> 0 (no disease)
        1-4 -> 1 (disease present)
    """

    openml_dataset = fetch_openml(data_id=194, as_frame=True)
    full_dataframe: pd.DataFrame = openml_dataset.frame.copy()

    if "num" not in full_dataframe.columns:
        raise ValueError(
            f"Expected target column 'num' not found. Columns available: {list(full_dataframe.columns)}"
        )

    # Convert all columns to numeric (coerce '?' to NaN)
    for column in full_dataframe.columns:
        full_dataframe[column] = pd.to_numeric(full_dataframe[column], errors="coerce")

    original_target_series = full_dataframe["num"]

    target_heart_disease = (original_target_series >= 1).astype(int)
    target_heart_disease.name = "target_heart_disease"

    clinical_features_df = full_dataframe.drop(columns=["num"])

    valid_target_mask = original_target_series.notna()

    clinical_features_df = clinical_features_df.loc[valid_target_mask].reset_index(drop=True)
    target_heart_disease = target_heart_disease.loc[valid_target_mask].reset_index(drop=True)

    return HeartDiseaseDataset(
        clinical_features_df=clinical_features_df,
        target_heart_disease=target_heart_disease,
    )