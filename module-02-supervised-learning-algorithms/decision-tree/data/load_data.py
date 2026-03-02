from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class StrokeDataset:
    clinical_features_df: pd.DataFrame
    target_stroke: pd.Series


def load_stroke_dataset_from_csv(csv_path: str | Path) -> StrokeDataset:
    """
    Loads the Stroke Prediction dataset from a local CSV file.

    Why local CSV?
    - This dataset is commonly distributed via Kaggle (not guaranteed on OpenML)
    - Keeps the exercise reproducible: same file -> same results
    - Enables realistic preprocessing (mixed types + missing BMI)
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stroke dataset CSV not found at: {csv_path}\n"
            "Expected you to place the file there (e.g., data/healthcare-dataset-stroke-data.csv)."
        )

    raw_dataset_df = pd.read_csv(csv_path)

    if "stroke" not in raw_dataset_df.columns:
        raise ValueError("Expected target column 'stroke' not found in the CSV.")

    # Drop non-clinical identifier column if present
    if "id" in raw_dataset_df.columns:
        raw_dataset_df = raw_dataset_df.drop(columns=["id"])

    target_stroke = raw_dataset_df["stroke"]
    clinical_features_df = raw_dataset_df.drop(columns=["stroke"])

    # Ensure target is numeric 0/1
    target_stroke_numeric = pd.to_numeric(target_stroke, errors="raise").astype(int)

    return StrokeDataset(
        clinical_features_df=clinical_features_df,
        target_stroke=target_stroke_numeric,
    )