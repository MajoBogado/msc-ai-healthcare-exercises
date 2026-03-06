from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class StrokeDataset:
    clinical_features_dataframe: pd.DataFrame
    target_stroke_series: pd.Series

def load_stroke_dataset_from_csv(csv_file_path: str) -> StrokeDataset:
    """
    Loads the Stroke Prediction Dataset from a local CSV file.

    Expected target column:
        - stroke (0 = no stroke, 1 = stroke)

    Returns:
        StrokeDataset:
            clinical_features_dataframe -> all predictor columns
            target_stroke_series -> target column
    """

    dataset_path = Path(csv_file_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path.resolve()}"
        )

    full_dataset_dataframe = pd.read_csv(dataset_path)

    if "stroke" not in full_dataset_dataframe.columns:
        raise ValueError(
            "The dataset must contain a 'stroke' column."
        )

    target_stroke_series = full_dataset_dataframe["stroke"].astype(int)
    clinical_features_dataframe = full_dataset_dataframe.drop(columns=["stroke"])

    return StrokeDataset(
        clinical_features_dataframe=clinical_features_dataframe,
        target_stroke_series=target_stroke_series,
    )