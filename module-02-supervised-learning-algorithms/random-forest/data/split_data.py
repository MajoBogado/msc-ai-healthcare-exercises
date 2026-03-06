from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class DatasetSplit:
    clinical_features_train: pd.DataFrame
    clinical_features_test: pd.DataFrame
    target_train: pd.Series
    target_test: pd.Series

def _print_class_distribution(
    target_series: pd.Series,
    label: str,
) -> None:
    counts = target_series.value_counts().sort_index()
    percentages = (counts / counts.sum()) * 100.0

    print(f"\nClass distribution ({label}):")
    print("value | count | percent")
    print("-----------------------")

    for class_value in counts.index:
        count = int(counts[class_value])
        percent = float(percentages[class_value])
        print(f"{class_value:>5} | {count:>5} | {percent:>7.2f}%")

def split_dataset_stratified(
    clinical_features_dataframe: pd.DataFrame,
    target_stroke_series: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    (
        clinical_features_train,
        clinical_features_test,
        target_train,
        target_test,
    ) = train_test_split(
        clinical_features_dataframe,
        target_stroke_series,
        test_size=test_size,
        stratify=target_stroke_series,
        random_state=random_state,
    )

    print("\n=== Step 3: Train/Test Split (Stratified) ===")
    print(f"Train size: {len(clinical_features_train)}")
    print(f"Test size:  {len(clinical_features_test)}")

    _print_class_distribution(target_train, label="Train")
    _print_class_distribution(target_test, label="Test")

    return DatasetSplit(
        clinical_features_train=clinical_features_train,
        clinical_features_test=clinical_features_test,
        target_train=target_train,
        target_test=target_test,
    )