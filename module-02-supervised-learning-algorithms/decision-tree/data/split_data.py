from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class StrokeTrainTestSplit:
    train_clinical_features_df: pd.DataFrame
    test_clinical_features_df: pd.DataFrame
    train_target_stroke: pd.Series
    test_target_stroke: pd.Series


def split_stroke_dataset(
    clinical_features_df: pd.DataFrame,
    target_stroke: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> StrokeTrainTestSplit:
    """
    Splits the dataset into train/test sets using stratification.

    Why stratify?
    - Stroke is typically an imbalanced target (few positive cases).
    - Stratification helps preserve the class distribution in both splits.
    """
    (
        train_clinical_features_df,
        test_clinical_features_df,
        train_target_stroke,
        test_target_stroke,
    ) = train_test_split(
        clinical_features_df,
        target_stroke,
        test_size=test_size,
        random_state=random_state,
        stratify=target_stroke,
    )

    return StrokeTrainTestSplit(
        train_clinical_features_df=train_clinical_features_df,
        test_clinical_features_df=test_clinical_features_df,
        train_target_stroke=train_target_stroke,
        test_target_stroke=test_target_stroke,
    )


def display_split_distribution(
    train_target_stroke: pd.Series,
    test_target_stroke: pd.Series,
) -> None:
    """
    Prints class distribution in the train and test splits so you can verify
    stratification worked as expected.
    """
    print("\nTarget distribution after split:")

    for split_name, split_target in [("TRAIN", train_target_stroke), ("TEST", test_target_stroke)]:
        counts = split_target.value_counts(dropna=False).sort_index()
        total = int(counts.sum())

        print(f"\n{split_name} set:")
        for target_value, count_value in counts.items():
            percentage = (count_value / total) * 100 if total > 0 else 0.0
            print(f" - stroke={target_value}: {int(count_value)} ({percentage:.2f}%)")