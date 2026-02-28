from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainTestSplit:
    train_clinical_features_df: pd.DataFrame
    test_clinical_features_df: pd.DataFrame
    train_target_heart_disease: pd.Series
    test_target_heart_disease: pd.Series


def split_dataset(
    clinical_features_df: pd.DataFrame,
    target_heart_disease: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainTestSplit:
    """
    Performs stratified train/test split.

    Stratification is critical in medical datasets to preserve
    disease distribution across train and test sets.
    """

    (
        train_clinical_features_df,
        test_clinical_features_df,
        train_target_heart_disease,
        test_target_heart_disease,
    ) = train_test_split(
        clinical_features_df,
        target_heart_disease,
        test_size=test_size,
        random_state=random_state,
        stratify=target_heart_disease,
    )

    return TrainTestSplit(
        train_clinical_features_df=train_clinical_features_df.reset_index(drop=True),
        test_clinical_features_df=test_clinical_features_df.reset_index(drop=True),
        train_target_heart_disease=train_target_heart_disease.reset_index(drop=True),
        test_target_heart_disease=test_target_heart_disease.reset_index(drop=True),
    )