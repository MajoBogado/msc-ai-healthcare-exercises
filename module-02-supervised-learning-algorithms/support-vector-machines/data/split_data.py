import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
    clinical_features_dataframe: pd.DataFrame,
    diagnosis_series: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits the dataset into train and test sets.

    Why stratify?
    - Because malignant cases are ~37% of the dataset.
    - Stratification keeps a similar malignant/benign ratio in both train and test,
      which is important for reliable sensitivity evaluation.
    """
    (
        clinical_features_train,
        clinical_features_test,
        diagnosis_train,
        diagnosis_test,
    ) = train_test_split(
        clinical_features_dataframe,
        diagnosis_series,
        test_size=test_size,
        random_state=random_state,
        stratify=diagnosis_series
    )

    return (
        clinical_features_train,
        clinical_features_test,
        diagnosis_train,
        diagnosis_test,
    )