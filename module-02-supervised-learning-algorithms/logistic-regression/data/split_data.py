from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(
    clinical_features: pd.DataFrame,
    diagnosis_labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Splits dataset into training and testing sets
    while preserving class distribution (stratification).
    """

    (
        clinical_features_train,
        clinical_features_test,
        diagnosis_labels_train,
        diagnosis_labels_test,
    ) = train_test_split(
        clinical_features,
        diagnosis_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=diagnosis_labels,  # critical for classification - because of stratify, the malignant/benign proportion should be preserved.
    )

    return (
        clinical_features_train,
        clinical_features_test,
        diagnosis_labels_train,
        diagnosis_labels_test,
    )

def display_split_distribution(
    diagnosis_labels_train,
    diagnosis_labels_test,
    diagnosis_categories,
):
    """
    Prints clean class distribution for train and test sets.
    """

    print("\nTrain:")
    train_distribution = diagnosis_labels_train.value_counts(normalize=True)
    for label_value, proportion in train_distribution.items():
        label_name = diagnosis_categories[label_value]
        print(f"{proportion * 100:.1f}% {label_name}")

    print("\nTest:")
    test_distribution = diagnosis_labels_test.value_counts(normalize=True)
    for label_value, proportion in test_distribution.items():
        label_name = diagnosis_categories[label_value]
        print(f"{proportion * 100:.1f}% {label_name}")
