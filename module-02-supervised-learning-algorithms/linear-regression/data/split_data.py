# data/split_data.py
from sklearn.model_selection import train_test_split

def split_dataset(patient_features, disease_progression, test_size=0.2, random_state=42):
    """
    Splits patient data into training and test sets.

    Returns:
    patient_features (X_train), disease_progression (X_test), represent the proportion of the dataset to include in the train split (y_train), represent the proportion of the dataset to include in the test split (y_test)
    """
    return train_test_split(
        patient_features,
        disease_progression,
        test_size=test_size,
        random_state=random_state
    )
