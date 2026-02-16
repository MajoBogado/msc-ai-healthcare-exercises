# data/load_data.py
from sklearn.datasets import load_diabetes

def load_dataset():
    """
    Loads a healthcare-related regression dataset.
    Returns:
        patient_features: DataFrame with patient characteristics - X: pandas.DataFrame of features
        disease_progression: Series with continuous outcome -  y: pandas.Series of target
    """
    patient_features, disease_progression = load_diabetes(
        return_X_y=True,
        as_frame=True
    )
    return patient_features, disease_progression
