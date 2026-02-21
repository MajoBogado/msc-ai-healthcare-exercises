from sklearn.preprocessing import StandardScaler
import pandas as pd


def standardize_clinical_features(
    clinical_features_train: pd.DataFrame,
    clinical_features_test: pd.DataFrame,
):
    """
    Fits a StandardScaler on training data only,
    then transforms both training and test sets.

    Returns:
        clinical_features_train_scaled
        clinical_features_test_scaled
        fitted_scaler
    """

    scaler = StandardScaler()

    clinical_features_train_np = clinical_features_train.to_numpy()
    clinical_features_test_np  = clinical_features_test.to_numpy()

    # Fit ONLY on training data
    scaler.fit(clinical_features_train_np)

    # Transform both sets
    clinical_features_train_scaled = scaler.transform(clinical_features_train_np)
    clinical_features_test_scaled = scaler.transform(clinical_features_test_np)

    return (
        clinical_features_train_scaled,
        clinical_features_test_scaled,
        scaler,
    )

def preview_scaling_comparison(
    clinical_features_train: pd.DataFrame,
    clinical_features_train_scaled,
    clinical_feature_names: list,
):
    """
    Displays first 5 features before and after scaling
    for the first patient in the training set.
    """

    original_values = clinical_features_train.iloc[0, :5].values
    scaled_values = clinical_features_train_scaled[0, :5]

    comparison_table = pd.DataFrame({
        "Feature": clinical_feature_names[:5],
        "Original Value": original_values,
        "Scaled Value": scaled_values,
    })

    print(comparison_table.to_string(index=False))