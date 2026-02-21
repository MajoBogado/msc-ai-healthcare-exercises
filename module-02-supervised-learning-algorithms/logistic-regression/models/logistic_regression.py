from sklearn.linear_model import LogisticRegression

def train_logistic_regression_model(
    clinical_features_train_scaled,
    diagnosis_labels_train,
    random_state: int = 42,
):
    """
    Trains a Logistic Regression classifier.

    Returns:
        trained_model
    """

    logistic_regression_model = LogisticRegression(
        max_iter=2000,      # ensures convergence
        random_state=random_state,
    )

    logistic_regression_model.fit(
        clinical_features_train_scaled,
        diagnosis_labels_train,
    )

    return logistic_regression_model
