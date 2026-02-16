from sklearn.metrics import mean_squared_error, r2_score

def evaluate_regression(disease_progression_test, test_predictions):
    """
    Returns basic regression metrics for test set predictions.
    """
    mse = mean_squared_error(disease_progression_test, test_predictions)
    rmse = mse ** 0.5
    r2 = r2_score(disease_progression_test, test_predictions)

    return {"mse": mse, "rmse": rmse, "r2": r2}