from data.load_data import load_dataset
from data.explore_data import explore_dataset
from data.split_data import split_dataset
from models.linear_regression import train_linear_regression
from models.predict import predict_disease_progression
from models.evaluate import evaluate_regression
from models.interpret_patient import interpret_single_feature_change
from models.sensitivity import compute_sensitivity_groups

def main():
    print("Linear Regression – Healthcare Exercise")

    print("Step 1: Loading dataset")
    patient_features, disease_progression = load_dataset()

    print("\nStep 2: Exploratory Data Analysis")
    explore_dataset(patient_features, disease_progression)

    print("\nStep 3: Train/Test Split")
    (
        patient_features_train,
        patient_features_test,
        disease_progression_train,
        disease_progression_test
    ) = split_dataset(patient_features, disease_progression)

    print(f"Training patients: {patient_features_train.shape[0]}")
    print(f"Test patients: {patient_features_test.shape[0]}")

    print("\nStep 4: Train Linear Regression Model")
    linear_model = train_linear_regression(patient_features_train, disease_progression_train)
    print("Model trained successfully.")

    print("\nStep 5: Generate Predictions on Test Set")
    test_predictions = predict_disease_progression(
        linear_model,
        patient_features_test
    )

    print(f"Generated {len(test_predictions)} predictions.")

    print("\nStep 6: Evaluate Model on Test Set")
    metrics = evaluate_regression(disease_progression_test, test_predictions)

    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R^2:  {metrics['r2']:.4f}")

    sensitivity_map, low_thr, high_thr = compute_sensitivity_groups(
    patient_features_train.columns,
    linear_model.coef_
    )

    print("\nSensitivity thresholds (based on abs(coefficients))")
    print(f"LOW  <= {low_thr:.2f}")
    print(f"HIGH >= {high_thr:.2f}")

    print("\nStep 7: Interpret Model Coefficients")
    coefficients = linear_model.coef_
    patient_feature_names = patient_features_train.columns

    for patient_feature_name, coef in zip(patient_feature_names, coefficients):
        print(f"{patient_feature_name}: {coef:.4f}")

    print("\nStep 8: Patient-level interpretation")

    # Take ONE patient from the test set
    single_patient = patient_features_test.iloc[[0]]

    # Example 1: change BMI by +1 (one standard deviation)
    result_bmi = interpret_single_feature_change(
        linear_model,
        single_patient,
        feature_name="bmi",
        delta=1.0,
        sensitivity_map=sensitivity_map
    )

    print(f"Feature: {result_bmi['feature']}")
    print(f"Sensitivity level: {result_bmi['sensitivity_group'].upper()}")
    print(f"Original value: {result_bmi['original_value']:.4f}")
    print(f"Modified value: {result_bmi['modified_value']:.4f}")
    print(f"Original prediction: {result_bmi['original_prediction']:.2f}")
    print(f"Modified prediction: {result_bmi['modified_prediction']:.2f}")
    print(f"Change in prediction: {result_bmi['prediction_change']:.2f}")

    # Example 2: change age by +1 (one standard deviation)
    result_age = interpret_single_feature_change(
        linear_model,
        single_patient,
        feature_name="age",
        delta=1.0,
        sensitivity_map=sensitivity_map
    )

    print(f"\nFeature: {result_age['feature']}")
    print(f"Sensitivity level: {result_age['sensitivity_group'].upper()}")
    print(f"Original value: {result_age['original_value']:.4f}")
    print(f"Modified value: {result_age['modified_value']:.4f}")
    print(f"Original prediction: {result_age['original_prediction']:.2f}")
    print(f"Modified prediction: {result_age['modified_prediction']:.2f}")
    print(f"Change in prediction: {result_age['prediction_change']:.2f}")

    #print("\nStep 9: New patient prediction (end-to-end)")


if __name__ == "__main__":
    main()
