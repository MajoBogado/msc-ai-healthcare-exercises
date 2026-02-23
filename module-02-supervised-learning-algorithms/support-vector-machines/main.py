from data.load_data import load_dataset
from data.explore_data import explore_dataset
from data.split_data import split_dataset
from models.standardize import fit_standard_scaler, standardize_features, print_scaling_preview_for_patient
from models.svm_linear import train_linear_svm, predict_with_linear_svm, train_weighted_linear_svm
from models.evaluate import evaluate_classification_model
from models.svm_rbf import train_rbf_svm, predict_with_rbf_svm

def main():
    
    print("\nSupport Vectore Machine – Healthcare Exercise")

    print("\nStep 1: Loading dataset")
    
    clinical_features_dataframe, diagnosis_series = load_dataset()

    print("\nStep 2: Exploratory Data Analysis")
    print("=== Full dataset overview ===")
    explore_dataset(clinical_features_dataframe, diagnosis_series)
    (
        clinical_features_train,
        clinical_features_test,
        diagnosis_train,
        diagnosis_test,
    ) = split_dataset(clinical_features_dataframe, diagnosis_series)

    print("\nStep 3: Train/Test Split")
    print("\n=== Train set diagnosis distribution ===")
    train_distribution = diagnosis_train.value_counts(normalize=True).sort_index()
    print(f"0 (benign):    {train_distribution.get(0, 0.0):.3f}")
    print(f"1 (malignant): {train_distribution.get(1, 0.0):.3f}")

    print("\n=== Test set diagnosis distribution ===")
    test_distribution = diagnosis_test.value_counts(normalize=True).sort_index()
    print(f"0 (benign):    {test_distribution.get(0, 0.0):.3f}")
    print(f"1 (malignant): {test_distribution.get(1, 0.0):.3f}")

    print("\nStep 4: Scaling (Preview of First Patient, First 5 Features)\n")
    fitted_standard_scaler = fit_standard_scaler(clinical_features_train)

    standardized_features_train = standardize_features(clinical_features_train, fitted_standard_scaler)
    standardized_features_test = standardize_features(clinical_features_test, fitted_standard_scaler)

    print_scaling_preview_for_patient(
        clinical_features_dataframe=clinical_features_train,
        standardized_features_dataframe=standardized_features_train,
        patient_index=clinical_features_train.index[0],
        number_of_features=5
    )

    print("\nStep 5: Train SVM Linear Model")
    linear_svm_model = train_linear_svm(
        standardized_features_train=standardized_features_train,
        diagnosis_train=diagnosis_train,
        regularization_parameter_C=1.0
    )

    linear_svm_predictions_test = predict_with_linear_svm(
        trained_linear_svm_model=linear_svm_model,
        standardized_features_dataframe=standardized_features_test
    )

    print("\nLinear SVM model trained and predictions generated.")

    print("\nStep 6: Model Evaluation (Test Set)\n")
    evaluate_classification_model(
        true_diagnosis=diagnosis_test,
        predicted_diagnosis=linear_svm_predictions_test
    )

    print("\nStep 7: Linear SVM Hyperparameter Tuning - to reduce false negatives\n")
    print("\n--- Linear SVM (C = 10.0) ---")
    linear_svm_model_C10 = train_linear_svm(
        standardized_features_train=standardized_features_train,
        diagnosis_train=diagnosis_train,
        regularization_parameter_C=10.0
    )

    linear_svm_predictions_test_C10 = predict_with_linear_svm(
        trained_linear_svm_model=linear_svm_model_C10,
        standardized_features_dataframe=standardized_features_test
    )

    evaluate_classification_model(
        true_diagnosis=diagnosis_test,
        predicted_diagnosis=linear_svm_predictions_test_C10
    )

    print("\n--- Linear SVM (C = 100.0) ---")
    linear_svm_model_C100 = train_linear_svm(
        standardized_features_train=standardized_features_train,
        diagnosis_train=diagnosis_train,
        regularization_parameter_C=100.0
    )

    linear_svm_predictions_test_C100 = predict_with_linear_svm(
        trained_linear_svm_model=linear_svm_model_C100,
        standardized_features_dataframe=standardized_features_test
    )

    evaluate_classification_model(
        true_diagnosis=diagnosis_test,
        predicted_diagnosis=linear_svm_predictions_test_C100
    )

    print("\n--- Linear SVM (Class-weighted: 2.0) ---")

    weighted_linear_svm_model = train_weighted_linear_svm(
        standardized_features_train=standardized_features_train,
        diagnosis_train=diagnosis_train,
        regularization_parameter_C=1.0,
        malignant_weight=2.0
    )

    weighted_linear_svm_predictions_test = predict_with_linear_svm(
        trained_linear_svm_model=weighted_linear_svm_model,
        standardized_features_dataframe=standardized_features_test
    )

    evaluate_classification_model(
        true_diagnosis=diagnosis_test,
        predicted_diagnosis=weighted_linear_svm_predictions_test
    )

    print("\nStep 8: Compare previous results against SVM Radial Basis Function (RBF) - trying to see if there are non-linear relationships between features\n")
    print("--- RBF SVM (C = 1.0, gamma = scale) ---")

    rbf_svm_model = train_rbf_svm(
        standardized_features_train=standardized_features_train,
        diagnosis_train=diagnosis_train,
        regularization_parameter_C=1.0,
        gamma="scale"
    )

    rbf_svm_predictions_test = predict_with_rbf_svm(
        trained_rbf_svm_model=rbf_svm_model,
        standardized_features_dataframe=standardized_features_test
    )

    evaluate_classification_model(
        true_diagnosis=diagnosis_test,
        predicted_diagnosis=rbf_svm_predictions_test
    )

if __name__ == "__main__":
    main()