from data.load_data import load_dataset
from data.explore_data import explore_dataset
from data.split_data import split_dataset, display_split_distribution
from models.standardize import standardize_clinical_features, preview_scaling_comparison
from models.logistic_regression import train_logistic_regression_model
from models.evaluate import evaluate_classification_model
from models.threshold_analysis import evaluate_with_custom_threshold
from models.interpret_patient import predict_new_patient
import numpy as np

def main():
    
    print("Logistic Regression – Healthcare Exercise")

    print("\nStep 1: Loading dataset")
    clinical_features, diagnosis_labels, clinical_feature_names, diagnosis_categories = load_dataset()

    print("Dataset shape:", clinical_features.shape)
    print("Diagnosis categories:", [str(category) for category in diagnosis_categories])
    print("First 3 feature names:", [str(feature) for feature in clinical_feature_names][:3])

    ##print("All feature names:")
    ##for feature in clinical_feature_names:
    ##    print(feature)

    print("\nStep 2: Exploratory Data Analysis")
    explore_dataset(clinical_features, diagnosis_labels, diagnosis_categories)

    print("\nStep 3: Train/Test Split")
    (
        clinical_features_train,
        clinical_features_test,
        diagnosis_labels_train,
        diagnosis_labels_test,
    ) = split_dataset(clinical_features, diagnosis_labels)

    print("\n--- SPLIT SUMMARY ---")
    print("Training patients:", clinical_features_train.shape[0])
    print("Testing patients:", clinical_features_test.shape[0])

    print("\n--- TO VERIFY STRATIFY ---")
    display_split_distribution(diagnosis_labels_train, diagnosis_labels_test, diagnosis_categories)

    print("\nStep 4: Scaling (Preview of First Patient, First 5 Features)\n")
    (
        clinical_features_train_scaled,
        clinical_features_test_scaled,
        fitted_scaler,
    ) = standardize_clinical_features(clinical_features_train, clinical_features_test)

    preview_scaling_comparison(
        clinical_features_train,
        clinical_features_train_scaled,
        clinical_feature_names,
    )

    print("\nStep 5: Train Logistic Regression Model")
    logistic_regression_model = train_logistic_regression_model(clinical_features_train_scaled, diagnosis_labels_train)
    print("\nModel Trained successfully.")

    print("\nStep 6: Model Evaluation (Test Set)\n")
    evaluate_classification_model(
        logistic_regression_model,
        clinical_features_test_scaled,
        diagnosis_labels_test,
        diagnosis_categories,
    )

    print("\nStep 7: Threshold Tuning\n")
    threshold_policy_metrics = evaluate_with_custom_threshold(
        logistic_regression_model,
        clinical_features_test_scaled,
        diagnosis_labels_test,
        diagnosis_categories,
        threshold=0.95,
    )

    threshold_policy_metrics = evaluate_with_custom_threshold(
        logistic_regression_model,
        clinical_features_test_scaled,
        diagnosis_labels_test,
        diagnosis_categories,
        threshold=0.3,
    )

    print("\nStep 8: Threshold Tuning")
    # Create a synthetic new patient using average feature values
    new_patient_raw_features = (
        clinical_features_train.mean().to_dict()
    )

    predict_new_patient(
        trained_model=logistic_regression_model,
        fitted_scaler=fitted_scaler,
        clinical_feature_names=clinical_feature_names,
        new_patient_raw_features=new_patient_raw_features,
        threshold_policy_metrics=threshold_policy_metrics,
    )

    print("\n")

if __name__ == "__main__":
    main()
