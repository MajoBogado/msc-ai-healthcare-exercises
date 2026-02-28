from data.load_data import load_heart_disease_cleveland_dataset
from data.split_data import split_dataset
from data.explore_data import display_dataset_overview, display_split_distribution
from preprocessing.numeric_pipeline import run_numeric_preprocessing_pipeline
from preprocessing.onehot_pipeline import run_onehot_preprocessing_pipeline
from models.knn_model import train_knn_classifier, predict_heart_disease
from models.evaluate import compute_classification_metrics, display_evaluation_report
from models.knn_tuning import tune_k_values
from models.knn_tuning_report import display_knn_tuning_results

def main() -> None:
    baseline_k_value = 5
    k_values_to_test = list(range(1, 16))
    
    print("\nK Nearest Neighbors – Healthcare Exercise\n")

    print("Step 1: Loading dataset")
    heart_disease_dataset = load_heart_disease_cleveland_dataset()

    print("\nStep 2: EDA")
    display_dataset_overview(
        clinical_features_df=heart_disease_dataset.clinical_features_df,
        target_heart_disease=heart_disease_dataset.target_heart_disease,
    )

    print("\nStep 3: Split dataset")

    train_test_split_result = split_dataset(
        clinical_features_df=heart_disease_dataset.clinical_features_df,
        target_heart_disease=heart_disease_dataset.target_heart_disease,
        test_size=0.2,
        random_state=42,
    )

    print("\n=== Train/Test Split Completed ===")
    print(
        f"Training set size: {train_test_split_result.train_clinical_features_df.shape[0]} samples"
    )
    print(
        f"Test set size: {train_test_split_result.test_clinical_features_df.shape[0]} samples"
    )

    display_split_distribution(
        train_target_heart_disease=train_test_split_result.train_target_heart_disease,
        test_target_heart_disease=train_test_split_result.test_target_heart_disease,
    )

    print("\nStep 4: Train and Evaluate numeric Pipeline (baseline k=5)")
    numeric_preprocessing_result = run_numeric_preprocessing_pipeline(
        train_clinical_features_df=train_test_split_result.train_clinical_features_df,
        test_clinical_features_df=train_test_split_result.test_clinical_features_df,
    )

    numeric_knn_model_result = train_knn_classifier(
        processed_train_features_df=numeric_preprocessing_result.scaled_train_features_df,
        train_target_heart_disease=train_test_split_result.train_target_heart_disease,
        number_of_neighbors=baseline_k_value,
        distance_weighting="uniform",
    )

    numeric_predicted_target_heart_disease = predict_heart_disease(
        fitted_knn_classifier=numeric_knn_model_result.fitted_knn_classifier,
        processed_test_features_df=numeric_preprocessing_result.scaled_test_features_df,
    )

    numeric_metrics = compute_classification_metrics(
        true_target_heart_disease=train_test_split_result.test_target_heart_disease,
        predicted_target_heart_disease=numeric_predicted_target_heart_disease,
    )

    display_evaluation_report(
        model_name=f"KNN Numeric Pipeline (baseline k={baseline_k_value})",
        metrics=numeric_metrics,
    )

    print("\nStep 5: Train and Evaluate One-Hot Pipeline (baseline k=5)")
    onehot_preprocessing_result = run_onehot_preprocessing_pipeline(
        train_clinical_features_df=train_test_split_result.train_clinical_features_df,
        test_clinical_features_df=train_test_split_result.test_clinical_features_df,
    )

    onehot_knn_model_result = train_knn_classifier(
        processed_train_features_df=onehot_preprocessing_result.processed_train_features_df,
        train_target_heart_disease=train_test_split_result.train_target_heart_disease,
        number_of_neighbors=baseline_k_value,
        distance_weighting="uniform",
    )

    onehot_predicted_target_heart_disease = predict_heart_disease(
        fitted_knn_classifier=onehot_knn_model_result.fitted_knn_classifier,
        processed_test_features_df=onehot_preprocessing_result.processed_test_features_df,
    )

    onehot_metrics = compute_classification_metrics(
        true_target_heart_disease=train_test_split_result.test_target_heart_disease,
        predicted_target_heart_disease=onehot_predicted_target_heart_disease,
    )

    display_evaluation_report(
        model_name=f"KNN One-Hot Pipeline (baseline k={baseline_k_value})",
        metrics=onehot_metrics,
    )

    print("\n=== Comparison Summary (Higher is better) ===")
    print(f"Numeric Pipeline  | Accuracy: {numeric_metrics.accuracy:.4f} | Recall: {numeric_metrics.recall:.4f} | F1: {numeric_metrics.f1:.4f}")
    print(f"One-Hot Pipeline  | Accuracy: {onehot_metrics.accuracy:.4f} | Recall: {onehot_metrics.recall:.4f} | F1: {onehot_metrics.f1:.4f}")

    print("\nStep 6: Tuning: modify k from 1..15")
    numeric_tuning_results = tune_k_values(
        processed_train_features_df=numeric_preprocessing_result.scaled_train_features_df,
        processed_test_features_df=numeric_preprocessing_result.scaled_test_features_df,
        train_target_heart_disease=train_test_split_result.train_target_heart_disease,
        test_target_heart_disease=train_test_split_result.test_target_heart_disease,
        k_values=k_values_to_test,
    )

    display_knn_tuning_results(
        experiment_name="Numeric Pipeline (median impute + standardize all features)",
        baseline_k=baseline_k_value,
        tuning_results=numeric_tuning_results,
    )

    onehot_tuning_results = tune_k_values(
        processed_train_features_df=onehot_preprocessing_result.processed_train_features_df,
        processed_test_features_df=onehot_preprocessing_result.processed_test_features_df,
        train_target_heart_disease=train_test_split_result.train_target_heart_disease,
        test_target_heart_disease=train_test_split_result.test_target_heart_disease,
        k_values=k_values_to_test,
    )

    display_knn_tuning_results(
        experiment_name="One-Hot Pipeline (impute + one-hot categorical + scale continuous)",
        baseline_k=baseline_k_value,
        tuning_results=onehot_tuning_results,
    )

if __name__ == "__main__":
    main()