from data.load_data import load_stroke_dataset_from_csv
from data.explore_data import run_eda
from data.split_data import split_dataset_stratified
from preprocessing.preprocess_pipeline import define_stroke_feature_groups, build_stroke_preprocessor
from preprocessing.inspect_preprocessing import inspect_preprocessing_step
from models.random_forest_model import RandomForestConfig, build_random_forest_pipeline
from models.evaluate import evaluate_binary_classifier
from models.oob_report import print_oob_score_if_available
from models.threshold_analysis import (
    evaluate_thresholds,
    get_positive_class_probabilities,
    print_threshold_report,
    select_threshold_by_best_f1,
    select_threshold_by_target_recall,
)

def main() -> None:
    print("\nRandom Forest – Stroke Prediction (Healthcare Exercise)\n")

    print("Step 1: Load dataset (local CSV)")
    stroke_dataset = load_stroke_dataset_from_csv(
        csv_file_path="data/healthcare-dataset-stroke-data.csv"
    )

    print("\nStep 2: Explore dataset (EDA)")
    run_eda(
        clinical_features_dataframe=stroke_dataset.clinical_features_dataframe,
        target_stroke_series=stroke_dataset.target_stroke_series,
    )

    print("\nStep 3: Train/Test Split (stratified)")
    dataset_split = split_dataset_stratified(
        clinical_features_dataframe=stroke_dataset.clinical_features_dataframe,
        target_stroke_series=stroke_dataset.target_stroke_series,
        test_size=0.2,
        random_state=42,
    )

    print("\nStep 4: Preprocessing - Imputation, One-Hot Encoding, Reusable for new patients")
    feature_groups = define_stroke_feature_groups(
        clinical_features_dataframe=dataset_split.clinical_features_train
    )
    preprocessor = build_stroke_preprocessor(feature_groups=feature_groups)

    print("\nFitting preprocessing pipeline on TRAIN data only (prevents data leakage)")
    preprocessor.fit(dataset_split.clinical_features_train)

    inspect_preprocessing_step(
        preprocessor=preprocessor,
        train_clinical_features_dataframe=dataset_split.clinical_features_train,
        test_clinical_features_dataframe=dataset_split.clinical_features_test,
    )

    print("\nStep 5: Baseline Random Forest (Pipeline + Train/Test evaluation)  - max_depth=None, min_samples_leaf=1, class_weight=None")
    baseline_forest_config = RandomForestConfig(
        number_of_trees=200,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        use_oob_estimation=True,
        bootstrap_samples=True
    )

    random_forest_pipeline = build_random_forest_pipeline(
        preprocessor=preprocessor,
        config=baseline_forest_config,
    )

    random_forest_pipeline.fit(
        dataset_split.clinical_features_train,
        dataset_split.target_train,
    )

    print_oob_score_if_available(
        model_pipeline=random_forest_pipeline,
        title="Step 5 (Baseline Random Forest)",
    )

    print("\nBaseline Random Forest Evaluation — TRAIN SET\n")
    evaluate_binary_classifier(
        model_pipeline=random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_train,
        target_stroke_series=dataset_split.target_train,
        label="Train",
    )

    print("\nBaseline Random Forest Evaluation — TEST SET\n")
    evaluate_binary_classifier(
        model_pipeline=random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_test,
        target_stroke_series=dataset_split.target_test,
        label="Test",
    )

    print("\nStep 6: Constrained Random Forest (Pipeline + Train/Test evaluation) — max_depth=4, min_samples_leaf=10, class_weight=None")
    constrained_forest_config = RandomForestConfig(
        number_of_trees=300,
        random_state=42,
        max_depth=4,
        min_samples_leaf=10,
        class_weight=None,
        use_oob_estimation=True,
        bootstrap_samples=True
    )

    constrained_random_forest_pipeline = build_random_forest_pipeline(
        preprocessor=preprocessor,
        config=constrained_forest_config,
    )

    constrained_random_forest_pipeline.fit(
        dataset_split.clinical_features_train,
        dataset_split.target_train,
    )

    print_oob_score_if_available(
        model_pipeline=constrained_random_forest_pipeline,
        title="Step 6 (Constrained Random Forest)",
    )

    print("\nConstrained Random Forest Evaluation — TRAIN SET\n")
    evaluate_binary_classifier(
        model_pipeline=constrained_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_train,
        target_stroke_series=dataset_split.target_train,
        label="Train",
    )

    print("\nConstrained Random Forest Evaluation — TEST SET\n")
    evaluate_binary_classifier(
        model_pipeline=constrained_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_test,
        target_stroke_series=dataset_split.target_test,
        label="Test",
    )

    print("\nStep 7: Random Forest with class_weight='balanced' (Baseline flexibility)")
    balanced_baseline_forest_config = RandomForestConfig(
        number_of_trees=200,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
    )

    balanced_baseline_random_forest_pipeline = build_random_forest_pipeline(
        preprocessor=preprocessor,
        config=balanced_baseline_forest_config,
    )

    balanced_baseline_random_forest_pipeline.fit(
        dataset_split.clinical_features_train,
        dataset_split.target_train,
    )

    print("\nBalanced Baseline Random Forest Evaluation — TRAIN SET\n")
    evaluate_binary_classifier(
        model_pipeline=balanced_baseline_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_train,
        target_stroke_series=dataset_split.target_train,
        label="Train",
    )

    print("\nBalanced Baseline Random Forest Evaluation — TEST SET\n")
    evaluate_binary_classifier(
        model_pipeline=balanced_baseline_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_test,
        target_stroke_series=dataset_split.target_test,
        label="Test",
    )

    print("\nStep 8: Random Forest with class_weight='balanced' (Constrained forest)")
    balanced_constrained_forest_config = RandomForestConfig(
        number_of_trees=300,
        random_state=42,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
    )

    balanced_constrained_random_forest_pipeline = build_random_forest_pipeline(
        preprocessor=preprocessor,
        config=balanced_constrained_forest_config,
    )

    balanced_constrained_random_forest_pipeline.fit(
        dataset_split.clinical_features_train,
        dataset_split.target_train,
    )

    print("\nBalanced Constrained Random Forest Evaluation — TRAIN SET\n")
    evaluate_binary_classifier(
        model_pipeline=balanced_constrained_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_train,
        target_stroke_series=dataset_split.target_train,
        label="Train",
    )

    print("\nBalanced Constrained Random Forest Evaluation — TEST SET\n")
    evaluate_binary_classifier(
        model_pipeline=balanced_constrained_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_test,
        target_stroke_series=dataset_split.target_test,
        label="Test",
    )

    print("\nStep 9: Threshold tuning (using Balanced Constrained Random Forest probabilities)\n")
    test_set_positive_probabilities = get_positive_class_probabilities(
        model_pipeline=balanced_constrained_random_forest_pipeline,
        clinical_features_dataframe=dataset_split.clinical_features_test,
    )

    threshold_values = [0.90, 0.80, 0.70, 0.60, 0.55, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]

    threshold_results = evaluate_thresholds(
        true_labels=dataset_split.target_test,
        predicted_probabilities=test_set_positive_probabilities,
        thresholds=threshold_values,
    )

    best_f1_result = select_threshold_by_best_f1(threshold_results)

    target_recall = 0.90
    selected_by_recall = select_threshold_by_target_recall(
        threshold_results=threshold_results,
        target_recall=target_recall,
    )

    print_threshold_report(
        threshold_results=threshold_results,
        highlighted_threshold=best_f1_result.threshold,
    )

    print(f"\n- Best F1 threshold: {best_f1_result.threshold:.2f} (F1={best_f1_result.f1:.4f}, Recall={best_f1_result.recall:.4f}, Precision={best_f1_result.precision:.4f})")

    if selected_by_recall is None:
        print(f"- No threshold in the sweep reached target recall ≥ {target_recall:.2f}")
    else:
        print(
            f"- Best Precision among thresholds with Recall ≥ {target_recall:.2f}: "
            f"threshold={selected_by_recall.threshold:.2f} "
            f"(Precision={selected_by_recall.precision:.4f}, Recall={selected_by_recall.recall:.4f}, F1={selected_by_recall.f1:.4f})"
        )

if __name__ == "__main__":
    main()