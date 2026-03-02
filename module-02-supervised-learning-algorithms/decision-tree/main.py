from data.load_data import load_stroke_dataset_from_csv
from data.explore_data import explore_stroke_dataset
from data.split_data import split_stroke_dataset, display_split_distribution
from preprocessing.preprocess_pipeline import define_stroke_feature_groups, build_stroke_preprocessor
from preprocessing.inspect_preprocessing import inspect_preprocessing_step
from models.decision_tree_model import build_decision_tree_pipeline
from models.evaluate import evaluate_classification_model
from models.grid_search import run_decision_tree_grid_search
from models.feature_importance import extract_decision_tree_feature_importance, print_top_feature_importance
from models.predict_patient import (
    build_high_risk_patient_from_schema,
    build_low_risk_patient_from_schema,
    predict_new_patient,
    print_patient_prediction,
    explain_tree_leaf_for_patient
)

def main() -> None:
    print("\nDecision Trees – Stroke Prediction (Healthcare Exercise)\n")

    print("Step 1: Load dataset (local CSV)")
    stroke_dataset = load_stroke_dataset_from_csv(
        csv_path="data/healthcare-dataset-stroke-data.csv"
    )

    print("\nStep 2: Explore dataset (EDA)")
    explore_stroke_dataset(
        clinical_features_df=stroke_dataset.clinical_features_df,
        target_stroke=stroke_dataset.target_stroke,
        preview_rows=5,
    )

    print("\nStep 3: Train/Test Split (stratified)")
    stroke_split = split_stroke_dataset(
        clinical_features_df=stroke_dataset.clinical_features_df,
        target_stroke=stroke_dataset.target_stroke,
        test_size=0.2,
        random_state=42,
    )

    print("\n=== Train/Test Split Completed ===")
    print(f"Training set size: {stroke_split.train_clinical_features_df.shape[0]} samples")
    print(f"Test set size: {stroke_split.test_clinical_features_df.shape[0]} samples")

    display_split_distribution(
        train_target_stroke=stroke_split.train_target_stroke,
        test_target_stroke=stroke_split.test_target_stroke,
    )

    print("\nStep 4: Preprocessing - Imputation (Replace missing), Encoding categorical variables (convert to numeric values), Reusable for new patients")
    feature_groups = define_stroke_feature_groups(
        clinical_features_df=stroke_split.train_clinical_features_df
    )
    preprocessor = build_stroke_preprocessor(feature_groups=feature_groups)

    print("\nFitting preprocessing pipeline on TRAIN data only (prevents data leakage)")
    preprocessor.fit(stroke_split.train_clinical_features_df)

    inspect_preprocessing_step(
        preprocessor=preprocessor,
        train_clinical_features_df=stroke_split.train_clinical_features_df,
        test_clinical_features_df=stroke_split.test_clinical_features_df,
    )

    print("\nStep 5: Train and Evaluate baseline Decision Tree - max_depth=None, min_samples_leaf=1, class_weight=None")
    baseline_pipeline = build_decision_tree_pipeline(preprocessor=preprocessor)

    baseline_pipeline.fit(
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
    )

    evaluate_classification_model(
        model_pipeline=baseline_pipeline,
        features_df=stroke_split.train_clinical_features_df,
        target_series=stroke_split.train_target_stroke,
        split_label="TRAIN",
    )

    evaluate_classification_model(
        model_pipeline=baseline_pipeline,
        features_df=stroke_split.test_clinical_features_df,
        target_series=stroke_split.test_target_stroke,
        split_label="TEST",
    )

    print("\nStep 6: Constrained Decision Tree (max_depth=5, class_weight=None)")

    constrained_pipeline = build_decision_tree_pipeline(
        preprocessor=preprocessor,
        max_depth=5,
        min_samples_leaf=10,
        class_weight=None,
    )

    constrained_pipeline.fit(
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
    )

    evaluate_classification_model(
        constrained_pipeline,
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
        "TRAIN (Constrained)",
    )

    evaluate_classification_model(
        constrained_pipeline,
        stroke_split.test_clinical_features_df,
        stroke_split.test_target_stroke,
        "TEST (Constrained)",
    )

    print("\nStep 7: Constrained + Balanced Decision Tree (max_depth=5, class_weight=balanced)")

    balanced_pipeline = build_decision_tree_pipeline(
        preprocessor=preprocessor,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
    )

    balanced_pipeline.fit(
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
    )

    evaluate_classification_model(
        balanced_pipeline,
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
        "TRAIN (Balanced)",
    )

    evaluate_classification_model(
        balanced_pipeline,
        stroke_split.test_clinical_features_df,
        stroke_split.test_target_stroke,
        "TEST (Balanced)",
    )

    print("\nStep 8: Grid Search (Optimize F1 on TRAIN using Cross-Validation)")

    # Start from a “neutral” pipeline. GridSearch will override tree params.
    grid_search_start_pipeline = build_decision_tree_pipeline(
        preprocessor=preprocessor,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
    )

    grid_result = run_decision_tree_grid_search(
        base_pipeline=grid_search_start_pipeline,
        train_features_df=stroke_split.train_clinical_features_df,
        train_target_series=stroke_split.train_target_stroke,
        cv_folds=5,
    )

    print("\nGrid Search Results:")
    print(f" - Best CV F1-score: {grid_result.best_cv_score_f1:.4f}")
    print(" - Best parameters:")
    for param_name, param_value in grid_result.best_params.items():
        print(f"   • {param_name}: {param_value}")

    print("\nStep 9: Evaluate Grid-Search Best Model on TRAIN and TEST")
    evaluate_classification_model(
        grid_result.best_pipeline,
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
        "TRAIN (Grid Best)",
    )

    evaluate_classification_model(
        grid_result.best_pipeline,
        stroke_split.test_clinical_features_df,
        stroke_split.test_target_stroke,
        "TEST (Grid Best)",
    )

    print("\nStep 10: Feature Importance Comparison (Baseline vs Grid Best)")

    baseline_importance_df = extract_decision_tree_feature_importance(baseline_pipeline)
    grid_best_importance_df = extract_decision_tree_feature_importance(grid_result.best_pipeline)

    print_top_feature_importance(
        baseline_importance_df,
        title="Top Feature Importance — Baseline Decision Tree",
        top_n=15,
    )

    print_top_feature_importance(
        grid_best_importance_df,
        title="Top Feature Importance — Grid-Search Best Decision Tree",
        top_n=15,
    )

    print("\nUnder regularization and F1 optimization, age becomes the most informative global splitting feature for distinguishing stroke vs non-stroke in this dataset.\n")


    print("\nStep 11: New patient prediction (end-to-end) - High risk vs. Low risk")
    high_risk_patient_df = build_high_risk_patient_from_schema(
        feature_columns=list(stroke_split.train_clinical_features_df.columns)
    )

    high_risk_prediction = predict_new_patient(
        model_pipeline=grid_result.best_pipeline,
        patient_row_df=high_risk_patient_df,
    )

    print_patient_prediction(patient_row_df=high_risk_patient_df, result=high_risk_prediction)

    explain_tree_leaf_for_patient(grid_result.best_pipeline, high_risk_patient_df)
    
    low_risk_patient_df = build_low_risk_patient_from_schema(
        feature_columns=list(stroke_split.train_clinical_features_df.columns)
    )

    low_risk_prediction = predict_new_patient(
        model_pipeline=grid_result.best_pipeline,
        patient_row_df=low_risk_patient_df,
    )

    print_patient_prediction(
        patient_row_df=low_risk_patient_df,
        result=low_risk_prediction,
    )

    explain_tree_leaf_for_patient(grid_result.best_pipeline, low_risk_patient_df)

    print("\nStep 12: Probability-smoother Decision Tree (min_samples_leaf=50, balanced)")

    smoother_tree_pipeline = build_decision_tree_pipeline(
        preprocessor=preprocessor,
        max_depth=5,
        min_samples_leaf=50,
        class_weight="balanced",
    )

    smoother_tree_pipeline.fit(
        stroke_split.train_clinical_features_df,
        stroke_split.train_target_stroke,
    )

    # Re-run patient predictions with smoother_tree_pipeline
    high_risk_prediction = predict_new_patient(smoother_tree_pipeline, high_risk_patient_df)
    print_patient_prediction(high_risk_patient_df, high_risk_prediction)
    explain_tree_leaf_for_patient(smoother_tree_pipeline, high_risk_patient_df)

    low_risk_prediction = predict_new_patient(smoother_tree_pipeline, low_risk_patient_df)
    print_patient_prediction(low_risk_patient_df, low_risk_prediction)
    explain_tree_leaf_for_patient(smoother_tree_pipeline, low_risk_patient_df)

if __name__ == "__main__":
    main()