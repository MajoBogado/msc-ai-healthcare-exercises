import pandas as pd

def explore_dataset(clinical_features_dataframe: pd.DataFrame, diagnosis_series: pd.Series) -> None:
    print("==== Dataset Shape ====")
    print(f"Number of patients (rows): {clinical_features_dataframe.shape[0]}")
    print(f"Number of clinical features (columns): {clinical_features_dataframe.shape[1]}")

    print("\n==== Target Distribution (Diagnosis) ====")
    # After flipping labels:
    # 1 = malignant, 0 = benign
    class_counts = diagnosis_series.value_counts().sort_index()
    benign_count = int(class_counts.get(0, 0))
    malignant_count = int(class_counts.get(1, 0))

    print("0 = benign, 1 = malignant")
    print(f"Benign count:    {benign_count}")
    print(f"Malignant count: {malignant_count}")

    malignant_ratio = malignant_count / len(diagnosis_series)
    print(f"\nMalignant ratio: {malignant_ratio:.3f}")

    print("\n==== Missing Values Check ====")
    missing_values_per_column = clinical_features_dataframe.isna().sum()
    total_missing = int(missing_values_per_column.sum())
    print(f"Total missing values in features: {total_missing}")

    if total_missing > 0:
        print("\nColumns with missing values (top 10):")
        missing_columns = missing_values_per_column[missing_values_per_column > 0].sort_values(ascending=False)
        for column_name, missing_count in missing_columns.head(10).items():
            print(f"{column_name}: {int(missing_count)}")

    print("\n==== Feature Ranges (Why scaling is needed for SVM) ====")

    feature_summary = pd.DataFrame({
        "min": clinical_features_dataframe.min(),
        "max": clinical_features_dataframe.max(),
        "mean": clinical_features_dataframe.mean(),
        "std": clinical_features_dataframe.std()
    })

    feature_summary["range"] = feature_summary["max"] - feature_summary["min"]

    feature_summary_sorted = feature_summary.sort_values("range", ascending=False)

    # Keep table format (clean and readable)
    print(feature_summary_sorted.head(10))

    print("\nTip: If feature ranges differ a lot, distance/margin-based models (SVM/KNN) NEED scaling.")