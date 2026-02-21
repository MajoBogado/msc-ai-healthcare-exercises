import pandas as pd

def explore_dataset(
    clinical_features: pd.DataFrame,
    diagnosis_labels: pd.Series,
    diagnosis_categories: list,
):
    print("\n--- DATASET OVERVIEW ---")
    print("Number of patients:", clinical_features.shape[0])
    print("Number of clinical features:", clinical_features.shape[1])

    print("\n--- CLASS DISTRIBUTION ---")
    class_counts = diagnosis_labels.value_counts()

    for label_value, count in class_counts.items():
        label_name = diagnosis_categories[label_value]
        percentage = (count / len(diagnosis_labels)) * 100
        print(f"{label_name}: {count} patients ({percentage:.2f}%)")

    print("\n--- FEATURE SCALE SNAPSHOT ---")
    summary_stats = clinical_features.describe().T
    print(summary_stats[["mean", "std", "min", "max"]].head(10))

    print("--- CONCLUSION: Scaling needed - to center all features around 0 and have variance 1. - Logistic Regression is sensitive to feature scale. ---")
