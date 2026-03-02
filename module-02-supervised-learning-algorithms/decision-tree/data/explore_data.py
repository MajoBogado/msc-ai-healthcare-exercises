from __future__ import annotations
import pandas as pd

def explore_stroke_dataset(
    clinical_features_df: pd.DataFrame,
    target_stroke: pd.Series,
    preview_rows: int = 5,
) -> None:
    """
    Lightweight EDA for the stroke dataset.

    Purpose (programming perspective):
    - Verify we loaded what we think we loaded
    - Identify missing values that require imputation (commonly BMI)
    - Check target imbalance (stroke cases are usually rare)
    - Confirm feature types (numeric vs categorical) to design preprocessing
    """

    print(f"\nDataset size (rows, columns): {clinical_features_df.shape[0]}, {clinical_features_df.shape[1]}")

    print("\nFeature columns:")
    for column_name in clinical_features_df.columns:
        print(f" - {column_name}")

    print("\nTarget distribution (stroke):")
    target_counts = target_stroke.value_counts(dropna=False).sort_index()
    total_count = int(target_counts.sum())

    for target_value, count_value in target_counts.items():
        percentage = (count_value / total_count) * 100 if total_count > 0 else 0.0
        print(f" - stroke={target_value}: {int(count_value)} ({percentage:.2f}%)")

    print("\nTarget meaning:")
    print(" - stroke = 1  → Stroke occurred (Positive class)")
    print(" - stroke = 0  → No stroke (Negative class)")

    positive_count = target_counts.get(1, 0)
    negative_count = target_counts.get(0, 0)

    if positive_count > 0:
        imbalance_ratio = negative_count / positive_count
        print(f"\nClass imbalance ratio (negative / positive): {imbalance_ratio:.2f} : 1")

    print("\nMissing values per feature (non-zero only):")
    missing_value_counts = clinical_features_df.isna().sum().sort_values(ascending=False)
    missing_value_counts_nonzero = missing_value_counts[missing_value_counts > 0]

    if missing_value_counts_nonzero.empty:
        print(" - No missing values found.")
    else:
        for column_name, missing_count in missing_value_counts_nonzero.items():
            missing_percentage = (missing_count / clinical_features_df.shape[0]) * 100
            print(f" - {column_name}: {int(missing_count)} missing ({missing_percentage:.2f}%)")

    print("\nData types (helps decide preprocessing strategy):")
    data_types_df = pd.DataFrame(
        {
            "feature": clinical_features_df.dtypes.index,
            "dtype": clinical_features_df.dtypes.values.astype(str),
        }
    )
    print(data_types_df.to_string(index=False))

    print(f"\nPreview (first {preview_rows} rows):")
    print(clinical_features_df.head(preview_rows).to_string(index=False))