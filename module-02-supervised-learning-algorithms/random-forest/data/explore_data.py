from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass(frozen=True)
class DatasetProfile:
    number_of_rows: int
    number_of_columns: int
    target_distribution_counts: Dict[int, int]
    target_distribution_percentages: Dict[int, float]
    missing_values_by_column: Dict[str, int]
    numeric_feature_names: List[str]
    categorical_feature_names: List[str]

def build_dataset_profile(
    clinical_features_dataframe: pd.DataFrame,
    target_stroke_series: pd.Series,
) -> DatasetProfile:
    number_of_rows = int(clinical_features_dataframe.shape[0])
    number_of_columns = int(clinical_features_dataframe.shape[1])

    target_counts_series = target_stroke_series.value_counts(dropna=False).sort_index()
    target_distribution_counts = {int(index): int(value) for index, value in target_counts_series.items()}

    target_percentages_series = (target_counts_series / target_counts_series.sum()) * 100.0
    target_distribution_percentages = {int(index): float(value) for index, value in target_percentages_series.items()}

    missing_values_series = clinical_features_dataframe.isna().sum().sort_values(ascending=False)
    missing_values_by_column = {str(column): int(count) for column, count in missing_values_series.items() if int(count) > 0}

    numeric_feature_names = sorted(
        clinical_features_dataframe.select_dtypes(include=["number"]).columns.tolist()
    )
    categorical_feature_names = sorted(
        clinical_features_dataframe.select_dtypes(exclude=["number"]).columns.tolist()
    )

    return DatasetProfile(
        number_of_rows=number_of_rows,
        number_of_columns=number_of_columns,
        target_distribution_counts=target_distribution_counts,
        target_distribution_percentages=target_distribution_percentages,
        missing_values_by_column=missing_values_by_column,
        numeric_feature_names=numeric_feature_names,
        categorical_feature_names=categorical_feature_names,
    )


def print_dataset_profile(profile: DatasetProfile) -> None:
    print(f"Rows: {profile.number_of_rows}")
    print(f"Columns (predictors only): {profile.number_of_columns}")

    print("\nTarget distribution (stroke):")
    print("value | count | percent")
    print("----------------------")
    for target_value in sorted(profile.target_distribution_counts.keys()):
        count = profile.target_distribution_counts[target_value]
        percent = profile.target_distribution_percentages[target_value]
        print(f"{target_value:>5} | {count:>5} | {percent:>7.2f}%")

    if profile.missing_values_by_column:
        print("\nMissing values by column (predictors):")
        for column_name, missing_count in profile.missing_values_by_column.items():
            print(f"- {column_name}: {missing_count}")
    else:
        print("\nMissing values by column (predictors): none")

    print("\nFeature types:")
    print(f"- Numeric features ({len(profile.numeric_feature_names)}): {', '.join(profile.numeric_feature_names)}")
    print(f"- Categorical features ({len(profile.categorical_feature_names)}): {', '.join(profile.categorical_feature_names)}")

    print("\nNotes for Random Forest:")
    print("- Random Forest does NOT require feature scaling to work well (unlike KNN/SVM with RBF),")
    print("  because tree splits depend on thresholds and ordering, not distance.")
    print("- We still use a consistent preprocessing Pipeline to handle missing values and encode categories,")
    print("  and to support end-to-end predictions from raw patient inputs.")


def run_eda(
    clinical_features_dataframe: pd.DataFrame,
    target_stroke_series: pd.Series,
) -> DatasetProfile:
    profile = build_dataset_profile(
        clinical_features_dataframe=clinical_features_dataframe,
        target_stroke_series=target_stroke_series,
    )
    print_dataset_profile(profile)
    return profile