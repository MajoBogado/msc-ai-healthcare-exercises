import pandas as pd

def display_dataset_overview(
    clinical_features_df: pd.DataFrame,
    target_heart_disease: pd.Series,
) -> None:
    """
    Displays:
    - Dataset shape
    - Clean target distribution (no pandas metadata)
    - First 5 rows of features
    """

    print("=== Heart Disease Dataset Loaded ===")
    print(f"Clinical features shape: {clinical_features_df.shape}")

    _print_target_distribution(target_heart_disease)

    print("\nFirst 5 rows of clinical features:")
    print(clinical_features_df.head().to_string(index=False))


def _print_target_distribution(target_series: pd.Series) -> None:
    counts_dict = target_series.value_counts().sort_index().to_dict()

    no_disease_count = counts_dict.get(0, 0)
    disease_count = counts_dict.get(1, 0)

    total_samples = no_disease_count + disease_count

    print("\nTarget distribution:")
    print(f"  No heart disease (0): {no_disease_count}")
    print(f"  Heart disease (1):    {disease_count}")
    print(f"  Total samples:        {total_samples}")

def display_split_distribution(
    train_target_heart_disease: pd.Series,
    test_target_heart_disease: pd.Series,
) -> None:
    """
    Displays clean class distribution for train and test sets
    to verify stratification worked properly.
    """

    def _format_distribution(target_series: pd.Series):
        counts = target_series.value_counts().sort_index().to_dict()
        total = sum(counts.values())

        no_disease = counts.get(0, 0)
        disease = counts.get(1, 0)

        no_disease_pct = (no_disease / total) * 100
        disease_pct = (disease / total) * 100

        return {
            "total": total,
            "no_disease": no_disease,
            "disease": disease,
            "no_disease_pct": no_disease_pct,
            "disease_pct": disease_pct,
        }

    train_stats = _format_distribution(train_target_heart_disease)
    test_stats = _format_distribution(test_target_heart_disease)

    print("\n=== Stratification Verification ===")

    print("\nTraining Set:")
    print(f"  Total samples: {train_stats['total']}")
    print(
        f"  No disease (0): {train_stats['no_disease']} "
        f"({train_stats['no_disease_pct']:.2f}%)"
    )
    print(
        f"  Disease (1):    {train_stats['disease']} "
        f"({train_stats['disease_pct']:.2f}%)"
    )

    print("\nTest Set:")
    print(f"  Total samples: {test_stats['total']}")
    print(
        f"  No disease (0): {test_stats['no_disease']} "
        f"({test_stats['no_disease_pct']:.2f}%)"
    )
    print(
        f"  Disease (1):    {test_stats['disease']} "
        f"({test_stats['disease_pct']:.2f}%)"
    )