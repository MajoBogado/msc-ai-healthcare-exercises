from typing import List
from models.knn_tuning import KNNKResult

def display_knn_tuning_results(
    *,
    experiment_name: str,
    baseline_k: int,
    tuning_results: List[KNNKResult],
) -> None:
    print(f"\n=== KNN Tuning Results: {experiment_name} ===")
    print(f"Baseline run used k = {baseline_k}")
    print("Now evaluating k values (modified k):")

    header = "k | accuracy | precision | recall | f1"
    print(header)
    print("-" * len(header))

    for result in tuning_results:
        print(
            f"{result.k:>2} | "
            f"{result.accuracy:.4f}  | "
            f"{result.precision:.4f}   | "
            f"{result.recall:.4f} | "
            f"{result.f1:.4f}"
        )

    best_by_recall = max(tuning_results, key=lambda r: r.recall)
    best_by_f1 = max(tuning_results, key=lambda r: r.f1)

    print("\nBest k (highest recall):")
    print(
        f"  k={best_by_recall.k} | recall={best_by_recall.recall:.4f} | "
        f"precision={best_by_recall.precision:.4f} | f1={best_by_recall.f1:.4f}"
    )

    print("\nBest k (highest F1):")
    print(
        f"  k={best_by_f1.k} | f1={best_by_f1.f1:.4f} | "
        f"recall={best_by_f1.recall:.4f} | precision={best_by_f1.precision:.4f}"
    )