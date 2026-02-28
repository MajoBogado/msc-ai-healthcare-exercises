from dataclasses import dataclass
import pandas as pd
from typing import List
from models.knn_model import train_knn_classifier, predict_heart_disease
from models.evaluate import compute_classification_metrics

@dataclass(frozen=True)
class KNNKResult:
    k: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def tune_k_values(
    processed_train_features_df: pd.DataFrame,
    processed_test_features_df: pd.DataFrame,
    train_target_heart_disease: pd.Series,
    test_target_heart_disease: pd.Series,
    k_values: List[int],
) -> List[KNNKResult]:

    results = []

    for k in k_values:
        model_result = train_knn_classifier(
            processed_train_features_df=processed_train_features_df,
            train_target_heart_disease=train_target_heart_disease,
            number_of_neighbors=k,
        )

        predictions = predict_heart_disease(
            fitted_knn_classifier=model_result.fitted_knn_classifier,
            processed_test_features_df=processed_test_features_df,
        )

        metrics = compute_classification_metrics(
            true_target_heart_disease=test_target_heart_disease,
            predicted_target_heart_disease=predictions,
        )

        results.append(
            KNNKResult(
                k=k,
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
            )
        )

    return results