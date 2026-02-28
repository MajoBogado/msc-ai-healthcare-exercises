from dataclasses import dataclass
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

@dataclass(frozen=True)
class KNNModelResult:
    fitted_knn_classifier: KNeighborsClassifier


def train_knn_classifier(
    processed_train_features_df: pd.DataFrame,
    train_target_heart_disease: pd.Series,
    *,
    number_of_neighbors: int = 5,
    distance_metric: str = "minkowski",
    minkowski_power: int = 2,
    distance_weighting: str = "uniform",
) -> KNNModelResult:
    """
    Trains a K-Nearest Neighbors classifier.

    Parameters:
      - number_of_neighbors: k in KNN
      - distance_metric:
          * "minkowski" with p=2 is Euclidean distance
          * "minkowski" with p=1 is Manhattan distance
      - minkowski_power: p parameter for Minkowski (ignored unless metric="minkowski")
      - distance_weighting:
          * "uniform": all neighbors contribute equally
          * "distance": closer neighbors contribute more
    """

    knn_classifier = KNeighborsClassifier(
        n_neighbors=number_of_neighbors,
        metric=distance_metric,
        p=minkowski_power,
        weights=distance_weighting,
    )

    knn_classifier.fit(processed_train_features_df, train_target_heart_disease)

    return KNNModelResult(fitted_knn_classifier=knn_classifier)


def predict_heart_disease(
    fitted_knn_classifier: KNeighborsClassifier,
    processed_test_features_df: pd.DataFrame,
) -> pd.Series:
    """
    Predicts class labels:
      0 = no disease
      1 = disease
    Returns a Series aligned to test rows.
    """
    predicted_labels = fitted_knn_classifier.predict(processed_test_features_df)
    return pd.Series(predicted_labels, name="predicted_heart_disease")


def predict_heart_disease_probability(
    fitted_knn_classifier: KNeighborsClassifier,
    processed_test_features_df: pd.DataFrame,
) -> pd.Series:
    """
    Predicts probability of disease (class 1).

    Note:
    KNN probabilities are based on the fraction of neighbors belonging to class 1
    (optionally weighted by distance).
    """
    probability_matrix = fitted_knn_classifier.predict_proba(processed_test_features_df)
    disease_probability = probability_matrix[:, 1]
    return pd.Series(disease_probability, name="predicted_probability_heart_disease")