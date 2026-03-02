from __future__ import annotations
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class GridSearchResult:
    best_pipeline: Pipeline
    best_params: dict
    best_cv_score_f1: float

def run_decision_tree_grid_search(
    base_pipeline: Pipeline,
    train_features_df,
    train_target_series,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> GridSearchResult:
    """
    Performs GridSearchCV on a DecisionTree pipeline.
    Optimizes F1-score to balance precision and recall.

    Important:
    - Grid search runs ONLY on training data (cross-validation)
    - We evaluate the chosen best model later on the separate test set
    """
    param_grid = {
        "decision_tree__max_depth": [3, 5, 7, 10, None],
        "decision_tree__min_samples_leaf": [1, 5, 10, 20],
        "decision_tree__min_samples_split": [2, 10, 20, 50],
        "decision_tree__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )

    grid_search.fit(train_features_df, train_target_series)

    return GridSearchResult(
        best_pipeline=grid_search.best_estimator_,
        best_params=grid_search.best_params_,
        best_cv_score_f1=float(grid_search.best_score_),
    )