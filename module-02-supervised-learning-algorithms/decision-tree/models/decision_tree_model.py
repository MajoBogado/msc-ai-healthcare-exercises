from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def build_decision_tree_pipeline(
    preprocessor,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
) -> Pipeline:
    """
    Builds a decision tree pipeline with configurable hyperparameters.
    """

    decision_tree_classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("decision_tree", decision_tree_classifier),
        ]
    )