from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def extract_decision_tree_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    """
    Extracts feature importance from a fitted Pipeline(preprocessing -> decision_tree).

    Returns a DataFrame sorted by importance descending.
    """
    preprocessor = model_pipeline.named_steps["preprocessing"]
    decision_tree = model_pipeline.named_steps["decision_tree"]

    feature_names = preprocessor.get_feature_names_out()
    importances = decision_tree.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    return importance_df

def print_top_feature_importance(importance_df: pd.DataFrame, title: str, top_n: int = 15) -> None:
    """
    Prints top features from a feature importance DataFrame.
    """
    print("\n====================")
    print(title)
    print("====================")

    top_features_df = importance_df.head(top_n)

    for _, row in top_features_df.iterrows():
        feature_name = str(row["feature_name"])
        importance_value = float(row["importance"])
        print(f" - {feature_name:30s} {importance_value:.4f}")