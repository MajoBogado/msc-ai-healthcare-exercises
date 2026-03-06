from dataclasses import dataclass
from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class RandomForestConfig:
    number_of_trees: int = 200
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    class_weight: Optional[str] = None  # use "balanced" later in the imbalance step
    use_oob_estimation: bool = False
    bootstrap_samples: bool = True

def build_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    config: RandomForestConfig,
) -> Pipeline:
    random_forest_classifier = RandomForestClassifier(
        n_estimators=config.number_of_trees,
        random_state=config.random_state,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        n_jobs=-1,
        bootstrap=config.bootstrap_samples,
        oob_score=config.use_oob_estimation
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest_classifier", random_forest_classifier),
        ]
    )

    return model_pipeline