from sklearn.pipeline import Pipeline

def print_oob_score_if_available(
    model_pipeline: Pipeline,
    title: str,
) -> None:
    """
    Prints the RandomForestClassifier OOB score if enabled.
    OOB score is a training-set generalization estimate (built-in validation).
    """

    random_forest_classifier = model_pipeline.named_steps.get("random_forest_classifier")
    if random_forest_classifier is None:
        return

    if not getattr(random_forest_classifier, "oob_score", False):
        return

    oob_score_value = getattr(random_forest_classifier, "oob_score_", None)
    if oob_score_value is None:
        return

    print("\nOut-of-Bag (OOB) estimation:")
    print(f"- {title}: OOB score = {float(oob_score_value):.4f}")
    print("  (Interpretation: estimated performance on unseen data using training samples not included")
    print("   in each tree’s bootstrap sample.)")