from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class PatientPrediction:
    predicted_label: int
    stroke_probability: float

def build_high_risk_patient_from_schema(feature_columns: list[str]) -> pd.DataFrame:
    """
    Creates a single-row DataFrame representing a high-risk patient, using the dataset schema.
    Values are chosen to be clinically plausible and intentionally high-risk.

    Important:
    - This is for demonstration/learning only (not clinical advice).
    - The column order matches the training schema to avoid mismatches.
    """
    high_risk_patient = {
        "gender": "Male",
        "age": 78.0,
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 220.0,
        "bmi": 34.0,
        "smoking_status": "smokes",
    }

    # Ensure we match the exact set/order of columns the model expects
    patient_row_df = pd.DataFrame([high_risk_patient])

    missing_columns = [col for col in feature_columns if col not in patient_row_df.columns]
    extra_columns = [col for col in patient_row_df.columns if col not in feature_columns]

    if missing_columns:
        raise ValueError(f"High-risk patient is missing expected columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"High-risk patient has unexpected columns: {extra_columns}")

    patient_row_df = patient_row_df[feature_columns]
    return patient_row_df

def predict_new_patient(model_pipeline, patient_row_df: pd.DataFrame) -> PatientPrediction:
    """
    Runs an end-to-end prediction for a single patient using the full pipeline.
    Prints both predicted class and predicted probability for stroke.

    Assumes positive class is stroke=1.
    """
    predicted_label = int(model_pipeline.predict(patient_row_df)[0])

    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(patient_row_df)[0]
        # In sklearn binary classification, proba columns correspond to classes_ order
        class_order = list(getattr(model_pipeline, "classes_", []))

        # Pipelines don't expose classes_ directly; extract from final estimator
        final_estimator = model_pipeline.named_steps["decision_tree"]
        class_order = list(final_estimator.classes_)

        if 1 in class_order:
            stroke_index = class_order.index(1)
            stroke_probability = float(probabilities[stroke_index])
        else:
            stroke_probability = float("nan")
    else:
        stroke_probability = float("nan")

    return PatientPrediction(
        predicted_label=predicted_label,
        stroke_probability=stroke_probability,
    )

def print_patient_prediction(patient_row_df: pd.DataFrame, result: PatientPrediction) -> None:
    """
    Prints:
    - Raw patient features
    - Predicted class
    - Model-estimated probability (clearly explained)
    """
    
    print("\nPatient Features:")
    for column_name, value in patient_row_df.iloc[0].items():
        print(f" - {column_name:20s}: {value}")

    label_meaning = (
        "Stroke occurred (Positive)"
        if result.predicted_label == 1
        else "No stroke (Negative)"
    )

    print("\nModel Classification:")
    print(f" - Predicted label: {result.predicted_label} → {label_meaning}")

    if result.stroke_probability == result.stroke_probability:
        print(
            f" - Model-estimated stroke probability "
            f"(based on training leaf frequency): "
            f"{result.stroke_probability:.4f}"
        )
        print(
            "   Note: For a Decision Tree, this value represents the "
            "proportion of stroke cases in the training samples "
            "that ended up in the same leaf node. "
            "It is not a calibrated clinical risk estimate."
        )
    else:
        print(" - Probability not available.")

def build_low_risk_patient_from_schema(feature_columns: list[str]) -> pd.DataFrame:
    """
    Creates a single-row DataFrame representing a low-risk patient.
    Values are intentionally chosen to reflect lower stroke risk.
    """

    low_risk_patient = {
        "gender": "Female",
        "age": 70.0,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Rural",
        "avg_glucose_level": 100.0,
        "bmi": 30.0,
        "smoking_status": "smokes",
    }

    patient_row_df = pd.DataFrame([low_risk_patient])

    missing_columns = [col for col in feature_columns if col not in patient_row_df.columns]
    extra_columns = [col for col in patient_row_df.columns if col not in feature_columns]

    if missing_columns:
        raise ValueError(f"Low-risk patient is missing expected columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Low-risk patient has unexpected columns: {extra_columns}")

    patient_row_df = patient_row_df[feature_columns]
    return patient_row_df

def explain_tree_leaf_for_patient(model_pipeline: Pipeline, patient_row_df: pd.DataFrame) -> None:
    """
    Explains which leaf the patient falls into and how many training samples
    of each class were in that leaf.

    NOTE: DecisionTreeClassifier stores counts in tree_.value[node_id].
    """
    preprocessor = model_pipeline.named_steps["preprocessing"]
    decision_tree = model_pipeline.named_steps["decision_tree"]

    patient_transformed = preprocessor.transform(patient_row_df)
    leaf_node_id = int(decision_tree.apply(patient_transformed)[0])

    # tree_.value has shape (n_nodes, 1, n_classes)
    class_counts = decision_tree.tree_.value[leaf_node_id][0].astype(float)

    # Map counts to class labels (order is decision_tree.classes_)
    class_labels = list(decision_tree.classes_)
    class_count_map = {int(label): float(count) for label, count in zip(class_labels, class_counts)}

    no_stroke_count = class_count_map.get(0, 0.0)
    stroke_count = class_count_map.get(1, 0.0)
    total_count = no_stroke_count + stroke_count

    leaf_stroke_probability = (stroke_count / total_count) if total_count > 0 else float("nan")

    print("\nLeaf explanation (Decision Tree internal):")
    print(f" - Leaf node id: {leaf_node_id}")
    print(f" - Training samples in leaf: {int(total_count)}")
    print(f" - No-stroke count (class 0): {int(no_stroke_count)}")
    print(f" - Stroke count (class 1):    {int(stroke_count)}")
    if leaf_stroke_probability == leaf_stroke_probability:
        print(f" - Leaf stroke probability:   {leaf_stroke_probability:.4f}")