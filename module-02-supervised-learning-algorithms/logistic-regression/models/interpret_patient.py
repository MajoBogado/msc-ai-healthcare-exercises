import pandas as pd


def categorize_risk(probability_malignant: float) -> str:
    """
    Simple risk buckets (you can tweak later).
    """
    if probability_malignant < 0.05:
        return "Low"
    if probability_malignant < 0.20:
        return "Moderate"
    return "High"


def predict_new_patient(
    trained_model,
    fitted_scaler,
    clinical_feature_names: list,
    new_patient_raw_features: dict,
    threshold_policy_metrics: dict,
):
    """
    new_patient_raw_features: dict mapping feature_name -> value (RAW, unscaled)

    Prints:
    Probability of malignancy
    Risk category
    Model sensitivity & specificity (from your chosen threshold evaluation)
    """

    # Build a single-row DataFrame in the correct column order
    patient_row = pd.DataFrame([new_patient_raw_features], columns=clinical_feature_names)
    
    # Scale with the SAME scaler fit on training data
    patient_row_scaled = fitted_scaler.transform(patient_row.to_numpy())

    # Predict probabilities
    probability_malignant = trained_model.predict_proba(patient_row_scaled)[0][0]
    risk_category = categorize_risk(probability_malignant)

    sensitivity = threshold_policy_metrics.get("sensitivity_malignant", None)
    specificity = threshold_policy_metrics.get("specificity_benign", None)

    print("\nNew Patient Prediction\n")
    print(f"Probability of malignancy: {probability_malignant * 100:.1f}%")
    print(f"Risk category: {risk_category}")

    # These are model-policy metrics from your test set at the chosen threshold
    if sensitivity is not None:
        print(f"Model sensitivity: {sensitivity * 100:.1f}%")
    if specificity is not None:
        print(f"Model specificity: {specificity * 100:.1f}%")
