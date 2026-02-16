import pandas as pd

def interpret_single_feature_change(
    model,
    patient_features,
    feature_name,
    delta=1.0, 
    sensitivity_map=None
):
    """
    Compares prediction before and after changing ONE feature
    by `delta` (in standardized units).
    """
    
    original_value = patient_features[feature_name].iloc[0]
    original_prediction = model.predict(patient_features)[0]

    modified_features = patient_features.copy()
    modified_features[feature_name] += delta

    modified_value = modified_features[feature_name].iloc[0]
    modified_prediction = model.predict(modified_features)[0]

    sensitivity = "unknown"
    if sensitivity_map is not None:
        sensitivity = sensitivity_map.get(feature_name, "unknown")

    return {
        "feature": feature_name,
        "sensitivity_group": sensitivity,
        "original_value": original_value,
        "modified_value": modified_value,
        "original_prediction": original_prediction,
        "modified_prediction": modified_prediction,
        "prediction_change": modified_prediction - original_prediction
    }
