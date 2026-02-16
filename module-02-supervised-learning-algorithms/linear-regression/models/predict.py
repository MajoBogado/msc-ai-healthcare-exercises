# models/predict.py

def predict_disease_progression(model, patient_features_test):
    """
    Generate predictions for unseen patients.
    """
    predictions = model.predict(patient_features_test)
    return predictions