# data/explore_data.py

def explore_dataset(patient_features, disease_progression):
    print("\n--- Feature Overview ---")
    print(patient_features.columns.tolist())

    print("\n--- Feature Summary Statistics ---")
    print(patient_features.describe())

    print("\n--- Target Summary ---")
    print(disease_progression.describe())

