def standardize_patient(raw_patient, feature_means, feature_stds):
    """
    Converts raw feature values into standardized values
    using training set statistics.
    """
    standardized = {}

    for feature in raw_patient:
        standardized[feature] = (
            raw_patient[feature] - feature_means[feature]
        ) / feature_stds[feature]

    return standardized
