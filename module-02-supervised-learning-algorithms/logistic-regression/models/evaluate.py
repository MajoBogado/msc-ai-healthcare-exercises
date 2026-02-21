from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def evaluate_classification_model(
    trained_model,
    clinical_features_test_scaled,
    diagnosis_labels_test,
    diagnosis_categories,
):
    """
    Evaluates logistic regression model on test data.
    Prints clinically meaningful metrics.
    """

    predicted_labels = trained_model.predict(clinical_features_test_scaled)
    predicted_probabilities = trained_model.predict_proba(clinical_features_test_scaled)[:, 1]

    accuracy = accuracy_score(diagnosis_labels_test, predicted_labels)
    precision = precision_score(diagnosis_labels_test, predicted_labels)
    ##recall = recall_score(diagnosis_labels_test, predicted_labels)

    malignant_recall = recall_score(
        diagnosis_labels_test,
        predicted_labels,
        pos_label=0,
    )

    benign_recall = recall_score(
        diagnosis_labels_test,
        predicted_labels,
        pos_label=1,
    )

    f1 = f1_score(diagnosis_labels_test, predicted_labels)
    roc_auc = roc_auc_score(diagnosis_labels_test, predicted_probabilities)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Malignant detection): {malignant_recall:.3f}")
    print(f"Recall (Benign detection):    {benign_recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")

    print("\nConfusion Matrix:")
    print("Evaluation with threshold: 0.5 (default)\n")
    cm = confusion_matrix(diagnosis_labels_test, predicted_labels)

    print(f"{diagnosis_categories[0]} predicted as {diagnosis_categories[0]}: {cm[0][0]}")
    print(f"{diagnosis_categories[0]} predicted as {diagnosis_categories[1]}: {cm[0][1]}")
    print(f"{diagnosis_categories[1]} predicted as {diagnosis_categories[0]}: {cm[1][0]}")
    print(f"{diagnosis_categories[1]} predicted as {diagnosis_categories[1]}: {cm[1][1]}")
