import pandas as pd
from sklearn.compose import ColumnTransformer

def inspect_preprocessing_step(
    preprocessor: ColumnTransformer,
    train_clinical_features_dataframe: pd.DataFrame,
    test_clinical_features_dataframe: pd.DataFrame,
    preview_feature_names: int = 25,
) -> None:
    """
    Prints basic sanity checks to confirm:
      - Fit happens on TRAIN only
      - Transform works on TRAIN/TEST
      - Output shapes and feature names look reasonable
    """

    transformed_train_matrix = preprocessor.transform(train_clinical_features_dataframe)
    transformed_test_matrix = preprocessor.transform(test_clinical_features_dataframe)

    print("\n=== Preprocessing Inspection ===")
    print(f"Raw TRAIN shape: {train_clinical_features_dataframe.shape}")
    print(f"Raw TEST shape:  {test_clinical_features_dataframe.shape}")
    print(f"Transformed TRAIN shape: {transformed_train_matrix.shape}")
    print(f"Transformed TEST shape:  {transformed_test_matrix.shape}")

    try:
        output_feature_names = list(preprocessor.get_feature_names_out())
        print(f"\nNumber of output features after preprocessing: {len(output_feature_names)}")
        print(f"First {min(preview_feature_names, len(output_feature_names))} output features:")
        for feature_name in output_feature_names[:preview_feature_names]:
            print(f"- {feature_name}")
    except Exception as feature_name_error:
        print("\nCould not extract feature names from preprocessor.")
        print(f"Reason: {feature_name_error}")