from sklearn.metrics import accuracy_score
from ml.model import inference


def evaluate_on_slices(model, X_test, y_test, test_data, cat_features):
    """
    Evaluate model performance on slices of the data based on categorical features.

    Parameters:
    - model: Trained model to evaluate.
    - X_test: Processed features of the test set (np.array).
    - y_test: True labels of the test set (np.array).
    - test_data: Original test DataFrame before processing.
    - cat_features: List of categorical features to slice on.
    """

    # Ensure test_data index matches X_test/y_test after processing
    test_data = test_data.reset_index(drop=True)

    for feature in cat_features:
        print(f"Evaluating performance on slices of {feature}...")

        # Get unique values of the current categorical feature
        unique_values = test_data[feature].unique()

        for value in unique_values:
            # Get indices of rows where feature equals the current value
            slice_indices = test_data.index[test_data[feature] == value].tolist()

            # Select corresponding slices of X_test and y_test
            X_slice = X_test[slice_indices]
            y_slice = y_test[slice_indices]

            # Run inference and calculate accuracy for the slice
            preds = inference(model, X_slice)
            accuracy = accuracy_score(y_slice, preds)
            print(f"Accuracy for {feature} = {value}: {accuracy:.4f}")
