from sklearn.metrics import accuracy_score
from ml.model import inference


def evaluate_on_slices(model, X_test, y_test, test_data, cat_features):
    """
    Evaluate model performance on slices of the data based on categorical features.

    Parameters:
    - model: Trained model to evaluate.
    - X_test: Processed features of the test set.
    - y_test: True labels of the test set.
    - test_data:
    Original test DataFrame before processing (for filtering based on categorical values).
    - cat_features: List of categorical features to slice on.
    """

    for feature in cat_features:
        print(f"Evaluating performance on slices of {feature}...")

        # Loop through each unique value in the feature
        unique_values = test_data[feature].unique()

        for value in unique_values:
            # Find the indices where the test data matches the current value
            # of the categorical feature
            slice_indices = test_data[test_data[feature] == value].index
            X_slice = X_test[slice_indices]
            y_slice = y_test[slice_indices]

            # Make predictions on this slice
            preds = inference(model, X_slice)

            # Calculate accuracy for this slice
            accuracy = accuracy_score(y_slice, preds)
            print(f"Accuracy for {feature} = {value}: {accuracy:.4f}")
