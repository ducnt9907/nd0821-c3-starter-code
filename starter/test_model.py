import pytest
import numpy as np
import pandas as pd
from starter.ml.model import train_model, inference
from starter.ml.data import process_data
from sklearn.ensemble import RandomForestClassifier

# Sample data
data = pd.DataFrame({
    "age": [25, 32, 47, 51],
    "workclass": ["Private", "Self-emp-not-inc", "Private", "State-gov"],
    "education": ["Bachelors", "HS-grad", "HS-grad", "Bachelors"],
    "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Exec-managerial", "Adm-clerical"],
    "relationship": ["Not-in-family", "Husband", "Unmarried", "Husband"],
    "race": ["White", "White", "Black", "White"],
    "sex": ["Male", "Female", "Male", "Female"],
    "native-country": ["United-States", "United-States", "United-States", "United-States"],
    "salary": [">50K", "<=50K", ">50K", "<=50K"]
})

cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

def test_process_data():
    """Test the process_data function."""
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert X.shape[0] == data.shape[0] 
    assert y.shape[0] == data.shape[0]
    assert encoder is not None 
    assert lb is not None  

def test_train_model():
    """Test the train_model function."""
    # Process data and train model
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)  
    assert hasattr(model, "predict")  

def test_inference():
    """Test the inference function."""
    # Process data, train model and run inference
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)  
    assert isinstance(preds, np.ndarray) 
    assert set(preds).issubset([0, 1]) 
