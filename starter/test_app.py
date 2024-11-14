import pytest
from fastapi.testclient import TestClient
from app import app  # Assuming the FastAPI app is named 'app.py'

# Initialize the TestClient
client = TestClient(app)

# Test the GET route (root)
def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the model inference API"}

# Test the POST route with a valid input
def test_post_predict_valid():
    data = [
        {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    ]
    
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "predictions" in response.json()

# Test the POST route with another valid input (different data for prediction)
def test_post_predict_another_valid():
    data = [
        {
            "age": 25,
            "workclass": "Private",
            "fnlgt": 234567,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 5000,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "India"
        }
    ]
    
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "predictions" in response.json()

