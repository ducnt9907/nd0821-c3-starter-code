from fastapi.testclient import TestClient
from app import app

# Initialize the TestClient
client = TestClient(app)


# Test the GET route (root)
def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the model inference API"}


# Test the POST route with a valid input
def test_predict_below_50k():
    data = [
        {
            'age': 39,
            'workclass': ' State-gov',
            'fnlgt': 77516,
            'education': ' Bachelors',
            'education-num': 13,
            'marital-status': ' Never-married',
            'occupation': ' Adm-clerical',
            'relationship': ' Not-in-family',
            'race': ' White',
            'sex': ' Male',
            'capital-gain': 2174,
            'capital-loss': 0,
            'hours-per-week': 40,
            'native-country': ' United-States',
        }
    ]

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"predictions": [0]}


# Test the POST route with another valid input (different data for prediction)
def test_predict_above_50k():
    data = [
        {
            'age': 36,
            'workclass': 'Private',
            'fnlgt': 128757,
            'education': 'Bachelors',
            'education-num': 13,
            'marital-status': 'Married-civ-spouse',
            'occupation': 'Other-service',
            'relationship': 'Husband',
            'race': 'Black',
            'sex': 'Male',
            'capital-gain': 7298,
            'capital-loss': 0,
            'hours-per-week': 36,
            'native-country': 'United-States',
        }
    ]

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"predictions": [1]}
