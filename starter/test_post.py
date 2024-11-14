import requests

# Define the live API URL
url = "https://your-app-name.herokuapp.com/predict"

# Define a sample data payload
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
        "native-country": "United-States",
    }
]

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.status_code)
print(response.json())
