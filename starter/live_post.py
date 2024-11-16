import requests

# Define the live API URL
url = "https://nd0821-c3-starter-code-ca93.onrender.com/predict"

# Define a sample data payload
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

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.status_code)
print(response.json())
