import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

# Load the model, encoder, and label binarizer at the start
model_path = 'model/random_forest_model.pkl'
encoder_path = 'model/encoder.pkl'
lb_path = 'model/label_binarizer.pkl'

# Load model, encoder, and label binarizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

with open(lb_path, 'rb') as lb_file:
    lb = pickle.load(lb_file)

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

@app.get("/")
def root():
    return {"message": "Welcome to the model inference API"}

@app.post("/predict")
def predict(data: List[InputData]):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([item.dict(by_alias=True) for item in data])

    # Process the input data
    X, _, _, _ = process_data(
        input_data, 
        categorical_features=["workclass", "education", "marital-status", "occupation", 
                              "relationship", "race", "sex", "native-country"], 
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make inference
    predictions = inference(model, X)
    return {"predictions": predictions.tolist()}
