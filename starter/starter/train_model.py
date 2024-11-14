# Script to train machine learning model.
from ml.data import process_data
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from ml.evaluate import evaluate_on_slices

data = pd.read_csv('data/processed_census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _= process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
from ml.model import train_model, inference

model = train_model(X_train, y_train)

with open('model/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/encoder.pkl', 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)

with open('model/label_binarizer.pkl', 'wb') as lb_file:
    pickle.dump(lb, lb_file)

# Evaluate model
evaluate_on_slices(data, cat_features, label="salary")