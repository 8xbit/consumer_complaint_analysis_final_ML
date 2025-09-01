from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Define feature input schema using Pydantic
class ComplaintFeatures(BaseModel):
    Product: str
    Issue: str
    Consumer_disputed: str  # e.g., 'Yes', 'No', or 'Missing'

# Load saved model and encoder artifacts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

app = FastAPI()

@app.post('/predict_timely_response')
def predict_timely_response(features: ComplaintFeatures):
    # Convert input dictionary to model-ready format
    input_data = [[features.Product, features.Issue, features.Consumer_disputed]]

    # Encode categorical features
    input_encoded = encoder.transform(input_data)

    # Predict class (timely response yes/no)
    pred_class = model.predict(input_encoded)[0]

    # Probability for positive class
    pred_proba = model.predict_proba(input_encoded)[0][1]

    return {
        'timely_response_prediction': str(pred_class),
        'probability': pred_proba
    }
