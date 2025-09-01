import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

class ComplaintFeatures(BaseModel):
    Product: str
    Issue: str
    Consumer_disputed: str
    Company_response: str = ""  # default empty string

# Load the pipeline (encoder + model) and feature columns
try:
    pipeline = joblib.load("../models/best_pipeline.pkl")  # Updated filename
    feature_cols = joblib.load("../models/feature_columns.pkl")  # Load feature column names
    print("Model and features loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure the model files exist in ../models/")
    pipeline = None
    feature_cols = None

app = FastAPI(title="Complaint Timely Response Predictor", version="1.0.0")

@app.get("/")
def root():
    return {"message": "Complaint Timely Response Prediction API"}

@app.post('/predict_timely_response')
def predict_timely_response(features: ComplaintFeatures):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create input DataFrame with the exact column names used in training
        input_df = pd.DataFrame([{
            "Product": features.Product,
            "Issue": features.Issue,
            "Consumer disputed?": features.Consumer_disputed,  # Note the question mark
            "Company response": features.Company_response
        }])
        
        # Ensure columns are in the same order as training
        input_df = input_df[feature_cols]
        
        # Make predictions using the pipeline
        pred_class = pipeline.predict(input_df)[0]
        pred_proba = pipeline.predict_proba(input_df)[0]
        
        # Get the probability for the positive class (timely response = 1)
        # Note: pred_class is encoded (0 or 1), we need to map back to meaningful labels
        timely_response_labels = {0: "No", 1: "Yes"}  # Adjust based on your encoding
        
        return {
            'timely_response_prediction': timely_response_labels.get(pred_class, str(pred_class)),
            'probability_timely': float(pred_proba[1]),  # Probability of timely response
            'probability_not_timely': float(pred_proba[0]),  # Probability of not timely response
            'confidence': float(max(pred_proba))  # Highest probability as confidence measure
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

