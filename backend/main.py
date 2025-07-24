from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import numpy as np
import pickle
import datetime
import os

app = FastAPI(title="Nexora.ai – A.P.F.D.S", description="AI-powered fraud detection system", version="1.0")

# Check if model exists
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found. Make sure it's included in the deployment.")

# Load the trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Input schema
class Transaction(BaseModel):
    amount: float
    location: str
    time: str  # e.g., "2025-07-23 10:00:00"

# Output schema
class PredictionResponse(BaseModel):
    prediction: str

# Preprocessing function
def preprocess(transaction: Transaction):
    try:
        # Example encoding: location (Dubai = 1, others = 0)
        location_encoded = 1 if transaction.location.lower() == "dubai" else 0

        # Extract hour from time
        time_obj = datetime.datetime.strptime(transaction.time, "%Y-%m-%d %H:%M:%S")
        hour = time_obj.hour

        return np.array([[transaction.amount, location_encoded, hour]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    x = preprocess(transaction)
    try:
        prediction = model.predict(x)
        result = "Fraud" if prediction[0] == 1 else "Safe"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

# Home route (Frontend landing)
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Nexora.ai – Fraud Detection</title>
        </head>
        <body style="font-family: Arial; text-align: center; padding-top: 50px;">
            <h1>🛡️ Nexora.ai – A.P.F.D.S</h1>
            <p>Welcome to our AI-powered fraud detection system.</p>
            <a href="/docs" style="font-size: 18px; color: blue;">Visit API Documentation</a>
        </body>
    </html>
    """



