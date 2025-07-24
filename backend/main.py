from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import pickle
import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Nexora.ai – A.P.F.D.S", description="AI-powered fraud detection system", version="1.0")

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoders.pkl"
DATA_PATH = "data/synthetic_fraud_dataset.csv"

# Auto-train if no model found
if not os.path.exists(MODEL_PATH):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found for training. Please upload it to 'data/synthetic_fraud_dataset.csv'.")

    df = pd.read_csv(DATA_PATH)
    df["hour"] = pd.to_datetime(df["time"]).dt.hour

    # Encode fields
    label_encoders = {}
    for col in ["location", "transaction_type", "device_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features & target
    X = df[["amount", "location", "hour", "transaction_type", "device_type"]]
    y = df["is_fraud"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoders, f)
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoders = pickle.load(f)

# Request body
class Transaction(BaseModel):
    amount: float
    location: str
    time: str
    transaction_type: str
    device_type: str

class PredictionResponse(BaseModel):
    prediction: str

def preprocess(transaction: Transaction):
    try:
        hour = datetime.datetime.strptime(transaction.time, "%Y-%m-%d %H:%M:%S").hour
        location_encoded = label_encoders["location"].transform([transaction.location.lower()])[0]
        trans_type_encoded = label_encoders["transaction_type"].transform([transaction.transaction_type.lower()])[0]
        device_encoded = label_encoders["device_type"].transform([transaction.device_type.lower()])[0]

        return np.array([[transaction.amount, location_encoded, hour, trans_type_encoded, device_encoded]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    x = preprocess(transaction)
    try:
        prediction = model.predict(x)
        result = "Fraud" if prediction[0] == 1 else "Safe"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Nexora.ai – Fraud Detection</title></head>
        <body style="text-align: center; padding-top: 50px;">
            <h1>🛡️ Nexora.ai – A.P.F.D.S</h1>
            <p>Welcome to our AI-powered fraud detection system.</p>
            <a href="/docs">Go to API Docs</a>
        </body>
    </html>
    """



