from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class Transaction(BaseModel):
    amount: float
    location: str
    time: str

# Output schema
class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # Dummy encoding — modify as per real feature logic
    x = np.array([[transaction.amount]])
    prediction = model.predict(x)
    result = "Fraud" if prediction[0] == 1 else "Safe"
    return {"prediction": result}

from fastapi.responses import HTMLResponse

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




