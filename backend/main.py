from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model (make sure model.pkl is present)
model = joblib.load("model.pkl")

class Transaction(BaseModel):
    amount: float
    location: str
    time: str

@app.post("/predict")
def predict(transaction: Transaction):
    # For demo: dummy encoding
    x = np.array([[transaction.amount]])
    prediction = model.predict(x)
    result = "Fraud" if prediction[0] == 1 else "Safe"
    return {"prediction": result}
