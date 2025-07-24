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
