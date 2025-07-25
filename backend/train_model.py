﻿import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Load dataset
df = pd.read_csv("data/synthetic_fraud_dataset.csv")


# Handle missing values if any
df = df.dropna()


# Encode categorical columns
label_encoders = {}
categorical_columns = ['location', 'merchant', 'device_type']


for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Feature and label split
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


print("✅ Model trained and saved as model.pkl")


# Optionally save label encoders for consistent API decoding
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
