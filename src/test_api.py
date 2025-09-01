"""
Test script for Titanic Prediction API.
Sends multiple sample passengers and prints + saves predictions.
"""

import requests
import pandas as pd
import os

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# API endpoint
URL = "http://127.0.0.1:8000/predict"

# Sample passengers
test_passengers = [
    {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25,
     "Embarked": "S", "FamilySize": 2, "IsSolo": 0, "Title": "Mr"},
    {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28,
     "Embarked": "C", "FamilySize": 2, "IsSolo": 0, "Title": "Mrs"},
    {"Pclass": 3, "Sex": "female", "Age": 26, "SibSp": 0, "Parch": 0, "Fare": 7.92,
     "Embarked": "Q", "FamilySize": 1, "IsSolo": 1, "Title": "Miss"},
    {"Pclass": 2, "Sex": "male", "Age": 35, "SibSp": 0, "Parch": 0, "Fare": 26.0,
     "Embarked": "S", "FamilySize": 1, "IsSolo": 1, "Title": "Mr"},
]

results = []

# Send requests
for idx, passenger in enumerate(test_passengers, start=1):
    try:
        response = requests.post(URL, json=passenger)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Error")
            results.append({"Passenger": idx, "Prediction": prediction})
        else:
            results.append({"Passenger": idx, "Prediction": f"Error {response.status_code}"})
    except Exception as e:
        results.append({"Passenger": idx, "Prediction": f"Request failed: {e}"})

# Print results
df = pd.DataFrame(results)
print("\n=== Titanic Survival Predictions ===")
print(df.to_string(index=False))

# Save results
df.to_csv("reports/api_test_results.csv", index=False)
print("✅ Predictions saved to reports/api_test_results.csv")
