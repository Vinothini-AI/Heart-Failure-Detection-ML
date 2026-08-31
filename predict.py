import pandas as pd
import joblib

# Load trained model
model = joblib.load(
    "models/heart_failure_model.pkl"
)

# Create a new patient
patient = pd.DataFrame({
    "age": [60],
    "anaemia": [0],
    "creatinine_phosphokinase": [250],
    "diabetes": [0],
    "ejection_fraction": [40],
    "high_blood_pressure": [1],
    "platelets": [250000],
    "serum_creatinine": [1.2],
    "serum_sodium": [137],
    "sex": [1],
    "smoking": [0],
    "time": [100]
})

# Prediction
prediction = model.predict(patient)

probability = model.predict_proba(patient)

if prediction[0] == 1:
    print("Prediction: DEATH EVENT")
else:
    print("Prediction: NO DEATH EVENT")

print(
    "Probability of death event:",
    round(probability[0][1] * 100, 2),
    "%"
)