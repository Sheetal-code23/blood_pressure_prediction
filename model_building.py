import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.model_building import train_linear_model  # your own function

# Load dataset
df = pd.read_csv("data/blood_pressure.csv")

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Feature and target selection
X = df[["Age", "Gender", "Weight", "BMI", "Smoking", "ExerciseHours", "StressLevel"]]  # 7 features
y = df["SystolicBP"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model, X_test, y_test = train_linear_model(X_scaled, y)

# Save model and scaler
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully.")
