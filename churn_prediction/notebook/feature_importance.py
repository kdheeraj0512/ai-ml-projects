import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/churn_model.pkl")

# Feature names (order matters)
features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Extract importance
importance = model.feature_importances_

# Display
for f, i in zip(features, importance):
    print(f"{f}: {i:.4f}")
