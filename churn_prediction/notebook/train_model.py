import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/churn.csv")

# Keep only required columns
df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]

# 🔥 FIX TotalCharges (THIS LINE IS CRITICAL)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values
df = df.dropna()

# Target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/churn_model.pkl")

print("✅ Model trained successfully (cleaned data)")
