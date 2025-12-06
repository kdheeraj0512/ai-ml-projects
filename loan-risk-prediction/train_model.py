import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

def generate_synthetic_loan_data(n=5000):
    rng = np.random.default_rng(42)
    income = rng.normal(80000, 20000, n).clip(20000, 200000)
    credit_score = rng.normal(700, 50, n).clip(500, 850)
    loan_amount = rng.normal(250000, 70000, n).clip(10000, 900000)
    tenure = rng.integers(5, 30, n)
    dti = loan_amount / (income * tenure)
    prob = 0.2 + (dti * 0.9) - ((credit_score - 650) / 1000)
    prob = np.clip(prob, 0, 1)
    default = rng.binomial(1, prob)
    return pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "tenure": tenure,
        "dti": dti,
        "default": default
    })

def main():
    df = generate_synthetic_loan_data()
    X = df[["income", "credit_score", "loan_amount", "tenure", "dti"]]
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = GradientBoostingClassifier()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    dump({"model": model, "scaler": scaler}, "model.joblib")
    print("Saved model.joblib")

if __name__ == "__main__":
    main()
