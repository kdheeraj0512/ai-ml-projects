import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

def generate_sensor_data(n=4000):
    rng = np.random.default_rng(42)
    vibration = rng.normal(50, 10, n)
    temperature = rng.normal(70, 15, n)
    pressure = rng.normal(30, 5, n)
    wear = (vibration*0.04)+(temperature*0.03)+(pressure*0.02)
    fail_prob = np.clip((wear-5)/10,0,1)
    failed = rng.binomial(1, fail_prob)
    return pd.DataFrame({
        "vibration": vibration,
        "temperature": temperature,
        "pressure": pressure,
        "failure": failed
    })

def main():
    df = generate_sensor_data()
    X = df[["vibration","temperature","pressure"]]
    y = df["failure"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,preds))
    print(classification_report(y_test,preds))
    dump(model,"maintenance_model.joblib")
    print("Model saved -> maintenance_model.joblib")

if __name__ == "__main__":
    main()
