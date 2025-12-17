import shap
import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")

X = pd.DataFrame(
    [[12, 75.0, 900.0]],
    columns=["tenure", "MonthlyCharges", "TotalCharges"]
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X
)
