from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import joblib
import pandas as pd

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🔥 PREDICT ROUTE (THIS WAS MISSING)
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    tenure: int = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)
):
    # Create dataframe (pipeline expects dataframe)
    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = f"Customer is likely to CHURN ❌ (Probability: {probability:.2%})"
    else:
        result = f"Customer will STAY ✅ (Probability: {1 - probability:.2%})"


    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result
        }
    )
