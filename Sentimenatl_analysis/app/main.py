from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import numpy as np

app = FastAPI(title="Sentiment Analysis App")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")

model = joblib.load(MODEL_PATH)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    probs = model.predict_proba([text])[0]
    labels = model.classes_

    max_prob = probs.max()
    prediction = labels[probs.argmax()]

    # 🔥 Neutral confidence rule
    if max_prob < 0.55:
        prediction = "Neutral"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": f"{max_prob:.2%}",
            "input_text": text
        }
    )
