# Customer Churn Prediction – ML + FastAPI

This project is an end-to-end Machine Learning web application that predicts customer churn and explains model decisions.

## Features
- Machine Learning model trained using Random Forest
- FastAPI backend for real-time predictions
- Web UI for user input
- Prediction probability output
- Feature importance & SHAP-based explainability

## Tech Stack
Python, Scikit-learn, FastAPI, Pandas, NumPy, SHAP

## How to Run Locally
```bash
pip install -r requirements.txt
python notebooks/train_model.py
uvicorn app.main:app --reload
