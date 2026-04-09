"""
AR/AP FastAPI Backend
=====================
Serves real ML model predictions to the React dashboard.

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Resolve paths relative to this file — no hardcoding needed
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

app = FastAPI(title="AR/AP ML API")

# Allow requests from the React dev server (localhost and 127.0.0.1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────
def load_models():
    required = [
        "linear_regression.joblib", "scaler_lr.joblib",
        "logistic_regression.joblib", "scaler_logr.joblib",
        "random_forest.joblib",
        "le_client.joblib", "le_resource.joblib",
        "le_risk.joblib", "le_aging.joblib",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise RuntimeError(f"Missing model files: {missing}. Run ar_ap_ml.py first.")

    return {
        "lr":          joblib.load(os.path.join(MODEL_DIR, "linear_regression.joblib")),
        "scaler_lr":   joblib.load(os.path.join(MODEL_DIR, "scaler_lr.joblib")),
        "logr":        joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib")),
        "scaler_logr": joblib.load(os.path.join(MODEL_DIR, "scaler_logr.joblib")),
        "rf":          joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib")),
        "le_client":   joblib.load(os.path.join(MODEL_DIR, "le_client.joblib")),
        "le_resource": joblib.load(os.path.join(MODEL_DIR, "le_resource.joblib")),
        "le_risk":     joblib.load(os.path.join(MODEL_DIR, "le_risk.joblib")),
        "le_aging":    joblib.load(os.path.join(MODEL_DIR, "le_aging.joblib")),
    }

MODELS = load_models()
print("✓ All models loaded.")

FEATURES = ["client_enc", "resource_enc", "risk_enc",
            "billing_rate", "working_days", "invoice_amount", "month_num",
            "contract_terms", "relationship_age_months", "payment_history_avg",
            "days_since_last_invoice", "invoice_seq", "reminder_sent"]

CONTRACT_TERMS = {
    "Infosys BPO": 30, "Accenture India": 30, "Cognizant": 30,
    "Wipro Digital": 45, "HCL Services": 45, "Capgemini": 60,
    "TechSpark Ltd": 60, "XYZ Corp": 90,
}
RELATIONSHIP_AGE = {
    "Infosys BPO": 36, "Accenture India": 48, "Cognizant": 24,
    "Wipro Digital": 18, "HCL Services": 30, "Capgemini": 12,
    "TechSpark Ltd": 6, "XYZ Corp": 3,
}

# ── Request schema ────────────────────────────────────────────
class InvoiceInput(BaseModel):
    client: str
    client_risk: str
    resource: str
    billing_rate: float
    working_days: int
    month_num: int              # 1–12
    payment_history_avg: float = 0.0   # avg days client took to pay historically
    days_since_last_invoice: int = 30  # days since last invoice for this client
    invoice_seq: int = 1               # how many invoices sent to this client so far
    reminder_sent: int = 0             # 1 if reminder was sent, 0 otherwise

# ── Predict endpoint ──────────────────────────────────────────
@app.post("/predict")
def predict(invoice: InvoiceInput):
    try:
        le_client   = MODELS["le_client"]
        le_resource = MODELS["le_resource"]
        le_risk     = MODELS["le_risk"]
        le_aging    = MODELS["le_aging"]

        if invoice.client not in le_client.classes_:
            raise HTTPException(400, f"Unknown client: {invoice.client}. Known: {list(le_client.classes_)}")
        if invoice.resource not in le_resource.classes_:
            raise HTTPException(400, f"Unknown resource: {invoice.resource}. Known: {list(le_resource.classes_)}")
        if invoice.client_risk not in le_risk.classes_:
            raise HTTPException(400, f"Unknown risk: {invoice.client_risk}. Known: {list(le_risk.classes_)}")

        invoice_amount = invoice.billing_rate * invoice.working_days

        # Use defaults for new features if not provided
        contract_terms       = CONTRACT_TERMS.get(invoice.client, 30)
        relationship_age     = RELATIONSHIP_AGE.get(invoice.client, 12)
        payment_history_avg  = invoice.payment_history_avg or float(contract_terms)

        features = np.array([[
            le_client.transform([invoice.client])[0],
            le_resource.transform([invoice.resource])[0],
            le_risk.transform([invoice.client_risk])[0],
            invoice.billing_rate,
            invoice.working_days,
            invoice_amount,
            invoice.month_num,
            contract_terms,
            relationship_age,
            payment_history_avg,
            invoice.days_since_last_invoice,
            invoice.invoice_seq,
            invoice.reminder_sent,
        ]])

        # Model 1: Linear Regression → days to payment
        X_lr      = MODELS["scaler_lr"].transform(features)
        pred_days = round(float(MODELS["lr"].predict(X_lr)[0]), 1)

        # Model 2: Logistic Regression → dispute probability
        X_logr            = MODELS["scaler_logr"].transform(features)
        dispute_prob      = round(float(MODELS["logr"].predict_proba(X_logr)[0][1]) * 100, 1)
        dispute_predicted = int(MODELS["logr"].predict(X_logr)[0])

        # Model 3: Random Forest → aging bucket
        pred_aging_enc    = MODELS["rf"].predict(features)[0]
        pred_aging_bucket = le_aging.inverse_transform([pred_aging_enc])[0]

        at_risk = int(dispute_prob > 40 or pred_aging_bucket in ["61-90", "90+"])

        return {
            "predicted_days_to_payment": pred_days,
            "dispute_probability_pct":   dispute_prob,
            "dispute_predicted":         dispute_predicted,
            "predicted_aging_bucket":    pred_aging_bucket,
            "invoice_amount":            invoice_amount,
            "at_risk":                   at_risk,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.get("/metrics")
def metrics():
    path = os.path.join(OUTPUT_DIR, "overall_kpis.json")
    if not os.path.exists(path):
        raise HTTPException(404, "overall_kpis.json not found. Run ar_ap_ml.py first.")
    with open(path) as f:
        return json.load(f)

def _csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} not found. Run ar_ap_ml.py first.")
    return pd.read_csv(path).round(2).to_dict(orient="records")

@app.get("/data/clients")
def data_clients():
    return _csv("kpi_by_client.csv")

@app.get("/data/aging")
def data_aging():
    return _csv("kpi_by_aging.csv")

@app.get("/data/resources")
def data_resources():
    return _csv("kpi_by_resource.csv")

@app.get("/data/monthly")
def data_monthly():
    return _csv("kpi_by_month.csv")
