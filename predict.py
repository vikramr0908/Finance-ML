"""
AR/AP Prediction Script
========================
Loads trained models and runs predictions on new invoice data.

Usage:
    python predict.py --input new_invoices.csv --output predictions.csv

Required columns in input CSV:
    client, client_risk, resource, billing_rate, working_days, invoice_amount, invoice_date

Outputs three predictions per invoice:
    - predicted_days_to_payment   (Linear Regression)
    - dispute_probability         (Logistic Regression)
    - dispute_predicted           (Logistic Regression)
    - predicted_aging_bucket      (Random Forest)
    - at_risk                     (dispute_prob > 0.4 OR aging bucket 61-90/90+)
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEATURES  = ["client_enc", "resource_enc", "risk_enc",
             "billing_rate", "working_days", "invoice_amount", "month_num"]

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
        print(f"ERROR: Missing model files in {MODEL_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun ar_ap_ml.py first to train and save the models.")
        sys.exit(1)

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


def encode_column(series, encoder, col_name):
    """Encode a column, handling unseen labels gracefully."""
    known = set(encoder.classes_)
    unseen = set(series.unique()) - known
    if unseen:
        print(f"WARNING: Unknown {col_name} values will be skipped: {unseen}")
        series = series.where(series.isin(known), other=encoder.classes_[0])
    return encoder.transform(series)


def predict(df, models):
    df = df.copy()

    # Feature engineering
    df["client_enc"]   = encode_column(df["client"],      models["le_client"],   "client")
    df["resource_enc"] = encode_column(df["resource"],    models["le_resource"], "resource")
    df["risk_enc"]     = encode_column(df["client_risk"], models["le_risk"],     "client_risk")
    df["month_num"]    = pd.to_datetime(df["invoice_date"]).dt.month

    X = df[FEATURES]

    # Model 1: Linear Regression → days_to_payment
    X_s = models["scaler_lr"].transform(X)
    df["predicted_days_to_payment"] = models["lr"].predict(X_s).round(1)

    # Model 2: Logistic Regression → dispute probability
    X_s2 = models["scaler_logr"].transform(X)
    df["dispute_probability"] = models["logr"].predict_proba(X_s2)[:, 1].round(4)
    df["dispute_predicted"]   = models["logr"].predict(X_s2)

    # Model 3: Random Forest → aging bucket
    df["predicted_aging_bucket"] = models["le_aging"].inverse_transform(models["rf"].predict(X))

    # Derived flag
    df["at_risk"] = (
        (df["dispute_probability"] > 0.4) |
        (df["predicted_aging_bucket"].isin(["61-90", "90+"]))
    ).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="Run AR/AP ML predictions on new invoice data.")
    parser.add_argument("--input",  required=True,  help="Path to input CSV file")
    parser.add_argument("--output", required=False, help="Path to output CSV file (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading models from {MODEL_DIR}...")
    models = load_models()
    print("  Models loaded.")

    print(f"\nReading input: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows found.")

    required_cols = ["client", "client_risk", "resource",
                     "billing_rate", "working_days", "invoice_amount", "invoice_date"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Input CSV is missing columns: {missing_cols}")
        sys.exit(1)

    print("\nRunning predictions...")
    result = predict(df, models)

    output_cols = list(df.columns) + [
        "predicted_days_to_payment",
        "dispute_probability",
        "dispute_predicted",
        "predicted_aging_bucket",
        "at_risk",
    ]
    result = result[output_cols]

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")
    else:
        print("\n── Predictions ──")
        print(result[[
            "predicted_days_to_payment",
            "dispute_probability",
            "dispute_predicted",
            "predicted_aging_bucket",
            "at_risk",
        ]].to_string(index=True))

    at_risk_count = result["at_risk"].sum()
    high_dispute  = (result["dispute_probability"] > 0.4).sum()
    print(f"\nSummary:")
    print(f"  Total invoices   : {len(result)}")
    print(f"  At-risk invoices : {at_risk_count}")
    print(f"  High dispute prob: {high_dispute}")
    print(f"  Avg predicted days to payment: {result['predicted_days_to_payment'].mean():.1f}")


if __name__ == "__main__":
    main()
