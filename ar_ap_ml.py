"""
AR/AP ML Pipeline
=================
Models:
  1. Linear Regression   → Predict days_to_payment (continuous)
  2. Logistic Regression → Predict dispute probability (binary)
  3. Random Forest       → Predict aging_bucket (multi-class: 0-30, 31-60, 61-90, 90+)

Outputs (under ./output/ next to this script):
  - ar_ap_data.csv        : raw synthetic dataset
  - ar_ap_predictions.csv : dataset with all model predictions + KPIs
  - kpi_*.csv             : KPI tables by client, month, aging, resource
  - model_metrics.txt     : accuracy / R2 / classification reports
  - overall_kpis.json     : rolled-up KPIs

Trained models are saved under ./models/
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)

np.random.seed(42)
random.seed(42)

# All outputs live next to this script (portable across machines)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "output")
MODEL_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

CLIENTS = [
    ("Infosys BPO",      "Low",    45),   # (name, risk_profile, avg_days_to_pay)
    ("TechSpark Ltd",    "High",   82),
    ("Accenture India",  "Low",    32),
    ("XYZ Corp",         "High",   95),
    ("Wipro Digital",    "Medium", 58),
    ("HCL Services",     "Medium", 51),
    ("Cognizant",        "Low",    38),
    ("Capgemini",        "Medium", 63),
]
    
RESOURCES = [
    ("Priya Sharma",    "Senior Analyst",   4800),
    ("Rahul Mehta",     "Data Engineer",    5200),
    ("Arun Kumar",      "Consultant",       6500),
    ("Divya Nair",      "Junior Analyst",   3200),
    ("Karan Joshi",     "Senior Engineer",  7000),
    ("Sneha Rao",       "Analyst",          4200),
    ("Vikram Singh",    "Lead Consultant",  8500),
    ("Meera Pillai",    "Data Analyst",     4600),
    ("Arjun Patel",     "Engineer",         5800),
    ("Lakshmi Iyer",    "Consultant",       6200),
]

def generate_invoice_date(year=2024):
    month = random.randint(1, 12)
    day = random.randint(1, 20)
    return datetime(year, month, day)

def compute_aging_bucket(days):
    if days <= 30:   return "0-30"
    elif days <= 60: return "31-60"
    elif days <= 90: return "61-90"
    else:            return "90+"

def generate_dataset(n=500):
    rows = []
    for i in range(n):
        client_name, risk, avg_days = random.choice(CLIENTS)
        resource_name, role, base_rate = random.choice(RESOURCES)

        # billing rate with some noise
        billing_rate = base_rate + np.random.randint(-500, 800)
        working_days = np.random.randint(15, 24)
        invoice_amount = billing_rate * working_days

        invoice_date = generate_invoice_date(2024)
        due_date = invoice_date + timedelta(days=30)

        # days_to_payment driven by client risk + noise
        noise = np.random.normal(0, 12)
        days_to_payment = max(5, int(avg_days + noise))

        payment_date = invoice_date + timedelta(days=days_to_payment)

        # dispute: higher for high-risk clients and large invoices
        dispute_prob = 0.05
        if risk == "High":    dispute_prob += 0.25
        if risk == "Medium":  dispute_prob += 0.10
        if invoice_amount > 120000: dispute_prob += 0.08
        disputed = int(np.random.rand() < dispute_prob)

        # payment outcome
        if disputed:
            short_chance = 0.6
        else:
            short_chance = 0.15
        r = np.random.rand()
        if r < short_chance:
            amount_received = round(invoice_amount * np.random.uniform(0.75, 0.97), 2)
            payment_status = "Short"
        elif r < short_chance + 0.05:
            amount_received = round(invoice_amount * np.random.uniform(1.01, 1.05), 2)
            payment_status = "Excess"
        else:
            amount_received = invoice_amount
            payment_status = "Exact"

        aging_bucket = compute_aging_bucket(days_to_payment)
        collection_efficiency = round((amount_received / invoice_amount) * 100, 2)

        rows.append({
            "invoice_id":           f"INV-2024-{i+1:04d}",
            "client":               client_name,
            "client_risk":          risk,
            "resource":             resource_name,
            "role":                 role,
            "billing_rate":         billing_rate,
            "working_days":         working_days,
            "invoice_amount":       invoice_amount,
            "invoice_date":         invoice_date.strftime("%Y-%m-%d"),
            "due_date":             due_date.strftime("%Y-%m-%d"),
            "payment_date":         payment_date.strftime("%Y-%m-%d"),
            "days_to_payment":      days_to_payment,
            "amount_received":      amount_received,
            "disputed":             disputed,
            "payment_status":       payment_status,
            "aging_bucket":         aging_bucket,
            "collection_efficiency": collection_efficiency,
            "month":                invoice_date.strftime("%Y-%m"),
        })

    return pd.DataFrame(rows)

print("=" * 60)
print("  AR/AP ML PIPELINE")
print("=" * 60)
print("\n[1] Generating synthetic dataset...")
df = generate_dataset(500)
df.to_csv(os.path.join(OUTPUT_DIR, "ar_ap_data.csv"), index=False)
print(f"    ✓ {len(df)} invoices generated across {df['client'].nunique()} clients, {df['resource'].nunique()} resources")
print(f"    ✓ Date range: {df['invoice_date'].min()} → {df['invoice_date'].max()}")
print(f"    ✓ Total invoice value: ₹{df['invoice_amount'].sum():,.0f}")
print(f"    ✓ Dispute rate: {df['disputed'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

le_client = LabelEncoder()
le_resource = LabelEncoder()
le_risk = LabelEncoder()

df["client_enc"]   = le_client.fit_transform(df["client"])
df["resource_enc"] = le_resource.fit_transform(df["resource"])
df["risk_enc"]     = le_risk.fit_transform(df["client_risk"])
df["month_num"]    = pd.to_datetime(df["invoice_date"]).dt.month

FEATURES = ["client_enc", "resource_enc", "risk_enc",
            "billing_rate", "working_days", "invoice_amount", "month_num"]

metrics_lines = []

# ─────────────────────────────────────────────
# 3. MODEL 1 — LINEAR REGRESSION (days_to_payment)
# ─────────────────────────────────────────────

print("\n[2] Model 1: Linear Regression → days_to_payment")

X_lr = df[FEATURES]
y_lr = df["days_to_payment"]

X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

r2  = r2_score(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)

print(f"    R²  : {r2:.4f}")
print(f"    MAE : {mae:.2f} days")

df["predicted_days_to_payment"] = lr.predict(scaler.transform(X_lr)).round(1)

metrics_lines += [
    "=" * 60,
    "MODEL 1: Linear Regression — Predict Days to Payment",
    "=" * 60,
    f"R² Score : {r2:.4f}",
    f"MAE      : {mae:.2f} days",
    "",
    "Feature Coefficients:",
]
for feat, coef in zip(FEATURES, lr.coef_):
    metrics_lines.append(f"  {feat:<20s}: {coef:+.4f}")
metrics_lines.append("")

# ─────────────────────────────────────────────
# 4. MODEL 2 — LOGISTIC REGRESSION (disputed)
# ─────────────────────────────────────────────

print("\n[3] Model 2: Logistic Regression → dispute_probability")

X_log = df[FEATURES]
y_log = df["disputed"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X_train2_s = scaler2.fit_transform(X_train2)
X_test2_s  = scaler2.transform(X_test2)

logr = LogisticRegression(max_iter=500, random_state=42)
logr.fit(X_train2_s, y_train2)
y_pred_log = logr.predict(X_test2_s)
y_prob_log  = logr.predict_proba(X_test2_s)[:, 1]

acc2 = accuracy_score(y_test2, y_pred_log)
print(f"    Accuracy: {acc2:.4f}")
print(f"    Report:\n{classification_report(y_test2, y_pred_log, target_names=['No Dispute','Disputed'])}")

df["dispute_probability"] = logr.predict_proba(scaler2.transform(X_log))[:, 1].round(4)
df["dispute_predicted"]   = logr.predict(scaler2.transform(X_log))

metrics_lines += [
    "=" * 60,
    "MODEL 2: Logistic Regression — Dispute Prediction",
    "=" * 60,
    f"Accuracy : {acc2:.4f}",
    "",
    classification_report(y_test2, y_pred_log, target_names=["No Dispute", "Disputed"]),
]

# ─────────────────────────────────────────────
# 5. MODEL 3 — RANDOM FOREST (aging_bucket)
# ─────────────────────────────────────────────

print("\n[4] Model 3: Random Forest → aging_bucket classification")

le_aging = LabelEncoder()
df["aging_enc"] = le_aging.fit_transform(df["aging_bucket"])

X_rf = df[FEATURES]
y_rf = df["aging_enc"]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train3, y_train3)
y_pred_rf = rf.predict(X_test3)

acc3 = accuracy_score(y_test3, y_pred_rf)
bucket_labels = le_aging.classes_
print(f"    Accuracy: {acc3:.4f}")
print(f"    Report:\n{classification_report(y_test3, y_pred_rf, target_names=bucket_labels)}")

df["predicted_aging_bucket"] = le_aging.inverse_transform(rf.predict(X_rf))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

metrics_lines += [
    "=" * 60,
    "MODEL 3: Random Forest — Aging Bucket Classification",
    "=" * 60,
    f"Accuracy : {acc3:.4f}",
    "",
    classification_report(y_test3, y_pred_rf, target_names=bucket_labels),
    "Feature Importances:",
]
for feat, imp in importances.items():
    metrics_lines.append(f"  {feat:<20s}: {imp:.4f}")
metrics_lines.append("")

# ─────────────────────────────────────────────
# 6. KPI CALCULATIONS
# ─────────────────────────────────────────────

print("\n[5] Computing KPIs...")

df["amount_outstanding"] = df["invoice_amount"] - df["amount_received"]
df["overdue"]            = (df["days_to_payment"] > 30).astype(int)
df["at_risk"]            = ((df["dispute_probability"] > 0.4) | (df["aging_bucket"].isin(["61-90","90+"]))).astype(int)

# ── KPI 1: By Client
kpi_client = df.groupby("client").agg(
    total_invoices        = ("invoice_id",              "count"),
    total_invoiced        = ("invoice_amount",          "sum"),
    total_collected       = ("amount_received",         "sum"),
    avg_days_to_payment   = ("days_to_payment",         "mean"),
    predicted_avg_days    = ("predicted_days_to_payment","mean"),
    dispute_count         = ("disputed",                "sum"),
    dispute_rate_pct      = ("disputed",                "mean"),
    collection_efficiency = ("collection_efficiency",   "mean"),
    at_risk_invoices      = ("at_risk",                 "sum"),
    amount_outstanding    = ("amount_outstanding",      "sum"),
    avg_dispute_prob      = ("dispute_probability",     "mean"),
).reset_index()
kpi_client["dispute_rate_pct"]      = (kpi_client["dispute_rate_pct"] * 100).round(2)
kpi_client["collection_efficiency"] = kpi_client["collection_efficiency"].round(2)
kpi_client["avg_days_to_payment"]   = kpi_client["avg_days_to_payment"].round(1)
kpi_client["predicted_avg_days"]    = kpi_client["predicted_avg_days"].round(1)
kpi_client["avg_dispute_prob"]      = (kpi_client["avg_dispute_prob"] * 100).round(2)

# ── KPI 2: By Month
kpi_month = df.groupby("month").agg(
    total_invoices        = ("invoice_id",              "count"),
    total_invoiced        = ("invoice_amount",          "sum"),
    total_collected       = ("amount_received",         "sum"),
    avg_days_to_payment   = ("days_to_payment",         "mean"),
    predicted_avg_days    = ("predicted_days_to_payment","mean"),
    dispute_rate_pct      = ("disputed",                "mean"),
    collection_efficiency = ("collection_efficiency",   "mean"),
    at_risk_invoices      = ("at_risk",                 "sum"),
).reset_index()
kpi_month["dispute_rate_pct"]      = (kpi_month["dispute_rate_pct"] * 100).round(2)
kpi_month["collection_efficiency"] = kpi_month["collection_efficiency"].round(2)
kpi_month["avg_days_to_payment"]   = kpi_month["avg_days_to_payment"].round(1)

# ── KPI 3: By Aging Bucket
kpi_aging = df.groupby("aging_bucket").agg(
    total_invoices        = ("invoice_id",              "count"),
    total_invoiced        = ("invoice_amount",          "sum"),
    total_collected       = ("amount_received",         "sum"),
    dispute_rate_pct      = ("disputed",                "mean"),
    collection_efficiency = ("collection_efficiency",   "mean"),
    amount_outstanding    = ("amount_outstanding",      "sum"),
    avg_dispute_prob      = ("dispute_probability",     "mean"),
).reset_index()
kpi_aging["dispute_rate_pct"]      = (kpi_aging["dispute_rate_pct"] * 100).round(2)
kpi_aging["collection_efficiency"] = kpi_aging["collection_efficiency"].round(2)
kpi_aging["avg_dispute_prob"]      = (kpi_aging["avg_dispute_prob"] * 100).round(2)

# ── KPI 4: By Role / Resource
kpi_resource = df.groupby(["resource","role"]).agg(
    total_invoices        = ("invoice_id",    "count"),
    total_invoiced        = ("invoice_amount","sum"),
    avg_billing_rate      = ("billing_rate",  "mean"),
    avg_working_days      = ("working_days",  "mean"),
    avg_days_to_payment   = ("days_to_payment","mean"),
    dispute_rate_pct      = ("disputed",      "mean"),
    collection_efficiency = ("collection_efficiency","mean"),
).reset_index()
kpi_resource["dispute_rate_pct"]      = (kpi_resource["dispute_rate_pct"] * 100).round(2)
kpi_resource["collection_efficiency"] = kpi_resource["collection_efficiency"].round(2)
kpi_resource["avg_days_to_payment"]   = kpi_resource["avg_days_to_payment"].round(1)

# ── Overall KPIs
overall = {
    "total_invoices":           len(df),
    "total_invoiced_inr":       round(df["invoice_amount"].sum(), 2),
    "total_collected_inr":      round(df["amount_received"].sum(), 2),
    "overall_collection_eff":   round(df["collection_efficiency"].mean(), 2),
    "overall_dispute_rate_pct": round(df["disputed"].mean() * 100, 2),
    "avg_days_to_payment":      round(df["days_to_payment"].mean(), 1),
    "predicted_avg_days":       round(df["predicted_days_to_payment"].mean(), 1),
    "at_risk_invoices":         int(df["at_risk"].sum()),
    "at_risk_amount_inr":       round(df[df["at_risk"]==1]["invoice_amount"].sum(), 2),
    "invoices_90plus":          int((df["aging_bucket"]=="90+").sum()),
    "rf_accuracy":              round(acc3, 4),
    "logr_accuracy":            round(acc2, 4),
    "lr_r2":                    round(r2, 4),
    "lr_mae_days":              round(mae, 2),
}

print(f"    ✓ Overall collection efficiency : {overall['overall_collection_eff']}%")
print(f"    ✓ Overall dispute rate          : {overall['overall_dispute_rate_pct']}%")
print(f"    ✓ Avg days to payment           : {overall['avg_days_to_payment']} days")
print(f"    ✓ At-risk invoices              : {overall['at_risk_invoices']}")

# ─────────────────────────────────────────────
# 7. SAVE MODELS
# ─────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(lr,       os.path.join(MODEL_DIR, "linear_regression.joblib"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler_lr.joblib"))
joblib.dump(logr,     os.path.join(MODEL_DIR, "logistic_regression.joblib"))
joblib.dump(scaler2,  os.path.join(MODEL_DIR, "scaler_logr.joblib"))
joblib.dump(rf,       os.path.join(MODEL_DIR, "random_forest.joblib"))
joblib.dump(le_client,   os.path.join(MODEL_DIR, "le_client.joblib"))
joblib.dump(le_resource, os.path.join(MODEL_DIR, "le_resource.joblib"))
joblib.dump(le_risk,     os.path.join(MODEL_DIR, "le_risk.joblib"))
joblib.dump(le_aging,    os.path.join(MODEL_DIR, "le_aging.joblib"))

print(f"\n[7] Models saved to {MODEL_DIR}")

# ─────────────────────────────────────────────
# 8. SAVE OUTPUTS
# ─────────────────────────────────────────────

df.to_csv(os.path.join(OUTPUT_DIR, "ar_ap_predictions.csv"), index=False)
kpi_client.to_csv(os.path.join(OUTPUT_DIR, "kpi_by_client.csv"), index=False)
kpi_month.to_csv(os.path.join(OUTPUT_DIR, "kpi_by_month.csv"), index=False)
kpi_aging.to_csv(os.path.join(OUTPUT_DIR, "kpi_by_aging.csv"), index=False)
kpi_resource.to_csv(os.path.join(OUTPUT_DIR, "kpi_by_resource.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
    f.write("\n".join(metrics_lines))

with open(os.path.join(OUTPUT_DIR, "overall_kpis.json"), "w") as f:
    json.dump({k: (float(v) if hasattr(v, 'item') else v) for k, v in overall.items()}, f, indent=2)

print("\n[8] All outputs saved:")
print("    ✓ ar_ap_data.csv")
print("    ✓ ar_ap_predictions.csv")
print("    ✓ kpi_by_client.csv")
print("    ✓ kpi_by_month.csv")
print("    ✓ kpi_by_aging.csv")
print("    ✓ kpi_by_resource.csv")
print("    ✓ model_metrics.txt")
print("    ✓ overall_kpis.json")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)

# Print KPI snapshot
print("\n── KPI Snapshot by Client ──")
print(kpi_client[["client","total_invoices","total_invoiced","collection_efficiency",
                   "avg_days_to_payment","predicted_avg_days","dispute_rate_pct","at_risk_invoices"]].to_string(index=False))

print("\n── KPI Snapshot by Aging Bucket ──")
print(kpi_aging.to_string(index=False))
