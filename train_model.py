"""
Train the churn prediction model using the EXACT preprocessing pipeline
from the original notebook (CUSTOMER_CHURN_PREDICTION.ipynb):

  - drop customerID
  - TotalCharges -> numeric, fill missing with median
  - Churn -> map Yes/No to 1/0
  - binary_cols -> LabelEncoder
  - multi_cols -> pd.get_dummies(drop_first=True)
  - StandardScaler on all features
  - LogisticRegression(max_iter=1000, random_state=42)

Run once locally:  python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "churn_model.pkl"

df = pd.read_csv(DATA_PATH)

# ---- 3.1 Drop customerID ----
df.drop(columns=["customerID"], inplace=True)

# ---- 3.2 Fix TotalCharges ----
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ---- 3.5 Encode target ----
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- 3.6 Label-encode binary columns ----
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---- 3.7 One-hot encode multi-class columns ----
multi_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
# remember original categories (order matters for the app's dropdowns)
multi_col_categories = {col: sorted(df[col].unique().tolist()) for col in multi_cols}

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# ---- Train/test split ----
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Scaling ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Train Logistic Regression ----
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
proba = model.predict_proba(X_test_scaled)[:, 1]
print("Accuracy :", round(accuracy_score(y_test, preds), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 4))

# ---- Save everything the app needs ----
bundle = {
    "model": model,
    "scaler": scaler,
    "encoders": encoders,              # LabelEncoders for binary_cols
    "binary_cols": binary_cols,
    "multi_cols": multi_cols,
    "multi_col_categories": multi_col_categories,  # dropdown options for the app
    "feature_order": X.columns.tolist(),           # exact columns after get_dummies
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

print(f"Saved model bundle -> {MODEL_PATH}")
