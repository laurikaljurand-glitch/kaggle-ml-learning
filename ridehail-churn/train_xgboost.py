"""XGBoost comparison using the exact data definition and split in train_compare.py.

Run from ridehail-churn/: python train_xgboost.py
Requires: pip install -r requirements.txt
Uses the same SHA-256-verified public dataset; no raw data is committed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEED = 42
SOURCE_COMMIT = "760192ed7ace0975bc273c8ef5ac596132439352"
SOURCE_URL = f"https://raw.githubusercontent.com/hsivasub/Uber-Rider-Churn-Prediction/{SOURCE_COMMIT}/attributes.json"
EXPECTED_SHA256 = "8197ac13ee1313d71fcd771a62c0080ee451adf3624b968108ad3cb3d7bc0e66"
OUT = Path("/kaggle/working/ridehail_churn_outputs") if Path("/kaggle/working").exists() else Path("ridehail_churn_outputs")
OUT.mkdir(parents=True, exist_ok=True)

def get_data_path():
    candidates = []
    if os.getenv("RIDEHAIL_DATA_PATH"):
        candidates.append(Path(os.environ["RIDEHAIL_DATA_PATH"]))
    candidates += [Path("attributes.json"), OUT / "attributes.json"]
    if Path("/kaggle/input").exists():
        candidates += sorted(Path("/kaggle/input").rglob("attributes.json"))
    for path in candidates:
        if path.is_file():
            return path
    path = OUT / "attributes.json"
    urlretrieve(SOURCE_URL, path)
    return path

path = get_data_path()
if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
    raise ValueError("Dataset SHA-256 differs from the Random Forest project.")
raw = pd.read_json(path, convert_dates=False)
required = {"city", "phone", "signup_date", "last_trip_date", "trips_in_first_30_days", "avg_dist", "upgraded_user"}
if not required.issubset(raw.columns):
    raise ValueError(f"Missing fields: {required - set(raw.columns)}")
df = raw.drop_duplicates().copy()
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="raise")
df["last_trip_date"] = pd.to_datetime(df["last_trip_date"], errors="raise")
if df[["signup_date", "last_trip_date", "trips_in_first_30_days"]].isna().any().any():
    raise ValueError("Missing target or required first-month feature")
if not df["signup_date"].between("2014-01-01", "2014-01-31").all():
    raise ValueError("Expected January 2014 signup cohort")
end = df["last_trip_date"].max()
window_start = end - pd.Timedelta(days=29)
if (df["signup_date"] + pd.Timedelta(days=30) >= window_start).any():
    raise ValueError("Outcome window overlaps prediction point")
y = (df["last_trip_date"] < window_start).astype(int)
X = df[["trips_in_first_30_days", "city", "phone"]].copy()
X["signup_day"] = df["signup_date"].dt.day
if X["trips_in_first_30_days"].lt(0).any():
    raise ValueError("Negative early trip count")
assert "last_trip_date" not in X.columns
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), ["trips_in_first_30_days", "signup_day"]),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), ["city", "phone"]),
])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
search = GridSearchCV(
    Pipeline([
        ("preprocess", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_estimators=200, n_jobs=1,
            random_state=SEED,
        )),
    ]),
    param_grid={
        "model__max_depth": [2, 4],
        "model__learning_rate": [0.03, 0.1],
        "model__min_child_weight": [5, 20],
        "model__reg_lambda": [1.0, 10.0],
    },
    cv=cv, scoring="roc_auc", n_jobs=2, refit=True,
)
search.fit(X_train, y_train)
prob = search.best_estimator_.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
metrics = {
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "average_precision": float(average_precision_score(y_test, prob)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "brier": float(brier_score_loss(y_test, prob)),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
}
results = {
    "dataset_sha256": EXPECTED_SHA256,
    "rows": len(df), "train_rows": len(X_train), "test_rows": len(X_test),
    "target": "No observed trip in final 30 calendar days of extract",
    "features": X.columns.tolist(),
    "cv_roc_auc": float(search.best_score_),
    "best_parameters": search.best_params_,
    "test_metrics": metrics,
}
(OUT / "xgboost_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
print("Random Forest held-out ROC AUC from REPORT.md: 0.7574")
print("Compare against the newly measured XGBoost ROC AUC above; do not assume improvement.")
