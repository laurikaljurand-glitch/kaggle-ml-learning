from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
SOURCE_COMMIT = "367bac236a9b885a77751e6535911d116bae58dd"
SOURCE_URL = (
    "https://raw.githubusercontent.com/dr-chidananda/ncr-ride-bookings-analysis/"
    f"{SOURCE_COMMIT}/data/ncr_ride_bookings.csv"
)
DATA_FILENAME = "ncr_ride_bookings.csv"
CUTOFF = pd.Timestamp("2024-10-01")

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = (
    Path("/kaggle/working/ridehail_conversion_outputs")
    if Path("/kaggle/working").exists()
    else HERE / "outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = ["hour", "weekday_num", "month_num", "day_of_month", "is_weekend"]
CATEGORICAL_FEATURES = ["vehicle_type", "pickup_location", "drop_location"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ConversionData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_datetime: pd.Series
    test_datetime: pd.Series
    quality: dict
    source_path: Path


def find_data() -> Path:
    candidates = []
    if os.getenv("RIDEHAIL_CONVERSION_DATA_PATH"):
        candidates.append(Path(os.environ["RIDEHAIL_CONVERSION_DATA_PATH"]))
    candidates.extend([HERE / DATA_FILENAME, OUTPUT_DIR / DATA_FILENAME])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.rglob(DATA_FILENAME)))
        candidates.extend(sorted(kaggle_input.rglob("*.csv")))

    for path in candidates:
        if path.is_file():
            return path

    destination = OUTPUT_DIR / DATA_FILENAME
    print("Downloading the pinned public NCR ride-bookings dataset.")
    urlretrieve(SOURCE_URL, destination)
    return destination


def _clean_identifier(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.strip('"').str.strip("'")


def load_and_split() -> ConversionData:
    path = find_data()
    raw = pd.read_csv(path, na_values=["null", "None", ""], keep_default_na=True)

    required = {
        "Date",
        "Time",
        "Booking ID",
        "Booking Status",
        "Customer ID",
        "Vehicle Type",
        "Pickup Location",
        "Drop Location",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw.copy()
    df["Booking ID"] = _clean_identifier(df["Booking ID"])
    df["Customer ID"] = _clean_identifier(df["Customer ID"])

    raw_rows = len(df)
    duplicate_booking_ids = int(df["Booking ID"].duplicated(keep="first").sum())
    df = df.drop_duplicates(subset=["Booking ID"], keep="first").copy()

    parsed_date = pd.to_datetime(df["Date"], errors="coerce")
    combined = parsed_date.dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str)
    df["booking_datetime"] = pd.to_datetime(combined, errors="coerce")
    bad_datetime = int(df["booking_datetime"].isna().sum())
    df = df.dropna(subset=["booking_datetime", "Booking Status"]).copy()
    df = df.sort_values("booking_datetime").reset_index(drop=True)

    status = df["Booking Status"].astype(str).str.strip()
    y = status.eq("Completed").astype(int)

    X = pd.DataFrame(index=df.index)
    X["hour"] = df["booking_datetime"].dt.hour.astype(int)
    X["weekday_num"] = df["booking_datetime"].dt.dayofweek.astype(int)
    X["month_num"] = df["booking_datetime"].dt.month.astype(int)
    X["day_of_month"] = df["booking_datetime"].dt.day.astype(int)
    X["is_weekend"] = df["booking_datetime"].dt.dayofweek.isin([5, 6]).astype(int)
    X["vehicle_type"] = df["Vehicle Type"].astype("string")
    X["pickup_location"] = df["Pickup Location"].astype("string")
    X["drop_location"] = df["Drop Location"].astype("string")

    train_mask = df["booking_datetime"] < CUTOFF
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Expected observations on both sides of the 2024-10-01 cutoff.")

    # Booking-time only: IDs and all post-booking operational/outcome fields are excluded.
    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    train_datetime = df.loc[train_mask, "booking_datetime"].reset_index(drop=True)
    test_datetime = df.loc[test_mask, "booking_datetime"].reset_index(drop=True)

    customer_unique_share = float(df["Customer ID"].nunique(dropna=True) / len(df))
    quality = {
        "source_rows": int(raw_rows),
        "duplicate_booking_ids_removed": duplicate_booking_ids,
        "unparseable_datetimes_removed": bad_datetime,
        "analysis_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_completion_rate": float(y_train.mean()),
        "test_completion_rate": float(y_test.mean()),
        "unique_customer_share": customer_unique_share,
        "train_start": str(train_datetime.min()),
        "train_end": str(train_datetime.max()),
        "test_start": str(test_datetime.min()),
        "test_end": str(test_datetime.max()),
    }

    return ConversionData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_datetime=train_datetime,
        test_datetime=test_datetime,
        quality=quality,
        source_path=path,
    )


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def evaluate_probabilities(y_true, probability, threshold: float = 0.5) -> dict:
    probability = np.asarray(probability)
    prediction = (probability >= threshold).astype(int)
    base_rate = float(np.mean(y_true))

    n_top = max(1, int(np.ceil(len(probability) * 0.10)))
    top_idx = np.argsort(probability)[-n_top:]
    top_rate = float(np.asarray(y_true)[top_idx].mean())
    top_decile_lift = float(top_rate / base_rate) if base_rate else None

    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "top_decile_completion_rate": top_rate,
        "top_decile_lift": top_decile_lift,
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
