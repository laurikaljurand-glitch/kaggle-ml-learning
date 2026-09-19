"""Reproducible comparison of a decision tree and random forest for rider inactivity.

Run: python train_compare.py
Or:  RIDEHAIL_DATA_PATH=/path/to/attributes.json python train_compare.py
The code downloads a pinned public challenge dataset if a local copy is absent.
"""

# %% [markdown]
# # Rider inactivity after the first 30 days
#
# We forecast whether a rider will have **no trip in the final 30 days** of the
# dataset, using information available 30 days after that rider signed up.
# This is a future inactivity proxy, not a verified account closure.
# The cities are fictional; original data provenance and license are not documented.

# %%
# 1. Import reproducible modeling and plotting tools.
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
SOURCE_COMMIT = "760192ed7ace0975bc273c8ef5ac596132439352"
SOURCE_URL = (
    "https://raw.githubusercontent.com/hsivasub/Uber-Rider-Churn-Prediction/"
    f"{SOURCE_COMMIT}/attributes.json"
)
EXPECTED_SHA256 = "8197ac13ee1313d71fcd771a62c0080ee451adf3624b968108ad3cb3d7bc0e66"
OUTPUT_DIR = Path("/kaggle/working/ridehail_churn_outputs") if Path("/kaggle/working").exists() else Path("ridehail_churn_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Load and inspect the source
#
# Place `attributes.json` alongside this notebook, set `RIDEHAIL_DATA_PATH`, or
# attach a Kaggle dataset containing that file. Otherwise Internet access is
# needed once. A hash check ensures the analysis uses the inspected data version.

# %%
def find_dataset() -> Path:
    """Prefer an explicitly supplied/local dataset; download only if needed."""
    paths = []
    if os.environ.get("RIDEHAIL_DATA_PATH"):
        paths.append(Path(os.environ["RIDEHAIL_DATA_PATH"]))
    paths.extend([Path("attributes.json"), OUTPUT_DIR / "attributes.json"])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        paths.extend(sorted(kaggle_input.rglob("attributes.json")))
    for path in paths:
        if path.is_file():
            return path
    destination = OUTPUT_DIR / "attributes.json"
    print("Downloading the pinned public dataset; this requires Internet access.")
    urlretrieve(SOURCE_URL, destination)
    return destination


data_path = find_dataset()
actual_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise ValueError(f"Unexpected dataset checksum at {data_path}: {actual_sha256}")

raw = pd.read_json(data_path, convert_dates=False)
required = {
    "city", "phone", "signup_date", "last_trip_date", "trips_in_first_30_days",
    "avg_dist", "upgraded_user",
}
if not required.issubset(raw.columns):
    raise ValueError(f"Missing expected columns: {sorted(required - set(raw.columns))}")

# Count contradictory fields before excluding them; don't delete riders selectively.
quality = {
    "source_rows": len(raw),
    "exact_duplicate_rows": int(raw.duplicated().sum()),
    "zero_early_trips_positive_avg_dist": int(((raw["trips_in_first_30_days"] == 0) & (raw["avg_dist"] > 0)).sum()),
    "zero_early_trips_upgraded_user": int(((raw["trips_in_first_30_days"] == 0) & raw["upgraded_user"]).sum()),
}
print("Dataset quality checks:", quality)

# %% [markdown]
# ## 3. Define the prediction point, target, and allowed features
#
# The latest trip is July 1, 2014. A rider is **inactive** if they did not ride
# from June 2 through July 1 inclusive. All riders signed up in January 2014,
# so every first-month snapshot is earlier than this outcome window.
#
# `last_trip_date` defines the label and MUST NOT be used as a predictor.
# Lifetime averages/ratings/weekday share are excluded because they can include
# later trips. `avg_dist` and `upgraded_user` are also excluded: their supposed
# first-month meaning contradicts the early trip count in many records.

# %%
# An exact duplicate might otherwise land in both train and test folds.
df = raw.drop_duplicates().copy()
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="raise")
df["last_trip_date"] = pd.to_datetime(df["last_trip_date"], errors="raise")
if df[["signup_date", "last_trip_date", "trips_in_first_30_days"]].isna().any().any():
    raise ValueError("The target or required first-month feature has missing values.")
if not df["signup_date"].between("2014-01-01", "2014-01-31").all():
    raise ValueError("This analysis requires the documented January 2014 signup cohort.")

end_date = df["last_trip_date"].max()
active_window_start = end_date - pd.Timedelta(days=29)
if (df["signup_date"] + pd.Timedelta(days=30) >= active_window_start).any():
    raise ValueError("The outcome window must begin after every first-month snapshot.")

# Label 1 means no observed trip in the last 30 calendar dates of the extract.
y = (df["last_trip_date"] < active_window_start).astype(int)

# This explicit allowlist prevents accidental inclusion of future-derived fields.
X = df[["trips_in_first_30_days", "city", "phone"]].copy()
X["signup_day"] = df["signup_date"].dt.day
if X["trips_in_first_30_days"].lt(0).any():
    raise ValueError("First-month trip counts must not be negative.")
assert "last_trip_date" not in X.columns

print(f"Riders: {len(df):,}; inactivity: {y.mean():.1%}")
print(f"Prediction: 30 days after signup; outcome window: {active_window_start.date()} through {end_date.date()}")
print("Available model features:", X.columns.tolist())

# %% [markdown]
# ## 4. Use the same train/test split and cross-validation for both models
#
# The test rows are held out until hyperparameters are selected by 3-fold
# cross-validation. Preprocessing lives inside each pipeline, so its imputers
# and category encoder learn only from their training fold.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
numeric_features = ["trips_in_first_30_days", "signup_day"]
categorical_features = ["city", "phone"]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Tune model complexity on identical folds, using ROC AUC as the selection score.
tree_search = GridSearchCV(
    Pipeline([("preprocess", preprocessor), ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
    param_grid={"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [50, 200]},
    scoring="roc_auc", cv=cv, n_jobs=2, refit=True,
)
forest_search = GridSearchCV(
    Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(
        n_estimators=120, random_state=RANDOM_STATE, n_jobs=1
    ))]),
    param_grid={
        "model__max_depth": [6, 12],
        "model__min_samples_leaf": [30, 100],
        "model__max_features": ["sqrt", 0.8],
    },
    scoring="roc_auc", cv=cv, n_jobs=2, refit=True,
)
tree_search.fit(X_train, y_train)
forest_search.fit(X_train, y_train)

# A majority-rate baseline tells us how much accuracy is available by guessing.
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
print("Tree CV ROC AUC:", round(tree_search.best_score_, 4), tree_search.best_params_)
print("Forest CV ROC AUC:", round(forest_search.best_score_, 4), forest_search.best_params_)

# %% [markdown]
# ## 5. Compare both models on the untouched test set
#
# ROC AUC measures ranking; average precision emphasizes finding inactive
# riders. Precision, recall and F1 use the same 0.5 probability threshold for
# both models. Brier score measures probability error (lower is better).

# %%
def evaluate(model, features, labels):
    """Collect comparable probabilities and classification scores."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }, probabilities


models = {
    "Majority-rate baseline": baseline,
    "Single decision tree": tree_search.best_estimator_,
    "Random forest": forest_search.best_estimator_,
}
scores = {}
probabilities = {}
for name, model in models.items():
    scores[name], probabilities[name] = evaluate(model, X_test, y_test)

summary = pd.DataFrame(scores).T[["roc_auc", "average_precision", "precision", "recall", "f1", "brier"]]
print("\nHeld-out test set (positive class = inactive):")
print(summary.round(4).to_string())
for name in ("Single decision tree", "Random forest"):
    print(f"{name} confusion matrix [true active, true inactive] x [pred active, pred inactive]: {scores[name]['confusion_matrix']}")

# Estimate the difference on paired resamples of the SAME test riders.
rng = np.random.default_rng(RANDOM_STATE)
y_array = y_test.to_numpy()
auc_differences = []
for _ in range(300):
    sample = rng.integers(0, len(y_array), len(y_array))
    if np.unique(y_array[sample]).size < 2:
        continue
    auc_differences.append(
        roc_auc_score(y_array[sample], probabilities["Random forest"][sample])
        - roc_auc_score(y_array[sample], probabilities["Single decision tree"][sample])
    )
auc_difference_interval = np.quantile(auc_differences, [0.025, 0.975]).tolist()
print("Forest minus tree test ROC AUC:", round(scores["Random forest"]["roc_auc"] - scores["Single decision tree"]["roc_auc"], 4))
print("Paired bootstrap 95% interval:", [round(v, 4) for v in auc_difference_interval])

# %% [markdown]
# ## 6. Explain what the models use and save reproducible outputs
#
# Permutation importance measures the drop in held-out ROC AUC when one original
# feature is shuffled. It is associative and does not establish cause or effect.
# The tree plot shows its top three levels; deeper branches are omitted in the image.

# %%
forest = forest_search.best_estimator_
importance_result = permutation_importance(
    forest, X_test, y_test, n_repeats=5, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=2
)
importance = pd.Series(importance_result.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\nRandom forest permutation importance (mean ROC AUC decrease):")
print(importance.round(4).to_string())

# Save a portable chart and model for local inspection; never serialize source data.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, ("Single decision tree", "Random forest")):
    ConfusionMatrixDisplay(np.array(scores[name]["confusion_matrix"]), display_labels=["Active", "Inactive"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(name)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
plt.close(fig)

tree = tree_search.best_estimator_
feature_names = tree.named_steps["preprocess"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(17, 7))
plot_tree(tree.named_steps["model"], feature_names=feature_names, class_names=["Active", "Inactive"],
          max_depth=2, filled=True, rounded=True, fontsize=9, ax=ax)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "decision_tree_top_levels.png", dpi=125)
plt.close(fig)

artifact = {
    "pipeline": forest,
    "features": X.columns.tolist(),
    "positive_class": "no trip in the last 30 days of the extract",
    "source_sha256": EXPECTED_SHA256,
    "training_library_versions": {"pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__},
}
joblib.dump(artifact, OUTPUT_DIR / "random_forest_churn.joblib", compress=3)

# Persist the measured numbers separately from this explanatory notebook.
results = {
    "source_url": SOURCE_URL,
    "source_sha256": EXPECTED_SHA256,
    "quality": quality,
    "analysis_rows": len(df),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "inactive_rate": float(y.mean()),
    "last_observed_date": str(end_date.date()),
    "active_window_start": str(active_window_start.date()),
    "features": X.columns.tolist(),
    "tree_best_params": tree_search.best_params_,
    "forest_best_params": forest_search.best_params_,
    "cv_roc_auc": {"tree": float(tree_search.best_score_), "forest": float(forest_search.best_score_)},
    "test_scores": scores,
    "forest_minus_tree_auc_ci95": auc_difference_interval,
    "forest_permutation_importance_auc_drop": importance.to_dict(),
    "library_versions": artifact["training_library_versions"],
}
(OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("\nWrote model, plots, and results to", OUTPUT_DIR.resolve())
