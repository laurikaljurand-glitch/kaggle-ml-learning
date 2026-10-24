from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUTPUT_DIR, load_and_split, write_json
from train_random_forest import fit_random_forest
from train_xgboost import fit_xgboost


def paired_auc_interval(y_true, rf_probability, xgb_probability, n_bootstrap=300, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    differences = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        differences.append(
            roc_auc_score(y[idx], xgb_probability[idx])
            - roc_auc_score(y[idx], rf_probability[idx])
        )
    return [float(v) for v in np.quantile(differences, [0.025, 0.975])]


def main():
    data = load_and_split()
    print("Data quality:")
    print(json.dumps(data.quality, indent=2))

    rf_model, rf_result, rf_probability = fit_random_forest(data)
    xgb_model, xgb_result, xgb_probability = fit_xgboost(data)

    auc_difference = (
        xgb_result["test_metrics"]["roc_auc"]
        - rf_result["test_metrics"]["roc_auc"]
    )
    interval = paired_auc_interval(
        data.y_test, rf_probability, xgb_probability
    )

    if interval[0] > 0:
        winner = "XGBoost"
        interpretation = "XGBoost has a higher held-out ROC AUC and the paired bootstrap interval is above zero."
    elif interval[1] < 0:
        winner = "Random Forest"
        interpretation = "Random Forest has a higher held-out ROC AUC and the paired bootstrap interval is below zero."
    else:
        winner = "No clear winner"
        interpretation = "The held-out ROC AUC difference is small enough that the paired bootstrap interval crosses zero."

    comparison = {
        "winner_by_test_roc_auc": winner,
        "interpretation": interpretation,
        "xgboost_minus_random_forest_roc_auc": float(auc_difference),
        "paired_bootstrap_95pct_interval": interval,
        "random_forest": rf_result,
        "xgboost": xgb_result,
    }

    joblib.dump(rf_model, OUTPUT_DIR / "random_forest_conversion.joblib", compress=3)
    joblib.dump(xgb_model, OUTPUT_DIR / "xgboost_conversion.joblib", compress=3)
    write_json(OUTPUT_DIR / "random_forest_results.json", rf_result)
    write_json(OUTPUT_DIR / "xgboost_results.json", xgb_result)
    write_json(OUTPUT_DIR / "comparison_results.json", comparison)

    report = f"""# Rider conversion model comparison

## Scope

This project predicts **booking request -> completed ride** at booking creation.
It is a booking-fulfillment conversion proxy, not app-open/quote -> booking conversion.

## Held-out future-period results

| Model | ROC AUC | Avg precision | Precision | Recall | F1 | Brier | Top-decile lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | {rf_result['test_metrics']['roc_auc']:.4f} | {rf_result['test_metrics']['average_precision']:.4f} | {rf_result['test_metrics']['precision']:.4f} | {rf_result['test_metrics']['recall']:.4f} | {rf_result['test_metrics']['f1']:.4f} | {rf_result['test_metrics']['brier']:.4f} | {rf_result['test_metrics']['top_decile_lift']:.3f} |
| XGBoost | {xgb_result['test_metrics']['roc_auc']:.4f} | {xgb_result['test_metrics']['average_precision']:.4f} | {xgb_result['test_metrics']['precision']:.4f} | {xgb_result['test_metrics']['recall']:.4f} | {xgb_result['test_metrics']['f1']:.4f} | {xgb_result['test_metrics']['brier']:.4f} | {xgb_result['test_metrics']['top_decile_lift']:.3f} |

**Winner by held-out ROC AUC:** {winner}

XGBoost - Random Forest ROC AUC: **{auc_difference:+.4f}**.
Paired bootstrap 95% interval: **[{interval[0]:+.4f}, {interval[1]:+.4f}]**.

{interpretation}

## Leakage-safe feature set

Only variables observable at booking creation are used:
- hour, weekday, month, day of month, weekend flag
- vehicle type
- pickup location
- drop location

Excluded: booking/customer IDs, Avg VTAT, Avg CTAT, cancellation fields/reasons,
booking value, distance, ratings and payment method. These are identifiers,
post-booking measurements, outcome fields, or structurally outcome-dependent.

## Dataset weaknesses

1. Public provenance is uncertain; treat the data as an educational proxy, not verified Uber production data.
2. The data starts at booking, so it cannot model quote/search -> booking conversion.
3. Most useful operational signals occur after booking and are deliberately excluded to prevent leakage.
4. Several downstream fields are structurally missing for failed rides, making them dangerous predictors.
5. Location and vehicle patterns may be synthetic or unusually regular, limiting external validity.
6. Customer ID is excluded: it is high-cardinality and the dataset does not provide reliable pre-booking rider history.
7. A single 2024 NCR market cannot establish generalization to other cities, countries, seasons, or marketplace regimes.

## Validation design

Training: bookings before 2024-10-01.
Test: bookings on/after 2024-10-01.
Hyperparameter selection: 3-fold expanding time-series cross-validation on the training period only.
Both algorithms use exactly the same target, features and held-out test period.
"""
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")

    print("\nCOMPARISON_RESULTS_JSON")
    print(json.dumps(comparison, indent=2))
    print("\nWrote outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
