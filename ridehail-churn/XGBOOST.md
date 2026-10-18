# Week 11: XGBoost on the same rider dataset

Run from the `ridehail-churn` directory:

```bash
pip install -r requirements.txt
python train_xgboost.py
```

The script reuses the exact pinned `attributes.json` source and checksum, duplicate removal, January 2014 cohort, 30-day inactivity target, four feature definitions, seed 42, stratified 80/20 split and 3-fold ROC AUC selection used by `train_compare.py`.

The dataset is public but its original provenance and license are not established; fictional city names appear. The label is **30-day inactivity**, not confirmed churn. Potential future-derived variables remain excluded to avoid leakage.

**What to inspect:** Cross-validation ROC AUC, held-out ROC AUC, average precision, precision, recall, F1, Brier score and confusion matrix. The script writes `ridehail_churn_outputs/xgboost_results.json` when run. Compare these results with the existing Random Forest ROC AUC of **0.7574** reported in `REPORT.md`. No XGBoost performance numbers are claimed until the code is executed.

**Learning notes:** Each boosting round adds a correction tree (for this binary setup). `learning_rate` shrinks each tree's contribution; `n_estimators` sets the boosting-round count. `max_depth`, `min_child_weight` and `reg_lambda` constrain complexity. A smaller learning rate often needs more rounds; too much model complexity can overfit. The parameter grid is intentionally small enough for a learning exercise.

**Evaluation caution:** Both models use the same held-out split, but the comparison remains an exercise on this dataset, not a validated production forecasting system. Avoid selecting additional hyperparameters by repeatedly inspecting the test results.
