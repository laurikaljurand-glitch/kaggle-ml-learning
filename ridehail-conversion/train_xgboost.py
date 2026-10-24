from __future__ import annotations

import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (
    FEATURES,
    OUTPUT_DIR,
    RANDOM_STATE,
    SOURCE_COMMIT,
    SOURCE_URL,
    evaluate_probabilities,
    load_and_split,
    make_preprocessor,
    write_json,
)


def fit_xgboost(data=None):
    data = data or load_and_split()
    cv = TimeSeriesSplit(n_splits=3)

    pipeline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=250,
            min_child_weight=10,
            reg_lambda=10.0,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.10],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,
        refit=True,
        return_train_score=True,
    )
    search.fit(data.X_train, data.y_train)

    model = search.best_estimator_
    test_probability = model.predict_proba(data.X_test)[:, 1]
    metrics = evaluate_probabilities(data.y_test, test_probability)

    result = {
        "model": "XGBoost",
        "target": "Booking request converts to Completed ride",
        "prediction_point": "Booking creation",
        "features": FEATURES,
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "quality": data.quality,
        "best_parameters": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
        "test_metrics": metrics,
    }
    return model, result, test_probability


def main():
    model, result, _ = fit_xgboost()
    joblib.dump(model, OUTPUT_DIR / "xgboost_conversion.joblib", compress=3)
    write_json(OUTPUT_DIR / "xgboost_results.json", result)
    print("XGBoost results")
    print(result)


if __name__ == "__main__":
    main()
