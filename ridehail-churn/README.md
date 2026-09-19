# Ride-hailing rider inactivity: tree versus random forest

This reproducible learning project forecasts **future 30-day inactivity** from first-month rider data. It compares a tuned single decision tree with a tuned random forest and includes [measured results and limitations](REPORT.md).

## Open the analysis

- [Self-contained, commented notebook](random-forest-vs-decision-tree.ipynb)
- [Commented Python script](train_compare.py)
- [Comparison report](REPORT.md)
- [Trained forest model](assets/random_forest_churn.joblib), [metrics](assets/results.json), and [plots](assets/)

## Reproduce the results

Use Python 3.12 and install the versions in `requirements.txt`. Then run:

```bash
python -m pip install -r requirements.txt
python train_compare.py
```

The script downloads the [public source JSON at a pinned commit](https://github.com/hsivasub/Uber-Rider-Churn-Prediction/blob/760192ed7ace0975bc273c8ef5ac596132439352/attributes.json) if it is not already present and verifies its SHA-256. If network access is off, download that file as `attributes.json` alongside the script, or set `RIDEHAIL_DATA_PATH` to its local path. In Kaggle, attach a dataset containing `attributes.json` under `/kaggle/input` and run the notebook; Internet access is not needed if that file is attached. Outputs appear in `ridehail_churn_outputs/` (or `/kaggle/working/ridehail_churn_outputs/`).

To use the saved research model with the exact dependency versions above:

```python
import joblib
import pandas as pd

artifact = joblib.load("assets/random_forest_churn.joblib")
rider = pd.DataFrame([{
    "trips_in_first_30_days": 2,
    "city": "Winterfell",
    "phone": "iPhone",
    "signup_day": 15,
}])
inactivity_probability = artifact["pipeline"].predict_proba(rider)[:, 1][0]
print(inactivity_probability)
```

Only load a `joblib` file you trust. This model uses fictional city names, has no later-cohort validation, and is for learning, **not production decisions**. The public mirror does not document the dataset's original provenance or license; the raw file is not included here.
