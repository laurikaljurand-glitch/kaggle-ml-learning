# Rider conversion model comparison

## Executive result

**Neither model is useful on the leakage-safe version of this public dataset.**

Random Forest has a held-out ROC AUC of **0.4992** and XGBoost **0.4983**. The difference is **-0.0008** in XGBoost minus Random Forest ROC AUC, with a paired bootstrap 95% interval of **[-0.0062, +0.0035]**. The interval crosses zero, so there is no meaningful winner.

If forced to pick one for this dataset, Random Forest is marginally better on the point estimates (ROC AUC, average precision and Brier score), but the difference is operationally negligible. The correct conclusion is that the current leakage-safe features contain essentially no conversion signal.

## Prediction task

The target is:

**booking request -> Completed ride**

Prediction happens at **booking creation**. This is therefore a booking-fulfillment conversion model, not app-open/quote -> booking conversion. The public dataset contains only bookings, so sessions or quotes that never became bookings are not observed.

## Dataset and split

Public NCR ride-bookings dataset, pinned to source commit:
367bac236a9b885a77751e6535911d116bae58dd

Data quality measured by the benchmark:
- Source rows: 150,000
- Duplicate Booking IDs removed: 1,233
- Analysis rows: 148,767
- Train rows: 111,771
- Test rows: 36,996
- Train completion rate: 61.96%
- Test completion rate: 62.17%
- Unique-customer share: 99.20%

Time split:
- Train: 2024-01-01 through 2024-09-30
- Test: 2024-10-01 through 2024-12-30
- Hyperparameter selection: 3-fold expanding time-series cross-validation on training data only

## Features

Both models use exactly the same booking-time variables:
- hour
- weekday number
- month
- day of month
- weekend flag
- vehicle type
- pickup location
- drop location

Excluded to prevent leakage or unstable memorization:
- Booking ID
- Customer ID
- Avg VTAT / Avg CTAT
- cancellation flags and cancellation reasons
- Booking Value
- Ride Distance
- Driver Ratings
- Customer Rating
- Payment Method

## Held-out results

| Model | CV ROC AUC | Test ROC AUC | Avg precision | Precision | Recall | F1 | Brier | Top-decile lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.4989 | **0.4992** | **0.6185** | 0.6217 | 1.0000 | **0.7667** | **0.2352** | **0.973** |
| XGBoost | **0.4992** | 0.4983 | 0.6178 | **0.6218** | 0.9986 | 0.7664 | 0.2361 | 0.970 |

Best parameters:
- Random Forest: max_depth=8, min_samples_leaf=50, n_estimators=200
- XGBoost: max_depth=5, learning_rate=0.10, n_estimators=250, min_child_weight=10, reg_lambda=10

### Why F1 looks high while the model is bad

The test completion rate is 62.17%. At the default 0.5 threshold, Random Forest predicts **every test booking as completed**:
- True failed bookings predicted failed: 0
- Failed bookings predicted completed: 13,997
- Completed bookings predicted failed: 0
- Completed bookings predicted completed: 22,999

That gives recall=1.0 and F1=0.7667, but no discrimination. ROC AUC near 0.50 exposes this immediately.

XGBoost behaves almost the same:
- True failed bookings predicted failed: 25
- Failed bookings predicted completed: 13,972
- Completed bookings predicted failed: 32
- Completed bookings predicted completed: 22,967

Top-decile lift is below 1 for both models, another sign that the ranking is not useful.

## Which model is better?

**No clear winner.**

Random Forest is slightly better by held-out ROC AUC (+0.0008 versus XGBoost), average precision and Brier score, but the difference is tiny and statistically inconclusive. XGBoost's extra complexity provides no benefit here.

For this dataset, use Random Forest only if you need a simpler learning baseline. Do not deploy either model.

## Main dataset weaknesses

1. **Wrong funnel stage for true rider conversion.** The dataset starts after a booking already exists. It cannot predict app-open/search/quote -> booking conversion.
2. **Weak pre-booking signal.** Once outcome-derived variables are removed, time, route and vehicle type do not separate completed from failed rides.
3. **Uncertain provenance.** The public dataset should be treated as an educational ride-hailing proxy rather than verified production data.
4. **Structural outcome-dependent missingness.** Fare, distance, payment and ratings are populated only for certain terminal outcomes. Using them would leak information.
5. **Post-booking ETA variables.** Avg VTAT/CTAT are operational measurements after the prediction point and can strongly inflate apparent performance.
6. **Almost no reusable rider history.** 99.20% of rows have a unique Customer ID, so useful repeat-rider behavior cannot be learned reliably.
7. **Single market/year.** NCR 2024 cannot establish generalization to other cities, countries, pricing systems or marketplace conditions.
8. **Potentially synthetic regularity.** Outcome and segment patterns are unusually flat, which is consistent with the near-random leakage-safe benchmark.

## Better data for a production-like conversion model

For true quote -> booking conversion, the ideal table would contain one row per ride-search/quote attempt, including both converted and non-converted sessions. Features should be timestamped before the conversion decision, for example:
- quoted ETA
- quoted price / surge
- pickup and destination zone
- supply-demand balance
- hour/day/season
- rider prior trips
- rider historical conversion / cancellation behavior
- app/platform
- payment method availability
- promo eligibility
- competitor price/ETA if legally and operationally available

Target:
- 1 = rider books within a defined window after quote
- 0 = quote/search does not turn into a booking

A later advanced dataset candidate is **SurvCancel**, a large real-world longitudinal ride-sharing cancellation dataset with 173,037 orders and 7.94M pre-pickup snapshots. It is more realistic for dynamic cancellation/fulfillment modeling, but at ~43.6 GB it is much heavier and still does not solve quote -> booking conversion.

## Reproducibility

Run:

    pip install -r ridehail-conversion/requirements.txt
    python ridehail-conversion/compare_models.py

The GitHub Actions benchmark for commit 2578c1465a7d7257ccc1c6e598bc6bb74c150168 completed successfully.
