# Ride-hailing rider conversion: Random Forest vs XGBoost

This folder mirrors the modeling discipline used in the churn project, but changes the target to rider conversion.

## What “conversion” means here

**Target = a booking request ends as a Completed ride.**

This is a **booking-to-completed-ride conversion / fulfillment model**. It is not a true quote-to-booking model because the public dataset contains bookings but no app sessions or quotes that did not become bookings.

Prediction is made **at booking creation**, before driver assignment.

## Dataset

Pinned public source:

- Repository: dr-chidananda/ncr-ride-bookings-analysis
- File: data/ncr_ride_bookings.csv
- Pinned commit: 367bac236a9b885a77751e6535911d116bae58dd
- Coverage: NCR ride bookings in 2024

The source is useful for learning because it has explicit booking outcomes, vehicle type, pickup/drop location and booking time. Its real-world provenance is not independently verified, so results must not be presented as official Uber performance.

## Leakage-safe feature set

Used by both models:

- hour
- weekday number
- month
- day of month
- weekend flag
- vehicle type
- pickup location
- drop location

Deliberately excluded:

- Booking ID and Customer ID
- Avg VTAT and Avg CTAT
- cancellation flags and cancellation reasons
- booking value and ride distance
- driver/customer ratings
- payment method

Those excluded fields are identifiers, downstream operational measurements, direct outcome fields, or fields whose missingness depends on the ride outcome.

## Fair comparison design

Both models use exactly the same:

- target
- booking-time feature set
- preprocessor
- training/test dates
- 3-fold expanding time-series cross-validation
- held-out test period
- evaluation metrics

Train: before 2024-10-01.  
Test: 2024-10-01 onward.

Metrics: ROC AUC, average precision, precision, recall, F1, Brier score and top-decile lift.

## Run

Install:

    pip install -r ridehail-conversion/requirements.txt

Random Forest only:

    python ridehail-conversion/train_random_forest.py

XGBoost only:

    python ridehail-conversion/train_xgboost.py

Fair side-by-side comparison:

    python ridehail-conversion/compare_models.py

Outputs are written to ridehail-conversion/outputs/ (or /kaggle/working/ridehail_conversion_outputs on Kaggle).

## Dataset limitations

- This is **not quote -> booking conversion**; non-booking sessions are absent.
- The public source's real-vs-synthetic provenance is uncertain.
- Strong operational variables such as ETA arise after booking and are excluded to avoid target leakage.
- Several downstream variables are only populated for successful/incomplete rides, so their presence itself can reveal the outcome.
- Customer identity is high-cardinality and does not provide a trustworthy historical rider feature set here.
- One market and one year do not establish transferability to other geographies or marketplace conditions.
- A model that performs only slightly above random is still an important result: it means the dataset does not contain enough pre-booking signal for a strong conversion model.

## Files

- common.py — shared data definition, feature engineering and metrics
- train_random_forest.py — Random Forest model
- train_xgboost.py — XGBoost model
- compare_models.py — identical-split benchmark and paired bootstrap comparison
- outputs/REPORT.md — generated benchmark report after a run
