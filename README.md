# 🏨 Hotel Booking Cancellation Predictor

Hotels lose revenue when guests cancel at the last minute. This project predicts, **at booking time**, how likely a booking is to be cancelled and suggests a deposit policy (none / partial / full).

## 🔗 Live app

**Try it here: https://asmaa-hotel-cancel.streamlit.app/**

| Low risk | High risk |
|---|---|
| ![Low risk](screenshot_low.png) | ![High risk](screenshot_high.png) |

## Results

Data: ~87K cleaned bookings from two hotels (80% train / 20% test). The baseline always predicts "not cancelled".

| Metric | Baseline (CV, train) | Final model (test set) |
|---|---|---|
| Accuracy | 0.723 | 0.82 |
| Precision | 0.000 | 0.64 |
| Recall | 0.000 | 0.78 |
| F1 | 0.000 | 0.70 |
| ROC-AUC | 0.500 | 0.892 |
| PR-AUC | 0.277 | 0.765 |

The final model is a tuned HistGradientBoosting pipeline with a decision threshold of 0.333, chosen on training data only. In plain words: out of every 100 bookings that really get cancelled, the model catches about 78; out of every 100 it flags, about 64 really are cancelled.

**Model comparison (5-fold cross-validation on the training set):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (majority) | 0.723 | 0.000 | 0.000 | 0.000 | 0.500 | 0.277 |
| Logistic Regression | 0.786 | 0.664 | 0.461 | 0.544 | 0.830 | 0.650 |
| Random Forest | 0.832 | 0.743 | 0.599 | 0.664 | 0.887 | 0.758 |
| HistGradientBoosting | 0.827 | 0.725 | 0.606 | 0.660 | 0.888 | 0.759 |

Tuning improved CV PR-AUC only slightly (0.759 → 0.767), and Random Forest was almost as good. HistGradientBoosting was chosen for its slightly higher PR-AUC, HistGradientBoosting was chosen for its slightly higher PR-AUC.

### Risk levels used by the app (test set)

| Risk level | Probability | Actual cancel rate | Share of bookings | Suggested action |
|---|---|---|---|---|
| Low | below 20% | 5.3% | 53.0% | no deposit |
| Medium | 20% to 50% | 32.9% | 23.4% | partial deposit |
| High | 50% or more | 73.0% | 23.5% | full deposit or phone confirmation |

## Key findings

- **Lead time:** cancel rate is 8.6% for bookings made up to 7 days ahead, and 39.9% for 180–365 days ahead.
- **Booking channel:** online travel agents cancel 35.6%, versus 12.4% for corporate and 14.7% for direct bookings.
- **Special requests and repeat guests:** cancel rate falls from 33.5% (no special requests) to 9.6% (4 or more); repeat guests cancel 8.7% versus 28.4% for new guests.
- **Unexplained:** non-refundable bookings are cancelled 94.3% of the time (823 bookings). The data does not explain why, so it is reported as an observation only.
- Most important features for the model: country, lead time, travel agent, special requests, market segment.

## Method

1. Removed 31,994 duplicate rows and 759 invalid rows (impossible price, zero guests, zero nights).
2. **Leakage control:** dropped `reservation_status`, `reservation_status_date`, `assigned_room_type` and `booking_changes` (not known at booking time). `required_car_parking_spaces` was also removed: all 5,806 bookings with a parking request were not cancelled, which suggests it is recorded after the stay. Removing it lowered CV PR-AUC from 0.781 to 0.759, and the lower, more honest score is the one reported.
3. Train/test split before any fitting; all preprocessing (imputation, scaling, one-hot encoding) lives inside a scikit-learn `Pipeline`.
4. Baseline + 3 algorithms, 5-fold stratified CV, `RandomizedSearchCV` on PR-AUC, threshold chosen from out-of-fold predictions on the training set. Test set used once.

## Limitations

- Data comes from two hotels in **Portugal** (2015–2017); results may not transfer to hotels in Egypt or to today's booking behaviour without retraining.
- Non-refundable bookings show an unexplained 94% cancel rate.
- `total_of_special_requests` may be recorded after booking; it was kept.
- `country` and `agent` are strong features. Using nationality to decide who pays a deposit could be unfair and should be reviewed before real use.
- Cut-offs are based on a statistical trade-off; a real hotel should set them from its own cost of an empty room versus a refused deposit.

## Run locally

```bash
git clone https://github.com/Asmaa12345666/hotel-cancellation-predictor.git
cd hotel-cancellation-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `model.joblib` | Trained pipeline, threshold and defaults |
| `requirements.txt` | Dependencies (scikit-learn pinned to the training version) |
| `hotel_cancellation_documented.ipynb` | Full analysis notebook |

## Data

Hotel Booking Demand dataset, Kaggle, licence CC BY 4.0: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
Original paper: Antonio, N., Almeida, A. & Nunes, L. (2019). *Hotel booking demand datasets.* Data in Brief.

*Built as a final project for the NextGen Institute Machine Learning internship.*
