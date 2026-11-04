# T7 Report: Patient Prediction for Improved Staffing (SIMULATED DATASET)
> [!IMPORTANT]
> **Data Notice:** The APTOS dataset contains no patient visit timestamps or longitudinal demographic records. All weekly schedules, visit volumes, time-series trajectories, and staffing projections presented in this module are **SIMULATED / SYNTHETIC** based on empirical class proportions from `data/dedup_records.csv`.

**Generated Files:**
- SIMULATED Schedule: `data/sim_weekly_screening_schedule.csv` (104 weeks)
- Forecast & Staffing Audit: `outputs/reports/sim_staffing_forecast_metrics.csv` (24-week horizon)
- Validation Figure: `outputs/figures/sim_t7_staffing_forecast.png` (300 DPI)

---

## 1. Executive Summary
Task 7 constructs a high-fidelity 2-year weekly screening time series incorporating real clinical severity proportions, annual seasonality (autumn/winter screening surges for diabetic care management), and stochastic clinic variations. We benchmark time-series lag forecasting for screening encounters and positive referrals (ICDR Grade 2+), and convert predicted demand into clinical workforce requirements (Primary Grader FTEs and Retinal Specialist FTEs).

---

## 2. Workforce Sizing & Operational Assumptions
- **Primary Screeners / Reading Center Graders:**
  - Standard throughput: **35 patient studies / grader / working day**
  - Weekly capacity (5-day week): **175 encounters / grader / week**
  - Minimum FTEs: $\lceil \text{Weekly Encounters} / 175 \rceil$
- **Retinal Specialists (Ophthalmologists):**
  - Handles referrals requiring slit-lamp biomicroscopy, optical coherence tomography (OCT), and therapeutic intervention (laser photocoagulation / anti-VEGF injection).
  - Standard throughput: **15 referral consultations / specialist / working day**
  - Weekly capacity (5-day week): **75 referrals / specialist / week**
  - Minimum FTEs: $\lceil \text{Weekly Referrals} / 75 \rceil$

---

## 3. Time-Series Forecasting Performance (24-Week Evaluation Horizon)

| Operational Metric | Forecasting Model | MAE | RMSE | MAPE |
| :--- | :--- | :---: | :---: | :---: |
| **Weekly Patient Encounters** | Ridge Regression (Lags 1-12, Rolling Mean/Std, Sin/Cos) | 11.0 pts | 13.8 | **2.59%** |
| **Weekly Specialist Referrals (Grade 2+)** | Gradient Boosting Regressor (Lags 1-12, Rolling Mean/Std) | 13.8 refs | 15.4 | **8.29%** |

### Workforce Management Insights
- Total weekly encounter volume averages **380-430 patients/week**, requiring a stable baseline of **2 to 3 Primary Graders**.
- Referral demand fluctuates between **140 and 190 urgent encounters/week**, requiring **2 to 3 Retinal Specialists**.
- Predictive scheduling prevents both clinical understaffing during peak autumn surges (preventing patient backlog and treatment delays) and wasteful overstaffing during summer troughs.

---

## 4. Visual Validation (300 DPI)
The multi-panel validation figure is archived at:
[`outputs/figures/sim_t7_staffing_forecast.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/sim_t7_staffing_forecast.png)
