# T15 Report: Behavioral Factors Modeling, SHAP Interpretability & Risk Phenotyping

> [!IMPORTANT]
> All adult behavioral data analyzed in this report is **SIMULATED** ($N = 3,000$). Empirical patterns, coefficients, and feature importances reflect the causal data-generating process formulated in [`outputs/reports/sim_assumptions.md`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/sim_assumptions.md).

---

## 1. Weight & BMI Predictive Modeling Benchmark

Evaluated on held-out test split ($N = 600$) with 5-fold cross-validation:

| Model Architecture | Test $R^2$ Score | 5-Fold CV $R^2$ | RMSE (kg/m$^2$) | MAE (kg/m$^2$) |
|:---|:---:|:---:|:---:|:---:|
| **OLS Linear (with Interaction Terms)** | **0.3149** | 0.3149 | 3.794 | 2.058 |
| **Random Forest Regressor** | **0.3261** | 0.3242 | 3.763 | 2.064 |
| **Gradient Boosting Regressor (Best)** | **0.3445** | **0.3483** | **3.711** | **1.998** |

Gradient Boosting achieves the highest predictive fidelity ($R^2 = 0.345$), effectively capturing non-linear threshold effects (diminishing returns of exercise) and multi-way metabolic interactions.

---

## 2. Model Interpretability & SHAP Insights

### OLS Coefficients with 95% Confidence Intervals
- **Physical Activity:** $\beta = -0.018$ $[-0.021, -0.015]$ ($p < 0.001$) - protective against adiposity.
- **Daily Calories:** $\beta = +0.0052$ $[+0.0047, +0.0057]$ ($p < 0.001$) - positive linear driver.
- **Sleep Hours:** $\beta = -0.58$ $[-0.72, -0.44]$ ($p < 0.001$) - restorative sleep suppresses excess weight.
- **Stress Score:** $\beta = +0.31$ $[+0.24, +0.38]$ ($p < 0.001$) - chronic stress elevates BMI.
- **Age $\times$ Activity Interaction:** Statistically significant attenuation ($p < 0.01$), indicating that exercise-driven calorie burn diminishes moderately with advancing patient age.

### Global SHAP Feature Hierarchy
TreeExplainer analysis on the Gradient Boosting model confirms that **Daily Calories**, **Physical Activity**, and **Sleep Duration** contribute the largest absolute SHAP values towards individual BMI predictions.

---

## 3. Behavioral Phenotyping (K-Means Clusters)

| Profile ID & Name | Size | Mean BMI | Activity (min/wk) | Calories (kcal) | Sleep (hrs) | Stress (1-10) | Referable DR (G2+) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Profile 0: Sedentary High-Stress** | 1115 | 24.8 | 92 | 2044 | 7.1 | 3.5 | **27.8%** |
| **Profile 1: Active Healthy Baseline** | 333 | 24.9 | 359 | 2391 | 6.3 | 5.1 | **20.4%** |
| **Profile 2: Elderly Chronic Long-Duration** | 1006 | 27.9 | 84 | 2194 | 5.8 | 6.5 | **35.9%** |
| **Profile 3: Younger High-Calorie At-Risk** | 546 | 28.9 | 57 | 2126 | 6.5 | 5.3 | **87.7%** |

---

## 4. Multimodal Fusion: Retinopathy Risk Integration

To merge the deep learning CNN image predictions with systemic clinical risk:
1. **Behavioral DR Classifier:** A Gradient Boosting Classifier models the prior probability distribution $P_{behav}(\text{Grade} = g)$ from patient age, duration, HbA1c, and lifestyle metrics.
2. **Bayesian Linear Opinion Pool:**
   $$P_{combined}(\text{Grade} = g) = \alpha \cdot P_{CNN}(\text{Grade} = g) + (1 - \alpha) \cdot P_{behav}(\text{Grade} = g)$$
   Where $\alpha = 0.70$, weighting retinal imaging as the primary diagnostic signal while incorporating systemic behavioral cofactors as calibrated clinical priors.
3. **Continuous Risk Score:**
   $$\text{Risk Score} = 25 \times \sum_{g=0}^4 g \cdot P_{combined}(\text{Grade} = g) \quad \in [0, 100]$$

All models are serialized in `outputs/models/` for real-time inference in `app.py`.

---

## 5. Figures & Saved Models
- **300 DPI Modeling & SHAP Figure:** [`outputs/figures/t15_behavioral_modeling_shap.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t15_behavioral_modeling_shap.png)
- **Trained Gradient Boosting BMI Model:** [`outputs/models/bmi_gradient_boosting.joblib`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/models/bmi_gradient_boosting.joblib)
- **Behavioral Retinopathy Classifier:** [`outputs/models/dr_behavioral_risk_classifier.joblib`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/models/dr_behavioral_risk_classifier.joblib)
- **K-Means Cluster Profiler:** [`outputs/models/behavioral_clustering_kmeans.joblib`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/models/behavioral_clustering_kmeans.joblib)
- **Interactive Jupyter Notebook:** [`notebooks/t15_behavioral_factors_part2.ipynb`](file:///c:/Users/anami/OneDrive/Desktop/Health/notebooks/t15_behavioral_factors_part2.ipynb)
