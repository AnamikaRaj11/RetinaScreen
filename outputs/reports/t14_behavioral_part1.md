# T14 Report: Statistical Analysis & Complexity of Adult Behavioral Factors

> [!IMPORTANT]
> All demographic, lifestyle, and behavioral records in this tutorial are **SIMULATED** ($N = 3,000$ adult diabetic records). The data incorporates non-linear relationships, multi-way interactions, missingness, and realistic outliers. All underlying generation equations are detailed in [`outputs/reports/sim_assumptions.md`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/sim_assumptions.md).

---

## 1. Data Cleaning & Integrity Audit

- **Raw Cohort Size:** 3,000 simulated patient encounters.
- **Missing Value Imputation:**
  - `physical_activity_min_week`: 120 missing $\rightarrow$ Imputed with sex-stratified median.
  - `sleep_hours`: 139 missing $\rightarrow$ Imputed with sex-stratified median.
  - `daily_calories`: 129 missing $\rightarrow$ Imputed with sex-stratified median.
  - `stress_score`: 103 missing $\rightarrow$ Imputed with median.
- **Outlier Winsorization:**
  - `weight_kg`: 18 physiological extremes capped to $[40.0, 180.0]$ kg; BMI recalculated.
  - `daily_calories`: 15 reporting anomalies capped to $[1000, 4800]$ kcal/day.
- **Clean Output File:** [`data/clean_sim_behavioral_factors.csv`](file:///c:/Users/anami/OneDrive/Desktop/Health/data/clean_sim_behavioral_factors.csv)

---

## 2. Statistical Findings & Complexity of Behavioral Factors

### Correlation Structure
- **Physical Activity:** Exhibited significant negative correlation with BMI ($r = -0.38$, Spearman $\rho = -0.41$), demonstrating diminishing returns past $180$ minutes/week.
- **Stress & Sleep:** Chronic stress was positively linked to daily caloric intake ($r = +0.28$) and negatively linked to restorative sleep ($r = -0.32$), driving elevated adiposity.
- **Glycemic Severity Link:** HbA1c showed strong coupling with diabetes duration ($r = +0.45$) and BMI ($r = +0.33$).

### Group Comparisons across Retinopathy Severity Grades (One-Way ANOVA)
| Clinical Biomarker | ANOVA F-Statistic | p-Value | Effect Size ($\eta^2$) | Interpretation |
|:---|:---:|:---:|:---:|:---|
| **HbA1c (%)** | **351.16** | **3.70e-248** | **0.3193** | Large biological effect on microvascular damage |
| **Diabetes Duration (yrs)** | **520.61** | **0.00e+00** | **0.4101** | Cumulative chronic vascular exposure |
| **Body Mass Index (kg/m$^2$)** | **59.91** | **9.80e-49** | **0.0741** | Moderate systemic metabolic influence |
| **Physical Activity (min/wk)** | **31.09** | **2.09e-25** | **0.0399** | Protective lifestyle cofactor |

### Multicollinearity Diagnostics (VIF)
All behavioral variables demonstrated Variance Inflation Factors (VIF) $< 3.5$, confirming the absence of fatal multicollinearity and validating feature stability for multivariate modeling in T15.

### Latent Behavioral Dimensions (Factor Analysis / PCA)
The top 3 principal components explained **60.1%** of behavioral variance:
1. **Dimension 1 (Metabolic Strain):** Heavy positive loadings on `daily_calories`, `stress_score`, and negative loadings on `sleep_hours`.
2. **Dimension 2 (Active Lifestyle):** Dominant positive loading on `physical_activity_min_week`.
3. **Dimension 3 (Glycemic Chronicity):** High loadings on `diabetes_duration_years` and `hba1c`.

---

## 3. Figures & Linked Data Artifacts
- **300 DPI Statistical EDA Figure:** [`outputs/figures/t14_behavioral_factors_eda.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t14_behavioral_factors_eda.png)
- **Cleaned Dataset:** [`data/clean_sim_behavioral_factors.csv`](file:///c:/Users/anami/OneDrive/Desktop/Health/data/clean_sim_behavioral_factors.csv)
- **Simulation Assumptions:** [`outputs/reports/sim_assumptions.md`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/sim_assumptions.md)
- **Interactive Jupyter Notebook:** [`notebooks/t14_behavioral_factors_part1.ipynb`](file:///c:/Users/anami/OneDrive/Desktop/Health/notebooks/t14_behavioral_factors_part1.ipynb)
