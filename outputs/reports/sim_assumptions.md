# Simulation Assumptions: Adult Behavioral Factors and Diabetic Retinopathy Risk
**Dataset:** `data/sim_behavioral_factors.csv` (N = 3,000 simulated adult diabetic encounters)

> [!IMPORTANT]
> This dataset is **100% SIMULATED / SYNTHETIC**. All patient demographics, lifestyle behaviors, biometric variables, and relationships were generated programmatically to study behavioral complexity and multimodal risk stratification in diabetic eye disease screening.

---

## 1. Demographic & Clinical Parameters

| Variable | Distribution / Source | Clinical Rationale / Formula | Bounding Range |
|:---|:---|:---|:---:|
| `age` | Normal $\mathcal{N}(56.5, 10.8)$ | Adult diabetic screening population profile | [25, 85] years |
| `sex` | Bernoulli ($p_{\text{male}} = 0.51$) | Balanced gender representation | 'Male', 'Female' |
| `height_cm` | Male $\mathcal{N}(175.2, 6.8)$, Female $\mathcal{N}(162.4, 6.2)$ | Standard anthropometric sex distributions | [142, 202] cm |
| `diabetes_duration_years` | Gamma($k=2.2, \theta=4.2$) + $0.08 \times (\text{age}-50)$ | Duration positively correlated with patient age | [0.5, 38.0] years |

---

## 2. Behavioral Lifestyle Factors

| Variable | Underlying Distribution | Functional Relationships & Non-linearities | Missingness Rate |
|:---|:---|:---|:---:|
| `physical_activity_min_week` | Log-normal ($\mu=4.7, \sigma=0.75$) | Declines with age ($-1.2$ min/yr) and duration ($-2.5$ min/yr) | 4.2% missing |
| `daily_calories_kcal` | Male $\mathcal{N}(2250, 280)$, Female $\mathcal{N}(1880, 240)$ | Positively modulated by stress ($+25$ kcal/pt); non-linear appetite | 3.8% missing |
| `sleep_hours` | Normal $\mathcal{N}(6.85, 1.15)$ | Negative correlation with stress score ($-0.18$ hrs/pt) | 3.5% missing |
| `stress_score` | Discrete scale 1 to 10 | Modulated by sleep deprivation, smoking, and diabetes duration | 4.0% missing |
| `smoking_status` | Categorical ('Never', 'Former', 'Current') | Prevalences: Never (54%), Former (28%), Current (18%) | 0.0% |
| `alcohol_drinks_week` | Zero-inflated Poisson ($\lambda=2.8$) | Right-skewed alcohol consumption pattern | 4.5% missing |

---

## 3. Biometric Weight & BMI Generating Formula
Body Mass Index (BMI) incorporates linear, non-linear, and interaction effects:

$$\text{BMI} = \text{BMI}_{\text{base}} + \Delta_{\text{cal}} + \Delta_{\text{act}} + \Delta_{\text{sleep}} + \Delta_{\text{stress}} + \Delta_{\text{age}\times\text{act}} + \Delta_{\text{duration}} + \epsilon$$

Where:
- $\text{BMI}_{\text{base}} = 24.2 + (0.5 \text{ if Female else } 0.0)$
- $\Delta_{\text{cal}} = \frac{\text{daily\_calories} - 2000}{190}$
- $\Delta_{\text{act}} = -1.9 \times \ln\left(1 + \frac{\text{activity}}{35}\right)$ *(Diminishing returns to physical activity)*
- $\Delta_{\text{sleep}} = 1.15 \times \max(0, 7.0 - \text{sleep\_hours})^{1.25}$ *(Non-linear metabolic penalty for sleep deprivation)*
- $\Delta_{\text{stress}} = 0.32 \times \text{stress\_score}$ *(Cortisol-mediated adiposity)*
- $\Delta_{\text{age}\times\text{act}} = 0.025 \times (\text{age} - 50) - 0.00028 \times (\text{age} - 50) \times \text{activity}$ *(Age-dependent metabolic efficiency)*
- $\Delta_{\text{duration}} = 0.11 \times \text{diabetes\_duration}$
- $\epsilon \sim \mathcal{N}(0, 2.15)$ *(Individual genetic and metabolic variance)*

**Body Weight (kg)** is derived deterministically:
$$\text{weight\_kg} = \text{BMI} \times \left(\frac{\text{height\_cm}}{100}\right)^2 + \mathcal{N}(0, 0.4)$$

---

## 4. Glycemic Control (HbA1c) & Retinopathy Severity Coupling
Glycemic control is modeled as:
$$\text{HbA1c} = 5.8 + 0.075 \times \text{diabetes\_duration} + 0.052 \times (\text{BMI} - 25) - 0.0028 \times \text{activity} + 0.07 \times \text{stress} + \mathcal{N}(0, 0.65)$$

**Diabetic Retinopathy Severity (Grades 0 to 4):**
Retinopathy severity is assigned through a latent microvascular risk index:
$$R_{\text{latent}} = 0.38 \times (\text{HbA1c} - 7.5) + 0.085 \times (\text{diabetes\_duration} - 8) + 0.035 \times (\text{BMI} - 28) + 0.015 \times (\text{age} - 55) + \mathcal{N}(0, 0.85)$$

The quantile thresholds of $R_{\text{latent}}$ are calibrated exactly to the **empirical APTOS 2019 class proportions**:
- Grade 0 (No DR): 49.3%
- Grade 1 (Mild NPDR): 10.1%
- Grade 2 (Moderate NPDR): 27.3%
- Grade 3 (Severe NPDR): 5.3%
- Grade 4 (Proliferative DR): 8.1%

---

## 5. Synthetic Outlier Injections
To simulate realistic clinical EHR noise and test data cleaning pipelines:
1. **Weight Outliers:** 0.8% of entries injected with extreme physiological weights (e.g., 185–215 kg or typographical scale drops < 38 kg).
2. **Activity Outliers:** 0.6% of entries injected with unverified activity logs (> 900 min/week).
3. **Calorie Outliers:** 0.7% of entries injected with extreme reporting errors (> 5,200 kcal/day).

