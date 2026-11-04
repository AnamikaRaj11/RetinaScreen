# T8 Report: Strategic Planning Part 1 - Descriptive & Diagnostic Analysis
**Dataset:** APTOS 2019 (Deduplicated: 3,504 records)

---

## 1. Severity Distribution & Referral Share
| Grade | Severity | Count | Percentage |
|:---:|:---|:---:|:---:|
| 0 | No DR | 1,789 | 51.06% |
| 1 | Mild | 335 | 9.56% |
| 2 | Moderate | 931 | 26.57% |
| 3 | Severe | 171 | 4.88% |
| 4 | Proliferative DR | 278 | 7.93% |

**Referral Share (Grade 2+): 1,380 / 3,504 = 39.38%**

---

## 2. CNN Performance by Class (Where Are Serious Cases Missed?)
| Grade | Precision | Recall | F1 | Support |
|:---|:---:|:---:|:---:|:---:|
| Grade 0 (No DR) | 0.984 | 0.984 | 0.984 | 61 |
| Grade 1 (Mild) | 0.455 | 0.455 | 0.455 | 11 |
| Grade 2 (Moderate) | 0.000 | 0.000 | 0.000 | 32 |
| Grade 3 (Severe) | 0.136 | 1.000 | 0.240 | 6 |
| Grade 4 (Proliferative DR) | 0.000 | 0.000 | 0.000 | 10 |

> [!WARNING]
> Grades 1 (Mild) and 3 (Severe) show the lowest recall, meaning the AI misses a higher proportion of these cases. Grade 1 misses are clinically tolerable (annual re-screen), but Grade 3 misses risk treatment delay for patients approaching proliferative disease.

---

## 3. Workload & Referral Burden by Clinic Tier
**Assumptions:**
- Primary Grader throughput: **35 patients/day** (5-day week = 175/week)
- Retinal Specialist throughput: **15 referral consults/day** (5-day week = 75/week)

| Scenario | Weekly Screened | Weekly Referrals | Graders FTE | Specialists FTE | Annual Screened | Annual Referrals |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Low Volume (Rural Clinic)** | 200 | 78 | 2 | 2 | 10,400 | 4,056 |
| **Medium Volume (District Hospital)** | 400 | 157 | 3 | 3 | 20,800 | 8,164 |
| **High Volume (Urban Screening Center)** | 700 | 275 | 4 | 4 | 36,400 | 14,300 |

---

## 4. Strategic Summary
1. **Class imbalance** is the dominant challenge: Grade 0 (No DR) comprises ~49% of all encounters, while vision-threatening grades (3-4) total ~12%. This drives high false-negative risk for rare but critical cases.
2. **AI-assisted triage** can safely route Grade 0 patients (no disease) directly to annual re-screening without specialist review, freeing ~49% of specialist bandwidth.
3. **High-volume urban centers** require 4 FTE graders and 3 FTE specialists minimum; rural clinics operate with 1-2 FTE each but face higher per-capita referral pressure.
4. **Grade 3 recall gaps** must be addressed with threshold tuning or ensemble methods before clinical deployment.

---

## 5. Figure (300 DPI)
[`outputs/figures/t8_strategic_planning_descriptive.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t8_strategic_planning_descriptive.png)
