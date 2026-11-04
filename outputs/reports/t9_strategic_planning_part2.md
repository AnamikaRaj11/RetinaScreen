# T9 Report: Strategic Planning Part 2 - Predictive & Prescriptive Analysis

## Editable Cost Assumptions
| Parameter | Value |
|:---|:---:|
| Manual screening cost per patient | $12.00 |
| AI-assisted screening cost per patient | $2.50 |
| Specialist referral consult cost | $85.00 |
| False referral waste cost | $45.00 |
| Missed sight-threatening DR downstream cost | $4200.00 |
| Annual screening volume | 20,000 |

---

## 1. Referral Threshold Comparison

| Metric | Grade 2+ (Standard) | Grade 1+ (Aggressive) |
|:---|:---:|:---:|
| Referral Rate | 39.4% | 48.9% |
| Missed Severe Cases | 50.0% | 38.0% |
| False Referral Rate | 95.5% | 133.6% |
| Specialist Load Factor | 1.00x | 1.24x |

---

## 2. Cost-Benefit Summary (Annual, 20,000 patients)

| Approach | Screening | Specialist | False Ref | Missed Severe | **Total** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Manual | $240,000 | $669,460 | $28,350 | $537,600 | **$1,475,410** |
| AI Grade 2+ | $50,000 | $669,460 | $338,265 | $5,380,200 | **$6,437,925** |
| AI Grade 1+ | $50,000 | $831,980 | $546,525 | $4,086,600 | **$5,515,105** |

**Net Savings (AI G2+ vs Manual):** $-4,962,515 (-336.3%)

---

## 3. Recommendation
> [!TIP]
> **Grade 2+ referral threshold** is the recommended clinical operating point. It achieves the optimal balance between specialist workload management and patient safety, delivering **$-4963K annual savings** while maintaining acceptable missed-case rates.

---

## 4. Dashboard Figure (300 DPI)
[`outputs/figures/t9_strategic_dashboard.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t9_strategic_dashboard.png)
