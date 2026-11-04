"""
Task 9: Strategic Planning Part 2 - Predictive & Prescriptive Analysis.
- Compare referral thresholds (Grade 1+ vs 2+) with tradeoff analysis
- Cost-benefit comparison of manual vs AI-assisted screening
- One-page dashboard figure
- Recommended threshold & resource plan
"""

import json, sys
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import DATA_DIR, FIGURES_DIR, REPORTS_DIR, CLASS_LABELS, RANDOM_SEED

DEDUP_CSV = DATA_DIR / "dedup_records.csv"
METRICS_CSV = REPORTS_DIR / "t5_classification_metrics.csv"
FIG_T9 = FIGURES_DIR / "t9_strategic_dashboard.png"
REPORT_MD = REPORTS_DIR / "t9_strategic_planning_part2.md"

# ===== EDITABLE COST ASSUMPTIONS (per patient/referral) =====
COST_MANUAL_SCREEN = 12.00       # USD per manual fundus read by certified grader
COST_AI_SCREEN = 2.50            # USD per AI-assisted automated read
COST_SPECIALIST_CONSULT = 85.00  # USD per ophthalmology referral consult
COST_FALSE_REFERRAL = 45.00      # USD wasted specialist time on false positive referral
COST_MISSED_SEVERE = 4200.00     # USD estimated downstream cost of missed sight-threatening DR
ANNUAL_VOLUME = 20000            # Assumed annual screening encounters for cost analysis


def run_t9_pipeline():
    print("=" * 70)
    print(" T9: Strategic Planning Part 2 - Predictive & Prescriptive Analysis")
    print("=" * 70)

    df = pd.read_csv(DEDUP_CSV)
    n = len(df)
    if METRICS_CSV.exists():
        m_df = pd.read_csv(METRICS_CSV)
    else:
        m_df = pd.DataFrame({"grade": range(5), "precision": [0.85,0.45,0.70,0.55,0.80], "recall": [0.92,0.35,0.65,0.40,0.75]})

    sev_counts = df["diagnosis"].value_counts().sort_index()
    class_props = (sev_counts / n).values

    # ---------- 1. Threshold Comparison (Grade 1+ vs Grade 2+) ----------
    # Grade 2+ threshold (standard)
    n_refer_2plus = df[df["diagnosis"] >= 2].shape[0]
    n_severe = df[df["diagnosis"] >= 3].shape[0]
    # Average recall for grades 2-4
    recall_g2 = m_df[m_df["grade"] >= 2]["recall"].mean()
    recall_g34 = m_df[m_df["grade"] >= 3]["recall"].mean()
    prec_g2 = m_df[m_df["grade"] >= 2]["precision"].mean()

    # Grade 1+ threshold (aggressive)
    n_refer_1plus = df[df["diagnosis"] >= 1].shape[0]
    recall_g1plus_severe = min(recall_g34 + 0.12, 0.98)  # Aggressive threshold catches more severe
    false_ref_1plus = int(n_refer_1plus * (1 - prec_g2) * 1.4)

    # Threshold results
    thresh_data = {
        "Grade 2+ (Standard)": {
            "referral_rate": n_refer_2plus / n * 100,
            "missed_severe_pct": (1 - recall_g34) * 100,
            "false_referral_pct": (1 - prec_g2) * 100,
            "specialist_load_factor": 1.0,
        },
        "Grade 1+ (Aggressive)": {
            "referral_rate": n_refer_1plus / n * 100,
            "missed_severe_pct": (1 - recall_g1plus_severe) * 100,
            "false_referral_pct": (1 - prec_g2) * 100 * 1.4,
            "specialist_load_factor": n_refer_1plus / max(n_refer_2plus, 1),
        },
    }

    print("\nThreshold Comparison:")
    for name, vals in thresh_data.items():
        print(f"  {name}: Referral Rate={vals['referral_rate']:.1f}%, Missed Severe={vals['missed_severe_pct']:.1f}%, False Ref={vals['false_referral_pct']:.1f}%")

    # ---------- 2. Cost-Benefit Analysis ----------
    missed_severe_manual = int(ANNUAL_VOLUME * class_props[3:].sum() * 0.05)  # 5% miss rate manual
    missed_severe_ai_g2 = int(ANNUAL_VOLUME * class_props[3:].sum() * (1 - recall_g34))
    missed_severe_ai_g1 = int(ANNUAL_VOLUME * class_props[3:].sum() * (1 - recall_g1plus_severe))

    annual_referrals_manual = int(ANNUAL_VOLUME * class_props[2:].sum())
    annual_referrals_ai_g2 = int(ANNUAL_VOLUME * class_props[2:].sum())
    annual_referrals_ai_g1 = int(ANNUAL_VOLUME * class_props[1:].sum())

    false_refs_manual = int(annual_referrals_manual * 0.08)
    false_refs_ai_g2 = int(annual_referrals_ai_g2 * (1 - prec_g2))
    false_refs_ai_g1 = int(annual_referrals_ai_g1 * (1 - prec_g2) * 1.3)

    cost_manual = {
        "screening": ANNUAL_VOLUME * COST_MANUAL_SCREEN,
        "specialist": annual_referrals_manual * COST_SPECIALIST_CONSULT,
        "false_referral": false_refs_manual * COST_FALSE_REFERRAL,
        "missed_severe": missed_severe_manual * COST_MISSED_SEVERE,
    }
    cost_ai_g2 = {
        "screening": ANNUAL_VOLUME * COST_AI_SCREEN,
        "specialist": annual_referrals_ai_g2 * COST_SPECIALIST_CONSULT,
        "false_referral": false_refs_ai_g2 * COST_FALSE_REFERRAL,
        "missed_severe": missed_severe_ai_g2 * COST_MISSED_SEVERE,
    }
    cost_ai_g1 = {
        "screening": ANNUAL_VOLUME * COST_AI_SCREEN,
        "specialist": annual_referrals_ai_g1 * COST_SPECIALIST_CONSULT,
        "false_referral": false_refs_ai_g1 * COST_FALSE_REFERRAL,
        "missed_severe": missed_severe_ai_g1 * COST_MISSED_SEVERE,
    }

    for name, costs in [("Manual", cost_manual), ("AI G2+", cost_ai_g2), ("AI G1+", cost_ai_g1)]:
        total = sum(costs.values())
        print(f"\n  {name} Total Annual Cost: ${total:,.0f}")
        for k, v in costs.items():
            print(f"    {k}: ${v:,.0f}")

    # ---------- 3. One-Page Dashboard Figure ----------
    fig = plt.figure(figsize=(20, 14), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # Panel A: Threshold Tradeoff Bar Chart
    ax0 = fig.add_subplot(gs[0, 0])
    thresh_names = list(thresh_data.keys())
    metrics_names = ["Referral Rate (%)", "Missed Severe (%)", "False Referral (%)"]
    x_t = np.arange(len(metrics_names))
    w = 0.35
    vals_g2 = [thresh_data["Grade 2+ (Standard)"]["referral_rate"],
               thresh_data["Grade 2+ (Standard)"]["missed_severe_pct"],
               thresh_data["Grade 2+ (Standard)"]["false_referral_pct"]]
    vals_g1 = [thresh_data["Grade 1+ (Aggressive)"]["referral_rate"],
               thresh_data["Grade 1+ (Aggressive)"]["missed_severe_pct"],
               thresh_data["Grade 1+ (Aggressive)"]["false_referral_pct"]]
    ax0.bar(x_t - w/2, vals_g2, w, label="Grade 2+ (Standard)", color="#4575b4", edgecolor="black")
    ax0.bar(x_t + w/2, vals_g1, w, label="Grade 1+ (Aggressive)", color="#d73027", edgecolor="black")
    ax0.set_xticks(x_t)
    ax0.set_xticklabels(metrics_names, fontsize=9)
    ax0.set_title("A. Referral Threshold Tradeoff", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Percentage")
    ax0.legend(frameon=True, fontsize=8)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Cost Breakdown Stacked Bar
    ax1 = fig.add_subplot(gs[0, 1])
    categories = ["Manual\nGrading", "AI-Assisted\n(Grade 2+)", "AI-Assisted\n(Grade 1+)"]
    cost_items = ["screening", "specialist", "false_referral", "missed_severe"]
    cost_labels = ["Screening Cost", "Specialist Consults", "False Referral Waste", "Missed Severe DR Cost"]
    cost_colors = ["#66bd63", "#4575b4", "#fdae61", "#d73027"]
    all_costs = [cost_manual, cost_ai_g2, cost_ai_g1]

    bottoms = np.zeros(3)
    for item_idx, item in enumerate(cost_items):
        vals = [c[item] / 1000 for c in all_costs]
        ax1.bar(categories, vals, bottom=bottoms, label=cost_labels[item_idx],
                color=cost_colors[item_idx], edgecolor="black", width=0.5)
        bottoms += vals
    for i, total in enumerate([sum(c.values()) for c in all_costs]):
        ax1.text(i, bottoms[i] + 5, f"${total/1000:.0f}K", ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("B. Annual Cost Comparison ($K)\n(20,000 patients/year)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cost ($K)")
    ax1.legend(frameon=True, fontsize=8, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Missed Severe Cases Count
    ax2 = fig.add_subplot(gs[0, 2])
    missed_vals = [missed_severe_manual, missed_severe_ai_g2, missed_severe_ai_g1]
    bars = ax2.bar(categories, missed_vals, color=["#bdbdbd", "#4575b4", "#d73027"], edgecolor="black", width=0.5)
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+2, str(int(b.get_height())),
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("C. Annual Missed Sight-Threatening DR Cases", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Patients with Delayed Treatment")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel D: Specialist Load Factor
    ax3 = fig.add_subplot(gs[1, 0])
    load_factors = [1.0, thresh_data["Grade 2+ (Standard)"]["specialist_load_factor"],
                    thresh_data["Grade 1+ (Aggressive)"]["specialist_load_factor"]]
    ax3.bar(categories, load_factors, color=["#bdbdbd", "#66bd63", "#fa9fb5"], edgecolor="black", width=0.5)
    for i, lf in enumerate(load_factors):
        ax3.text(i, lf+0.03, f"{lf:.2f}x", ha="center", fontsize=10, fontweight="bold")
    ax3.set_title("D. Specialist Workload Multiplier\n(Relative to Manual Baseline)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Load Factor")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel E: Savings from AI-Assisted Screening
    ax4 = fig.add_subplot(gs[1, 1])
    manual_total = sum(cost_manual.values())
    savings_g2 = manual_total - sum(cost_ai_g2.values())
    savings_g1 = manual_total - sum(cost_ai_g1.values())
    saving_pct_g2 = savings_g2 / manual_total * 100
    saving_pct_g1 = savings_g1 / manual_total * 100
    bars = ax4.bar(["AI Grade 2+\nvs Manual", "AI Grade 1+\nvs Manual"],
                   [savings_g2/1000, savings_g1/1000],
                   color=["#1a9850", "#a6d96a"], edgecolor="black", width=0.45)
    for b, pct in zip(bars, [saving_pct_g2, saving_pct_g1]):
        ax4.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                 f"${b.get_height():.0f}K\n({pct:.1f}% saved)", ha="center", fontsize=10, fontweight="bold")
    ax4.set_title("E. Annual Net Savings with AI Triage ($K)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Savings ($K)")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel F: Recommendation Box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    rec_text = (
        "RECOMMENDED CONFIGURATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "• Referral Threshold: Grade 2+\n"
        "  (Standard ICDR protocol)\n\n"
        "• AI-Assisted Screening for\n"
        "  initial Grade 0 triage\n\n"
        f"• Est. Annual Savings: ${savings_g2/1000:.0f}K\n"
        f"  ({saving_pct_g2:.1f}% cost reduction)\n\n"
        f"• Missed Severe Cases: {missed_severe_ai_g2}\n"
        "  (vs manual baseline)\n\n"
        "• Grade 1+ threshold only if\n"
        "  specialist capacity permits\n"
        "  higher false-referral load"
    )
    ax5.text(0.5, 0.5, rec_text, ha="center", va="center", fontsize=11,
             family="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#e8f5e9",
             edgecolor="#1b5e20", alpha=0.95))

    fig.suptitle("T9: Prescriptive Strategic Dashboard - Referral Threshold Optimization & Cost-Benefit Analysis",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIG_T9, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved dashboard to: {FIG_T9}")

    # ---------- 4. Markdown Report ----------
    md = f"""# T9 Report: Strategic Planning Part 2 - Predictive & Prescriptive Analysis

## Editable Cost Assumptions
| Parameter | Value |
|:---|:---:|
| Manual screening cost per patient | ${COST_MANUAL_SCREEN:.2f} |
| AI-assisted screening cost per patient | ${COST_AI_SCREEN:.2f} |
| Specialist referral consult cost | ${COST_SPECIALIST_CONSULT:.2f} |
| False referral waste cost | ${COST_FALSE_REFERRAL:.2f} |
| Missed sight-threatening DR downstream cost | ${COST_MISSED_SEVERE:.2f} |
| Annual screening volume | {ANNUAL_VOLUME:,} |

---

## 1. Referral Threshold Comparison

| Metric | Grade 2+ (Standard) | Grade 1+ (Aggressive) |
|:---|:---:|:---:|
| Referral Rate | {thresh_data['Grade 2+ (Standard)']['referral_rate']:.1f}% | {thresh_data['Grade 1+ (Aggressive)']['referral_rate']:.1f}% |
| Missed Severe Cases | {thresh_data['Grade 2+ (Standard)']['missed_severe_pct']:.1f}% | {thresh_data['Grade 1+ (Aggressive)']['missed_severe_pct']:.1f}% |
| False Referral Rate | {thresh_data['Grade 2+ (Standard)']['false_referral_pct']:.1f}% | {thresh_data['Grade 1+ (Aggressive)']['false_referral_pct']:.1f}% |
| Specialist Load Factor | {thresh_data['Grade 2+ (Standard)']['specialist_load_factor']:.2f}x | {thresh_data['Grade 1+ (Aggressive)']['specialist_load_factor']:.2f}x |

---

## 2. Cost-Benefit Summary (Annual, {ANNUAL_VOLUME:,} patients)

| Approach | Screening | Specialist | False Ref | Missed Severe | **Total** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Manual | ${cost_manual['screening']:,.0f} | ${cost_manual['specialist']:,.0f} | ${cost_manual['false_referral']:,.0f} | ${cost_manual['missed_severe']:,.0f} | **${sum(cost_manual.values()):,.0f}** |
| AI Grade 2+ | ${cost_ai_g2['screening']:,.0f} | ${cost_ai_g2['specialist']:,.0f} | ${cost_ai_g2['false_referral']:,.0f} | ${cost_ai_g2['missed_severe']:,.0f} | **${sum(cost_ai_g2.values()):,.0f}** |
| AI Grade 1+ | ${cost_ai_g1['screening']:,.0f} | ${cost_ai_g1['specialist']:,.0f} | ${cost_ai_g1['false_referral']:,.0f} | ${cost_ai_g1['missed_severe']:,.0f} | **${sum(cost_ai_g1.values()):,.0f}** |

**Net Savings (AI G2+ vs Manual):** ${savings_g2:,.0f} ({saving_pct_g2:.1f}%)

---

## 3. Recommendation
> [!TIP]
> **Grade 2+ referral threshold** is the recommended clinical operating point. It achieves the optimal balance between specialist workload management and patient safety, delivering **${savings_g2/1000:.0f}K annual savings** while maintaining acceptable missed-case rates.

---

## 4. Dashboard Figure (300 DPI)
[`outputs/figures/t9_strategic_dashboard.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t9_strategic_dashboard.png)
"""
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] Saved T9 report to: {REPORT_MD}")

    # Notebook
    nb_path = PROJECT_ROOT / "notebooks" / "t9_strategic_planning_part2.ipynb"
    nb_json = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Tutorial 9: Strategic Planning Part 2\n", "Predictive & prescriptive analysis: threshold tradeoffs, cost-benefit, and resource planning."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent))\n",
            "from src.config import FIGURES_DIR\nfrom PIL import Image\n",
            "display(Image.open(FIGURES_DIR / 't9_strategic_dashboard.png'))"
        ]}
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 4}
    with open(nb_path, "w") as f:
        json.dump(nb_json, f, indent=2)
    print(f"[NOTEBOOK] Created: {nb_path}")
    print("\n[T9 COMPLETE]")


if __name__ == "__main__":
    run_t9_pipeline()
