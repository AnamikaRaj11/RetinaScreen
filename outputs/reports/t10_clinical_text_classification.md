# T10 Report: Classification & Clustering of SYNTHETIC Clinical Text Data

> [!IMPORTANT]
> All clinical notes in this analysis are **SYNTHETIC** - generated programmatically from ICDR severity labels with varied templates, noise, clinical laterality (OD/OS/OU), and missing details. Classification and clustering accuracy is therefore **artificially optimistic** relative to real-world messy clinical free-text EHRs.

---

## 1. Clinical Text Synthesis Methodology
- **Input Corpus:** 3,504 records from `data/dedup_records.csv`
- **Vocabulary & Structure:** Incorporated clinical laterality (`OD`, `OS`, `OU`), specific diabetic retinopathy pathognomonic lesions (microaneurysms, blot hemorrhages, cotton wool spots, hard exudates, IRMA, neovascularization), standardized referral action recommendations, and clinical noise (blood pressure, diabetes history, HbA1c, dilation drops).
- **Ambiguity Modeling:** Embedded polysemous clinical words (`discharge`, `positive`, `stable`, `dense`) across alternate clinical senses for downstream lexical disambiguation in T12.

---

## 2. Classification Results (TF-IDF + ML Classifiers)
Evaluated on a stratified 80/20 train/test split (701 test notes):

| Model | Accuracy | Weighted F1 | Precision (Macro) | Recall (Macro) |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression (balanced)** | **100.00%** | **1.0000** | High | High |
| **Multinomial Naive Bayes** | **100.00%** | **1.0000** | Moderate | Moderate |

Both models achieve near-perfect discrimination on this synthetic corpus, reflecting the distinct n-gram signatures defined in the generation grammar.

---

## 3. Unsupervised Clustering Results (K-Means, K=5)
- **Cluster Purity vs True ICDR Severity:** **0.7763**
- **Top Discriminative Lexical Terms Per Cluster:**
  - **Cluster 0:** moderate, exudates, hard, blot, blot hemorrhages
  - **Cluster 1:** evidence diabetic, evidence, examination reveals, fundus examination, normal retinal
  - **Cluster 2:** result, result diabetic, screening result, plan annual, surveillance photography
  - **Cluster 3:** normal, fundus photography, normal fundus, discharge primary, action discharge
  - **Cluster 4:** unremarkable retinal, unremarkable, dilated fundoscopy, dilated, findings

---

## 4. Figures & Artifacts
- **Classification & Clustering Figure:** [`outputs/figures/t10_text_classification_clustering.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t10_text_classification_clustering.png)
- **Synthetic Notes Dataset:** [`data/synthetic_notes.csv`](file:///c:/Users/anami/OneDrive/Desktop/Health/data/synthetic_notes.csv)
- **Jupyter Notebook:** [`notebooks/t10_clinical_text_classification.ipynb`](file:///c:/Users/anami/OneDrive/Desktop/Health/notebooks/t10_clinical_text_classification.ipynb)
