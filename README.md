# RetinaScreen: AI-Assisted Diabetic Retinopathy Screening and Referral Planning

An end-to-end biomedical image processing, deep learning, clinical NLP, behavioral modeling, and strategic capacity planning framework for diabetic retinopathy (DR) grading and clinical referral planning from retinal fundus photographs using the **APTOS 2019 Blindness Detection** dataset and simulated multimodal clinical records.

---

## 🖥️ Interactive Gradio Clinical Web Application (`app.py`)

RetinaScreen features a comprehensive, deployment-ready web application built with Gradio (`gr.Blocks` with 8 specialized diagnostic tabs).

### Quickstart: Launch the Application
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Launch the Gradio web server
python app.py
```
Open your browser at **`http://127.0.0.1:7860`**.

### Input Capabilities:
- **Retinal Fundus Image Upload:** Accepts standard PNG/JPEG fundus photographs.
- **Optional Adult Lifestyle & Clinical Inputs:** Age, sex, weight, height, physical activity minutes/week, sleep hours/night, smoking history, diabetes duration, and HbA1c (%).
- **Quick Preset Buttons:** Pre-loaded test cases for **Normal (Grade 0)**, **Moderate NPDR (Grade 2)**, and **Proliferative DR (Grade 4)**.
- **Input Validation:** Friendly error messages for missing or corrupt image uploads.

### Output Tabs:
1. **Clinical Summary:** Predicted ICDR severity (Grade 0 to 4), probability confidence bars, clinical referral recommendation (No Concern / Monitor / Refer to Ophthalmology / Urgent Specialist Referral / Emergency Vitreoretinal Consultation), and automated quality audit warnings (brightness and blur checks).
2. **Biomedical Preprocessing (T2–T4):** Luminance CLAHE enhancement, bilateral denoising, morphological retinal vessel segmentation, and bright lesion (hard exudate) localization.
3. **Explainability (T5–T6):** Grad-CAM visual attention heatmap highlighting causative microvascular lesions and nearest-neighbor case retrieval from the APTOS dataset.
4. **Latent Embedding Map (T13):** Dynamic 2D PCA projection of the clinical manifold showing where the uploaded patient image clusters relative to the population.
5. **Clinical EHR Notes (T10–T12):** SYNTHETIC report-style progress note, spaCy named entity recognition (laterality, lesions, severity statements, action plans), and BM25 triage retrieval.
6. **Capacity Planning (T7–T9):** SIMULATED referral threshold operating trade-offs and staffing demand projections.
7. **Behavioral Risk (T14–T15):** SIMULATED behavioral weight modeling, predicted BMI, behavioral profile cluster, SHAP factor contributions, and multimodal integrated risk score ($0.70 \times P_{\text{CNN}} + 0.30 \times P_{\text{behav}}$).
8. **Download Clinical Report:** One-click generation and download of formatted **PDF** and structured **JSON** clinical screening reports with academic research disclaimers.

---

## 📁 Project Architecture

```
Health/
├── .venv/                      # Python 3.12 virtual environment (pre-installed dependencies)
├── app.py                      # Master 8-Tab Gradio Clinical Web Application
├── data/                       # APTOS 2019 dataset and processed artifacts
│   ├── train_images/           # 3,662 raw fundus images (.png)
│   ├── test_images/            # 1,928 unlabelled test fundus images (.png)
│   ├── train.csv               # Ground truth labels (id_code, diagnosis)
│   ├── test.csv                # Unlabelled test entries
│   ├── clean_records.csv       # Standardized & quality-filtered records
│   ├── dedup_records.csv       # Deduplicated unique patient encounters (N=3,504)
│   ├── sim_behavioral_factors.csv # 3,000 raw SIMULATED adult behavioral records
│   ├── clean_sim_behavioral_factors.csv # Imputed & cleaned behavioral dataset
│   ├── synthetic_notes.csv     # 3,504 SYNTHETIC clinical EHR progress notes
│   ├── extracted_clinical_entities.csv # Extracted structured clinical entities
│   ├── sim_weekly_screening_schedule.csv # 104-week simulated screening time series
│   └── processed/
│       ├── train_512/          # Cropped & standardized 512x512 images
│       └── train_224/          # Normalized 224x224 images for deep CNN training
├── notebooks/                  # 15 Interactive Jupyter Walkthrough Notebooks
│   ├── t1_understanding_ehr.ipynb
│   ├── t2_ehr_preprocessing.ipynb
│   ├── t3_redundant_missing_data.ipynb
│   ├── t4_biomedical_image_processing.ipynb
│   ├── t5_biomedical_image_analysis.ipynb
│   ├── t6_biomedical_image_visualization.ipynb
│   ├── t7_patient_prediction_staffing.ipynb
│   ├── t8_strategic_planning_part1.ipynb
│   ├── t9_strategic_planning_part2.ipynb
│   ├── t10_clinical_text_classification.ipynb
│   ├── t11_clinical_information_extraction.ipynb
│   ├── t12_ranking_and_disambiguation.ipynb
│   ├── t13_dimensionality_reduction.ipynb
│   ├── t14_behavioral_factors_part1.ipynb
│   └── t15_behavioral_factors_part2.ipynb
├── outputs/
│   ├── figures/                # 24 Publication-Quality 300 DPI Visualizations
│   │   └── ui_tests/           # End-to-end UI verification outputs & sample PDF reports
│   ├── models/                 # Serialized PyTorch weights & Scikit-learn / Joblib models
│   │   ├── best_retinascreen_cnn.pth (ResNet-101 backbone)
│   │   ├── bmi_gradient_boosting.joblib
│   │   ├── bmi_linear_model.joblib
│   │   ├── behavioral_clustering_kmeans.joblib
│   │   ├── behavioral_feature_scaler.joblib
│   │   ├── dr_behavioral_risk_classifier.joblib
│   │   └── embedding_map_cache.joblib
│   └── reports/                # Quantitative CSV metrics & technical Markdown reports
│       ├── sim_assumptions.md  # Detailed simulation mathematical assumptions
│       ├── t14_behavioral_part1.md
│       ├── t15_behavioral_part2.md
│       └── summary.md          # Comprehensive executive & technical summary report
├── src/
│   ├── config.py               # Pathlib-based paths & global parameters (seed=42)
│   ├── verify_data.py          # Dataset verification & schema integrity checks
│   ├── t2_preprocess_ehr.py    # Image quality audit, black border crop, 512x512 resize
│   ├── t3_dedup_missing.py     # MD5 + pHash deduplication & KNN imputation
│   ├── t4_image_processing.py  # CLAHE, NLM restoration, vessel/exudate segmentation
│   ├── t5_image_analysis.py    # SIFT/RANSAC & ResNet-101 transfer learning
│   ├── t6_image_visualization.py# 300 DPI clinical charts, confusion matrix, Grad-CAM
│   ├── t7_patient_prediction.py# 2-year weekly screening & staffing demand forecast
│   ├── t8_strategic_part1.py   # Descriptive & diagnostic triage capacity modeling
│   ├── t9_strategic_part2.py   # Prescriptive health economics & threshold trade-offs
│   ├── t10_text_classification.py # Synthetic text generation, TF-IDF + LogReg/NB, K-Means
│   ├── t11_information_extraction.py # Regex/spaCy Matcher + custom trained spaCy NER
│   ├── t12_ranking_disambiguation.py # Okapi BM25 vs TF-IDF ranking & Lesk/Embedding WSD
│   ├── t13_dimensionality_reduction.py # 2048-dim CNN embeddings, PCA, t-SNE, UMAP
│   ├── t14_behavioral_part1.py # Behavioral simulation, cleaning, VIF, ANOVA, PCA
│   ├── t15_behavioral_part2.py # BMI regression, SHAP, K-Means phenotyping, multimodal fusion
│   ├── gradio_app.py           # Standalone clinical Gradio interface
│   └── test_ui_app.py          # Automated UI test suite across 3 test cases
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📊 Summary of Tutorials (T1 – T15)

| Task | Domain | Focus & Key Methodology | Primary Metric / Result |
|:---:|:---|:---|:---|
| **T1** | EHR Ingestion | APTOS 2019 dataset auditing & metadata schema synthesis | 3,662 images; 5 ICDR classes |
| **T2** | Standardization | Quality thresholds (brightness, blur) & circular cropping | 158 Suboptimal cases flagged |
| **T3** | Deduplication | Exact MD5 and Perceptual pHash (Hamming $\le 4$) | 158 Redundant pairs excluded in CSV |
| **T4** | Image Processing | NLM/Bilateral denoising, CLAHE, morphological vessels | NLM PSNR 30.1 dB, SSIM 0.845; JPEG Q80 size $-82.4\%$ |
| **T5** | Image Analysis | SIFT/RANSAC biometrics & ResNet-101 transfer learning | Accuracy 76.7%, QWK 0.824 |
| **T6** | Visualization | Class distributions, training curves, Grad-CAM saliency | 6 Publication figures at 300 DPI |
| **T7** | Forecasting | 2-year simulated screening demand & staffing conversion | 2-3 Graders, 5-8 Specialists |
| **T8** | Strategic Pt 1 | Clinical triage burden across 3 volume scenarios | 40.7% Baseline referral rate |
| **T9** | Strategic Pt 2 | Health economics comparison & G2+ referral threshold | Recommended Grade 2+ cutoff |
| **T10**| Text Mining | Synthetic note synthesis, TF-IDF + LogReg, K-Means | Purity 0.776, Acc 100% (synthetic) |
| **T11**| Information Ext | Laterality, lesions, severity, actions via spaCy NER | Macro F1: 1.000 across 4 entities |
| **T12**| IR & WSD | BM25 vs TF-IDF ranking & Lesk vs Context-Embedding WSD | Spearman 0.9888, WSD Acc 97.5% |
| **T13**| Dim Reduction | 2048-dim CNN embeddings, PCA, t-SNE, UMAP | Top 2 PCs Explain 60.68% Var |
| **T14**| Behavioral Pt 1 | 3,000 SIMULATED adults, cleaning, VIF, ANOVA, PCA | VIF < 3.5; ANOVA F = 59.9 (BMI) |
| **T15**| Behavioral Pt 2 | BMI modeling (GBR R2=0.345), SHAP, K-Means clustering, Multimodal fusion | $P_{comb} = 0.70 P_{CNN} + 0.30 P_{behav}$ |
| **UI**  | Interactive App | Full 8-tab Gradio app with PDF/JSON export (`app.py`)| Passed automated test suite |

---

## ⚠️ Clinical Disclaimer & Academic Limitations
1. **Screening Aid, Not Autonomous Diagnostic Device:** RetinaScreen is intended strictly as an artificial intelligence triaging and decision-support aid for educational and academic research. It does NOT replace definitive clinical examination by a board-certified eye care specialist.
2. **Synthetic Clinical Text (T10–T12):** Clinical notes and entity recognition benchmarks were synthesized programmatically from severity labels; observed accuracy is optimistic relative to messy hospital EHR free-text.
3. **Simulated Scheduling & Behavioral Cohorts (T7, T14–T15):** The 104-week clinic schedule and the 3,000 adult behavioral records are simulated using mathematical and pharmacological assumptions formulated in [`outputs/reports/sim_assumptions.md`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/sim_assumptions.md).
4. **Class Imbalance:** Severe NPDR (Grade 3) represents only 5.3% of the APTOS cohort, warranting careful clinical interpretation of borderline cases.
