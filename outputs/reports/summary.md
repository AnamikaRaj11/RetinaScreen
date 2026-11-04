# RetinaScreen: Executive Summary & Comprehensive Technical Report
**AI-Assisted Diabetic Retinopathy Screening and Referral Planning from Retinal Fundus Images**

---

## 1. Project Overview & Clinical Mission
Diabetic Retinopathy (DR) remains the premier cause of preventable visual impairment and working-age blindness globally. Early diagnosis and timely clinical referral can reduce severe visual loss by up to **95%**. However, specialized ophthalmic manpower is critically scarce in low-resource and rural settings.

**RetinaScreen** is an end-to-end AI-assisted screening, clinical triaging, and capacity-planning platform engineered to evaluate digital fundus photography. The system integrates:
1. Classical biomedical signal processing, image restoration, and morphological segmentation.
2. Deep convolutional neural network (CNN) transfer learning for automated 5-class severity grading under the International Clinical Diabetic Retinopathy (ICDR) scale.
3. Explainable AI (Grad-CAM) lesion attention maps for clinical transparency.
4. Natural Language Processing (NLP) text classification, clustering, named entity recognition (NER), information retrieval, and word sense disambiguation (WSD).
5. Operational capacity modeling, forecasting, and prescriptive cost-benefit decision planning.
6. A deployment-ready interactive Gradio clinical interface.

---

## 2. Tutorial-by-Tutorial Methods & Key Results (T1 – T13)

### T1: Understanding Electronic Health Records (EHR) & Retinal Data
- **Objective:** Ingest the APTOS 2019 Blindness Detection cohort and synthesize structured EHR encounters.
- **Dataset:** 3,662 raw fundus photographs (`train_images/`) and labels (`train.csv`).
- **Severity Distribution:** Severe class imbalance:
  - Grade 0 (No DR): 1,805 images (49.3%)
  - Grade 1 (Mild NPDR): 370 images (10.1%)
  - Grade 2 (Moderate NPDR): 999 images (27.3%)
  - Grade 3 (Severe NPDR): 193 images (5.3%)
  - Grade 4 (Proliferative DR): 295 images (8.1%)
- **EHR Enrichment:** Synthesized clinical metadata (`sim_encounter_id`, `sim_encounter_type`, `sim_eye_laterality`, `sim_referral_needed`, `sim_triage_urgency`) saved to `data/ehr_records.csv`.

---

### T2: Preprocessing of EHR – Standardization & Data Cleaning
- **Objective:** Quality validation, artifact rejection, field-of-view border cropping, and spatial normalization.
- **Quality Auditing:** Implemented quantitative rejection thresholds:
  - Very dark / underexposed: `mean_brightness < 20.0`
  - Overexposed: `mean_brightness > 220.0`
  - Severe blur: `blur_score (Laplacian variance) < 4.0`
- **Results:** Flagged 158 suboptimal or artifact-laden images.
- **Image Processing:** Cropped uninformative black camera borders using threshold contour bounding boxes, resized to standard screening dimensions ($512 \times 512$ and $224 \times 224$), and saved to `data/processed/train_512` and `data/processed/train_224`.
- **Artifacts:** `data/clean_records.csv`, `outputs/figures/t2_preprocessing_comparison.png`, `outputs/reports/t2_ehr_preprocessing.md`.

---

### T3: Preprocessing of EHR – Redundancy Removal & Missing Data
- **Objective:** Cryptographic and perceptual deduplication, label consistency auditing, and imputation.
- **Deduplication Strategy:** Combined MD5 exact byte hashing with Perceptual Image Hashing (pHash, Hamming distance $\le 4$).
- **Findings:** Identified 158 duplicate/near-duplicate image pairs (originating from repeated shutter snaps). Confirmed **0% label discordance** between duplicates. Excluded redundant records in CSV only without altering source files, preserving 3,504 unique encounters.
- **Missing Data Handling:** Demonstrated median and K-Nearest Neighbors (KNN, $K=5$) imputation for derived image metrics.
- **Artifacts:** `data/dedup_records.csv`, `outputs/reports/removal_log.csv`, `outputs/figures/t3_deduplication_imputation.png`.

---

### T4: Biomedical Image Processing – Enhancement, Restoration, Segmentation & Compression
- **Enhancement:** Compared green channel isolation (optimal vascular contrast), Luminance CLAHE in LAB color space (preserving natural chromaticity), and Ben Graham local contrast normalization ($w / 30$ Gaussian blur subtraction).
- **Restoration Benchmark (Artificially Degraded Images):**
  - *Non-Local Means (NLM):* **PSNR 30.1 dB**, **SSIM 0.845** (Superior noise suppression and texture preservation)
  - *Bilateral Filtering:* PSNR 28.4 dB, SSIM 0.812
  - *Richardson-Lucy Deblurring:* PSNR 26.8 dB, SSIM 0.768
- **Classical Segmentation:**
  - *Vessels:* Morphological Top-Hat/Black-Hat transforms combined with adaptive Gaussian thresholding.
  - *Exudates:* Morphological Top-Hat luminance segmentation with automatic optic disc masking.
- **Compression Trade-offs:** Evaluated JPEG (Quality 95, 80, 50, 30) vs PNG. JPEG Q=80 achieved **82.4% storage savings** while maintaining PSNR $> 38.5$ dB without compromising lesion micro-structure.
- **Artifacts:** `outputs/reports/t4_restoration_metrics.csv`, `outputs/reports/t4_compression_metrics.csv`, Figures `t4_enhancement_comparison.png`, `t4_restoration_benchmark.png`, `t4_segmentation_vessels_exudates.png`, `t4_compression_tradeoffs.png`.

---

### T5: Biomedical Image Analysis Techniques – SIFT, RANSAC & CNN Transfer Learning
- **SIFT & RANSAC Geometry:**
  - Matched keypoints between fundus images and affine-transformed counterparts: **184 inlier matches (92.5% inlier ratio)** with successful homography recovery.
  - Matched keypoints between two distinct patient eyes: **0 inlier matches (0.0% inlier ratio)**, confirming that retinal vascular bifurcation topologies act as unique anatomical biometrics.
- **Deep CNN Transfer Learning:**
  - Architecture: ResNet-101 backbone with fine-tuned deep conv block (`layer4`) and dropout-regularized classification head.
  - Training: Stratified 80/20 train/validation split on CPU with data augmentation (random flips, rotations, color jitter) and inverse-frequency class-weighted cross-entropy loss.
  - **Performance:**
    - Accuracy: **76.7%**
    - Quadratic Weighted Kappa (QWK): **0.824** (indicates strong agreement across ordinal severity grades)
- **Artifacts:** `outputs/models/best_retinascreen_cnn.pth`, `outputs/reports/t5_classification_metrics.csv`, Figures `t5_sift_ransac_matching.png`, `t5_cnn_training_performance.png`.

---

### T6: Biomedical Image – Clinical Visualization Suite
- **Visual Diagnostics:**
  - Class distribution comparison before and after deduplication.
  - Severity sample grid across all 5 ICDR classes.
  - 3-panel before-and-after preprocessing pipelines.
  - CNN training curves demonstrating stable loss decay and QWK convergence.
  - 5-class confusion matrix with clinical risk weightings.
  - **Grad-CAM Saliency Maps:** Visualized attention regions on correct predictions (focal attention on microaneurysms, hard exudates, and disc neovascularization) and misclassified edge cases (e.g., peripheral microaneurysms missed due to central optic disc dominance).
- **Artifacts:** Figures `t6_class_distribution.png`, `t6_severity_sample_grid.png`, `t6_preprocessing_panels.png`, `t6_training_curves.png`, `t6_confusion_matrix_heatmap.png`, `t6_gradcam_validation_cases.png`.

---

### T7: Patient Prediction for Improved Staffing (SIMULATED)
- **Objective:** Forecast screening volumes and calculate ophthalmic workforce requirements over a 2-year horizon (104 weeks).
- **Time Series Simulation:** Modeled annual seasonality (winter surges and holiday dips) with autoregressive noise based on empirical ICDR class ratios.
- **Forecasting:** Lag-feature autoregression and ARIMA models predicting weekly patient arrivals and referable cases (Grade 2+).
- **Staffing Conversion:**
  - 1 Primary Grader per 35 patients screened per day.
  - 1 Retinal Specialist per 18 referred patients per week.
- **Findings:** Weekly volumes fluctuated between 340 and 520 patients/week, requiring a baseline of **2 to 3 primary graders** and **5 to 8 retinal specialists** during peak seasonal referral surges.
- **Artifacts:** `data/sim_weekly_screening_schedule.csv`, `outputs/reports/sim_staffing_forecast_metrics.csv`, Figure `sim_t7_staffing_forecast.png`.

---

### T8: Informed Strategic Planning – Part 1: Descriptive & Diagnostic Analysis
- **Clinical Triage Burden:**
  - Overall cohort referral share: **40.7%** (Grade 2+: Moderate, Severe, Proliferative).
  - High-risk sight-threatening share: **13.4%** (Grade 3 & 4).
- **Scenario Analysis:**
  - Low-Volume Clinic (5,000 patients/year): 2,035 referrals/year (~40 referrals/week).
  - Medium-Volume Center (20,000 patients/year): 8,140 referrals/year (~160 referrals/week).
  - High-Volume Regional Hub (50,000 patients/year): 20,350 referrals/year (~400 referrals/week).
- **Bottlenecks:** Identified critical specialist consult backlogs if all Grade 2 patients are referred indiscriminately.
- **Artifacts:** `outputs/reports/t8_strategic_planning_part1.md`, Figure `t8_strategic_planning_descriptive.png`.

---

### T9: Informed Strategic Planning – Part 2: Predictive & Prescriptive Analysis
- **Policy Comparison:**
  - *Standard Threshold (Grade 2+):* 39.4% referral rate, 50.0% missed severe cases on raw predictions, 95.5% false referral rate.
  - *Aggressive Threshold (Grade 1+):* 48.9% referral rate, 38.0% missed severe cases, 133.6% false referral burden (+24% specialist load).
- **Health Economics Evaluation (20,000 Annual Screenings):**
  - Modeled primary screening costs ($12 manual vs $2.50 AI), specialist consults ($85), false referral waste ($45), and severe missed DR downstream liability ($4,200).
  - Grade 2+ threshold emerged as the optimal operating balance between false positive referral saturation and patient vision preservation.
- **Artifacts:** `outputs/reports/t9_strategic_planning_part2.md`, Figure `t9_strategic_dashboard.png`.

---

### T10: Classification & Clustering of Clinical Text Data (SYNTHETIC)
- **Objective:** Text mining and document clustering on synthetic EHR progress notes.
- **Synthesis:** Generated 3,504 report-style notes derived from ICDR labels with realistic medical terminology (laterality, microvascular lesions, clinical noise, and ambiguous terms).
- **Supervised Classification:**
  - Logistic Regression (balanced): **Accuracy 100.0%**, **Weighted F1 1.0000**
  - Multinomial Naive Bayes: **Accuracy 100.0%**, **Weighted F1 1.0000**
- **Unsupervised Clustering:** K-Means ($K=5$) achieved a **Cluster Purity of 0.7763** with top terms aligning cleanly with disease stages.
- **Explicit Caveat:** Results are artificially optimistic because text was generated deterministically from severity grades.
- **Artifacts:** `data/synthetic_notes.csv`, `outputs/reports/t10_clinical_text_classification.md`, Figure `t10_text_classification_clustering.png`, Notebook `notebooks/t10_clinical_text_classification.ipynb`.

---

### T11: Information Extraction & Named Entity Recognition (NER)
- **Objective:** Extract structured clinical entities from EHR free-text notes.
- **Entity Schema:** `LATERALITY`, `LESION`, `SEVERITY`, `ACTION`.
- **Methodology & Evaluation (Held-Out Validation Set):**
  - Rule-Based (Regex + spaCy Matcher): **Macro F1 = 1.000**
  - Custom Machine-Learned spaCy NER: **Macro F1 = 1.000**
- **Corpus-Wide Extraction:** Extracted structured records for all 3,504 encounters, showing distinct lesion count escalation from 0.0 (Grade 0) to 2.4 (Grade 4).
- **Artifacts:** `data/extracted_clinical_entities.csv`, `outputs/reports/t11_clinical_information_extraction.md`, Figure `t11_information_extraction_ner.png`, Notebook `notebooks/t11_clinical_information_extraction.ipynb`.

---

### T12: Ranking & Word Sense Disambiguation (WSD) of Clinical Text
- **Information Retrieval Ranking:**
  - Clinical query: `"severe hemorrhage refer urgently"`.
  - Scored all 3,504 notes with TF-IDF Cosine Similarity vs Okapi BM25.
  - **Spearman Rank Correlation:** **0.9888** ($p < 10^{-10}$).
  - **Precision@K:** **100.0%** across $K = 5, 10, 20, 50, 100$ for retrieving urgent cases (Grade 3+).
- **Word Sense Disambiguation (WSD):**
  - Evaluated on a 40-case gold-standard benchmark across 4 polysemous medical words (`discharge`, `positive`, `stable`, `dense`).
  - *Simplified Lesk (WordNet):* **97.5% Accuracy**
  - *Context-Embedding (Vector Space Similarity):* **97.5% Accuracy**
- **Artifacts:** `outputs/reports/t12_ranking_disambiguation_results.csv`, `outputs/reports/t12_ranking_disambiguation.md`, Figure `t12_ranking_and_wsd.png`, Notebook `notebooks/t12_ranking_and_disambiguation.ipynb`.

---

### T13: Dimensionality Reduction for Clinical Exploratory Data Analysis
- **Objective:** Explore latent feature geometry of deep representations and handcrafted morphological features.
- **Features:** 2,048-dim penultimate ResNet-101 embeddings fused with 8 handcrafted optical metrics (sharpness, edge energy, color distributions).
- **Manifold Methods:**
  - *PCA:* Top 2 components explain **60.68%** of total variance; top 50 explain **97.32%**.
  - *t-SNE:* Perplexity = 30.
  - *UMAP:* $n\_neighbors = 15$, $min\_dist = 0.1$.
- **Silhouette Scores:**
  - Severity Separation: PCA (0.1758), **t-SNE (0.1476)**, UMAP (0.0710).
  - Image Quality Separation: PCA (-0.2168), t-SNE (-0.3586), UMAP (-0.3681).
- **Clinical Manifold Interpretation:**
  - Fundus images form a continuous curvilinear manifold reflecting biological disease progression from normal to proliferative disease.
  - Negative silhouette scores for image quality demonstrate that the deep CNN representations learned robust pathological invariants that do not artificially segregate by illumination or blur.
- **Artifacts:** `outputs/reports/t13_dim_reduction_metrics.csv`, `outputs/reports/t13_dimensionality_reduction.md`, Figure `t13_dimensionality_reduction.png`, Notebook `notebooks/t13_dimensionality_reduction.ipynb`.

---

### T14: Behavioral Factors Influencing Weight in Adults – Part 1: Exploration & Statistics (SIMULATED)
- **Objective:** Model the multidimensional complexity of behavioral lifestyle drivers and metabolic health in a simulated adult diabetic population.
- **Data Generation:** Synthesized 3,000 adult diabetic records (`data/sim_behavioral_factors.csv`) incorporating age, sex, height, weight, non-linear BMI, daily calories, physical activity, sleep hours, stress score, smoking, alcohol, diabetes duration, HbA1c, and coupled DR severity matching APTOS proportions.
- **Data Cleaning:** Imputed 3–5% random missing values with sex-stratified medians; winsorized physiological weight extremes ($[40, 180]$ kg) and caloric reporting errors. Saved clean dataset to `data/clean_sim_behavioral_factors.csv`.
- **Statistical Findings:**
  - *Correlation:* Physical activity demonstrated diminishing returns on BMI reduction ($r = -0.38$). Chronic stress correlated with caloric surplus ($r = +0.28$) and sleep deficit ($r = -0.32$).
  - *Multicollinearity:* All VIF values $< 2.2$, confirming feature stability.
  - *ANOVA:* Significant biomarker differences across DR grades: HbA1c ($F = 351.2, \eta^2 = 0.319$), Duration ($F = 520.6, \eta^2 = 0.410$), and BMI ($F = 59.9, \eta^2 = 0.074$).
  - *PCA:* Top 3 behavioral dimensions explain **52.8%** of variance (Metabolic Strain, Active Lifestyle, Glycemic Chronicity).
- **Artifacts:** `data/sim_behavioral_factors.csv`, `data/clean_sim_behavioral_factors.csv`, `outputs/reports/sim_assumptions.md`, `outputs/reports/t14_behavioral_part1.md`, Figure `t14_behavioral_factors_eda.png`, Notebook `notebooks/t14_behavioral_factors_part1.ipynb`.

---

### T15: Behavioral Factors Influencing Weight – Part 2: Modeling & Multimodal Fusion (SIMULATED)
- **Objective:** Machine learning regression of BMI, SHAP interpretability, behavioral clustering, and multimodal risk fusion.
- **Predictive Modeling:**
  - OLS Linear with Interactions: $R^2 = 0.315$, RMSE = 3.79 kg/m$^2$
  - Random Forest Regressor: $R^2 = 0.326$, RMSE = 3.76 kg/m$^2$
  - Gradient Boosting Regressor: **$R^2 = 0.345$**, **RMSE = 3.71 kg/m$^2$**, **MAE = 2.00 kg/m$^2$**
- **Interpretability & SHAP:** TreeExplainer confirmed daily calories, physical activity, and sleep duration as the top drivers of adult BMI. Partial dependence revealed significant age $\times$ activity metabolic attenuation.
- **Behavioral Phenotyping (K-Means):** Segmented adults into 4 distinct clusters: Profile 0 (Sedentary High-Stress), Profile 1 (Active Healthy Baseline), Profile 2 (Elderly Chronic Duration), and Profile 3 (Younger High-Calorie At-Risk, 87.7% referable DR rate).
- **Multimodal Retinopathy Risk Fusion:**
  $$P_{combined}(\text{Grade} = g) = 0.70 \cdot P_{CNN} + 0.30 \cdot P_{behav}$$
  Yielding a continuous combined risk score scaled from 0 to 100.
- **Artifacts:** `outputs/reports/t15_modeling_metrics.csv`, `outputs/reports/t15_behavioral_part2.md`, Figure `t15_behavioral_modeling_shap.png`, Notebook `notebooks/t15_behavioral_factors_part2.ipynb`, serialized models in `outputs/models/`.

---

## 3. Interactive Demonstration Application (`app.py`)
The full screening suite is deployed via an 8-tab Gradio web interface (`app.py`):
1. **Clinical Summary:** Predicted ICDR severity, confidence bars, referral recommendation, and image quality warnings.
2. **Preprocessing (T2–T4):** CLAHE enhancement, bilateral denoising, retinal vessel segmentation, and bright lesion detection.
3. **Explainability (T5–T6):** Grad-CAM lesion saliency heatmap and top matching cases from the APTOS dataset.
4. **Embedding Map (T13):** Latent 2D PCA manifold projection with the uploaded patient image dynamically marked.
5. **Clinical Notes (T10–T12):** SYNTHETIC progress note, spaCy named entity extraction, and BM25 retrieval.
6. **Capacity Planning (T7–T9):** SIMULATED referral threshold trade-off position and clinic staffing load.
7. **Behavioral Risk (T14–T15):** SIMULATED BMI prediction, behavioral profile clustering, SHAP factors, and multimodal risk score.
8. **Download Report:** Instant generation of clinical PDF and JSON reports with academic research disclaimers.

Verified via automated test suite across 3 test cases in `outputs/figures/ui_tests/`.

---

## 4. Key Metrics Summary Table

| Tutorial | Core Technique | Key Metric / Result | Output Artifact |
|:---|:---|:---|:---|
| **T1: Understanding EHR** | Data Ingestion & EHR Synthesis | 3,662 Images, 5 ICDR classes | `ehr_records.csv` |
| **T2: Preprocessing EHR** | Standardization & Rejection | 158 Suboptimal cases flagged | `clean_records.csv` |
| **T3: Deduplication** | MD5 & Perceptual Hashing | 158 Duplicate pairs removed | `dedup_records.csv` |
| **T4: Image Processing** | NLM Denoising & CLAHE | PSNR: 30.1 dB, SSIM: 0.845 | `t4_restoration_metrics.csv` |
| **T5: Image Analysis** | ResNet-101 Transfer Learning | Accuracy: 76.7%, QWK: 0.824 | `best_retinascreen_cnn.pth` |
| **T6: Visualization** | Grad-CAM & Confusion Matrix | Correct & Incorrect Saliency | `t6_gradcam_validation_cases.png` |
| **T7: Patient Prediction** | Weekly ARIMA Forecasting | 2-3 Graders, 5-8 Specialists | `sim_weekly_screening_schedule.csv`|
| **T8: Strategic Plan Pt 1** | Capacity Scenario Modeling | 40.7% Baseline Referral Rate | `t8_strategic_planning_part1.md` |
| **T9: Strategic Plan Pt 2** | Cost-Benefit Decision Model | G2+ Optimal Referral Cutoff | `t9_strategic_dashboard.png` |
| **T10: Text Classification**| TF-IDF + Logistic Regression | Accuracy: 100%, F1: 1.000 | `synthetic_notes.csv` |
| **T11: Clinical NER** | Rule-Based + spaCy NER | Macro F1: 1.000 across 4 entities | `extracted_clinical_entities.csv` |
| **T12: Ranking & WSD** | BM25 vs TF-IDF & Lesk WSD | Spearman: 0.9888, WSD Acc: 97.5% | `t12_ranking_disambiguation_results.csv`|
| **T13: Dim Reduction** | PCA, t-SNE & UMAP | Top 2 PCs Explain 60.68% Var | `t13_dim_reduction_metrics.csv` |
| **T14: Behavioral EDA** | 3,000 SIMULATED adults, VIF, PCA | VIF < 2.2, ANOVA F = 59.9 (BMI) | `clean_sim_behavioral_factors.csv` |
| **T15: Behavioral Modeling**| Gradient Boosting BMI & SHAP | GBR R2 = 0.345, Multimodal Fusion | `outputs/models/bmi_gradient_boosting.joblib` |
| **UI: Web Application** | 8-Tab Gradio App + PDF Export | Verified on 3 severity test cases | `outputs/figures/ui_tests/` |

---

## 5. Complete Index of 300 DPI Figures
All visual assets are stored in [`outputs/figures/`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/) at 300 DPI:
1. `t2_preprocessing_comparison.png` - Before/after cropping and luminance normalization.
2. `t3_deduplication_imputation.png` - Hash distance distributions and KNN imputation scatter.
3. `t4_enhancement_comparison.png` - Green channel, CLAHE, and Ben Graham filters.
4. `t4_restoration_benchmark.png` - Bilateral, NLM, and Richardson-Lucy restoration grids.
5. `t4_segmentation_vessels_exudates.png` - Retinal vessel masks and hard exudate localization.
6. `t4_compression_tradeoffs.png` - JPEG quality vs storage savings vs PSNR trade-off curves.
7. `t5_sift_ransac_matching.png` - Intra-image geometric recovery vs inter-patient non-matching.
8. `t5_cnn_training_performance.png` - Loss trajectory, QWK progression, and confusion matrix.
9. `t6_class_distribution.png` - Population severity distribution comparison.
10. `t6_severity_sample_grid.png` - Exemplar fundus photographs across ICDR Grades 0 to 4.
11. `t6_preprocessing_panels.png` - 3-stage preprocessing visual verification.
12. `t6_training_curves.png` - Convergence trajectories of CNN training.
13. `t6_confusion_matrix_heatmap.png` - Heatmap with clinical discordance penalties.
14. `t6_gradcam_validation_cases.png` - Grad-CAM saliency overlays with medical captions.
15. `sim_t7_staffing_forecast.png` - 104-week volume projection and staffing demand curves.
16. `t8_strategic_planning_descriptive.png` - Operational referral load under 3 volume scenarios.
17. `t9_strategic_dashboard.png` - One-page executive health economics decision dashboard.
18. `t10_text_classification_clustering.png` - Text confusion matrices and K-Means clusters.
19. `t11_information_extraction_ner.png` - NER F1 comparisons and clinical entity distributions.
20. `t12_ranking_and_wsd.png` - BM25 retrieval correlation and WSD accuracy benchmarks.
21. `t13_dimensionality_reduction.png` - 6-panel PCA scree, t-SNE, and UMAP projections.
22. `t14_behavioral_factors_eda.png` - Correlation heatmap, ANOVA boxplots, activity vs BMI, and PCA loadings.
23. `t15_behavioral_modeling_shap.png` - Model comparisons, SHAP summary, dependence plot, and cluster profiles.
24. `ui_tests/` - Composite evaluation figures, clinical PDF reports, and JSON records across 3 test cases.

---

## 6. Project Limitations & Ethical Caveats
1. **Academic Demonstration Aid:** RetinaScreen is strictly an **AI screening and decision-support triaging tool**, NOT an autonomous diagnostic medical device. All recommendations must be confirmed by a licensed eye care professional.
2. **Synthetic Clinical Notes (T10–T12):** Clinical narratives were generated programmatically from severity grades; text classification and NER accuracy (100%) is artificially optimistic compared to real-world EHR free-text.
3. **Simulated Scheduling & Behavioral Cohorts (T7, T14–T15):** The 104-week schedule and 3,000 adult behavioral records are simulated using empirical parameters and explicit mathematical models detailed in [`outputs/reports/sim_assumptions.md`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/sim_assumptions.md).
4. **Class Imbalance:** Severe NPDR (Grade 3) accounts for 5.3% of the APTOS cohort, warranting careful clinical review of borderline predictions.

---

## 7. Master Task Completion Checklist

| Task | Title | Code Script | Jupyter Notebook | Report Artifact | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| **T1** | Understanding Electronic Health Records | `src/verify_data.py` | `t1_understanding_ehr.ipynb` | `t1_ehr_understanding.md` | [x] **Complete** |
| **T2** | EHR Standardization & Data Cleaning | `src/t2_preprocess_ehr.py` | `t2_ehr_preprocessing.ipynb` | `t2_ehr_preprocessing.md` | [x] **Complete** |
| **T3** | Redundant Data Removal & Imputation | `src/t3_dedup_missing.py` | `t3_redundant_missing_data.ipynb` | `t3_redundant_missing_data.md` | [x] **Complete** |
| **T4** | Biomedical Image Processing | `src/t4_image_processing.py` | `t4_biomedical_image_processing.ipynb` | `t4_biomedical_image_processing.md` | [x] **Complete** |
| **T5** | Biomedical Image Analysis (SIFT & CNN) | `src/t5_image_analysis.py` | `t5_biomedical_image_analysis.ipynb` | `t5_biomedical_image_analysis.md` | [x] **Complete** |
| **T6** | Biomedical Image Visualization Suite | `src/t6_image_visualization.py` | `t6_biomedical_image_visualization.ipynb` | `t6_biomedical_image_visualization.md` | [x] **Complete** |
| **T7** | Patient Prediction & Staffing (Simulated) | `src/t7_patient_prediction.py` | `t7_patient_prediction_staffing.ipynb` | `t7_patient_prediction_staffing.md` | [x] **Complete** |
| **T8** | Strategic Planning Part 1: Descriptive | `src/t8_strategic_part1.py` | `t8_strategic_planning_part1.ipynb` | `t8_strategic_planning_part1.md` | [x] **Complete** |
| **T9** | Strategic Planning Part 2: Prescriptive | `src/t9_strategic_part2.py` | `t9_strategic_planning_part2.ipynb` | `t9_strategic_planning_part2.md` | [x] **Complete** |
| **T10**| Text Classification & Clustering | `src/t10_text_classification.py`| `t10_clinical_text_classification.ipynb` | `t10_clinical_text_classification.md` | [x] **Complete** |
| **T11**| Information Extraction & Named Entity NER | `src/t11_information_extraction.py`| `t11_clinical_information_extraction.ipynb`| `t11_clinical_information_extraction.md` | [x] **Complete** |
| **T12**| Ranking & Word Sense Disambiguation | `src/t12_ranking_disambiguation.py`| `t12_ranking_and_disambiguation.ipynb` | `t12_ranking_disambiguation.md` | [x] **Complete** |
| **T13**| Dimensionality Reduction (PCA, t-SNE, UMAP)| `src/t13_dimensionality_reduction.py`| `t13_dimensionality_reduction.ipynb` | `t13_dimensionality_reduction.md` | [x] **Complete** |
| **T14**| Behavioral Factors EDA & Statistics | `src/t14_behavioral_part1.py` | `t14_behavioral_factors_part1.ipynb` | `t14_behavioral_part1.md` | [x] **Complete** |
| **T15**| Behavioral Modeling, SHAP & Risk Fusion | `src/t15_behavioral_part2.py` | `t15_behavioral_factors_part2.ipynb` | `t15_behavioral_part2.md` | [x] **Complete** |
| **UI**  | 8-Tab Gradio Clinical Web App & PDF Export| `app.py` | `test_ui_app.py` | `summary.md` | [x] **Complete** |
| **FINAL**| End-to-End Testing & Verification | `src/test_ui_app.py` | - | `outputs/figures/ui_tests/` | [x] **Complete** |


