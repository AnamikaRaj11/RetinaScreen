# T5 Report: Biomedical Image Analysis - SIFT, RANSAC & CNN Transfer Learning
**Dataset:** Kaggle APTOS 2019 Blindness Detection  
**Generated Files:**
- Saved Model Checkpoint: `outputs/models/best_retinascreen_cnn.pth`
- Per-Class Metrics: `outputs/reports/t5_classification_metrics.csv`
- Figures (300 DPI):
  - `outputs/figures/t5_sift_ransac_matching.png`
  - `outputs/figures/t5_cnn_training_performance.png`

---

## 1. Executive Summary
Task 5 demonstrates two core biomedical image analysis paradigms: classical feature-based geometric registration using SIFT and RANSAC, and modern deep convolutional transfer learning (ResNet-101) for 5-class diabetic retinopathy severity grading. The pipeline addresses severe real-world clinical class imbalance using class-weighted loss and reports standardized medical benchmark metrics (Accuracy, Quadratic Weighted Kappa, Per-Class Precision/Recall).

---

## 2. Geometric Keypoint Analysis: SIFT & RANSAC

### Registration Findings
- **Same Eye (Synthetically Transformed):**
  - Lowe's Ratio Matches: **65**
  - RANSAC Inlier Matches: **54** (83.1% inlier ratio)
  - Geometric homography accurately solves for camera rotation (15°) and translation, validating longitudinal retinal alignment.
- **Inter-Patient Retinal Comparison:**
  - Lowe's Candidate Matches: **5**
  - RANSAC Inlier Matches: **0**
  - Structural vascular bifurcation patterns are unique to individual human retinas, resulting in low geometric consensus across different patients.

---

## 3. Deep Learning Severity Grading: Transfer Learning

### Model Architecture & Configuration
- **Backbone:** ResNet-101 (2048-dim feature representation pre-trained on ImageNet/COCO).
- **Optimization Strategy:** Lower convolutional blocks frozen; Layer 4 and Classification Head fine-tuned.
- **Input Resolution:** 224x224 normalized Ben Graham images.
- **Imbalance Mitigation:** Inverse-frequency class-weighted Cross-Entropy loss.
- **Validation Protocol:** Stratified 80/20 train/validation split (Seed 42).

### Clinical Grading Performance
- **Validation Accuracy:** **59.17%**
- **Quadratic Weighted Kappa (QWK):** **0.8049** (reflects clinical penalty for distant misclassifications)

| ICDR Severity Grade | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Grade 0** (No DR) | 0.984 | 0.984 | 0.984 | 61 |
| **Grade 1** (Mild) | 0.455 | 0.455 | 0.455 | 11 |
| **Grade 2** (Moderate) | 0.000 | 0.000 | 0.000 | 32 |
| **Grade 3** (Severe) | 0.136 | 1.000 | 0.240 | 6 |
| **Grade 4** (Proliferative DR) | 0.000 | 0.000 | 0.000 | 10 |

---

## 4. Visual Validation (300 DPI)
All high-resolution validation figures are saved in [`outputs/figures/`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/).
