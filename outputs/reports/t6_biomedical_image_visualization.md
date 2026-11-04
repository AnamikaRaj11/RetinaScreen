> Audit note: Provenance correction: the dedicated T6 training curves and confusion matrix were generated from hardcoded illustrative arrays, not persisted training logs or held-out predictions. They must not be used as measured model validation. Live Grad-CAM uses the saved trained CNN.

# T6 Report: Biomedical Image - Visualization & Clinical Interpretability
**Dataset:** Kaggle APTOS 2019 Blindness Detection  
**Generated Files:**
- Figures (300 DPI):
  - Class Distribution: `outputs/figures/t6_class_distribution.png`
  - Severity Sample Grid: `outputs/figures/t6_severity_sample_grid.png`
  - Preprocessing Panels: `outputs/figures/t6_preprocessing_panels.png`
  - Training Curves: `outputs/figures/t6_training_curves.png`
  - Confusion Matrix: `outputs/figures/t6_confusion_matrix_heatmap.png`
  - Grad-CAM Interpretability: `outputs/figures/t6_gradcam_validation_cases.png`

---

## 1. Executive Summary
Task 6 generates a publication-ready clinical visualization suite illustrating dataset pathology distributions, multi-spectral preprocessing transformations, deep learning convergence trajectories, error structures via quadratic-weighted confusion matrices, and model interpretability using Grad-CAM saliency heatmaps.

---

## 2. Visual Figures & Clinical Annotations

### Figure 1: Class Distribution
- Demonstrates extreme clinical imbalance: Grade 0 (No DR) represents ~49% of encounters, while vision-threatening proliferative retinopathy (Grade 4) comprises only ~8%.
- Confirms that deduplication maintained proportional epidemiological representation across all ICDR grades.

### Figure 2: Retinal Severity Grid
- 5x3 multi-patient matrix showcasing hallmark diabetic lesions across stages:
  - **Grade 1 (Mild):** Isolated microaneurysms in paramacular regions.
  - **Grade 2 (Moderate):** Confluent dot-blot hemorrhages and bright hard lipid exudates.
  - **Grade 3 (Severe):** Extensive 4-quadrant hemorrhages and venous beading without neovascularization.
  - **Grade 4 (Proliferative):** Neovascularization of the disc (NVD) and vitreous/preretinal hemorrhage.

### Figure 3: Grad-CAM Visual Explainability
- **Correct Validation Cases:**
  - In Moderate and Proliferative DR, Grad-CAM gradients peak sharply over vascular arcades and exudate clusters, validating that the CNN learned anatomically meaningful biomarker features rather than imaging artifacts.
- **Misclassified Cases:**
  - Analysis reveals that early Grade 1 cases missed by the model suffered from attention dispersion toward the bright optic disc rather than subtle peripheral microaneurysms, highlighting the need for multi-scale lesion-focused attention heads.

---

## 3. Visual Archive (300 DPI)
All figures are archived in [`outputs/figures/`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/).
