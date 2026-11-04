# T2 Report: Preprocessing of EHR - Standardization, Data Cleaning & Image Normalization
**Dataset:** Kaggle APTOS 2019 Blindness Detection  
**Generated Files:**
- Cleaned Metadata: `data/clean_records.csv` (3,662 records)
- Normalized Images (512x512): `data/processed/train_512/` (3,662 files)
- Normalized Images (224x224): `data/processed/train_224/` (3,662 files for CPU training)
- Visualization Figure: `outputs/figures/t2_preprocessing_comparison.png` (300 DPI)

---

## 1. Executive Summary
Task 2 standardizes clinical EHR schema types, validates diagnostic integrity, detects image quality anomalies, removes non-informative circular camera borders, and applies standard retinal color normalization (Ben Graham technique). The resulting dataset eliminates illumination and hardware bias while preparing standardized 512x512 and 224x224 spatial formats for downstream analytics and model training.

---

## 2. Standardization & Diagnostic Validation
- **Schema Validation:**
  - `record_id`: Verified alphanumeric string identifier.
  - `diagnosis`: Enforced standard integer `[0, 4]` matching the International Clinical Diabetic Retinopathy (ICDR) scale.
  - **Result:** 100% of the 3,662 records passed diagnostic range validation. Zero missing or out-of-range diagnosis codes.

---

## 3. Image Quality Screening & Anomaly Flagging

### Clinical Rationale & Thresholds
Retinal photography is susceptible to patient-induced artifacts (eyelash shadows, blinks, cataracts) and operator error (flash misalignment). We established automated screening filters:

| Quality Flag | Operational Threshold | Clinical Significance | Flagged Count | Percentage |
| :--- | :--- | :--- | :---: | :---: |
| **Corrupt** | Unreadable image / header fault | File I/O breakdown | 0 | 0.00% |
| **Very Dark** | `mean_brightness < 25.0` | Underexposed / pupillary constriction | 49 | 1.34% |
| **Overexposed** | `mean_brightness > 115.0` | Flash glare / retinal washout | 8 | 0.22% |
| **Blurry** | `blur_score < 7.0` (Laplacian Var) | Motion blur / severe cataract opacity | 532 | 14.53% |
| **Quality Pass**| Passes all checks | Certified diagnostic-grade visual clarity | 3,074 | 83.94% |

> [!NOTE]
> Flagged images are **not deleted** from the dataset to preserve clinical reality and test model robustness under noisy real-world clinical conditions. Quality flags are recorded in `data/clean_records.csv` to enable stratified quality analyses in T8-T9.

---

## 4. Biomedical Image Preprocessing Pipeline

### 1. Black Border Cropping
Fundus cameras produce circular apertures surrounded by non-informative black margins. We locate the active ocular mask ($I_{gray} > 10$), detect its bounding rectangle $[y_{min}:y_{max}, x_{min}:x_{max}]$, and crop away unneeded margins.

### 2. Ben Graham Color Normalization
Multi-site clinical screening encounters high color temperature and illumination variance due to different camera sensors. The Ben Graham algorithm standardizes color by:
$$I_{norm} = 4 \cdot I_{orig} - 4 \cdot \text{GaussianBlur}(I_{orig}, \sigma) + 128$$
where $\sigma = \text{target\_size} / 30$. This acts as a high-pass frequency filter, removing uneven background lighting while emphasizing microvascular lesions (microaneurysms, hemorrhages, exudates).

### 3. Aperture Masking & Dual-Scale Export
- A smooth circular mask ($r = 0.485 \times \text{size}$) is applied to maintain clinical fundus anatomy.
- Dual exports:
  - `train_512/`: 512x512 resolution for fine lesion segmentation and feature extraction (T4, T5, T6).
  - `train_224/`: 224x224 resolution for high-throughput CPU transfer learning and embedding extraction (T5, T13).
- Feature processing took 108.2 seconds across all 3,662 images.

---

## 5. Visual Comparison (300 DPI)
The multi-panel visual validation figure is saved at:
[`outputs/figures/t2_preprocessing_comparison.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t2_preprocessing_comparison.png)

It confirms that across all 5 severity levels:
1. Uneven illumination and sensor color biases are successfully neutralized.
2. Blood vessel architecture and pathology boundaries are noticeably enhanced.
3. Aspect ratios and camera border discrepancies are standardized to uniform geometry.
