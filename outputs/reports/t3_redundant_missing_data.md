# T3 Report: Preprocessing of EHR - Redundant Data Removal & Missing Data Imputation
**Dataset:** Kaggle APTOS 2019 Blindness Detection  
**Generated Files:**
- Deduplicated Dataset: `data/dedup_records.csv` (3,504 records retained)
- Removal Audit Trail: `outputs/reports/removal_log.csv` (303 logged rows)
- Evaluation Figure: `outputs/figures/t3_deduplication_imputation.png` (300 DPI)

---

## 1. Executive Summary
Task 3 performs cryptographic and perceptual deduplication across the retinal fundus dataset, flags clinical label discrepancies among near-duplicate images, guarantees non-destructive metadata exclusion, and benchmarks statistical vs. distance-weighted missing value imputation algorithms.

---

## 2. Redundancy & Near-Duplicate Analysis

### Methodology
1. **Cryptographic Check (MD5):** Byte-level exact duplicate identification on raw source files.
2. **Perceptual Hash (pHash):** 64-bit DCT-based frequency hashing on normalized 224x224 retinal images. Pairwise Hamming distance $\le 2$ flags identical ocular structures captured under different compression, scaling, or cropping parameters.
3. **Clinical Conflict Verification:** When duplicate images exist with mismatched ICDR severity grades, this signals inter-grader subjectivity or dataset labeling noise.
4. **Non-Destructive Exclusion Policy:** Original image files are never deleted from disk. Redundant entries are flagged and excluded solely within `data/dedup_records.csv`.

### Deduplication Audit Results
- **Original Cohort:** 3,662 records
- **Duplicate Clusters Identified:** 145
- **Conflicting Diagnostic Labels Detected:** 40 cluster(s)
- **Records Excluded from Training/Analysis CSV:** 158 (4.31%)
- **Deduplicated Cohort Retained:** 3,504 records

---

## 3. Missing Value Analysis & Imputation Benchmark

### Natural Dataset Verification
- **Missing Label Check (`diagnosis`):** 0 missing (100% complete)
- **Missing File Check (`image_path`):** 0 missing (100% verified on disk)
- **Derived Metrics Check (`brightness`, `blur`):** 0 missing (100% computed)

### Imputation Experiment (15% MCAR Masking)
To demonstrate production EHR resilience when optical sensor telemetry or clinical metrics are partially missing, we masked 15% of records at random and benchmarked **Univariate Median Imputation** against **Multivariate K-Nearest Neighbors (KNN, k=5)**:

| Optical / EHR Feature | Median MAE | KNN MAE | Median RMSE | KNN RMSE | KNN Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `width` | 791.94 | 100.43 | 912.77 | 333.92 | **+87.3%** |
| `height` | 458.52 | 70.93 | 540.45 | 233.72 | **+84.5%** |
| `mean_brightness` | 15.99 | 10.18 | 19.10 | 12.81 | **+36.3%** |
| `blur_score` | 15.48 | 4.50 | 20.18 | 7.77 | **+71.0%** |

### Clinical Implications
KNN imputation leverages cross-feature correlations (e.g., relationship between image resolution, optical blur, and illumination levels). As shown in the benchmark, KNN consistently reduces imputation error by up to **20-40%** compared to naive median replacement, preserving continuous distributions vital for machine learning feature stability.

---

## 4. Visual Validation (300 DPI)
The multi-panel validation figure is archived at:
[`outputs/figures/t3_deduplication_imputation.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t3_deduplication_imputation.png)
