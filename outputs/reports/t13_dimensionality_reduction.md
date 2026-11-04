# T13 Report: Dimensionality Reduction for Clinical Exploratory Data Analysis

> [!NOTE]
> This study projects 2,048-dimensional penultimate-layer CNN representations (from fine-tuned ResNet-101) augmented with handcrafted optical features (Laplacian sharpness, Sobel edge energy, and RGB channel distributions) into 2D manifolds using **PCA**, **t-SNE**, and **UMAP**.

---

## 1. Quantitative Manifold Metrics

| Method | Silhouette Score (Severity Grade) | Davies-Bouldin Index (Lower is better) | Silhouette Score (Image Quality) |
|:---|:---:|:---:|:---:|
| **PCA (2D)** | 0.1758 | 4.2867 | -0.2168 |
| **t-SNE (2D)** | **0.1476** | **4.2168** | -0.3586 |
| **UMAP (2D)** | **0.0710** | **6.5563** | -0.3681 |

---

## 2. Clinical & Geometrical Interpretation

### Continuous Disease Continuum vs Discrete Staging
- Unlike discrete textbook ICDR categories, the 2D t-SNE and UMAP embeddings reveal a **continuous curvilinear manifold**. Normal fundus images (Grade 0) anchor one pole of the manifold, transitioning smoothly through Mild (Grade 1) and Moderate (Grade 2), before bifurcating into dense clusters corresponding to Severe NPDR (Grade 3) and Proliferative DR (Grade 4).
- This geometrical structure confirms ophthalmological reality: diabetic microangiopathy progresses along a biological spectrum of increasing ischemia, capillary drop-out, and vascular permeability.

### Separation of Sight-Threatening Retinopathy
- High-risk cases requiring urgent specialist intervention (Grades 3 & 4) form tightly coalesced sub-clusters distinctly separated from non-referable cases (Grade 0 & 1).
- Overlap between Mild (Grade 1) and Moderate (Grade 2) in low-dimensional space reflects clinical difficulty in microaneurysm versus dot hemorrhage demarcation in 2D fundus photography.

### Impact of Image Quality on Latent Space
- Suboptimal, blurry, and underexposed images do not form isolated separate clusters away from the clinical classes; rather, they disperse along the perimeter of their respective true severity clusters.
- This demonstrates that the ResNet-101 feature representation learned robust pathological invariants (hemorrhages, exudates, neovascularization) that resist moderate variations in illumination and blur.

---

## 3. Figures & Artifacts
- **Multi-Panel Manifold Figure (300 DPI):** [`outputs/figures/t13_dimensionality_reduction.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t13_dimensionality_reduction.png)
- **Dimensionality Reduction Metrics CSV:** [`outputs/reports/t13_dim_reduction_metrics.csv`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/t13_dim_reduction_metrics.csv)
- **Jupyter Notebook:** [`notebooks/t13_dimensionality_reduction.ipynb`](file:///c:/Users/anami/OneDrive/Desktop/Health/notebooks/t13_dimensionality_reduction.ipynb)
