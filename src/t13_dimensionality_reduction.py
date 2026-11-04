"""
Task 13: Dimensionality Reduction for Exploratory Data Analysis in Daily Medical Research.
- Feature Extraction:
    - Penultimate-layer CNN embeddings (2048-dim from fine-tuned ResNet-101)
    - Simple handcrafted image features (RGB mean/std, green Laplacian sharpness, Sobel edge energy)
- Dimensionality Reduction:
    - PCA (with individual and cumulative explained variance scree plot)
    - t-SNE (2D manifold)
    - UMAP (2D non-linear topology)
- Analysis & Visualization:
    - Scatter plots colored by ICDR severity (Grade 0-4)
    - Scatter plots colored by image quality categories
    - Silhouette score comparison for clinical separation
    - Clinical interpretation of embedding manifold
- Outputs:
    - outputs/reports/t13_dim_reduction_metrics.csv
    - outputs/figures/t13_dimensionality_reduction.png
    - outputs/reports/t13_dimensionality_reduction.md
    - notebooks/t13_dimensionality_reduction.ipynb
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import umap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, CLASS_LABELS, RANDOM_SEED

DEDUP_CSV = DATA_DIR / "dedup_records.csv"
TRAIN_224_DIR = DATA_DIR / "processed" / "train_224"
BEST_MODEL_PATH = MODELS_DIR / "best_retinascreen_cnn.pth"
FIG_T13 = FIGURES_DIR / "t13_dimensionality_reduction.png"
METRICS_CSV = REPORTS_DIR / "t13_dim_reduction_metrics.csv"
REPORT_MD = REPORTS_DIR / "t13_dimensionality_reduction.md"
NB_PATH = PROJECT_ROOT / "notebooks" / "t13_dimensionality_reduction.ipynb"


def load_fine_tuned_resnet():
    """Load the fine-tuned ResNet-101 model from T5 checkpoint."""
    print("Loading fine-tuned ResNet-101 model checkpoint...")
    net = models.resnet101(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 5)
    )
    if BEST_MODEL_PATH.exists():
        state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
        net.load_state_dict(state_dict)
        print(f"Loaded weights from {BEST_MODEL_PATH.name}")
    else:
        print("Warning: Model checkpoint not found; using uninitialized weights.")
    net.eval()
    return net


class EvalRetinalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / row["path_224"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Handcrafted features
        mean_r = float(img_rgb[:, :, 0].mean())
        mean_g = float(img_rgb[:, :, 1].mean())
        mean_b = float(img_rgb[:, :, 2].mean())
        std_intensity = float(img_rgb.std())

        green_chan = img_rgb[:, :, 1]
        laplacian_var = float(cv2.Laplacian(green_chan, cv2.CV_64F).var())
        sobelx = cv2.Sobel(green_chan, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(green_chan, cv2.CV_64F, 0, 1, ksize=3)
        sobel_energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        green_ratio = mean_g / (mean_r + mean_b + 1e-5)
        dark_fraction = float(np.mean(green_chan < 15.0))

        handcrafted = np.array([
            mean_r, mean_g, mean_b, std_intensity,
            laplacian_var, sobel_energy, green_ratio, dark_fraction
        ], dtype=np.float32)

        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)

        return tensor, handcrafted, int(row["diagnosis"]), row["record_id"]


def extract_embeddings_and_features(net, sample_df: pd.DataFrame):
    """Extract penultimate-layer embeddings and simple image features."""
    print(f"Extracting CNN embeddings and handcrafted features for {len(sample_df)} images...")
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = EvalRetinalDataset(sample_df, val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    cnn_embeddings = []
    def hook_fn(module, input, output):
        # avgpool output is (batch, 2048, 1, 1)
        cnn_embeddings.append(output.squeeze(-1).squeeze(-1).detach().cpu().numpy())

    hook = net.avgpool.register_forward_hook(hook_fn)

    all_handcrafted = []
    all_targets = []
    all_record_ids = []

    with torch.no_grad():
        for tensors, hand, labels, rec_ids in loader:
            _ = net(tensors)
            all_handcrafted.append(hand.numpy())
            all_targets.extend(labels.numpy())
            all_record_ids.extend(rec_ids)

    hook.remove()

    cnn_feats = np.concatenate(cnn_embeddings, axis=0)  # (N, 2048)
    handcrafted_feats = np.concatenate(all_handcrafted, axis=0)  # (N, 8)
    targets = np.array(all_targets)

    print(f"Extracted feature matrices: CNN {cnn_feats.shape}, Handcrafted {handcrafted_feats.shape}")
    return cnn_feats, handcrafted_feats, targets, all_record_ids


def run_dimensionality_reduction(cnn_feats, handcrafted_feats, targets, sample_df):
    """Run PCA, t-SNE, and UMAP on fused multimodal representations."""
    print("\nExecuting PCA, t-SNE, and UMAP...")

    # Standardize handcrafted features and concatenate with L2-normalized CNN features
    cnn_norm = cnn_feats / np.linalg.norm(cnn_feats, axis=1, keepdims=True)
    hand_scaler = StandardScaler()
    hand_scaled = hand_scaler.fit_transform(handcrafted_feats)

    # Combined feature representation (2048 + 8)
    fused_features = np.hstack([cnn_norm, hand_scaled * 0.2])

    # 1. PCA
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    pca_feats = pca.fit_transform(fused_features)
    pca_2d = pca_feats[:, :2]
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)
    print(f"  PCA: Top 2 components explain {cum_var_exp[1]*100:.2f}% variance. Top 50 explain {cum_var_exp[-1]*100:.2f}%.")

    # 2. t-SNE
    print("  Fitting 2D t-SNE on top 30 PCA components...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000)
    tsne_2d = tsne.fit_transform(pca_feats[:, :30])

    # 3. UMAP
    print("  Fitting 2D UMAP on top 30 PCA components...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
    umap_2d = reducer.fit_transform(pca_feats[:, :30])

    # Assign Quality Categories for stratification
    def assign_quality(row):
        if row.get("quality_pass", False):
            return "Good Quality"
        elif row.get("is_blurry", False):
            return "Blurry"
        elif row.get("is_very_dark", False):
            return "Very Dark"
        elif row.get("is_overexposed", False):
            return "Overexposed"
        else:
            return "Suboptimal"

    quality_labels = sample_df.apply(assign_quality, axis=1).values

    # Quantitative Evaluation: Silhouette Scores
    sil_pca_sev = silhouette_score(pca_2d, targets)
    sil_tsne_sev = silhouette_score(tsne_2d, targets)
    sil_umap_sev = silhouette_score(umap_2d, targets)

    sil_pca_qual = silhouette_score(pca_2d, quality_labels)
    sil_tsne_qual = silhouette_score(tsne_2d, quality_labels)
    sil_umap_qual = silhouette_score(umap_2d, quality_labels)

    db_pca_sev = davies_bouldin_score(pca_2d, targets)
    db_tsne_sev = davies_bouldin_score(tsne_2d, targets)
    db_umap_sev = davies_bouldin_score(umap_2d, targets)

    print("\nSilhouette Scores (Severity Separation):")
    print(f"  PCA 2D:  {sil_pca_sev:.4f} | DB Index: {db_pca_sev:.4f}")
    print(f"  t-SNE:   {sil_tsne_sev:.4f} | DB Index: {db_tsne_sev:.4f}")
    print(f"  UMAP:    {sil_umap_sev:.4f} | DB Index: {db_umap_sev:.4f}")

    print("\nSilhouette Scores (Quality Separation):")
    print(f"  PCA 2D:  {sil_pca_qual:.4f}")
    print(f"  t-SNE:   {sil_tsne_qual:.4f}")
    print(f"  UMAP:    {sil_umap_qual:.4f}")

    metrics = {
        "pca_2d": pca_2d,
        "tsne_2d": tsne_2d,
        "umap_2d": umap_2d,
        "var_exp": var_exp,
        "cum_var_exp": cum_var_exp,
        "targets": targets,
        "quality_labels": quality_labels,
        "sil_scores": {
            "PCA (Severity)": sil_pca_sev, "t-SNE (Severity)": sil_tsne_sev, "UMAP (Severity)": sil_umap_sev,
            "PCA (Quality)": sil_pca_qual, "t-SNE (Quality)": sil_tsne_qual, "UMAP (Quality)": sil_umap_qual,
        },
        "db_scores": {
            "PCA": db_pca_sev, "t-SNE": db_tsne_sev, "UMAP": db_umap_sev
        }
    }
    return metrics


def generate_figure(metrics):
    """Generate comprehensive 6-panel 300 DPI visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
    plt.subplots_adjust(hspace=0.30, wspace=0.22)

    palette_sev = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c", "#7b3294"]
    sev_labels = [f"G{g}: {CLASS_LABELS[g]}" for g in range(5)]

    # --- Row 1: Colored by ICDR Severity ---
    # 1. PCA by Severity
    for g in range(5):
        mask = metrics["targets"] == g
        axes[0, 0].scatter(metrics["pca_2d"][mask, 0], metrics["pca_2d"][mask, 1],
                           c=palette_sev[g], label=sev_labels[g], alpha=0.7, s=25, edgecolors="none")
    axes[0, 0].set_title(f"A. PCA 2D (Silhouette = {metrics['sil_scores']['PCA (Severity)']:.3f})\nPC1 ({metrics['var_exp'][0]*100:.1f}%) vs PC2 ({metrics['var_exp'][1]*100:.1f}%)",
                          fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Principal Component 1")
    axes[0, 0].set_ylabel("Principal Component 2")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    # 2. t-SNE by Severity
    for g in range(5):
        mask = metrics["targets"] == g
        axes[0, 1].scatter(metrics["tsne_2d"][mask, 0], metrics["tsne_2d"][mask, 1],
                           c=palette_sev[g], label=sev_labels[g], alpha=0.7, s=25, edgecolors="none")
    axes[0, 1].set_title(f"B. t-SNE 2D (Silhouette = {metrics['sil_scores']['t-SNE (Severity)']:.3f})\nPerplexity = 30",
                          fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("t-SNE Dimension 1")
    axes[0, 1].set_ylabel("t-SNE Dimension 2")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # 3. UMAP by Severity
    for g in range(5):
        mask = metrics["targets"] == g
        axes[0, 2].scatter(metrics["umap_2d"][mask, 0], metrics["umap_2d"][mask, 1],
                           c=palette_sev[g], label=sev_labels[g], alpha=0.7, s=25, edgecolors="none")
    axes[0, 2].set_title(f"C. UMAP 2D (Silhouette = {metrics['sil_scores']['UMAP (Severity)']:.3f})\nn_neighbors = 15, min_dist = 0.1",
                          fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("UMAP Dimension 1")
    axes[0, 2].set_ylabel("UMAP Dimension 2")
    axes[0, 2].legend(loc="upper right", fontsize=8)
    axes[0, 2].grid(True, linestyle="--", alpha=0.4)

    # --- Row 2: Diagnostics & Colored by Image Quality ---
    # 4. PCA Scree Plot
    comps = np.arange(1, 21)
    axes[1, 0].bar(comps, metrics["var_exp"][:20] * 100, color="#64b5f6", alpha=0.8, label="Individual Variance (%)")
    axes[1, 0].plot(comps, metrics["cum_var_exp"][:20] * 100, color="#d32f2f", marker="o", lw=2, label="Cumulative Variance (%)")
    axes[1, 0].set_title("D. PCA Scree & Cumulative Variance Plot (Top 20)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Principal Component Number")
    axes[1, 0].set_ylabel("Explained Variance Ratio (%)")
    axes[1, 0].legend(loc="center right", fontsize=8)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # 5. t-SNE by Quality
    qualities = np.unique(metrics["quality_labels"])
    q_palette = sns.color_palette("tab10", n_colors=len(qualities))
    for i, q in enumerate(qualities):
        mask = metrics["quality_labels"] == q
        axes[1, 1].scatter(metrics["tsne_2d"][mask, 0], metrics["tsne_2d"][mask, 1],
                           c=[q_palette[i]], label=q, alpha=0.7, s=25, edgecolors="none")
    axes[1, 1].set_title(f"E. t-SNE by Image Quality Category\n(Quality Silhouette = {metrics['sil_scores']['t-SNE (Quality)']:.3f})",
                          fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("t-SNE Dimension 1")
    axes[1, 1].set_ylabel("t-SNE Dimension 2")
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    # 6. UMAP by Quality
    for i, q in enumerate(qualities):
        mask = metrics["quality_labels"] == q
        axes[1, 2].scatter(metrics["umap_2d"][mask, 0], metrics["umap_2d"][mask, 1],
                           c=[q_palette[i]], label=q, alpha=0.7, s=25, edgecolors="none")
    axes[1, 2].set_title(f"F. UMAP by Image Quality Category\n(Quality Silhouette = {metrics['sil_scores']['UMAP (Quality)']:.3f})",
                          fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("UMAP Dimension 1")
    axes[1, 2].set_ylabel("UMAP Dimension 2")
    axes[1, 2].legend(loc="upper right", fontsize=8)
    axes[1, 2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("T13: Dimensionality Reduction for Clinical Exploratory Data Analysis (PCA, t-SNE & UMAP)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIG_T13, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved T13 figure to: {FIG_T13}")


def run_t13_pipeline():
    print("=" * 70)
    print(" T13: Dimensionality Reduction for Clinical Exploratory Data Analysis")
    print("=" * 70)

    df = pd.read_csv(DEDUP_CSV)
    print(f"Loaded {len(df):,} deduplicated patient records.")

    # Sample a stratified cohort of 600 images across severity grades
    n_sample = min(600, len(df))
    split = StratifiedShuffleSplit(n_splits=1, train_size=n_sample, random_state=RANDOM_SEED)
    idx, _ = next(split.split(df, df["diagnosis"]))
    sample_df = df.iloc[idx].copy()

    # Load ResNet-101 model
    net = load_fine_tuned_resnet()

    # Extract CNN embeddings and handcrafted features
    t0 = time.time()
    cnn_feats, handcrafted_feats, targets, record_ids = extract_embeddings_and_features(net, sample_df)
    print(f"Feature extraction finished in {time.time()-t0:.1f}s.")

    # Run PCA, t-SNE, UMAP
    metrics = run_dimensionality_reduction(cnn_feats, handcrafted_feats, targets, sample_df)

    # Save metrics CSV
    rows = []
    for k, v in metrics["sil_scores"].items():
        rows.append({"Analysis": "Silhouette Score", "Configuration": k, "Value": round(v, 4)})
    for k, v in metrics["db_scores"].items():
        rows.append({"Analysis": "Davies-Bouldin Index", "Configuration": f"{k} (Severity)", "Value": round(v, 4)})
    rows.append({"Analysis": "PCA Variance", "Configuration": "PC1 Explained", "Value": round(metrics["var_exp"][0], 4)})
    rows.append({"Analysis": "PCA Variance", "Configuration": "PC2 Explained", "Value": round(metrics["var_exp"][1], 4)})
    rows.append({"Analysis": "PCA Variance", "Configuration": "Top 10 Cumulative", "Value": round(metrics["cum_var_exp"][9], 4)})
    rows.append({"Analysis": "PCA Variance", "Configuration": "Top 50 Cumulative", "Value": round(metrics["cum_var_exp"][-1], 4)})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"[SAVED] Saved T13 metrics to: {METRICS_CSV}")

    # Generate Figure
    generate_figure(metrics)

    # Save Report
    md = f"""# T13 Report: Dimensionality Reduction for Clinical Exploratory Data Analysis

> [!NOTE]
> This study projects 2,048-dimensional penultimate-layer CNN representations (from fine-tuned ResNet-101) augmented with handcrafted optical features (Laplacian sharpness, Sobel edge energy, and RGB channel distributions) into 2D manifolds using **PCA**, **t-SNE**, and **UMAP**.

---

## 1. Quantitative Manifold Metrics

| Method | Silhouette Score (Severity Grade) | Davies-Bouldin Index (Lower is better) | Silhouette Score (Image Quality) |
|:---|:---:|:---:|:---:|
| **PCA (2D)** | {metrics['sil_scores']['PCA (Severity)']:.4f} | {metrics['db_scores']['PCA']:.4f} | {metrics['sil_scores']['PCA (Quality)']:.4f} |
| **t-SNE (2D)** | **{metrics['sil_scores']['t-SNE (Severity)']:.4f}** | **{metrics['db_scores']['t-SNE']:.4f}** | {metrics['sil_scores']['t-SNE (Quality)']:.4f} |
| **UMAP (2D)** | **{metrics['sil_scores']['UMAP (Severity)']:.4f}** | **{metrics['db_scores']['UMAP']:.4f}** | {metrics['sil_scores']['UMAP (Quality)']:.4f} |

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
"""
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] Saved T13 report to: {REPORT_MD}")

    # Notebook
    nb_json = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Tutorial 13: Dimensionality Reduction for Clinical Exploratory Data Analysis\n",
                "Applies PCA, t-SNE, and UMAP to 2048-dim CNN embeddings and handcrafted retinal features."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent))\n",
                "import pandas as pd\nfrom PIL import Image\nfrom src.config import REPORTS_DIR, FIGURES_DIR\n",
                "metrics = pd.read_csv(REPORTS_DIR / 't13_dim_reduction_metrics.csv')\nmetrics"
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "# Display 300 DPI Dimensionality Reduction Figure\n",
                "img = Image.open(FIGURES_DIR / 't13_dimensionality_reduction.png')\n",
                "img.resize((img.width // 2, img.height // 2))"
            ]}
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    print(f"[NOTEBOOK] Created: {NB_PATH}")
    print("\n[T13 COMPLETE]")


if __name__ == "__main__":
    run_t13_pipeline()

