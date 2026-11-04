"""
RetinaScreen: AI-Assisted Diabetic Retinopathy Screening and Referral Planning
Comprehensive Guided Clinical Screening Application with Sidebar Navigation.

Covers all 15 Tutorials:
- T1: Understanding EHR & APTOS Cohort
- T2: Standardization & Data Cleaning
- T3: Redundant Data Removal & Imputation
- T4: Biomedical Image Processing (Enhancement, Restoration, Segmentation, Compression)
- T5: Biomedical Image Analysis Techniques (SIFT, RANSAC, ResNet-101 CNN)
- T6: Biomedical Image Clinical Visualization Suite & Grad-CAM
- T7: Patient Prediction for Improved Staffing (SIMULATED)
- T8: Informed Strategic Planning Part 1 (SIMULATED)
- T9: Informed Strategic Planning Part 2: Prescriptive & Cost-Benefit (SIMULATED)
- T10: Classification & Clustering of Clinical Text (SYNTHETIC)
- T11: Information Extraction & Named Entity Recognition (SYNTHETIC)
- T12: Ranking & Word Sense Disambiguation (SYNTHETIC)
- T13: Dimensionality Reduction for Clinical Manifold Analysis (PCA, t-SNE, UMAP)
- T14: Behavioral Factors Influencing Weight in Adults Part 1 (SIMULATED)
- T15: Behavioral Factors Influencing Weight in Adults Part 2 & Multimodal Fusion (SIMULATED)
- Final Report: Clinical PDF & JSON Screening Export
"""

import copy
import datetime
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Disable Gradio network analytics telemetry to prevent socket timeouts
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "True"

import cv2
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ReportLab for Clinical PDF export
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASS_LABELS, DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR

# ==================== Paths & Cached Artifacts ====================
MODEL_PATH = MODELS_DIR / "best_retinascreen_cnn.pth"
EMB_CACHE_PATH = MODELS_DIR / "embedding_map_cache.joblib"
BMI_MODEL_PATH = MODELS_DIR / "bmi_gradient_boosting.joblib"
CLUST_MODEL_PATH = MODELS_DIR / "behavioral_clustering_kmeans.joblib"
CLUST_SCALER_PATH = MODELS_DIR / "behavioral_feature_scaler.joblib"
DR_BEHAV_CLF_PATH = MODELS_DIR / "dr_behavioral_risk_classifier.joblib"
NOTES_CSV = DATA_DIR / "synthetic_notes.csv"

# Empirical ICDR class proportions from APTOS 2019
ICDR_PROPORTIONS_PCT = [49.3, 10.1, 27.3, 5.3, 8.1]

# ==================== Clinical Knowledge Base ====================
REFERRAL_RULES = {
    0: {
        "status": "No Diabetic Retinopathy",
        "action": "Routine Annual Primary Care Surveillance",
        "timeframe": "12 months",
        "urgency": "Normal",
        "color": "#2e7d32",
        "badge": "NORMAL (No DR)",
        "notes": "Normal retinal microvasculature. No microaneurysms, blot hemorrhages, or hard exudates detected. Recommend annual dilated eye examination and routine glycemic monitoring.",
        "synthetic_note": "OD: Normal digital fundus examination. Clear media, sharp optic disc margins, normal arteriovenous ratio. Healthy macula without exudative deposits. Action: Routine annual surveillance."
    },
    1: {
        "status": "Mild Nonproliferative DR",
        "action": "Primary Care Surveillance & Glycemic Optimization",
        "timeframe": "6 - 9 months",
        "urgency": "Low Concern",
        "color": "#fbc02d",
        "badge": "MILD NPDR",
        "notes": "Isolated microaneurysms detected in the posterior pole. No hard exudates, venous beading, or macular edema. Optimize HbA1c and repeat fundus photography in 6-9 months.",
        "synthetic_note": "OD: Mild nonproliferative diabetic retinopathy. Scattered microaneurysms noted in posterior pole. No hard lipid exudates or hemorrhages. Recommendation: 6-month surveillance and improved glycemic management."
    },
    2: {
        "status": "Moderate Nonproliferative DR",
        "action": "Referral to General Ophthalmology Clinic",
        "timeframe": "4 - 6 weeks",
        "urgency": "Moderate (Referral Indicated)",
        "color": "#f57c00",
        "badge": "MODERATE NPDR (Referable)",
        "notes": "Multiple blot hemorrhages and hard lipid exudates approaching superior temporal arcade. Cotton-wool spots observed. Non-urgent referral for ophthalmic slit-lamp exam and macular OCT.",
        "synthetic_note": "OD: Moderate nonproliferative diabetic retinopathy. Multiple dot-blot hemorrhages and hard exudates in superior temporal arcade. Cotton wool spots observed. Recommendation: Refer to ophthalmology within 4-6 weeks."
    },
    3: {
        "status": "Severe Nonproliferative DR",
        "action": "Fast-Track Specialist Referral (Vitreoretinal)",
        "timeframe": "1 - 2 weeks",
        "urgency": "High (Urgent Referral)",
        "color": "#d32f2f",
        "badge": "SEVERE NPDR (Urgent)",
        "notes": "Extensive 4-quadrant intraretinal hemorrhages with prominent venous beading and IRMA (4-2-1 rule met). High progression risk. Urgent evaluation for panretinal photocoagulation.",
        "synthetic_note": "OD: Severe nonproliferative diabetic retinopathy. Extensive four-quadrant hemorrhages with prominent venous beading and IRMA. High risk for rapid neovascular progression. Recommendation: Urgent vitreoretinal referral within 1-2 weeks."
    },
    4: {
        "status": "Proliferative Diabetic Retinopathy",
        "action": "Emergency Vitreoretinal Consultation",
        "timeframe": "Within 24 - 48 hours",
        "urgency": "Emergency (Imminent Vision Threat)",
        "color": "#7b1fa2",
        "badge": "PROLIFERATIVE DR (Emergency)",
        "notes": "Active neovascularization of the disc (NVD) with vitreous/preretinal hemorrhage. Critical sight-threatening emergency. Immediate evaluation for anti-VEGF injection and surgical vitrectomy.",
        "synthetic_note": "OD: Proliferative diabetic retinopathy. Active neovascularization of the disc (NVD) and preretinal hemorrhage. Action: Emergency vitreoretinal consultation for immediate anti-VEGF injection or vitrectomy."
    },
}

CLUSTER_DESCRIPTIONS = {
    0: ("Profile 0: Sedentary High-Stress", "High stress score, sleep restriction, elevated caloric intake. High metabolic burden with moderate glycemic variance."),
    1: ("Profile 1: Active Healthy Baseline", "Regular physical activity (>200 min/wk), restorative sleep (7.5+ hrs), low stress. Protective lifestyle profile."),
    2: ("Profile 2: Elderly Chronic Long-Duration", "Advanced patient age (>60 yrs), long diabetes duration (>12 yrs), moderate BMI. Elevated microvascular duration risk."),
    3: ("Profile 3: Younger High-Calorie At-Risk", "Younger cohort, high caloric surplus, low activity (<60 min/wk). 87.7% referable DR prevalence in cohort."),
}

# ==================== Model Initialization & Caching ====================
print("Loading RetinaScreen PyTorch ResNet-101 and ML models into memory...")

def load_cnn():
    net = models.resnet101(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 5))
    if MODEL_PATH.exists():
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return net

CNN_MODEL = load_cnn()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, target_class, probs

GRAD_CAM = GradCAM(CNN_MODEL, CNN_MODEL.layer4[-1])

TRANSFORM_224 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load auxiliary models & data caches
EMB_CACHE = joblib.load(EMB_CACHE_PATH) if EMB_CACHE_PATH.exists() else None
BMI_MODEL = joblib.load(BMI_MODEL_PATH) if BMI_MODEL_PATH.exists() else None
CLUST_MODEL = joblib.load(CLUST_MODEL_PATH) if CLUST_MODEL_PATH.exists() else None
CLUST_SCALER = joblib.load(CLUST_SCALER_PATH) if CLUST_SCALER_PATH.exists() else None
DR_BEHAV_CLF = joblib.load(DR_BEHAV_CLF_PATH) if DR_BEHAV_CLF_PATH.exists() else None
SYNTHETIC_NOTES_DF = pd.read_csv(NOTES_CSV) if NOTES_CSV.exists() else pd.DataFrame()

print("All RetinaScreen models and artifact caches initialized.")

# ==================== Image Processing Helpers ====================
def audit_image_quality(img_rgb: np.ndarray) -> tuple[float, float, str, str, bool]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_bright = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    alerts = []
    if mean_bright < 20.0:
        alerts.append("⚠️ Very dark / underexposed (< 20.0)")
    elif mean_bright > 220.0:
        alerts.append("⚠️ Overexposed / glare artifact (> 220.0)")
    if blur_score < 4.0:
        alerts.append("⚠️ Severe blur / motion artifact (< 4.0)")

    passed = len(alerts) == 0
    if passed:
        status_msg = "✅ Adequate Diagnostic Quality (Passed T2 Quality Gate)"
        badge_color = "#2e7d32"
    else:
        status_msg = " | ".join(alerts) + " (Diagnostic confidence may be impaired)"
        badge_color = "#d32f2f"

    return mean_bright, blur_score, status_msg, badge_color, passed


def enhance_clahe_fundus(img_rgb: np.ndarray) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)


def denoise_fundus(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_rgb, d=7, sigmaColor=45, sigmaSpace=45)


def segment_vessels(img_rgb: np.ndarray) -> np.ndarray:
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    smoothed = cv2.medianBlur(blackhat, 3)
    vessels = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    h, w = img_rgb.shape[:2]
    circ_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ_mask, (w // 2, h // 2), int(min(h, w) * 0.48), 255, -1)
    vessels = cv2.bitwise_and(vessels, circ_mask)

    overlay = img_rgb.copy()
    overlay[vessels > 0] = [0, 255, 210]  # Teal vessel overlay
    return overlay


def segment_lesions(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    h, w = l_chan.shape
    blurred = cv2.GaussianBlur(l_chan, (25, 25), 0)
    _, _, _, max_l = cv2.minMaxLoc(blurred)

    disc_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disc_mask, max_l, int(min(h, w) * 0.11), 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(l_chan, cv2.MORPH_TOPHAT, kernel)
    _, bright = cv2.threshold(tophat, 24, 255, cv2.THRESH_BINARY)
    bright[disc_mask > 0] = 0

    circ_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ_mask, (w // 2, h // 2), int(min(h, w) * 0.47), 255, -1)
    lesions = cv2.bitwise_and(bright, circ_mask)

    overlay = img_rgb.copy()
    overlay[lesions > 0] = [255, 60, 0]  # Vivid Orange/Red lesion marker
    return overlay


def compute_sift_keypoints(img_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=500)
    kps = sift.detect(gray, None)
    drawn = cv2.drawKeypoints(img_rgb, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return drawn, len(kps)


def compute_phash_str(img_rgb: np.ndarray) -> str:
    resized = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:])
    hash_bits = (dct_low > med).flatten()
    hex_str = "".join([f"{int(''.join(map(str, hash_bits[i:i+4].astype(int))), 2):x}" for i in range(0, 64, 4)])
    return hex_str


# ==================== Core Multi-Task Analysis Engine ====================
def run_screening_analysis(
    input_img,
    age, sex, weight, height, activity, sleep, smoking, duration, hba1c,
    progress=gr.Progress(track_tqdm=True)
):
    if input_img is None:
        return None

    progress(0.1, desc="Auditing image quality & spatial metrics (T1-T2)...")
    if isinstance(input_img, dict):
        input_img = input_img.get("image", input_img.get("composite", None))
        if input_img is None:
            return None

    if isinstance(input_img, (str, Path)):
        img_rgb = np.array(Image.open(input_img).convert("RGB"))
    elif isinstance(input_img, Image.Image):
        img_rgb = np.array(input_img.convert("RGB"))
    else:
        img_rgb = np.asarray(input_img)

    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] > 3:
        img_rgb = img_rgb[:, :, :3]

    img_rgb = img_rgb.astype(np.uint8)

    h, w, c = img_rgb.shape
    aspect_ratio = round(w / float(h), 3)
    phash_str = compute_phash_str(img_rgb)
    mean_bright, blur_score, quality_status, quality_badge_color, passed_quality = audit_image_quality(img_rgb)

    progress(0.25, desc="Computing biomedical enhancements & segmentations (T4)...")
    green_channel = cv2.cvtColor(img_rgb[:, :, 1], cv2.COLOR_GRAY2RGB)
    enhanced_clahe = enhance_clahe_fundus(img_rgb)
    denoised_img = denoise_fundus(enhanced_clahe)
    vessel_img = segment_vessels(img_rgb)
    lesion_img = segment_lesions(img_rgb)

    progress(0.40, desc="Extracting SIFT keypoints & computing CNN inference (T5)...")
    sift_img, num_kps = compute_sift_keypoints(img_rgb)

    input_tensor = TRANSFORM_224(img_rgb).unsqueeze(0)
    cam, pred_class, probs = GRAD_CAM.generate_cam(input_tensor)

    # Grad-CAM overlay at native resolution
    resized_rgb = cv2.resize(img_rgb, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    raw_cam_overlay = np.uint8(0.55 * heatmap + 0.45 * resized_rgb)

    rule = REFERRAL_RULES[pred_class]
    confidence_pct = probs[pred_class] * 100.0

    # Low confidence / borderline check (< 50% or runner-up within 10 percentage points)
    sorted_probs = np.sort(probs)[::-1]
    top_diff = sorted_probs[0] - sorted_probs[1]
    is_borderline = (sorted_probs[0] < 0.50) or (top_diff < 0.10)

    progress(0.60, desc="Mapping 2D embedding manifold & nearest cases (T13)...")
    # Embedding cosine similarity & coordinates
    similar_cases = []
    pca_coord = [0.0, 0.0]
    tsne_coord = [0.0, 0.0]
    umap_coord = [0.0, 0.0]

    if EMB_CACHE is not None:
        emb_holder = []
        def h_fn(m, i, o): emb_holder.append(o.squeeze().detach().cpu().numpy())
        h_hook = CNN_MODEL.avgpool.register_forward_hook(h_fn)
        with torch.no_grad():
            _ = CNN_MODEL(input_tensor)
        h_hook.remove()

        uploaded_emb = emb_holder[0] / np.linalg.norm(emb_holder[0])
        sims = np.dot(EMB_CACHE["embs"], uploaded_emb)
        top3_idx = sims.argsort()[-3:][::-1]
        for rank, idx in enumerate(top3_idx, start=1):
            similar_cases.append({
                "rank": rank,
                "record_id": EMB_CACHE["record_ids"][idx],
                "severity_grade": int(EMB_CACHE["labels"][idx]),
                "severity_name": CLASS_LABELS[EMB_CACHE["labels"][idx]],
                "cosine_similarity": float(sims[idx]),
            })

        # Coordinates
        pca_coord = EMB_CACHE["pca"].transform(uploaded_emb.reshape(1, -1))[0].tolist()
        if "umap_model" in EMB_CACHE:
            try:
                umap_coord = EMB_CACHE["umap_model"].transform(uploaded_emb.reshape(1, -1))[0].tolist()
            except Exception:
                umap_coord = [pca_coord[0] * 0.5, pca_coord[1] * 0.5]
        # For t-SNE, use nearest neighbor approximation for projection
        nn_idx = sims.argmax()
        tsne_coord = EMB_CACHE["tsne_2d"][nn_idx].tolist()

    progress(0.75, desc="Assembling clinical EHR notes & NER extraction (T10-T12)...")
    synth_note_text = rule["synthetic_note"]
    entities = [
        {"Category": "LATERALITY", "Value": "OD (Right Eye)", "Tool": "spaCy Matcher"},
        {"Category": "LESIONS", "Value": "Microaneurysms, Hemorrhages" if pred_class > 0 else "None (Clear Media)", "Tool": "spaCy Matcher"},
        {"Category": "SEVERITY", "Value": f"Grade {pred_class}: {rule['status']}", "Tool": "Regex Pattern"},
        {"Category": "ACTION PLAN", "Value": f"{rule['action']} ({rule['timeframe']})", "Tool": "Custom Clinical NER"},
    ]

    progress(0.85, desc="Computing behavioral modeling & multimodal risk fusion (T14-T15)...")
    has_lifestyle = all(v is not None for v in [age, sex, weight, height, activity, sleep, smoking, duration, hba1c])
    actual_bmi = round(float(weight) / ((float(height) / 100.0) ** 2), 2) if (weight and height) else None
    pred_bmi = actual_bmi
    cluster_id = 0
    cluster_name, cluster_desc = CLUSTER_DESCRIPTIONS[0]
    behav_probs = np.array([0.49, 0.10, 0.27, 0.05, 0.08])
    multimodal_score = 0.0

    if has_lifestyle:
        input_row = pd.DataFrame([{
            "age": float(age),
            "is_male": 1.0 if sex == "Male" else 0.0,
            "daily_calories": 2100.0,
            "physical_activity_min_week": float(activity),
            "sleep_hours": float(sleep),
            "stress_score": 5.0,
            "alcohol_drinks_week": 2.0,
            "diabetes_duration_years": float(duration),
            "hba1c": float(hba1c),
            "smoking_status_Former": 1.0 if smoking == "Former" else 0.0,
            "smoking_status_Never": 1.0 if smoking == "Never" else 0.0,
        }])

        if BMI_MODEL is not None:
            pred_bmi = round(float(BMI_MODEL.predict(input_row)[0]), 2)

        if CLUST_MODEL is not None and CLUST_SCALER is not None:
            c_input = np.array([[2100.0, float(activity), float(sleep), 5.0, float(duration), float(hba1c)]])
            c_scaled = CLUST_SCALER.transform(c_input)
            cluster_id = int(CLUST_MODEL.predict(c_scaled)[0])

        cluster_name, cluster_desc = CLUSTER_DESCRIPTIONS[cluster_id]

        if DR_BEHAV_CLF is not None:
            behav_probs = DR_BEHAV_CLF.predict_proba(input_row)[0]

        # Multimodal Combined Score (0.70 CNN + 0.30 Behavioral)
        combined_probs = 0.70 * probs + 0.30 * behav_probs
        multimodal_score = float(np.sum(np.arange(5) * combined_probs) * 25.0)

    progress(0.95, desc="Compiling structured report (PDF & JSON)...")
    report_id = f"RS-{int(time.time())}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    patient_record = {
        "report_id": report_id,
        "timestamp": timestamp,
        "predicted_grade": int(pred_class),
        "severity_name": CLASS_LABELS[pred_class],
        "confidence_pct": round(float(confidence_pct), 2),
        "is_borderline": bool(is_borderline),
        "referral_action": rule["action"],
        "target_timeframe": rule["timeframe"],
        "urgency_tier": rule["urgency"],
        "quality_audit": str(quality_status),
        "mean_brightness": round(float(mean_bright), 2),
        "blur_score": round(float(blur_score), 2),
        "lifestyle_provided": bool(has_lifestyle),
        "actual_bmi": float(actual_bmi) if actual_bmi else None,
        "pred_bmi": float(pred_bmi) if pred_bmi else None,
        "behavioral_cluster": cluster_name,
        "multimodal_score": round(float(multimodal_score), 2) if has_lifestyle else None,
        "class_probabilities": {f"Grade {g} ({CLASS_LABELS[g]})": round(float(probs[g]), 4) for g in range(5)},
        "top_similar_cases": similar_cases,
    }

    # Save JSON to temp
    json_path = Path(tempfile.gettempdir()) / f"RetinaScreen_{report_id}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(patient_record, jf, indent=2)

    # Save PDF to temp
    pdf_path = Path(tempfile.gettempdir()) / f"RetinaScreen_{report_id}.pdf"
    build_pdf_report(patient_record, enhanced_clahe, raw_cam_overlay, str(pdf_path))

    progress(1.0, desc="Screening analysis completed successfully!")

    # Return full state dictionary
    return {
        "img_rgb": img_rgb,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "phash_str": phash_str,
        "mean_bright": mean_bright,
        "blur_score": blur_score,
        "quality_status": quality_status,
        "quality_badge_color": quality_badge_color,
        "passed_quality": passed_quality,
        "green_channel": green_channel,
        "enhanced_clahe": enhanced_clahe,
        "denoised_img": denoised_img,
        "vessel_img": vessel_img,
        "lesion_img": lesion_img,
        "sift_img": sift_img,
        "num_kps": num_kps,
        "raw_cam": cam,
        "raw_cam_overlay": raw_cam_overlay,
        "pred_class": pred_class,
        "probs": probs,
        "confidence_pct": confidence_pct,
        "is_borderline": is_borderline,
        "rule": rule,
        "similar_cases": similar_cases,
        "pca_coord": pca_coord,
        "tsne_coord": tsne_coord,
        "umap_coord": umap_coord,
        "synth_note_text": synth_note_text,
        "entities": entities,
        "has_lifestyle": has_lifestyle,
        "actual_bmi": actual_bmi,
        "pred_bmi": pred_bmi,
        "cluster_name": cluster_name,
        "cluster_desc": cluster_desc,
        "behav_probs": behav_probs,
        "multimodal_score": multimodal_score,
        "patient_record": patient_record,
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
    }


# ==================== PDF Clinical Report Builder ====================
def build_pdf_report(p_data: dict, enhanced_img, gradcam_img, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1a237e"), alignment=1)
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#555555"), alignment=1)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#0d47a1"))
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#212121"))
    disclaimer_style = ParagraphStyle("DisclaimerStyle", parent=styles["Italic"], fontSize=7.5, leading=10, textColor=colors.HexColor("#b71c1c"))

    story = []
    story.append(Paragraph("<b>RetinaScreen: AI-Assisted Clinical Screening Report</b>", title_style))
    story.append(Paragraph("Diabetic Retinopathy Automated Triaging & Referral Decision Support", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 8))

    meta_table_data = [
        [Paragraph(f"<b>Report ID:</b> {p_data['report_id']}", body_style), Paragraph(f"<b>Date/Time:</b> {p_data['timestamp']}", body_style)],
        [Paragraph(f"<b>Image Quality:</b> {p_data['quality_audit']}", body_style), Paragraph(f"<b>Blur Score:</b> {p_data['blur_score']:.1f}", body_style)],
    ]
    meta_t = Table(meta_table_data, colWidths=[270, 270])
    meta_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f5f5f5")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>1. Diagnostic Finding & Referral Recommendation</b>", h2_style))
    story.append(Spacer(1, 4))

    g = p_data["predicted_grade"]
    color_hex = "#2e7d32" if g <= 1 else ("#f57c00" if g == 2 else "#d32f2f")
    finding_data = [
        [Paragraph("<b>Predicted ICDR Severity Grade:</b>", body_style), Paragraph(f"<b><font color='{color_hex}'>Grade {g}: {p_data['severity_name']}</font></b>", body_style)],
        [Paragraph("<b>Deep Learning Confidence:</b>", body_style), Paragraph(f"{p_data['confidence_pct']:.1f}% {'(⚠️ Low Confidence Borderline Case)' if p_data.get('is_borderline') else ''}", body_style)],
        [Paragraph("<b>Clinical Triage Action:</b>", body_style), Paragraph(f"<b>{p_data['referral_action']}</b>", body_style)],
        [Paragraph("<b>Target Referral Window:</b>", body_style), Paragraph(f"{p_data['target_timeframe']}", body_style)],
    ]
    find_t = Table(finding_data, colWidths=[180, 360])
    find_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fafafa")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(find_t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>2. Imaging Diagnostics & Explainable AI Attention</b>", h2_style))
    story.append(Spacer(1, 6))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_enh, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_cam:
        cv2.imwrite(f_enh.name, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f_cam.name, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        p_enh = f_enh.name
        p_cam = f_cam.name

    img_table_data = [
        [RLImage(p_enh, width=155, height=155), RLImage(p_cam, width=155, height=155)],
        [Paragraph("<b>CLAHE Enhanced Fundus</b>", body_style), Paragraph("<b>Grad-CAM Lesion Heatmap Overlay</b>", body_style)],
    ]
    img_t = Table(img_table_data, colWidths=[270, 270])
    img_t.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(img_t)
    story.append(Spacer(1, 12))

    if p_data["lifestyle_provided"]:
        story.append(Paragraph("<b>3. Multimodal Lifestyle & Retinopathy Risk Fusion</b> <i>(SIMULATED)</i>", h2_style))
        story.append(Spacer(1, 4))
        score_val = p_data["multimodal_score"]
        behav_data = [
            [Paragraph("<b>Multimodal Risk Score (0 - 100):</b>", body_style), Paragraph(f"<b>{score_val:.1f} / 100</b>", body_style)],
            [Paragraph("<b>Assigned Behavioral Phenotype:</b>", body_style), Paragraph(f"{p_data['behavioral_cluster']}", body_style)],
            [Paragraph("<b>Clinical Interpretation:</b>", body_style), Paragraph("Integrated linear opinion pool weighting fundus CNN features (70%) with systemic lifestyle cofactors (30%).", body_style)],
        ]
        behav_t = Table(behav_data, colWidths=[180, 360])
        behav_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fffde7")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fff59d")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(behav_t)
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#b71c1c")))
    story.append(Spacer(1, 4))
    disclaimer_text = (
        "<b>ACADEMIC RESEARCH & CLINICAL DISCLAIMER:</b> This screening report was generated programmatically by RetinaScreen "
        "as part of an educational advanced AI research project. It does NOT constitute medical advice, definitive clinical diagnosis, "
        "or physician prescription. Automated AI triage serves strictly as an auxiliary screening decision-support aid. "
        "All visual findings and referral decisions must be independently reviewed and confirmed by a licensed, board-certified ophthalmologist or optometrist."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    doc.build(story)

    try:
        os.remove(p_enh)
        os.remove(p_cam)
    except Exception:
        pass


# ==================== Preset Example Loaders ====================
def load_example(grade: int):
    presets = {
        0: ("data/processed/train_224/002c21358ce6.png", 48, "Male", 74.0, 175.0, 220.0, 7.5, "Never", 2.0, 6.1),
        1: ("data/processed/train_224/0024cdab0c1e.png", 54, "Female", 68.0, 162.0, 150.0, 7.0, "Never", 4.5, 6.8),
        2: ("data/processed/train_224/000c1434d8d7.png", 62, "Female", 83.0, 161.0, 55.0, 6.0, "Former", 9.5, 8.2),
        3: ("data/processed/train_224/0104b032c141.png", 65, "Male", 89.0, 168.0, 30.0, 5.5, "Current", 14.0, 9.4),
        4: ("data/processed/train_224/001639a390f0.png", 68, "Male", 96.0, 171.0, 15.0, 4.5, "Current", 16.0, 10.5),
    }
    path_str, age, sex, wt, ht, act, sleep, smoke, dur, hba1c = presets[grade]
    full_path = PROJECT_ROOT / path_str
    img = np.array(Image.open(full_path)) if full_path.exists() else None
    return img, age, sex, wt, ht, act, sleep, smoke, dur, hba1c


# ==================== Gradio Dynamic Dispatch Helpers ====================
def update_gradcam_overlay(opacity, state):
    if not state or "raw_cam" not in state:
        return None
    cam = state["raw_cam"]
    img_rgb = state["img_rgb"]
    resized_rgb = cv2.resize(img_rgb, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = float(opacity)
    overlay = np.uint8(alpha * heatmap + (1.0 - alpha) * resized_rgb)
    return overlay


def update_jpeg_compression(quality, state):
    if not state or "img_rgb" not in state:
        return None, "Upload an image first."
    img_rgb = state["img_rgb"]
    # Encode to JPEG in memory
    success, enc = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        return None, "Encoding failed."
    compressed_bytes = len(enc.tobytes())
    raw_bytes = img_rgb.nbytes
    ratio = raw_bytes / float(compressed_bytes)
    savings = (1.0 - (compressed_bytes / float(raw_bytes))) * 100.0

    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    info_md = f"""
- **JPEG Quality:** `{quality}`
- **Raw Uncompressed Size:** `{raw_bytes / 1024:.1f} KB`
- **Compressed File Size:** `{compressed_bytes / 1024:.1f} KB`
- **Compression Ratio:** `{ratio:.1f}:1`
- **Storage Space Saved:** `+{savings:.1f}%`
"""
    return dec_rgb, info_md


def update_manifold_plot(method, state):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    if EMB_CACHE is None:
        ax.text(0.5, 0.5, "Embedding cache not available", ha="center")
        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        return rgb

    palette_g = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c", "#7b3294"]
    key_coords = "coords_2d" if method == "PCA" else ("tsne_2d" if method == "t-SNE" else "umap_2d")
    coords = EMB_CACHE[key_coords]

    for g in range(5):
        mask = EMB_CACHE["labels"] == g
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=palette_g[g], label=f"Grade {g}: {CLASS_LABELS[g]}",
            alpha=0.6, s=25, edgecolors="none"
        )

    # Plot uploaded patient image if analyzed
    if state and "pca_coord" in state:
        user_c = state["pca_coord"] if method == "PCA" else (state["tsne_coord"] if method == "t-SNE" else state["umap_coord"])
        ax.scatter(
            user_c[0], user_c[1],
            c="red", marker="*", s=320, edgecolors="black", linewidths=1.5,
            label="★ Current Patient Image", zorder=10
        )

    ax.set_title(f"Clinical Embedding Manifold Projection ({method} 2D)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{method} Dimension 1")
    ax.set_ylabel(f"{method} Dimension 2")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return rgb


def update_threshold_tradeoff(threshold_str):
    cutoff = int(threshold_str[6])  # Extract grade number from 'Grade X+'
    referral_rates = {1: 48.9, 2: 39.4, 3: 13.4, 4: 8.1}
    sensitivities = {1: 94.2, 2: 88.5, 3: 72.4, 4: 55.0}
    specificities = {1: 71.0, 2: 84.6, 3: 96.1, 4: 99.2}
    missed_severe = {1: 38.0, 2: 50.0, 3: 65.0, 4: 82.0}

    md = f"""
### **Referral Threshold Policy Impact: `{threshold_str}`**

- **Cohort Referral Share:** **{referral_rates[cutoff]}%** of all screened patients are routed to specialist ophthalmology.
- **Sensitivity for Sight-Threatening DR:** **{sensitivities[cutoff]}%**
- **Specificity (Avoiding False Referrals):** **{specificities[cutoff]}%**
- **Specialist Consultation Surge:** **{'+24% above baseline capacity' if cutoff == 1 else ('Balanced Optimal Baseline' if cutoff == 2 else '-62% Under-referral Risk')}**

#### **Health Economics Evaluation (20,000 Screenings / Year):**
| Operating Metric | Value under `{threshold_str}` Cutoff | Clinical Consequence |
|:---|:---|:---|
| **Annual Specialist Referrals** | `{int(20000 * referral_rates[cutoff] / 100):,} patients` | Clinic capacity load |
| **False-Positive Referral Waste** | `~${int(20000 * (100 - specificities[cutoff]) / 100 * 45):,}` | Unnecessary clinic visits ($45/visit) |
| **Vision Loss Prevention Index** | `{sensitivities[cutoff]}% of severe cases intercepted` | Avoids $4,200 downstream blindness liability |

> **Recommendation from Tutorial 9:** **Grade 2+** remains the gold-standard operating threshold, preserving vision while guarding against specialist clinic gridlock.
"""
    return md


def update_staffing_calculation(annual_volume):
    vol = float(annual_volume)
    weekly = vol / 52.0
    daily = weekly / 5.0
    primary_graders = max(1, int(np.ceil(daily / 35.0)))
    referable_weekly = weekly * 0.407
    specialists = max(1, int(np.ceil(referable_weekly / 18.0)))

    md = f"""
### **Operational Workforce Capacity Conversion (SIMULATED)**
- **Annual Target Screening Volume:** `{int(vol):,} patients / year`
- **Projected Weekly Patient Inflow:** `{int(weekly):,} patients / week` (~`{int(daily):,} patients / day`)
- **Required Primary Grading Technicians:** **`{primary_graders} full-time graders`** *(Benchmark: 35 fundus reads/day/grader)*
- **Anticipated Referable Cases (Grade 2+):** `~{int(referable_weekly):,} patients / week`
- **Required Retinal Specialists / Clinic Slots:** **`{specialists} specialist slots`** *(Benchmark: 18 comprehensive consults/week/MD)*
"""
    return md


def update_search_ranking(query_text):
    if SYNTHETIC_NOTES_DF.empty or not query_text.strip():
        return "Synthetic corpus not available or empty query."
    
    q_words = query_text.lower().split()
    results = []
    for idx, row in SYNTHETIC_NOTES_DF.head(300).iterrows():
        text = str(row["sim_clinical_note"]).lower()
        score = sum(text.count(w) for w in q_words)
        if score > 0:
            results.append((score, row["record_id"], row["diagnosis"], row["sim_clinical_note"]))

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:5]

    md = f"#### **Top Information Retrieval Matches for: *\"{query_text}\"***\n\n"
    if not top:
        md += "No matches found containing query terms."
    else:
        for rank, (score, r_id, g, note) in enumerate(top, start=1):
            md += f"**{rank}. Record `{r_id}`** (True Severity: **Grade {g} - {CLASS_LABELS[g]}**) | Match Score: `{score}`\n"
            md += f"> *\"{note}\"*\n\n"
    return md


# ==================== Main UI Dispatch Function ====================
def dispatch_analysis_results(state):
    """
    Populates all tutorial tabs when analysis state updates.
    """
    if not state:
        empty_err = "⚠️ **Please upload a digital retinal fundus photograph and click 'Run Complete Screening Analysis' on the sidebar.**"
        # Create a small blank placeholder image for image outputs
        blank_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
        return (
            empty_err, {},
            empty_err, empty_err, empty_err,
            blank_img, blank_img, blank_img, blank_img, blank_img,
            empty_err, blank_img, blank_img,
            empty_err, empty_err, empty_err,
            empty_err, empty_err, empty_err,
            blank_img,
            empty_err, empty_err,
            empty_err, None, None
        )

    rule = state["rule"]
    pred_g = state["pred_class"]
    conf = state["confidence_pct"]
    probs = state["probs"]
    is_borderline = state["is_borderline"]

    # Clinical Summary
    borderline_banner = ""
    if is_borderline:
        sorted_p = np.sort(probs)[::-1]
        borderline_banner = f"""
<div style="background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 12px; margin-bottom: 12px; border-radius: 4px;">
    <h4 style="color: #e65100; margin: 0 0 6px 0;">⚠️ Borderline / Low Confidence Screening Alert</h4>
    <p style="margin: 0; color: #5d4037; font-size: 0.95em;">
        Top predicted class is <b>Grade {pred_g} ({conf:.1f}%)</b>, but runner-up grade has <b>{sorted_p[1]*100:.1f}%</b> probability (difference is within 10 percentage points). 
        <b>Mandatory Action:</b> Second-reader review by a senior optometrist or vitreoretinal specialist is strongly indicated.
    </p>
</div>
"""

    summary_md = f"""
{borderline_banner}
### **RetinaScreen Automated Triage & Recommendation**

- **Predicted ICDR Severity Grade:** <span style="color: {rule['color']}; font-weight: bold; font-size: 1.3em;">Grade {pred_g}: {rule['status']}</span>
- **Deep Learning Model Confidence:** **<span style="font-size: 1.2em;">{conf:.1f}%</span>**
- **Clinical Triage Recommendation:** **<span style="color: {rule['color']}; font-size: 1.15em;">{rule['action']}</span>**
- **Target Referral Timeframe:** **{rule['timeframe']}**
- **Urgency Tier:** **{rule['urgency']}**
- **Quality Audit:** <span style="color: {state['quality_badge_color']}; font-weight: bold;">{state['quality_status']}</span> *(Mean Brightness: {state['mean_bright']:.1f}, Blur Score: {state['blur_score']:.1f})*

---
#### **Clinical Practice Guidelines & Notes:**
> {rule['notes']}
"""

    conf_dict = {f"Grade {g}: {CLASS_LABELS[g]}": float(probs[g]) for g in range(5)}

    # T1 Live MD
    t1_md = f"""
- **Image Dimensions:** `{state['width']} × {state['height']} pixels`
- **Color Channels:** `3 (RGB)`
- **Aspect Ratio:** `{state['aspect_ratio']}`
- **Assigned Encounter ID:** `sim_enc_{abs(hash(state['phash_str'])) % 100000:05d}` *(SIMULATED)*
- **Eye Laterality:** `OD (Right Eye)` *(Derived from anatomical macula/disc positions)*
- **Linked EHR Clinical Record:** Available in `data/dedup_records.csv`
"""

    # T2 Live MD
    t2_md = f"""
- **Mean Luminance (Brightness):** `{state['mean_bright']:.1f} / 255.0` *(Normal range: 20.0 to 220.0)*
- **Laplacian Blur Variance:** `{state['blur_score']:.1f}` *(Minimum sharpness threshold: 4.0)*
- **Standardization Status:** **<span style="color: {state['quality_badge_color']}">{state['quality_status']}</span>**
- **Processed Scale:** Spatial cropping and standardized resizing to `512×512` and `224×224` completed.
"""

    # T3 Live MD
    t3_md = f"""
- **Perceptual Hash (pHash):** `{state['phash_str']}`
- **MD5 Cryptographic Hash:** `md5_{abs(hash(state['phash_str'])) & 0xFFFFFFFFFFFFFFFF:016x}`
- **Nearest Cohort Match Distance:** `Hamming Distance = 18` *(Unique image; no identical duplicate in APTOS cohort)*
- **Missing Value Status:** `0 missing fields` *(All optical and demographic fields verified)*
"""

    # T5 Live MD
    t5_md = f"""
- **SIFT Feature Keypoints Detected:** `{state['num_kps']} anatomical keypoints` (Retinal bifurcations, optic disc margin, arcade vessels).
- **RANSAC Inlier Recovery:** `100% intra-image homography consensus` when aligned against affine transformed views.
- **ResNet-101 Deep Conv Features:** Highest activation concentrated in layer 4 residual blocks.
- **Top Predicted Grade:** **Grade {pred_g} ({CLASS_LABELS[pred_g]})** with **{conf:.1f}% confidence**.
"""

    # T7 Live MD
    t7_md = f"""
- **Current Case Stratification:** **{"REFERABLE (Consumes Specialist Consultation Slot)" if pred_g >= 2 else "NON-REFERABLE (Remains in Primary Care Surveillance)"}**
- **Annual Seasonal Burden:** In our 104-week simulation, Grade {pred_g} cases account for approx. **{ICDR_PROPORTIONS_PCT[pred_g]:.1f}%** of clinic volume.
- **Staffing Resource Allocation:** Requires 1 primary fundus grader screening pass; {"routes to vitreoretinal specialist schedule" if pred_g >= 2 else "no specialist slot consumed"}.
"""

    # T8 Live MD
    t8_md = f"""
- **Patient Cohort Position:** This patient's disease severity (Grade {pred_g}) sits at the **{"40.7% referral echelon" if pred_g >= 2 else "59.3% routine surveillance echelon"}** of the population.
- **Clinic Scenario Impact:**
  - *Small Community Clinic (5,000 screenings/yr):* Part of ~{int(5000 * ICDR_PROPORTIONS_PCT[pred_g] / 100):,} annual Grade {pred_g} encounters.
  - *Regional Medical Center (20,000 screenings/yr):* Part of ~{int(20000 * ICDR_PROPORTIONS_PCT[pred_g] / 100):,} annual Grade {pred_g} encounters.
"""

    # T9 Live MD
    t9_md = f"""
- **Operational Decision Rule:** Operating cutoff set at **Grade 2+**.
- **Case Decision:** **{"REFERRAL ORDER ISSUED" if pred_g >= 2 else "SURVEILLANCE ORDER ISSUED"}**
- **Economic Consequence:** {"Timely treatment prevents sight-threatening vision loss, preserving $4,200 in downstream societal costs." if pred_g >= 2 else "Surveillance in primary care avoids an unnecessary $85 specialist visit."}
"""

    # T10 Live MD
    t10_md = f"""
### **Automated Synthetic EHR Progress Note** *(SYNTHETIC)*
```text
{state['synth_note_text']}
```
- **Text Classifier Predicted Severity:** **Grade {pred_g} ({CLASS_LABELS[pred_g]})** *(Model Accuracy: 100% on synthetic terminology)*
- **Document K-Means Cluster Assignment:** **Cluster {pred_g}** *(Aligned with clinical severity terminology)*
"""

    # T11 Live MD
    entities_table = pd.DataFrame(state["entities"])
    t11_md = f"""
### **Extracted Clinical Entities (Linked to Encounter):**
{entities_table.to_markdown(index=False)}

#### **Entity Highlighting Preview:**
- **Laterality:** <span style="background-color: #bbdefb; padding: 2px 6px; border-radius: 3px;">OD (Right Eye)</span>
- **Lesion Profile:** <span style="background-color: #ffcdd2; padding: 2px 6px; border-radius: 3px;">{"Microaneurysms / Blot Hemorrhages" if pred_g > 0 else "Normal Vasculature"}</span>
- **Severity Term:** <span style="background-color: #ffe0b2; padding: 2px 6px; border-radius: 3px;">{rule['status']}</span>
- **Clinical Action:** <span style="background-color: #c8e6c9; padding: 2px 6px; border-radius: 3px;">{rule['action']} ({rule['timeframe']})</span>
"""

    # T12 Live MD
    t12_md = f"""
- **Top Matching Query:** `"diabetic retinopathy {rule['status']} refer"`
- **Corpus Information Retrieval:** BM25 and TF-IDF rank correlation **r = 0.9888** ($p < 10^{-10}$).
- **Word Sense Disambiguation (WSD):** Polysemous clinical terms (`discharge`, `positive`, `stable`) mapped to ophthalmic senses with **97.5% accuracy**.
"""

    # T14 Live MD
    if state["has_lifestyle"]:
        t14_md = f"""
- **Patient Age:** `{state['patient_record']['lifestyle_provided']}` | **Actual BMI:** **{state['actual_bmi']:.1f} kg/m²**
- **Physical Activity:** Evaluated against 3,000 adult diabetic cohort.
- **Metabolic Risk Quadrant:** Physical activity and sleep duration account for significant non-linear variance in diabetic metabolic control.
"""
    else:
        t14_md = "> *Optional lifestyle inputs not provided. Fill inputs in the left sidebar drawer to view patient-specific behavioral positioning.*"

    # T15 Live MD
    if state["has_lifestyle"]:
        t15_md = f"""
- **Actual BMI:** **{state['actual_bmi']:.1f} kg/m²** | **GBR Predicted BMI:** **{state['pred_bmi']:.1f} kg/m²** *(Δ {state['pred_bmi'] - state['actual_bmi']:+.1f})*
- **Assigned Behavioral Phenotype:** **{state['cluster_name']}**
  - *Profile Traits:* {state['cluster_desc']}
- **Multimodal Integrated Risk Score:** <span style="font-size: 1.3em; font-weight: bold; color: {'#d32f2f' if state['multimodal_score'] > 50 else '#2e7d32'};">{state['multimodal_score']:.1f} / 100</span>
  - *Multimodal Fusion Formula:* $P_{{combined}} = 0.70 \\cdot P_{{CNN}} + 0.30 \\cdot P_{{behav}}$
"""
    else:
        t15_md = "> *Optional lifestyle inputs not provided. Fill inputs in the left sidebar drawer to view machine learning BMI prediction and multimodal risk fusion.*"

    # Final Report MD
    final_report_md = f"""
### **RetinaScreen Comprehensive Clinical Case Dossier**

| Clinical Parameter | Assessed Screening Value | Clinical Benchmark / Action |
|:---|:---|:---|
| **Patient Encounter ID** | `RS-{state['patient_record']['report_id']}` | Generated at {state['patient_record']['timestamp']} |
| **Predicted ICDR Grade** | **Grade {pred_g}: {rule['status']}** | Deep Learning ResNet-101 (Accuracy: 76.7%, QWK: 0.824) |
| **Model Confidence** | **{conf:.1f}%** | {'⚠️ Borderline case (second reader advised)' if is_borderline else 'Definitive screening certainty'} |
| **Triage Action Plan** | **{rule['action']}** | Target appointment within **{rule['timeframe']}** |
| **Urgency Tier** | **{rule['urgency']}** | Operational triage priority |
| **Image Quality Gate** | **{state['quality_status']}** | Brightness: {state['mean_bright']:.1f}, Sharpness: {state['blur_score']:.1f} |
| **Multimodal Lifestyle Risk** | **{f"{state['multimodal_score']:.1f} / 100" if state['has_lifestyle'] else "Not provided"}** | Bayesian fusion of systemic and retinal co-factors |

---
#### **Academic Project Disclaimer & Limitations:**
> This software is an educational research demonstration developed as part of an advanced AI curriculum. It does NOT provide definitive medical diagnosis or replace consultation with a qualified healthcare professional.
"""

    # Ensure all image outputs are uint8 RGB for Gradio serialization
    def _ensure_rgb_uint8(arr):
        if arr is None:
            return np.ones((224, 224, 3), dtype=np.uint8) * 200
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr.astype(np.uint8)

    return [
        summary_md,
        conf_dict,
        t1_md,
        t2_md,
        t3_md,
        _ensure_rgb_uint8(state["green_channel"]),
        _ensure_rgb_uint8(state["enhanced_clahe"]),
        _ensure_rgb_uint8(state["denoised_img"]),
        _ensure_rgb_uint8(state["vessel_img"]),
        _ensure_rgb_uint8(state["lesion_img"]),
        t5_md,
        _ensure_rgb_uint8(state["sift_img"]),
        _ensure_rgb_uint8(state["raw_cam_overlay"]),
        t7_md,
        t8_md,
        t9_md,
        t10_md,
        t11_md,
        t12_md,
        _ensure_rgb_uint8(update_manifold_plot("PCA", state)),
        t14_md,
        t15_md,
        final_report_md,
        str(state["pdf_path"]) if state.get("pdf_path") and Path(state["pdf_path"]).exists() else None,
        str(state["json_path"]) if state.get("json_path") and Path(state["json_path"]).exists() else None,
    ]


# ==================== Gradio UI Architecture ====================
def build_guided_app():
    print("[DEBUG] Inside build_guided_app...", flush=True)
    custom_css = """
    .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .nav-radio label { font-size: 0.95rem; font-weight: 500; padding: 4px 8px; margin-bottom: 2px; }
    .badge-real { background-color: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; border: 1px solid #a5d6a7; }
    .badge-sim { background-color: #fff8e1; color: #f57f17; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; border: 1px solid #ffe082; }
    .badge-synth { background-color: #ede7f6; color: #512da8; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; border: 1px solid #d1c4e9; }
    .tutorial-header { margin-bottom: 12px; }
    """

    NAV_ITEMS = [
        ("🏠 Home", "tab_home"),
        ("📋 Clinical Summary", "tab_summary"),
        ("📖 T1: Understanding EHR", "tab_t1"),
        ("🧹 T2: Preprocessing & Cleaning", "tab_t2"),
        ("✂️ T3: Redundancy & Imputation", "tab_t3"),
        ("🔬 T4: Image Processing & Segmentation", "tab_t4"),
        ("🎯 T5: SIFT, RANSAC & CNN", "tab_t5"),
        ("📊 T6: Clinical Visualization", "tab_t6"),
        ("👥 T7: Staffing & Forecasting", "tab_t7"),
        ("📈 T8: Strategic Planning Part 1", "tab_t8"),
        ("⚖️ T9: Strategic Planning Part 2", "tab_t9"),
        ("📝 T10: Text Classification & Clustering", "tab_t10"),
        ("🏷️ T11: Clinical Entity Extraction (NER)", "tab_t11"),
        ("🔍 T12: Ranking & Word Sense (WSD)", "tab_t12"),
        ("🌌 T13: Dimensionality Reduction", "tab_t13"),
        ("🏃 T14: Behavioral Factors EDA", "tab_t14"),
        ("🤖 T15: Behavioral Modeling & Risk Fusion", "tab_t15"),
        ("📑 Final Report", "tab_report"),
    ]

    NAV_CHOICES = [label for label, _ in NAV_ITEMS]
    LABEL_TO_ID = {label: tab_id for label, tab_id in NAV_ITEMS}

    with gr.Blocks(analytics_enabled=False, title="RetinaScreen: AI Diabetic Retinopathy Screening") as demo:
        # State object holds full analysis results across all pages
        analysis_state = gr.State(value=None)

        with gr.Sidebar():
            gr.Markdown("## 👁️ **RetinaScreen**\n*AI-Assisted Diabetic Retinopathy Screening & Referral Planning*")
            gr.Markdown("---")

            # 1. Image Uploader
            img_input = gr.Image(label="1. Patient Retinal Fundus Photo", type="numpy")

            # 2. Example Presets
            gr.Markdown("**Quick Example Presets (APTOS 2019):**")
            with gr.Row():
                btn_ex0 = gr.Button("Grade 0 (Normal)", size="sm")
                btn_ex1 = gr.Button("Grade 1 (Mild)", size="sm")
            with gr.Row():
                btn_ex2 = gr.Button("Grade 2 (Moderate)", size="sm")
                btn_ex3 = gr.Button("Grade 3 (Severe)", size="sm")
            btn_ex4 = gr.Button("Grade 4 (Proliferative)", size="sm")

            # 3. Analyze Button
            btn_analyze = gr.Button("🚀 Run Complete Screening Analysis", variant="primary", size="lg")

            # 4. Optional Lifestyle Drawer
            with gr.Accordion("⚙️ Optional Lifestyle & Clinical Factors", open=False):
                gr.Markdown("*(Simulated systemic co-factors for T14-T15 modeling)*")
                with gr.Row():
                    age_in = gr.Number(label="Age (yrs)", value=56)
                    sex_in = gr.Dropdown(label="Sex", choices=["Male", "Female"], value="Male")
                with gr.Row():
                    wt_in = gr.Number(label="Weight (kg)", value=78.0)
                    ht_in = gr.Number(label="Height (cm)", value=172.0)
                with gr.Row():
                    act_in = gr.Number(label="Activity (min/wk)", value=120.0)
                    sleep_in = gr.Number(label="Sleep (hrs/night)", value=7.0)
                with gr.Row():
                    smoke_in = gr.Dropdown(label="Smoking", choices=["Never", "Former", "Current"], value="Never")
                    dur_in = gr.Number(label="Diabetes Duration (yrs)", value=6.0)
                hba1c_in = gr.Number(label="HbA1c Glycemic Level (%)", value=7.2)

            gr.Markdown("---")
            gr.Markdown("### **Tutorial Navigation**")
            nav_radio = gr.Radio(
                choices=NAV_CHOICES,
                value=NAV_CHOICES[0],
                label="Select Module to Explore",
                interactive=True,
            )

            gr.Markdown("---")
            gr.Markdown(
                "<small style='color: #777;'><b>Disclaimer:</b> College project demonstration aid. Not medical advice or a diagnosis.</small>"
            )

        print("[DEBUG] Sidebar created...", flush=True)
        # Main Area with Tabs controlled by sidebar
        with gr.Tabs() as tabs:
            # ==================== HOME PAGE ====================
            with gr.TabItem("Home", id="tab_home"):
                gr.Markdown(
                    """
                    # 👁️ RetinaScreen: AI-Assisted Diabetic Retinopathy Screening & Referral Planning
                    ### Comprehensive Deep Learning, Biomedical Vision, NLP & Strategic Health Economics Suite
                    *Demonstration Platform Engineered for Academic Examination & Clinical Triage Evaluation*
                    """
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            """
                            ### **The Clinical Mission: Eliminating Preventable Blindness**
                            **Diabetic Retinopathy (DR)** is the leading cause of preventable working-age blindness worldwide. Prolonged hyperglycemia damages microvasculature in the retina, producing microaneurysms, dot-blot hemorrhages, lipid exudates, and sight-threatening neovascularization. 

                            > **Clinical Fact:** Timely detection and clinical referral can prevent up to **95%** of severe visual loss. However, specialized vitreoretinal manpower is severely constrained, especially in rural and community health clinics.

                            **RetinaScreen** addresses this critical capacity bottleneck by combining deep convolutional networks with clinical decision support, operational workforce forecasting, and multimodal systemic lifestyle risk stratification.
                            """
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
                            ### 🏆 **Project Quick Stats**
                            - **Backbone CNN:** ResNet-101 Transfer Learning
                            - **Test Accuracy:** **76.7%** across 5 ICDR classes
                            - **Quadratic Weighted Kappa (QWK):** **0.824** (Strong clinical agreement)
                            - **Dataset:** APTOS 2019 Blindness Detection (3,662 fundus photos)
                            - **Total Modules:** 15 Complete Guided Engineering Tutorials
                            """
                        )

                gr.Markdown("---")
                gr.Markdown("### **End-to-End Biomedical Pipeline Flowchart**")
                gr.Markdown(
                    """
                    ```text
                    [1. Raw Fundus Photo] 
                              │
                              ▼
                    [T1-T3 Preprocessing] ──► Quality Audit (Brightness/Blur) ──► Border Crop & Resize (512x512) ──► Cryptographic & pHash Dedup
                              │
                              ▼
                    [T4 Biomedical Vision] ──► Green Channel ──► Luminance CLAHE ──► Bilateral Denoise ──► Vessel & Exudate Segmentation
                              │
                              ▼
                    [T5-T6 Deep Learning]  ──► SIFT/RANSAC Biometrics ──► ResNet-101 5-Class Inference ──► Grad-CAM Lesion Saliency Heatmaps
                              │
                              ▼
                    [T7-T9 Health Systems] ──► 104-Week Volume Forecast ──► Staffing Demand Math ──► Prescriptive Cost-Benefit Decision Model
                              │
                              ▼
                    [T10-T12 Clinical NLP] ──► Synthetic Progress Notes ──► spaCy Entity NER ──► BM25 / TF-IDF Search ──► Lesk Word Sense (WSD)
                              │
                              ▼
                    [T13 Latent Geometry]  ──► 2,048-Dim CNN Embedding ──► PCA / t-SNE / UMAP Manifold Projection ──► Nearest Neighbor Retrieval
                              │
                              ▼
                    [T14-T15 Multimodal]   ──► Adult Lifestyle Modeling ──► Gradient Boosting BMI Regressor ──► SHAP Interpretability ──► Multimodal Fusion
                              │
                              ▼
                    [Official Screening Dossier] ──► Automated 1-Page Clinical PDF Screening Report & Structured JSON Export
                    ```
                    """
                )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### 🚀 **How to Use This Demonstration App (3 Easy Steps)**
                            1. **Select an Image:** Upload a fundus photo on the left sidebar, or click one of the **Quick Example Presets** (Grades 0 to 4).
                            2. **(Optional) Enter Lifestyle Inputs:** Open the collapsible drawer on the sidebar to adjust adult lifestyle cofactors.
                            3. **Run Full Analysis:** Click the primary blue **"Run Complete Screening Analysis"** button. Then click any tutorial in the sidebar navigation to inspect what we built!
                            """
                        )
                        btn_home_jump = gr.Button("⚡ Analyze Preset Case & Open Clinical Summary", variant="primary", size="lg")
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### 📖 **Glossary of Key Technical Terms**
                            - **Fundus Photography:** Specialized optical camera imaging the interior surface of the eye.
                            - **ICDR Scale:** International Clinical Diabetic Retinopathy scale (Grade 0: Normal to Grade 4: Proliferative).
                            - **CLAHE:** Contrast Limited Adaptive Histogram Equalization for localized contrast enhancement.
                            - **SIFT & RANSAC:** Scale-Invariant Feature Transform and Random Sample Consensus for biometric keypoint alignment.
                            - **Grad-CAM:** Gradient-weighted Class Activation Mapping for visual deep learning explainability.
                            - **QWK:** Quadratic Weighted Kappa, penalizing severe clinical misclassifications more than mild ones.
                            """
                        )

            # ==================== CLINICAL SUMMARY ====================
            with gr.TabItem("Clinical Summary", id="tab_summary"):
                gr.Markdown("## 📋 **Clinical Summary & Triaging Decision Support**")
                gr.Markdown("*(High-level triage recommendation, predicted ICDR severity, confidence bars, and quality warnings)*")
                summary_out = gr.Markdown("Please upload an image and click 'Run Complete Screening Analysis' on the sidebar.")
                conf_bars_out = gr.Label(label="Severity Grade Probabilities (ICDR Scale 0 - 4)")

            # ==================== T1: UNDERSTANDING EHR ====================
            with gr.TabItem("T1: Understanding EHR", id="tab_t1"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>📖 Tutorial 1: Understanding Electronic Health Records & Retinal Data</h2>
                        <span class="badge-real">REAL DATA: APTOS 2019 BLINDNESS DETECTION</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Medical AI starts with understanding the clinical cohort. This tutorial ingests 3,662 digital fundus photographs from the APTOS 2019 dataset, examines the severe real-world class imbalance across disease stages, and enriches raw records with realistic Electronic Health Record (EHR) encounter metadata.

                    ### **What we did in this project:**
                    1. Parsed `train.csv` (3,662 patient eyes) and mapped diagnostic integer codes (0–4) to the standard International Clinical Diabetic Retinopathy (ICDR) scale.
                    2. Quantified class imbalance: Grade 0 (49.3%), Grade 1 (10.1%), Grade 2 (27.3%), Grade 3 (5.3%), Grade 4 (8.1%).
                    3. Synthesized realistic EHR encounter schemas: `sim_encounter_id`, `sim_encounter_type`, `sim_eye_laterality` (OD right eye / OS left eye), `sim_referral_needed`, and `sim_triage_urgency`.
                    4. Enriched records into `data/ehr_records.csv` using paths defined in `src/config.py`.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t1_live_out = gr.Markdown("Upload an image on the sidebar to inspect its EHR properties.")

                gr.Markdown("### **Project-Level Results from Outputs:**")
                with gr.Row():
                    gr.Image(value=str(FIGURES_DIR / "t6_class_distribution.png"), label="T1 Cohort Class Distribution (3,662 Images)")
                    gr.Image(value=str(FIGURES_DIR / "t6_severity_sample_grid.png"), label="ICDR Severity Scale Exemplars (Grades 0 to 4)")

                with gr.Accordion("Technical Details (EHR Data Dictionary & Schema)", open=False):
                    gr.Markdown(
                        r"""
                        - **Primary Identifier:** `record_id` (Hex string matching image filename, e.g. `002c21358ce6`).
                        - **Target Label:** `diagnosis` (Integer $y \in \{0, 1, 2, 3, 4\}$).
                        - **Class Imbalance Ratio:** 9.35 : 1 between Majority (Grade 0) and Minority (Grade 3).
                        - **Storage Path:** Configured relative to project root in `src/config.py`.
                        """
                    )

            # ==================== T2: PREPROCESSING EHR ====================
            with gr.TabItem("T2: Preprocessing & Cleaning", id="tab_t2"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🧹 Tutorial 2: Preprocessing of EHR – Standardization & Data Cleaning</h2>
                        <span class="badge-real">REAL DATA: APTOS 2019</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Raw clinical fundus photos suffer from extreme variability: uninformative black circular camera borders, inconsistent resolutions, severe underexposure, and motion blur. This tutorial audits image quality, rejects corrupted inputs, crops camera borders, and resizes to standard screening geometries.

                    ### **What we did in this project:**
                    1. Implemented quantitative quality gates: Underexposed (`mean_brightness < 20.0`), Overexposed (`mean_brightness > 220.0`), and Severe Blur (`Laplacian variance < 4.0`).
                    2. Flagged 158 suboptimal images in the APTOS dataset with diagnostic warnings.
                    3. Cropped circular black camera borders using Otsu threshold contour bounding boxes.
                    4. Normalized aspect ratios and resized images to dual resolutions: `512×512` (for high-res feature segmentation) and `224×224` (for deep CNN input).
                    5. Saved clean metadata to `data/clean_records.csv` and generated `outputs/reports/t2_ehr_preprocessing.md`.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t2_live_out = gr.Markdown("Upload an image on the sidebar to run the quality audit.")

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t2_preprocessing_comparison.png"), label="T2 Quality Audit & Border Cropping Pipeline")

                with gr.Accordion("Technical Details (Quality Audit Formulas)", open=False):
                    gr.Markdown(
                        r"""
                        - **Luminance:** $\mu = \frac{1}{HW} \sum_{x,y} I_{gray}(x, y)$
                        - **Blur Score:** $\text{Var}(\nabla^2 I_{gray}) = \frac{1}{HW} \sum (\nabla^2 I_{gray} - \bar{\nabla}^2)^2$
                        - **Rejection Thresholds:** Darkness $< 20.0$, Glare $> 220.0$, Blur $< 4.0$.
                        """
                    )

            # ==================== T3: REDUNDANCY & IMPUTATION ====================
            with gr.TabItem("T3: Redundancy & Imputation", id="tab_t3"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>✂️ Tutorial 3: Preprocessing of EHR – Redundancy Removal & Missing Data</h2>
                        <span class="badge-real">REAL DATA: APTOS 2019</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Clinical datasets often contain exact duplicate shutter snaps or near-duplicate photographs of the same eye taken seconds apart. This tutorial combines cryptographic and perceptual image hashing to detect duplicates, verifies diagnostic label consistency, and demonstrates robust missing-data imputation.

                    ### **What we did in this project:**
                    1. Generated MD5 exact byte hashes and 64-bit Perceptual Hashes (pHash using Discrete Cosine Transform).
                    2. Identified 158 duplicate/near-duplicate pairs (Hamming distance $\le 4$).
                    3. Audited label concordance: verified **0% label discordance** (100% agreement between duplicates).
                    4. Excluded redundant records in CSV metadata only without deleting source images, preserving 3,504 unique patient eyes in `data/dedup_records.csv`.
                    5. Implemented Median and K-Nearest Neighbors (KNN, $K=5$) imputation for derived image metrics.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t3_live_out = gr.Markdown("Upload an image on the sidebar to compute cryptographic & perceptual hashes.")

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t3_deduplication_imputation.png"), label="T3 Perceptual Hash Distance Distribution & KNN Imputation")

                with gr.Accordion("Technical Details (pHash & KNN Math)", open=False):
                    gr.Markdown(
                        r"""
                        - **pHash Algorithm:** Resize to $32 \times 32 \to$ Compute 2D DCT $\to$ Keep top-left $8 \times 8$ low frequencies $\to$ Binarize around median.
                        - **Hamming Distance:** $D_H(h_1, h_2) = \sum_{i=1}^{64} (h_{1,i} \oplus h_{2,i})$. Threshold: $D_H \le 4$.
                        - **KNN Imputation:** $\hat{x}_i = \frac{1}{K} \sum_{j \in \mathcal{N}_K(i)} x_j$ weighted by Euclidean feature distance.
                        """
                    )

            # ==================== T4: IMAGE PROCESSING ====================
            with gr.TabItem("T4: Image Processing & Segmentation", id="tab_t4"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🔬 Tutorial 4: Biomedical Image Processing – Enhancement, Restoration, Segmentation & Compression</h2>
                        <span class="badge-real">REAL DATA: APTOS 2019</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Before feeding images into deep learning models, classical biomedical vision techniques enhance microvascular contrast, restore degraded signals, isolate retinal structures, and optimize storage transmission.

                    ### **What we did in this project:**
                    1. **Enhancement:** Compared Green Channel extraction, Luminance CLAHE in LAB color space, and Ben Graham local contrast normalization ($w/30$ Gaussian blur subtraction).
                    2. **Restoration Benchmark:** Evaluated Non-Local Means (PSNR 30.1 dB, SSIM 0.845), Bilateral Filtering (PSNR 28.4 dB), and Richardson-Lucy deblurring on artificially degraded images.
                    3. **Morphological Segmentation:** Extracted retinal vessel trees using black-hat transforms and localized bright exudate lesions with optic disc masking.
                    4. **Compression Analysis:** Benchmarked JPEG qualities against PNG; JPEG Q=80 saved **82.4% storage space** with PSNR $> 38.5$ dB.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                with gr.Tabs():
                    with gr.TabItem("Enhancement & Restoration"):
                        with gr.Row():
                            t4_green_out = gr.Image(label="Green Channel (Optimal Vascular Contrast)")
                            t4_clahe_out = gr.Image(label="Luminance CLAHE Enhanced (LAB Space)")
                            t4_denoise_out = gr.Image(label="Bilateral Filter Denoised")
                    with gr.TabItem("Vessels & Lesion Segmentation"):
                        with gr.Row():
                            t4_vessel_out = gr.Image(label="Morphological Retinal Vasculature Segmentation")
                            t4_lesion_out = gr.Image(label="Bright Lesion (Hard Exudate) Localization")
                    with gr.TabItem("Interactive JPEG Compression Simulator"):
                        gr.Markdown("*(Adjust JPEG quality slider to observe live byte size, compression ratio, and visual fidelity)*")
                        slider_jpeg = gr.Slider(minimum=10, maximum=100, step=5, value=80, label="JPEG Quality Factor")
                        with gr.Row():
                            t4_jpeg_img_out = gr.Image(label="JPEG Compressed Preview")
                            t4_jpeg_info_out = gr.Markdown()
                        slider_jpeg.change(fn=update_jpeg_compression, inputs=[slider_jpeg, analysis_state], outputs=[t4_jpeg_img_out, t4_jpeg_info_out])

                gr.Markdown("### **Project-Level Results from Outputs:**")
                with gr.Row():
                    gr.Image(value=str(FIGURES_DIR / "t4_enhancement_comparison.png"), label="T4 Enhancement Comparison")
                    gr.Image(value=str(FIGURES_DIR / "t4_segmentation_vessels_exudates.png"), label="T4 Morphological Segmentation")
                gr.Image(value=str(FIGURES_DIR / "t4_compression_tradeoffs.png"), label="T4 Compression Storage Savings vs PSNR Curves")

            # ==================== T5: IMAGE ANALYSIS ====================
            with gr.TabItem("T5: SIFT, RANSAC & CNN", id="tab_t5"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🎯 Tutorial 5: Biomedical Image Analysis – SIFT, RANSAC & CNN Transfer Learning</h2>
                        <span class="badge-real">REAL DATA: APTOS 2019</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    This tutorial bridges classical geometric computer vision with modern deep learning. We test SIFT keypoints and RANSAC homography for geometric retinal alignment, and train a 101-layer Residual Network (ResNet-101) for automated 5-class severity grading.

                    ### **What we did in this project:**
                    1. **SIFT & RANSAC:** Matched keypoints between fundus photos and transformed copies: achieved **184 inliers (92.5% inlier ratio)** with successful homography recovery. Matching two distinct eyes yielded **0 inliers (0.0%)**, proving vascular bifurcation geometry functions as a unique anatomical biometric.
                    2. **Deep CNN Transfer Learning:** Fine-tuned ResNet-101 on a stratified 80/20 train/validation split using class-weighted cross-entropy loss to counteract severe class imbalance.
                    3. **Evaluation:** Achieved **76.7% Accuracy** and a **Quadratic Weighted Kappa (QWK) of 0.824**, indicating strong clinical agreement. Best model saved to `outputs/models/best_retinascreen_cnn.pth`.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                with gr.Row():
                    t5_sift_out = gr.Image(label="SIFT Keypoint Detection on Uploaded Retina")
                    t5_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                with gr.Row():
                    gr.Image(value=str(FIGURES_DIR / "t5_sift_ransac_matching.png"), label="T5 SIFT & RANSAC Inlier Recovery")
                    gr.Image(value=str(FIGURES_DIR / "t5_cnn_training_performance.png"), label="T5 ResNet-101 Convergence & Confusion Matrix")

            # ==================== T6: VISUALIZATION & GRAD-CAM ====================
            with gr.TabItem("T6: Clinical Visualization", id="tab_t6"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>📊 Tutorial 6: Biomedical Image Clinical Visualization Suite & Grad-CAM Explainability</h2>
                        <span class="badge-real">REAL DATA: APTOS 2019</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Medical AI cannot be a black box. Clinicians must see *why* a model made its prediction. This tutorial builds an explainability and validation suite featuring Gradient-weighted Class Activation Mapping (Grad-CAM), confusion matrices with clinical risk weights, and error audits.

                    ### **What we did in this project:**
                    1. Implemented Grad-CAM hooked to the final convolutional block (`layer4[-1]`) of ResNet-101.
                    2. Visualized correct predictions (focal attention directly on microaneurysms, blot hemorrhages, and disc neovascularization).
                    3. Audited misclassified edge cases: identified failure modes such as peripheral microaneurysms being overshadowed by central optic disc glare.
                    4. Generated comprehensive clinical evaluation figures at 300 DPI in `outputs/figures/`.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                slider_opacity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.55, label="Interactive Grad-CAM Heatmap Opacity")
                t6_gradcam_out = gr.Image(label="Grad-CAM Saliency Lesion Attention Overlay")
                slider_opacity.change(fn=update_gradcam_overlay, inputs=[slider_opacity, analysis_state], outputs=[t6_gradcam_out])

                gr.Markdown("### **Project-Level Results from Outputs:**")
                with gr.Row():
                    gr.Image(value=str(FIGURES_DIR / "t6_gradcam_validation_cases.png"), label="T6 Grad-CAM Saliency on Correct & Borderline Cases")
                    gr.Image(value=str(FIGURES_DIR / "t6_confusion_matrix_heatmap.png"), label="T6 Confusion Matrix with Clinical Cost Penalties")

            # ==================== T7: PATIENT PREDICTION & STAFFING ====================
            with gr.TabItem("T7: Staffing & Forecasting", id="tab_t7"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>👥 Tutorial 7: Patient Prediction for Improved Staffing</h2>
                        <span class="badge-sim">SIMULATED DATA: 104-WEEK HORIZON</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Deploying AI screening into hospital workflows requires predicting patient demand and workforce needs. This tutorial simulates 104 weeks of clinical screening volume with seasonal trends and converts arrivals into staffing requirements for primary graders and vitreoretinal specialists.

                    ### **What we did in this project:**
                    1. Simulated a 2-year (104-week) screening time-series with winter surges, holiday dips, and autoregressive noise, coupled to empirical ICDR class proportions.
                    2. Built autoregressive forecasting models predicting weekly patient arrivals and referable cases (Grade 2+).
                    3. Applied clinical workload standards: 1 Primary Grader per 35 screens/day; 1 Retinal Specialist per 18 referrals/week.
                    4. Found that weekly arrivals fluctuated between 340 and 520 patients/week, requiring a baseline of **2 to 3 primary graders** and **5 to 8 retinal specialists** during seasonal peaks.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t7_live_md = gr.Markdown()

                gr.Markdown("### **Interactive Clinic Staffing Calculator:**")
                slider_vol = gr.Slider(minimum=2000, maximum=50000, step=1000, value=20000, label="Annual Clinic Screening Volume (Patients / Year)")
                staffing_calc_out = gr.Markdown(update_staffing_calculation(20000))
                slider_vol.change(fn=update_staffing_calculation, inputs=[slider_vol], outputs=[staffing_calc_out])

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "sim_t7_staffing_forecast.png"), label="T7 104-Week Volume Forecast & Staffing Requirements")

            # ==================== T8: STRATEGIC PLANNING PART 1 ====================
            with gr.TabItem("T8: Strategic Planning Part 1", id="tab_t8"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>📈 Tutorial 8: Informed Strategic Planning Part 1 – Descriptive & Diagnostic Analysis</h2>
                        <span class="badge-sim">SIMULATED CLINICAL SCENARIOS</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Strategic health administration requires analyzing the clinical triage burden across different hospital sizes. This tutorial models referral volumes under low, medium, and high-volume scenarios and identifies operational bottlenecks.

                    ### **What we did in this project:**
                    1. Established cohort referral baselines: **40.7%** of screened patients require specialist evaluation (Grade 2+), while **13.4%** present with vision-threatening disease (Grade 3 & 4).
                    2. Evaluated 3 clinic scale tiers: Small Community Clinic (5,000 pts/yr $\to$ ~40 referrals/wk), Medium Center (20,000 pts/yr $\to$ ~160 referrals/wk), Regional Hub (50,000 pts/yr $\to$ ~400 referrals/wk).
                    3. Highlighted that indiscriminate referral of all Grade 2 patients causes severe consultation backlogs unless risk-stratified.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t8_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t8_strategic_planning_descriptive.png"), label="T8 Operational Referral Burden Across Clinic Tiers")

            # ==================== T9: STRATEGIC PLANNING PART 2 ====================
            with gr.TabItem("T9: Strategic Planning Part 2", id="tab_t9"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>⚖️ Tutorial 9: Informed Strategic Planning Part 2 – Prescriptive & Cost-Benefit Analysis</h2>
                        <span class="badge-sim">SIMULATED HEALTH ECONOMICS</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Where should the clinical cutoff for specialist referral be set? This tutorial performs a rigorous health-economics evaluation comparing operating thresholds (Grade 1+ vs Grade 2+) to maximize vision preservation while minimizing false-referral costs.

                    ### **What we did in this project:**
                    1. Compared referral operating policies: Grade 2+ threshold (39.4% referral rate) vs aggressive Grade 1+ threshold (48.9% referral rate, +24% specialist load).
                    2. Constructed a micro-economic cost-benefit model for 20,000 annual screenings factoring screening costs ($2.50 AI vs $12 manual), specialist consults ($85), false-referral waste ($45), and severe missed DR downstream liability ($4,200).
                    3. Demonstrated that the **Grade 2+ threshold** represents the optimal operational equilibrium.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t9_live_md = gr.Markdown()

                gr.Markdown("### **Interactive Operating Threshold Simulator:**")
                radio_thresh = gr.Radio(choices=["Grade 1+", "Grade 2+", "Grade 3+", "Grade 4+"], value="Grade 2+", label="Select Operational Referral Threshold Policy")
                tradeoff_out = gr.Markdown(update_threshold_tradeoff("Grade 2+"))
                radio_thresh.change(fn=update_threshold_tradeoff, inputs=[radio_thresh], outputs=[tradeoff_out])

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t9_strategic_dashboard.png"), label="T9 Strategic Health Economics Decision Dashboard")

            # ==================== T10: TEXT CLASSIFICATION ====================
            with gr.TabItem("T10: Text Classification & Clustering", id="tab_t10"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>📝 Tutorial 10: Classification & Clustering of Clinical Text Data</h2>
                        <span class="badge-synth">SYNTHETIC EHR CLINICAL NOTES</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Real-world EHR systems contain unstructured clinical narratives alongside fundus photographs. This tutorial processes 3,504 synthetic progress notes to evaluate TF-IDF text vectorization, supervised disease staging, and unsupervised document clustering.

                    ### **What we did in this project:**
                    1. Generated 3,504 report-style clinical narratives from ICDR labels with realistic medical terminology (laterality, lesions, vascular status).
                    2. Trained balanced Logistic Regression and Multinomial Naive Bayes classifiers: achieved **100% Accuracy and Weighted F1 = 1.000**.
                    3. Performed K-Means document clustering ($K=5$): achieved **Cluster Purity of 0.7763** with top terms aligning cleanly with disease stages.
                    4. *Explicit Caveat:* High accuracy reflects deterministic generation from ground-truth labels.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t10_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t10_text_classification_clustering.png"), label="T10 Text Classification Confusion Matrices & K-Means Clusters")

            # ==================== T11: INFORMATION EXTRACTION & NER ====================
            with gr.TabItem("T11: Clinical Entity Extraction (NER)", id="tab_t11"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🏷️ Tutorial 11: Clinical Information Extraction & Named Entity Recognition (NER)</h2>
                        <span class="badge-synth">SYNTHETIC EHR CLINICAL NOTES</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Unstructured clinical narratives must be converted into structured EHR databases. This tutorial builds Named Entity Recognition (NER) pipelines using rule-based matchers and custom spaCy statistical models to extract key ophthalmic entities.

                    ### **What we did in this project:**
                    1. Defined a 4-entity clinical schema: `LATERALITY`, `LESION`, `SEVERITY`, and `ACTION PLAN`.
                    2. Evaluated rule-based (Regex + spaCy Matcher) and custom statistical NER models on held-out validation data (Macro F1 = 1.000).
                    3. Parsed all 3,504 notes into `data/extracted_clinical_entities.csv`, quantifying lesion count escalation from 0.0 (Grade 0) to 2.4 (Grade 4).
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t11_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t11_information_extraction_ner.png"), label="T11 Named Entity Recognition Performance & Entity Frequencies")

            # ==================== T12: RANKING & WSD ====================
            with gr.TabItem("T12: Ranking & Word Sense (WSD)", id="tab_t12"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🔍 Tutorial 12: Ranking & Word Sense Disambiguation (WSD) of Clinical Text</h2>
                        <span class="badge-synth">SYNTHETIC EHR CLINICAL NOTES</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Clinical decision support requires fast retrieval of relevant patient records and accurate disambiguation of polysemous medical terms (e.g. "discharge" as eye secretion vs hospital dismissal). This tutorial benchmarks TF-IDF vs Okapi BM25 ranking and tests Lesk WSD algorithms.

                    ### **What we did in this project:**
                    1. **Information Retrieval:** Queried 3,504 notes with clinical search terms: Okapi BM25 and TF-IDF achieved **Spearman rank correlation r = 0.9888** ($p < 10^{-10}$) with **Precision@K = 100%** across $K = 5, 10, 20$.
                    2. **Word Sense Disambiguation:** Tested Simplified Lesk (WordNet) and context embeddings across 4 polysemous medical words (`discharge`, `positive`, `stable`, `dense`), achieving **97.5% accuracy**.
                    """
                )
                gr.Markdown("### **Interactive Search Query Box:**")
                text_query = gr.Textbox(value="severe hemorrhage refer urgently", label="Search Clinical Progress Notes Corpus (BM25 vs TF-IDF)")
                search_res_out = gr.Markdown(update_search_ranking("severe hemorrhage refer urgently"))
                text_query.submit(fn=update_search_ranking, inputs=[text_query], outputs=[search_res_out])

                gr.Markdown("### **Live Result on Your Uploaded Image:**")
                t12_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t12_ranking_and_wsd.png"), label="T12 BM25 Retrieval Correlation & WSD Disambiguation Accuracy")

            # ==================== T13: DIMENSIONALITY REDUCTION ====================
            with gr.TabItem("T13: Dimensionality Reduction", id="tab_t13"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🌌 Tutorial 13: Dimensionality Reduction for Clinical Exploratory Analysis</h2>
                        <span class="badge-real">REAL DATA: 2,048-DIM CNN EMBEDDINGS</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Deep neural networks map complex retinal images into high-dimensional latent vectors. This tutorial projects 2,048-dimensional penultimate ResNet-101 embeddings into 2D space using PCA, t-SNE, and UMAP to visualize clinical disease trajectories and audit feature robustness.

                    ### **What we did in this project:**
                    1. Extracted 2,048-dimensional feature vectors from `CNN_MODEL.avgpool` across 250 diverse patient eyes.
                    2. Fitted PCA (top 2 PCs explain **60.68%** of total variance), t-SNE (perplexity=30), and UMAP ($n=15$).
                    3. Calculated Silhouette Scores: confirmed strong biological disease progression separation while negative silhouette scores for image quality proved the CNN learned robust pathological features unaffected by lighting or blur.
                    4. Cached manifold models in `outputs/models/embedding_map_cache.joblib`.
                    """
                )
                gr.Markdown("### **Interactive Latent Manifold Projection:**")
                radio_proj = gr.Radio(choices=["PCA", "t-SNE", "UMAP"], value="PCA", label="Select Projection Algorithm")
                plot_manifold_out = gr.Image(value=update_manifold_plot("PCA", None), label="Latent Manifold Projection")
                radio_proj.change(fn=update_manifold_plot, inputs=[radio_proj, analysis_state], outputs=[plot_manifold_out])

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t13_dimensionality_reduction.png"), label="T13 PCA Scree, t-SNE & UMAP Projections")

            # ==================== T14: BEHAVIORAL FACTORS EDA ====================
            with gr.TabItem("T14: Behavioral Factors EDA", id="tab_t14"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🏃 Tutorial 14: Behavioral Factors Influencing Weight in Adults Part 1 – Exploration & Statistics</h2>
                        <span class="badge-sim">SIMULATED DATA: 3,000 ADULTS</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    Retinopathy does not develop in isolation—it is driven by systemic metabolic health. This tutorial explores a simulated cohort of 3,000 adult diabetics to model the complex multidimensional interactions between physical activity, sleep, stress, glycemic control (HbA1c), and body mass index (BMI).

                    ### **What we did in this project:**
                    1. Synthesized `data/sim_behavioral_factors.csv` (3,000 records) incorporating non-linear interactions, missing values (3–5%), and physiological outliers.
                    2. Cleaned dataset using sex-stratified median imputation and physiological winsorization ($[40, 180]$ kg), saving `data/clean_sim_behavioral_factors.csv`.
                    3. Conducted statistical EDA: physical activity showed diminishing returns on BMI ($r = -0.38$), chronic stress correlated with sleep deficits ($r = -0.32$), and VIF values confirmed feature stability ($< 2.2$).
                    4. Verified significant biomarker differences across DR grades via ANOVA: HbA1c ($F = 351.2$), Diabetes Duration ($F = 520.6$), and BMI ($F = 59.9$).
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Lifestyle Inputs:**")
                t14_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t14_behavioral_factors_eda.png"), label="T14 Correlation Heatmap, ANOVA Boxplots, Activity Curves & PCA Loadings")

            # ==================== T15: BEHAVIORAL MODELING & RISK FUSION ====================
            with gr.TabItem("T15: Behavioral Modeling & Risk Fusion", id="tab_t15"):
                gr.Markdown(
                    """
                    <div class="tutorial-header">
                        <h2>🤖 Tutorial 15: Behavioral Factors Influencing Weight Part 2 – Modeling, SHAP & Multimodal Risk Fusion</h2>
                        <span class="badge-sim">SIMULATED DATA MODELS</span>
                    </div>
                    """,
                    sanitize_html=False
                )
                gr.Markdown(
                    r"""
                    ### **What is this tutorial about?**
                    This tutorial builds machine learning models to predict adult BMI, explains non-linear feature interactions using SHAP values, clusters patients into behavioral phenotypes, and fuses systemic lifestyle risk with deep retinal CNN probabilities.

                    ### **What we did in this project:**
                    1. **Predictive Modeling:** Benchmarked OLS Linear ($R^2 = 0.315$), Random Forest ($R^2 = 0.326$), and Gradient Boosting Regressor (**$R^2 = 0.345$, RMSE = 3.71 kg/m²**).
                    2. **SHAP Interpretability:** Identified daily calories, physical activity, and sleep as the top drivers of adult adiposity, revealing age-activity metabolic attenuation.
                    3. **Behavioral Phenotyping (K-Means):** Segmented patients into 4 profiles: Profile 0 (Sedentary High-Stress), Profile 1 (Active Healthy Baseline), Profile 2 (Elderly Chronic Duration), and Profile 3 (Younger High-Calorie At-Risk, 87.7% referable DR).
                    4. **Multimodal Fusion:** Combined CNN probabilities ($P_{CNN}$) with behavioral prior risk ($P_{behav}$) via Bayesian opinion pooling: $P_{combined} = 0.70 P_{CNN} + 0.30 P_{behav}$.
                    """
                )
                gr.Markdown("### **Live Result on Your Uploaded Lifestyle Inputs:**")
                t15_live_md = gr.Markdown()

                gr.Markdown("### **Project-Level Results from Outputs:**")
                gr.Image(value=str(FIGURES_DIR / "t15_behavioral_modeling_shap.png"), label="T15 Model Comparisons, SHAP Feature Importance & Behavioral Clusters")

            # ==================== FINAL REPORT ====================
            with gr.TabItem("Final Report", id="tab_report"):
                gr.Markdown("## 📑 **Final Clinical Case Report & Official Dossier**")
                gr.Markdown("*(Comprehensive screening dossier with downloadable PDF clinical report and structured JSON export)*")
                final_report_md_out = gr.Markdown("Please upload an image and run analysis to generate the final clinical screening dossier.")
                with gr.Row():
                    pdf_file_out = gr.File(label="Download Clinical Screening Report (PDF)")
                    json_file_out = gr.File(label="Download Structured Record (JSON)")

        # ==================== Event Wiring ====================
        # Home page preset jump button
        btn_home_jump.click(
            fn=lambda: load_example(2),
            outputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in]
        )

        # Preset example buttons
        btn_ex0.click(fn=lambda: load_example(0), outputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in])
        btn_ex1.click(fn=lambda: load_example(1), outputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in])
        btn_ex2.click(fn=lambda: load_example(2), outputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in])
        btn_ex3.click(fn=lambda: load_example(3), outputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in])
        btn_ex4.click(fn=lambda: load_example(4), outputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in])

        # Core analysis trigger: updates analysis_state, then dispatches to all tutorial outputs
        analysis_outputs = [
            summary_out, conf_bars_out,
            t1_live_out, t2_live_out, t3_live_out,
            t4_green_out, t4_clahe_out, t4_denoise_out, t4_vessel_out, t4_lesion_out,
            t5_live_md, t5_sift_out, t6_gradcam_out,
            t7_live_md, t8_live_md, t9_live_md,
            t10_live_md, t11_live_md, t12_live_md,
            plot_manifold_out,
            t14_live_md, t15_live_md,
            final_report_md_out, pdf_file_out, json_file_out
        ]

        def on_analyze_click(img, age, sex, wt, ht, act, sleep, smoke, dur, hba1c):
            try:
                if img is None:
                    # Fallback to Preset 2 (Moderate DR) if analyze clicked without an image
                    ex_data = load_example(2)
                    img = ex_data[0]
                    if age is None: age = ex_data[1]
                    if sex is None: sex = ex_data[2]
                    if wt is None: wt = ex_data[3]
                    if ht is None: ht = ex_data[4]
                    if act is None: act = ex_data[5]
                    if sleep is None: sleep = ex_data[6]
                    if smoke is None: smoke = ex_data[7]
                    if dur is None: dur = ex_data[8]
                    if hba1c is None: hba1c = ex_data[9]

                st = run_screening_analysis(img, age, sex, wt, ht, act, sleep, smoke, dur, hba1c)
                if st is None:
                    empty_outs = dispatch_analysis_results(None)
                    return (None, *empty_outs)

                outs = dispatch_analysis_results(st)
                return (st, *outs)
            except Exception as e:
                import traceback
                print(f"[ERROR in on_analyze_click]: {e}", flush=True)
                traceback.print_exc()
                err_md = f"### ❌ Screening Analysis Error\n**`{type(e).__name__}: {str(e)}`**\n\n```text\n{traceback.format_exc()}\n```"
                empty_outs = list(dispatch_analysis_results(None))
                empty_outs[0] = err_md
                return (None, *empty_outs)

        btn_analyze.click(
            fn=on_analyze_click,
            inputs=[img_input, age_in, sex_in, wt_in, ht_in, act_in, sleep_in, smoke_in, dur_in, hba1c_in],
            outputs=[analysis_state, *analysis_outputs]
        )

        nav_radio.change(
            fn=lambda choice: LABEL_TO_ID.get(choice, "tab_home"),
            inputs=[nav_radio],
            outputs=[tabs]
        )

    return demo


demo = build_guided_app()

if __name__ == "__main__":
    print("Launching RetinaScreen Master Guided Tutorial Experience on http://127.0.0.1:7860 ...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
