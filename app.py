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
import uuid
from threading import RLock
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
import matplotlib
matplotlib.use("Agg")
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
from src.tutorial_services import extract_entities, image_identity, registration_analysis, classify_text, search_notes
MODEL_LOCK = RLock()

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
        "color": "#34d399",
        "badge": "NORMAL (No DR)",
        "notes": "Normal retinal microvasculature. No microaneurysms, blot hemorrhages, or hard exudates detected. Recommend annual dilated eye examination and routine glycemic monitoring.",
        "synthetic_note": "OD: Normal digital fundus examination. Clear media, sharp optic disc margins, normal arteriovenous ratio. Healthy macula without exudative deposits. Action: Routine annual surveillance."
    },
    1: {
        "status": "Mild Nonproliferative DR",
        "action": "Primary Care Surveillance & Glycemic Optimization",
        "timeframe": "6 - 9 months",
        "urgency": "Low Concern",
        "color": "#fbbf24",
        "badge": "MILD NPDR",
        "notes": "Isolated microaneurysms detected in the posterior pole. No hard exudates, venous beading, or macular edema. Optimize HbA1c and repeat fundus photography in 6-9 months.",
        "synthetic_note": "OD: Mild nonproliferative diabetic retinopathy. Scattered microaneurysms noted in posterior pole. No hard lipid exudates or hemorrhages. Recommendation: 6-month surveillance and improved glycemic management."
    },
    2: {
        "status": "Moderate Nonproliferative DR",
        "action": "Referral to General Ophthalmology Clinic",
        "timeframe": "4 - 6 weeks",
        "urgency": "Moderate (Referral Indicated)",
        "color": "#fb923c",
        "badge": "MODERATE NPDR (Referable)",
        "notes": "Multiple blot hemorrhages and hard lipid exudates approaching superior temporal arcade. Cotton-wool spots observed. Non-urgent referral for ophthalmic slit-lamp exam and macular OCT.",
        "synthetic_note": "OD: Moderate nonproliferative diabetic retinopathy. Multiple dot-blot hemorrhages and hard exudates in superior temporal arcade. Cotton wool spots observed. Recommendation: Refer to ophthalmology within 4-6 weeks."
    },
    3: {
        "status": "Severe Nonproliferative DR",
        "action": "Fast-Track Specialist Referral (Vitreoretinal)",
        "timeframe": "1 - 2 weeks",
        "urgency": "High (Urgent Referral)",
        "color": "#f87171",
        "badge": "SEVERE NPDR (Urgent)",
        "notes": "Extensive 4-quadrant intraretinal hemorrhages with prominent venous beading and IRMA (4-2-1 rule met). High progression risk. Urgent evaluation for panretinal photocoagulation.",
        "synthetic_note": "OD: Severe nonproliferative diabetic retinopathy. Extensive four-quadrant hemorrhages with prominent venous beading and IRMA. High risk for rapid neovascular progression. Recommendation: Urgent vitreoretinal referral within 1-2 weeks."
    },
    4: {
        "status": "Proliferative Diabetic Retinopathy",
        "action": "Emergency Vitreoretinal Consultation",
        "timeframe": "Within 24 - 48 hours",
        "urgency": "Emergency (Imminent Vision Threat)",
        "color": "#c084fc",
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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained CNN checkpoint missing: {MODEL_PATH}")
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
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
        alerts.append("Warning: Very dark / underexposed (< 20.0)")
    elif mean_bright > 220.0:
        alerts.append("Warning: Overexposed / glare artifact (> 220.0)")
    if blur_score < 4.0:
        alerts.append("Warning: Severe blur / motion artifact (< 4.0)")

    passed = len(alerts) == 0
    if passed:
        status_msg = "Passed: Adequate Diagnostic Quality (Passed T2 Quality Gate)"
        badge_color = "#34d399"
    else:
        status_msg = " | ".join(alerts) + " (Diagnostic confidence may be impaired)"
        badge_color = "#f87171"

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
    progress=gr.Progress(track_tqdm=True),
    daily_calories=2100.0, stress_score=5.0, alcohol_drinks_week=2.0,
    image_is_preprocessed=False
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

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3 or min(img_rgb.shape[:2]) < 16:
        raise ValueError("Use a valid retinal image at least 16 pixels wide and high.")
    if not np.isfinite(img_rgb).all():
        raise ValueError("Image contains non-finite values.")
    img_rgb = img_rgb.astype(np.uint8)
    for label, value, low, high in [("Age", age, 18, 110), ("Weight", weight, 20, 350), ("Height", height, 100, 230), ("Activity", activity, 0, 3000), ("Sleep", sleep, 1, 16), ("Duration", duration, 0, 90), ("HbA1c", hba1c, 3, 20), ("Calories", daily_calories, 500, 6000), ("Stress", stress_score, 0, 10), ("Alcohol", alcohol_drinks_week, 0, 100)]:
        if value is not None and (not np.isfinite(float(value)) or not low <= float(value) <= high):
            raise ValueError(f"{label} must be between {low} and {high}.")

    h, w, c = img_rgb.shape
    aspect_ratio = round(w / float(h), 3)
    md5_digest, phash_str, duplicate_match, compared_images = image_identity(img_rgb)
    mean_bright, blur_score, quality_status, quality_badge_color, passed_quality = audit_image_quality(img_rgb)

    progress(0.25, desc="Computing biomedical enhancements & segmentations (T4)...")
    green_channel = cv2.cvtColor(img_rgb[:, :, 1], cv2.COLOR_GRAY2RGB)
    enhanced_clahe = enhance_clahe_fundus(img_rgb)
    denoised_img = denoise_fundus(enhanced_clahe)
    vessel_img = segment_vessels(img_rgb)
    lesion_img = segment_lesions(img_rgb)

    progress(0.40, desc="Extracting SIFT keypoints & computing CNN inference (T5)...")
    sift_img, num_kps = compute_sift_keypoints(img_rgb)

    from src.t2_preprocess_ehr import ben_graham_normalize
    if image_is_preprocessed:
        normalized_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        normalized_512 = ben_graham_normalize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), target_size=512)
        normalized_rgb = cv2.cvtColor(cv2.resize(normalized_512, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    registration_data = registration_analysis(normalized_rgb)
    input_tensor = TRANSFORM_224(normalized_rgb).unsqueeze(0)
    with MODEL_LOCK:
        cam, pred_class, probs = GRAD_CAM.generate_cam(input_tensor)

    # Grad-CAM overlay at native resolution
    resized_rgb = cv2.resize(normalized_rgb, (224, 224))
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
    umap_coord = None

    if EMB_CACHE is not None:
        emb_holder = []
        def h_fn(m, i, o): emb_holder.append(o.squeeze().detach().cpu().numpy())
        h_hook = CNN_MODEL.avgpool.register_forward_hook(h_fn)
        with torch.no_grad():
            with MODEL_LOCK:
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
                umap_coord = None
        # For t-SNE, use nearest neighbor approximation for projection
        nn_idx = sims.argmax()
        tsne_coord = EMB_CACHE["tsne_2d"][nn_idx].tolist()

    progress(0.75, desc="Assembling clinical EHR notes & NER extraction (T10-T12)...")
    synth_note_text = rule["synthetic_note"]
    entities = extract_entities(synth_note_text)

    progress(0.85, desc="Computing behavioral modeling & multimodal risk fusion (T14-T15)...")
    has_lifestyle = all(v is not None for v in [age, sex, weight, height, activity, sleep, smoking, duration, hba1c])
    actual_bmi = round(float(weight) / ((float(height) / 100.0) ** 2), 2) if (weight and height) else None
    pred_bmi = actual_bmi
    cluster_id = 0
    cluster_name, cluster_desc = CLUSTER_DESCRIPTIONS[0]
    behav_probs = np.array([0.49, 0.10, 0.27, 0.05, 0.08])
    multimodal_score = 0.0

    if has_lifestyle:
        if any(model is None for model in (BMI_MODEL, CLUST_MODEL, CLUST_SCALER, DR_BEHAV_CLF)):
            raise FileNotFoundError("Behavioral experiment requires the saved BMI, clustering, scaler and risk model artifacts.")
        input_row = pd.DataFrame([{
            "age": float(age),
            "is_male": 1.0 if sex == "Male" else 0.0,
            "daily_calories": float(daily_calories),
            "physical_activity_min_week": float(activity),
            "sleep_hours": float(sleep),
            "stress_score": float(stress_score),
            "alcohol_drinks_week": float(alcohol_drinks_week),
            "diabetes_duration_years": float(duration),
            "hba1c": float(hba1c),
            "smoking_status_Former": 1.0 if smoking == "Former" else 0.0,
            "smoking_status_Never": 1.0 if smoking == "Never" else 0.0,
        }])

        if BMI_MODEL is not None:
            pred_bmi = round(float(BMI_MODEL.predict(input_row)[0]), 2)

        if CLUST_MODEL is not None and CLUST_SCALER is not None:
            c_input = pd.DataFrame([[float(daily_calories), float(activity), float(sleep), float(stress_score), float(duration), float(hba1c)]], columns=["daily_calories", "physical_activity_min_week", "sleep_hours", "stress_score", "diabetes_duration_years", "hba1c"])
            c_scaled = CLUST_SCALER.transform(c_input)
            cluster_id = int(CLUST_MODEL.predict(c_scaled)[0])

        cluster_name, cluster_desc = CLUSTER_DESCRIPTIONS[cluster_id]

        if DR_BEHAV_CLF is not None:
            behav_probs = DR_BEHAV_CLF.predict_proba(input_row)[0]

        # Multimodal Combined Score (0.70 CNN + 0.30 Behavioral)
        combined_probs = 0.70 * probs + 0.30 * behav_probs
        multimodal_score = float(np.sum(np.arange(5) * combined_probs) * 25.0)

    progress(0.95, desc="Compiling structured report (PDF & JSON)...")
    report_id = f"RS-{uuid.uuid4().hex[:12]}"
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
        "behavioral_cluster": cluster_name if has_lifestyle else None,
        "behavioral_inputs": input_row.iloc[0].to_dict() if has_lifestyle else None,
        "model_checkpoint": MODEL_PATH.name,
        "input_already_preprocessed": bool(image_is_preprocessed),
        "synthetic_note": synth_note_text,
        "entity_method": "regex and spaCy Matcher (not a persisted trained NER model)",
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
        "normalized_rgb": normalized_rgb,
        "age": age,
        "md5_digest": md5_digest,
        "duplicate_match": duplicate_match,
        "compared_images": compared_images,
        "registration": registration_data["summary"],
        "registration_data": registration_data,
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
        [Paragraph("<b>Deep Learning Confidence:</b>", body_style), Paragraph(f"{p_data['confidence_pct']:.1f}% {'(Low Confidence Borderline Case)' if p_data.get('is_borderline') else ''}", body_style)],
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
        [Paragraph("<b>CLAHE Enhanced Fundus</b>", body_style), Paragraph("<b>Grad-CAM Model Attention Overlay</b>", body_style)],
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
    img_rgb = state["normalized_rgb"]
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
    fig.patch.set_facecolor('#f8fbfc')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#698095')
    ax.xaxis.label.set_color('#486075')
    ax.yaxis.label.set_color('#486075')
    ax.title.set_color('#18354b')
    for spine in ax.spines.values():
        spine.set_color('#dce6ed')

    if EMB_CACHE is None:
        ax.text(0.5, 0.5, "Embedding cache not available", ha="center", color='#698095')
        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        return rgb

    palette_g = ["#38bdf8", "#34d399", "#fbbf24", "#fb923c", "#c084fc"]
    key_coords = "coords_2d" if method == "PCA" else ("tsne_2d" if method == "t-SNE" else "umap_2d")
    coords = EMB_CACHE[key_coords]

    for g in range(5):
        mask = EMB_CACHE["labels"] == g
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=palette_g[g], label=f"Grade {g}: {CLASS_LABELS[g]}",
            alpha=0.75, s=26, edgecolors="none"
        )

    # Plot uploaded patient image if analyzed
    if state and method != "t-SNE" and state.get("pca_coord") is not None and (method != "UMAP" or state.get("umap_coord") is not None):
        user_c = state["pca_coord"] if method == "PCA" else (state["tsne_coord"] if method == "t-SNE" else state["umap_coord"])
        ax.scatter(
            user_c[0], user_c[1],
            c="#f43f5e", marker="o", s=220, edgecolors="#ffffff", linewidths=2.0,
            label="Uploaded image", zorder=10
        )

    ax.set_title(f"CNN embeddings ({method})" + (" - reference cohort only" if method == "t-SNE" else ""), fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{method} Dimension 1")
    ax.set_ylabel(f"{method} Dimension 2")
    ax.legend(loc="upper right", fontsize=8, facecolor='#ffffff', edgecolor='#dce6ed', labelcolor='#486075')
    ax.grid(True, linestyle="--", alpha=0.25, color="#b8c9d5")
    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return rgb


def update_threshold_tradeoff(threshold_str):
    cutoff = int(threshold_str.split()[1].replace('+', ''))
    cohort = pd.read_csv(DATA_DIR / "dedup_records.csv")
    share = float((cohort.diagnosis >= cutoff).mean())
    referrals = round(20000 * share)
    return (f"### Simulated policy: {threshold_str}\n\n"
            f"Cohort share: **{share:.1%}**. At 20,000 screenings/year: **{referrals:,} referrals**.\n\n"
            f"Consultation budget at $85/referral: **${referrals * 85:,}**. "
            "Screening cost: $240,000 manual ($12 each) or $50,000 AI-assisted ($2.50 each).\n\n"
            "These are scenario costs using observed label proportions, not measured threshold sensitivity or specificity. "
            "Patient-level held-out predictions are needed to estimate false referrals and missed cases.")


def update_staffing_calculation(annual_volume):
    weekly = float(annual_volume) / 52
    share = float((pd.read_csv(DATA_DIR / "dedup_records.csv").diagnosis >= 2).mean())
    return (f"### Simulated workforce requirements\n\nWeekly screenings: **{weekly:,.0f}**; "
            f"expected referrals: **{weekly * share:,.0f}** ({share:.1%} cohort share).\n\n"
            f"Graders: **{int(np.ceil(weekly / 175))}** at 35 reads/day, 5 days/week.\n\n"
            f"Specialists: **{int(np.ceil(weekly * share / 75))}** at 15 consultations/day, 5 days/week (75/week). "
            "Capacities match the T7 forecasting pipeline.")


def update_search_ranking(query_text):
    result = search_notes(query_text)
    return result.to_markdown(index=False) if not result.empty else "No matching notes."


# ==================== Main UI Dispatch Function ====================
def dispatch_analysis_results(state):
    if not state:
        message = "Please upload a retinal image and run screening."
        return (message, {}, message, message, message, None, None, None, None, None,
                message, None, None, message, message, message, message, message,
                message, None, message, message, message, None, None)
    s = state
    g = s["pred_class"]
    from src.presentation import render_screening_summary
    summary = render_screening_summary(s)
    nearest = s['duplicate_match']
    identity = (f"Pixel MD5: `{s['md5_digest']}`\n\npHash: `{s['phash_str']}`\n\n"
                + (f"Closest available processed image: `{nearest[1]}`, Hamming distance **{nearest[0]}**, exact decoded-pixel match: **{nearest[2]}**. " if nearest else "No comparison images available. ")
                + f"Compared {s['compared_images']} locally available processed images. No full-cohort uniqueness claim is made.")
    t1 = f"Uploaded image: **{s['width']} × {s['height']}**, RGB. Aspect ratio: **{s['aspect_ratio']}**. Laterality and patient identity are unknown; an upload is not automatically linked to an EHR record."
    t2 = f"Brightness: **{s['mean_bright']:.1f}**; Laplacian variance: **{s['blur_score']:.1f}**. {s['quality_status']}\n\nCNN input follows T2 crop / Ben Graham normalization and 224 × 224 resizing, unless the input is explicitly marked already processed."
    t5 = f"SIFT keypoints: **{s['num_kps']}**. {s['registration']}\n\nTrained ResNet-101 prediction: **Grade {g}**."
    t7 = "Historical simulated demand and held-out forecasts are shown below. Staffing is a population calculation, independent of this image's predicted grade."
    t8 = "Descriptive cohort distributions and clinic-volume scenarios are shown below. These describe the dataset, not a patient's population percentile."
    t9 = update_threshold_tradeoff('Grade 2+')
    t10 = f"Synthetic demonstration note (generated from the image grade; not a real patient record):\n\n{s['synth_note_text']}\n\n" + classify_text(s['synth_note_text'])
    entities = pd.DataFrame(s['entities'])
    t11 = (entities.to_markdown(index=False) if not entities.empty else "No entities matched.") + "\n\nRule-based regex / spaCy Matcher spans from the synthetic note. Mentions may be negated; they are not confirmed clinical findings."
    t12 = "Search the complete synthetic note corpus with BM25 or TF-IDF and compare word senses using the controls below."
    t14 = f"Age: **{s['age']}** · Measured BMI: **{s['actual_bmi']:.1f} kg/m²**. Cohort statistics below describe simulated adults." if s['has_lifestyle'] else "Enter optional lifestyle data to compare BMI."
    t15 = (f"Predicted BMI: **{s['pred_bmi']:.1f} kg/m²**; measured BMI: **{s['actual_bmi']:.1f} kg/m²**.\n\n"
           f"{s['cluster_name']}: {s['cluster_desc']}\n\nExperimental combined score: **{s['multimodal_score']:.1f}/100** (70% image / 30% simulated behavioral model). This score does not change the image grade or referral rule.") if s['has_lifestyle'] else "Lifestyle modeling not requested."
    return (summary, {f"Grade {i}: {CLASS_LABELS[i]}": float(s['probs'][i]) for i in range(5)},
            t1, t2, identity, s['green_channel'], s['enhanced_clahe'], s['denoised_img'], s['vessel_img'], s['lesion_img'],
            t5, s['sift_img'], s['raw_cam_overlay'], t7, t8, t9, t10, t11, t12, update_manifold_plot('PCA', s),
            t14, t15, summary, s['pdf_path'], s['json_path'])


def build_guided_app():
    from src.workspace_ui import build_workspace
    return build_workspace(sys.modules[__name__])


demo, dark_theme, custom_css, js_dark = build_guided_app()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
        share=False, theme=dark_theme, css=custom_css)
