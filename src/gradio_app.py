"""
RetinaScreen: AI-Assisted Diabetic Retinopathy Screening & Clinical Referral Planning
Interactive Gradio Application.

Features:
1. Retinal Image Upload
2. Automated Preprocessing & Contrast Enhancement (CLAHE / Ben Graham)
3. Morphological Retinal Vasculature Segmentation
4. CNN Severity Classification (ResNet-101 Transfer Learning, Grades 0-4)
5. Grad-CAM Explainable AI Lesion Saliency Heatmap Overlay
6. Clinical Referral & Triage Recommendation
"""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASS_LABELS, DATA_DIR, MODELS_DIR

MODEL_PATH = MODELS_DIR / "best_retinascreen_cnn.pth"
PROCESSED_DIR = DATA_DIR / "processed" / "train_224"

# ==================== Clinical Knowledge Base ====================
REFERRAL_RULES = {
    0: {
        "status": "No Diabetic Retinopathy (ICDR Grade 0)",
        "action": "Routine Annual Surveillance",
        "timeframe": "12 months",
        "urgency": "Normal",
        "color": "#2e7d32",
        "notes": "Retinal vasculature and macula appear normal. Patient counseled on glycemic control. Routine annual diabetic eye screening recommended."
    },
    1: {
        "status": "Mild Nonproliferative DR (ICDR Grade 1)",
        "action": "Primary Care Follow-Up & Glycemic Optimization",
        "timeframe": "6 - 9 months",
        "urgency": "Low",
        "color": "#fbc02d",
        "notes": "Scattered microaneurysms detected without hard exudates or macular edema. Optimize HbA1c and monitor with repeat photography in 6 months."
    },
    2: {
        "status": "Moderate Nonproliferative DR (ICDR Grade 2)",
        "action": "Referral to Comprehensive Ophthalmology",
        "timeframe": "4 - 6 weeks",
        "urgency": "Medium (Referral Indicated)",
        "color": "#f57c00",
        "notes": "Multiple blot hemorrhages, hard lipid exudates, or cotton wool spots observed. Comprehensive slit-lamp biomicroscopy and macular OCT indicated."
    },
    3: {
        "status": "Severe Nonproliferative DR (ICDR Grade 3)",
        "action": "Urgent Referral to Vitreoretinal Specialist",
        "timeframe": "1 - 2 weeks",
        "urgency": "High (Sight-Threatening)",
        "color": "#d32f2f",
        "notes": "Widespread four-quadrant hemorrhages, venous beading, or prominent IRMA. High risk for rapid progression to neovascularization. Urgent specialist review."
    },
    4: {
        "status": "Proliferative Diabetic Retinopathy (ICDR Grade 4)",
        "action": "Emergency Vitreoretinal Intervention",
        "timeframe": "Within 24 - 48 hours",
        "urgency": "Emergency (Imminent Vision Loss)",
        "color": "#7b1fa2",
        "notes": "Active neovascularization (NVD/NVE), preretinal/vitreous hemorrhage, or fibrovascular proliferation. Urgent panretinal photocoagulation or anti-VEGF injection required."
    },
}

# ==================== Model & Grad-CAM Loader ====================
def load_model():
    net = models.resnet101(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 5)
    )
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        net.load_state_dict(state_dict)
    net.eval()
    return net

MODEL = load_model()


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

GRAD_CAM = GradCAM(MODEL, MODEL.layer4[-1])

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==================== Image Processing Functions ====================
def enhance_fundus(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE Luminance enhancement in LAB color space."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)


def segment_retinal_vessels(img_rgb: np.ndarray) -> np.ndarray:
    """Classical morphological vessel segmentation with green channel extraction."""
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    smoothed = cv2.medianBlur(blackhat, 3)
    vessels = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
    )
    h, w = img_rgb.shape[:2]
    circ_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ_mask, (w // 2, h // 2), int(min(h, w) * 0.48), 255, -1)
    vessels = cv2.bitwise_and(vessels, circ_mask)

    # Return RGB vessel map overlay
    vessel_overlay = img_rgb.copy()
    vessel_overlay[vessels > 0] = [0, 255, 200]
    return vessel_overlay


# ==================== Inference Pipeline ====================
def analyze_retinal_image(input_img):
    if input_img is None:
        return None, None, None, {}, "Please upload a retinal fundus image."

    # Convert to RGB numpy array
    if isinstance(input_img, Image.Image):
        img_rgb = np.array(input_img.convert("RGB"))
    else:
        img_rgb = input_img

    # 1. Image Enhancement
    enhanced = enhance_fundus(img_rgb)

    # 2. Vessel Segmentation
    vessel_overlay = segment_retinal_vessels(img_rgb)

    # 3. Model Inference & Grad-CAM
    input_tensor = IMAGE_TRANSFORM(img_rgb).unsqueeze(0)
    cam, pred_class, probs = GRAD_CAM.generate_cam(input_tensor)

    # Grad-CAM heatmap overlay
    resized_rgb = cv2.resize(img_rgb, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    gradcam_overlay = np.uint8(0.55 * heatmap + 0.45 * resized_rgb)

    # 4. Probabilities dictionary
    prob_dict = {
        f"Grade {g}: {CLASS_LABELS[g]}": float(probs[g])
        for g in range(5)
    }

    # 5. Referral & Triage Recommendation
    rule = REFERRAL_RULES[pred_class]
    conf_pct = float(probs[pred_class]) * 100

    triage_md = f"""
### **RetinaScreen Automated Clinical Triage Report**

- **Predicted Severity:** <span style="color: {rule['color']}; font-weight: bold; font-size: 1.15em;">{rule['status']}</span>
- **Model Confidence:** **{conf_pct:.1f}%**
- **Clinical Triage Urgency:** **<span style="color: {rule['color']}">{rule['urgency']}</span>**
- **Recommended Action:** **{rule['action']}**
- **Target Referral Window:** **{rule['timeframe']}**

---
#### **Clinical Recommendation:**
> {rule['notes']}

*(Disclaimer: RetinaScreen is an AI screening and triage decision-support tool. It does not replace definitive examination by a licensed ophthalmologist or optometrist.)*
"""

    return enhanced, vessel_overlay, gradcam_overlay, prob_dict, triage_md


# ==================== Gradio UI ====================
def build_app():
    # Find example images
    example_files = []
    if PROCESSED_DIR.exists():
        example_files = [str(p) for p in list(PROCESSED_DIR.glob("*.png"))[:4]]

    with gr.Blocks(title="RetinaScreen: AI Diabetic Retinopathy Screening") as demo:
        gr.Markdown(
            """
            # RetinaScreen: AI-Assisted Diabetic Retinopathy Screening & Referral Planning
            ### Deep Learning Triaging from Digital Retinal Fundus Photography
            Upload a retinal fundus image to view **image enhancement**, **retinal vasculature segmentation**, **deep learning grading**, **Grad-CAM explainable saliency**, and **referral recommendations**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Retinal Fundus Image", type="numpy")
                btn_run = gr.Button("Run Screening Analysis", variant="primary")
                
                if example_files:
                    gr.Examples(
                        examples=example_files,
                        inputs=input_image,
                        label="Example Retinal Fundus Images",
                    )

            with gr.Column(scale=2):
                with gr.Tab("Clinical Summary & Referral Recommendation"):
                    triage_output = gr.Markdown(value="*Upload an image and click 'Run Screening Analysis' to see recommendations.*")
                    confidence_output = gr.Label(label="Severity Grade Probabilities (ICDR Scale 0 - 4)")

                with gr.Tab("Explainable AI & Biomedical Processing"):
                    with gr.Row():
                        enhanced_output = gr.Image(label="Luminance-Preserving CLAHE Enhancement")
                        vessel_output = gr.Image(label="Retinal Vessel Segmentation Overlay")
                        gradcam_output = gr.Image(label="Grad-CAM Lesion Attention Map")

        btn_run.click(
            fn=analyze_retinal_image,
            inputs=[input_image],
            outputs=[enhanced_output, vessel_output, gradcam_output, confidence_output, triage_output],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    print("Launching RetinaScreen Gradio Demonstration Application...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
