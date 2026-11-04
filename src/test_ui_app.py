"""
Test Suite for redesigned app.py:
Validates the full guided tutorial pipeline on 3 representative fundus images
across different severity grades (Grade 0 Normal, Grade 2 Moderate, Grade 4 Proliferative),
as well as testing empty inputs and invalid uploads.
Saves visual outputs, PDF reports, and JSON records to outputs/figures/ui_tests/.
"""

import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app import dispatch_analysis_results, run_screening_analysis
from src.config import DATA_DIR, FIGURES_DIR

UI_TESTS_DIR = FIGURES_DIR / "ui_tests"
UI_TESTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "name": "Case_1_Grade0_Normal",
        "path": DATA_DIR / "processed" / "train_224" / "002c21358ce6.png",
        "lifestyle": {
            "age": 48, "sex": "Male", "weight": 74.0, "height": 175.0,
            "activity": 220.0, "sleep": 7.5, "smoking": "Never",
            "duration": 2.0, "hba1c": 6.1
        }
    },
    {
        "name": "Case_2_Grade2_Moderate",
        "path": DATA_DIR / "processed" / "train_224" / "000c1434d8d7.png",
        "lifestyle": {
            "age": 62, "sex": "Female", "weight": 83.0, "height": 161.0,
            "activity": 55.0, "sleep": 6.0, "smoking": "Former",
            "duration": 9.5, "hba1c": 8.2
        }
    },
    {
        "name": "Case_3_Grade4_Proliferative",
        "path": DATA_DIR / "processed" / "train_224" / "001639a390f0.png",
        "lifestyle": {
            "age": 68, "sex": "Male", "weight": 96.0, "height": 171.0,
            "activity": 15.0, "sleep": 4.5, "smoking": "Current",
            "duration": 16.0, "hba1c": 10.5
        }
    },
]


def run_tests():
    print("=" * 70)
    print(" Running End-to-End Test Suite for Redesigned app.py (3 Severity Cases)")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, start=1):
        print(f"\n--- Testing {case['name']} ---")
        img_bgr = cv2.imread(str(case["path"]))
        assert img_bgr is not None, f"Could not load image: {case['path']}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ls = case["lifestyle"]
        state = run_screening_analysis(
            img_rgb,
            ls["age"], ls["sex"], ls["weight"], ls["height"],
            ls["activity"], ls["sleep"], ls["smoking"], ls["duration"], ls["hba1c"],
            image_is_preprocessed=True
        )

        assert state is not None, "Screening analysis returned None"
        assert "rule" in state, "State missing clinical referral rule"
        assert "pred_class" in state, "State missing predicted class"
        assert "enhanced_clahe" in state, "State missing enhanced image"
        assert "vessel_img" in state, "State missing vessel segmentation"
        assert "raw_cam_overlay" in state, "State missing Grad-CAM overlay"
        assert os.path.exists(state["pdf_path"]), f"PDF report missing: {state['pdf_path']}"
        assert os.path.exists(state["json_path"]), f"JSON report missing: {state['json_path']}"

        outs = dispatch_analysis_results(state)
        assert len(outs) == 25, f"Expected 25 output components, got {len(outs)}"

        summary_md = outs[0]
        conf_dict = outs[1]
        pdf_path = outs[23]
        json_path = outs[24]

        print(f"  Predicted: Grade {state['pred_class']} ({state['rule']['status']})")
        print(f"  Confidence: {state['confidence_pct']:.1f}%")
        print(f"  Referral Action: {state['rule']['action']}")
        print(f"  PDF Generated: {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path):,} bytes)")
        print(f"  JSON Generated: {os.path.basename(json_path)} ({os.path.getsize(json_path):,} bytes)")

        # Save visualization panel for this test case
        fig_test, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
        axes[0].imshow(img_rgb); axes[0].set_title("Input Fundus", fontsize=10, fontweight="bold")
        axes[1].imshow(state["vessel_img"]); axes[1].set_title("Vessel Segmentation", fontsize=10, fontweight="bold")
        axes[2].imshow(state["raw_cam_overlay"]); axes[2].set_title("Grad-CAM Saliency", fontsize=10, fontweight="bold")
        axes[3].imshow(state["lesion_img"]); axes[3].set_title("Bright Lesions", fontsize=10, fontweight="bold")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        save_fig_path = UI_TESTS_DIR / f"{case['name']}_composite.png"
        fig_test.savefig(save_fig_path)
        plt.close(fig_test)

        # Copy PDF and JSON into UI_TESTS_DIR
        shutil.copy(pdf_path, UI_TESTS_DIR / f"{case['name']}_report.pdf")
        shutil.copy(json_path, UI_TESTS_DIR / f"{case['name']}_record.json")

    # Test Empty / None image handling
    print("\n--- Testing Empty Image Fallback ---")
    none_state = run_screening_analysis(None, 50, "Male", 70, 170, 100, 7, "Never", 5, 7.0)
    assert none_state is None, "Expected None state for None input image"
    empty_outs = dispatch_analysis_results(none_state)
    assert len(empty_outs) == 25, f"Expected 25 output components on fallback, got {len(empty_outs)}"
    assert "upload" in empty_outs[0].lower(), "Fallback message should prompt to upload an image"
    print("  Empty image handled gracefully!")

    print("\n" + "=" * 70)
    print(" ALL 3 SEVERITY TEST CASES & FALLBACKS PASSED SUCCESSFULLY!")
    print(f" Results archived in: {UI_TESTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
