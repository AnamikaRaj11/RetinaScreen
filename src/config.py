"""
Configuration module for RetinaScreen.
Defines project-relative paths (using pathlib) and global parameters.
All later scripts and notebooks must import paths from this file.
"""

from pathlib import Path

# Project root directory: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths (APTOS 2019 Blindness Detection)
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"
TEST_IMG_DIR = DATA_DIR / "test_images"

# Code & Notebooks paths
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Global Random Seed for reproducibility
RANDOM_SEED = 42

# Clinical Diabetic Retinopathy Grading Scale (ICDR)
CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# Ensure all output directories exist
for directory in (FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
