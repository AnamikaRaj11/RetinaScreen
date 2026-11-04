"""
Dataset Verification Script for RetinaScreen:
AI-Assisted Diabetic Retinopathy Screening and Referral Planning.

Verifies:
1. train.csv column schema ('id_code', 'diagnosis').
2. Image file extensions used in dataset folders.
3. 1-to-1 matching between train.csv id_code entries and train_images files (missing/extra files).
4. Image counts in both train_images/ and test_images/.
5. Integrity check by opening a sample of images with PIL/OpenCV to detect unreadable files.
6. Class distribution across diagnosis severity grades (0 to 4).
7. Validation note: test.csv is unlabelled, so all training & validation must use train.csv.
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image

# Import project-relative paths from src/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_IMG_DIR,
    TEST_IMG_DIR,
    CLASS_LABELS,
    RANDOM_SEED,
)


def verify_dataset() -> bool:
    print("=" * 70)
    print(" RetinaScreen - APTOS 2019 Blindness Detection Dataset Verification")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Confirm train.csv schema
    # ---------------------------------------------------------
    if not TRAIN_CSV.exists():
        print(f"[FAIL] train.csv not found at: {TRAIN_CSV}", file=sys.stderr)
        return False

    train_df = pd.read_csv(TRAIN_CSV)
    required_cols = {"id_code", "diagnosis"}
    found_cols = set(train_df.columns)

    print("\n[1] Verifying train.csv Schema:")
    print(f"    - Path: {TRAIN_CSV}")
    print(f"    - Total records: {len(train_df):,}")
    print(f"    - Columns present: {list(train_df.columns)}")

    if not required_cols.issubset(found_cols):
        print(f"    [FAIL] Missing required columns! Expected: {required_cols}, Found: {found_cols}", file=sys.stderr)
        return False
    print("    [PASS] Required columns ('id_code', 'diagnosis') confirmed.")

    # ---------------------------------------------------------
    # 2. Count images & detect file extensions
    # ---------------------------------------------------------
    print("\n[2] Detecting Image Folders & File Extensions:")
    if not TRAIN_IMG_DIR.exists():
        print(f"    [FAIL] train_images folder not found at: {TRAIN_IMG_DIR}", file=sys.stderr)
        return False

    train_files = [f for f in TRAIN_IMG_DIR.iterdir() if f.is_file()]
    test_files = [f for f in TEST_IMG_DIR.iterdir() if f.is_file()] if TEST_IMG_DIR.exists() else []

    train_exts = {f.suffix.lower() for f in train_files}
    test_exts = {f.suffix.lower() for f in test_files}

    detected_ext = list(train_exts)[0] if len(train_exts) == 1 else str(train_exts)
    print(f"    - train_images count: {len(train_files):,} files")
    print(f"    - test_images count : {len(test_files):,} files")
    print(f"    - Detected train image extension(s): {train_exts} (primary: '{detected_ext}')")
    print(f"    - Detected test image extension(s) : {test_exts}")

    # ---------------------------------------------------------
    # 3. Check every id_code has matching image in train_images
    # ---------------------------------------------------------
    print("\n[3] Checking ID Code to Image File Matching:")
    train_file_stems = {f.stem: f for f in train_files}
    csv_ids = set(train_df["id_code"].astype(str))

    missing_in_images = csv_ids - set(train_file_stems.keys())
    extra_in_images = set(train_file_stems.keys()) - csv_ids

    if missing_in_images:
        print(f"    [WARNING] {len(missing_in_images)} id_code(s) in train.csv have NO matching image file!")
        print(f"    Sample missing IDs: {list(missing_in_images)[:5]}")
    else:
        print(f"    [PASS] Every id_code in train.csv has a matching image in train_images ({len(csv_ids):,} / {len(csv_ids):,}).")

    if extra_in_images:
        print(f"    [INFO] {len(extra_in_images)} extra image file(s) found in train_images not listed in train.csv.")
    else:
        print("    [PASS] No orphaned or extra images in train_images (exact 1-to-1 mapping).")

    # ---------------------------------------------------------
    # 4. Open a sample of images to test readability & integrity
    # ---------------------------------------------------------
    sample_size = min(150, len(train_files))
    print(f"\n[4] Testing Image Integrity (sampling {sample_size} images using PIL):")
    sample_files = train_df["id_code"].sample(n=sample_size, random_state=RANDOM_SEED)

    unreadable = []
    resolutions = set()
    for id_code in sample_files:
        img_path = train_file_stems.get(str(id_code))
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()  # Check for header corruption
            # Reopen to read dimensions & format
            with Image.open(img_path) as img:
                resolutions.add(img.size)
        except Exception as e:
            unreadable.append((img_path.name, str(e)))

    if unreadable:
        print(f"    [FAIL] Detected {len(unreadable)} unreadable image(s):", file=sys.stderr)
        for name, err in unreadable[:5]:
            print(f"      - {name}: {err}", file=sys.stderr)
        return False
    else:
        sample_res = list(resolutions)[:3]
        print(f"    [PASS] All {sample_size} sampled images opened and verified successfully without corruption.")
        print(f"    - Sample image dimensions (W x H): {sample_res}")

    # ---------------------------------------------------------
    # 5. Class Counts for diagnosis 0 to 4
    # ---------------------------------------------------------
    print("\n[5] Class Counts for Diagnosis Severity (0 to 4):")
    print("    " + "-" * 55)
    print(f"    {'Class':<7} {'Clinical Severity':<20} {'Count':>8} {'Percentage':>12}")
    print("    " + "-" * 55)

    counts = train_df["diagnosis"].value_counts().sort_index()
    for grade in range(5):
        count = counts.get(grade, 0)
        pct = (count / len(train_df)) * 100
        label = CLASS_LABELS.get(grade, "Unknown")
        print(f"    {grade:<7} {label:<20} {count:>8,d} {pct:>11.2f}%")
    print("    " + "-" * 55)
    print(f"    {'Total':<27} {len(train_df):>8,d} {'100.00%':>12}")

    # ---------------------------------------------------------
    # 6. Training/Validation Note regarding test.csv
    # ---------------------------------------------------------
    print("\n[6] Dataset Splitting & Label Availability Notice:")
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV)
        print(f"    - test.csv found with {len(test_df):,} rows and columns: {list(test_df.columns)}")
        print("    - [IMPORTANT] Notice: test.csv contains NO ground-truth labels ('diagnosis').")
        print("    - [CRITICAL] ALL training, cross-validation, and held-out validation MUST use train.csv only!")
    else:
        print("    - test.csv not found.")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Data verification completed successfully.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
