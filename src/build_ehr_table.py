"""
Build structured EHR records table from APTOS 2019 dataset.
Extracts image-derived metrics (width, height, size, brightness, blur score),
merges with clinical severity grades, attaches simulated encounter metadata,
and exports to data/ehr_records.csv and outputs/reports/t1_ehr_understanding.md.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path and import project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    DATA_DIR,
    TRAIN_CSV,
    TRAIN_IMG_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)

EHR_OUTPUT_CSV = DATA_DIR / "ehr_records.csv"
REPORT_OUTPUT_MD = REPORTS_DIR / "t1_ehr_understanding.md"

# Severity labels matching user specification:
# (No DR, Mild, Moderate, Severe, Proliferative)
SEVERITY_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

# Clinical referral guidelines associated with ICDR severity grades
TRIAGE_RECOMMENDATIONS = {
    0: "Routine Annual Screening (12 months)",
    1: "Routine Annual Monitoring (12 months)",
    2: "Ophthalmology Referral (3-6 months)",
    3: "Prompt Specialist Referral (2-4 weeks)",
    4: "Urgent Retinal Specialist / Vitreoretinal Referral (1-2 weeks)",
}


def extract_image_features(img_path: Path) -> dict:
    """
    Extracts physical and image-derived features from a retinal fundus image:
    - width, height, aspect ratio
    - file size (bytes and KB)
    - mean brightness (grayscale mean intensity)
    - blur score (variance of the Laplacian)
    """
    file_size_bytes = img_path.stat().st_size
    file_size_kb = round(file_size_bytes / 1024.0, 2)

    # Read image in grayscale for brightness and edge variance computation
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not decode image at {img_path}")

    height, width = gray.shape
    aspect_ratio = round(float(width) / float(height), 4)

    # Mean brightness (0-255 scale)
    mean_brightness = round(float(np.mean(gray)), 2)

    # Blur score using Laplacian variance (higher = sharper, lower = blurrier)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = round(float(laplacian_var), 2)

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": aspect_ratio,
        "file_size_bytes": int(file_size_bytes),
        "file_size_kb": file_size_kb,
        "mean_brightness": mean_brightness,
        "blur_score": blur_score,
    }


def process_record(row_tuple: tuple) -> dict:
    """Worker function to process a single train.csv record."""
    idx, id_code, diagnosis = row_tuple
    img_path = TRAIN_IMG_DIR / f"{id_code}.png"

    metrics = extract_image_features(img_path)

    # Severity labels
    severity_name = SEVERITY_NAMES.get(int(diagnosis), "Unknown")

    # Simulated clinical encounter fields (strictly prefixed with 'sim_')
    # Note: Deterministic simulation for reproducibility. NO patient age or sex simulated.
    sim_encounter_id = f"ENC-2019-{idx + 1:05d}"
    sim_encounter_type = "Diabetic Retinopathy Tele-Screening"
    # Simulated eye laterality (OD = Oculus Dexter / Right Eye, OS = Oculus Sinister / Left Eye)
    # Alternates deterministically based on hash of id_code
    sim_eye_laterality = "OD" if (hash(id_code) % 2 == 0) else "OS"
    sim_referral_needed = bool(diagnosis >= 2)  # Referable DR is typically grade >= 2
    sim_triage_urgency = TRIAGE_RECOMMENDATIONS.get(int(diagnosis), "Clinical Evaluation")

    return {
        "record_id": str(id_code),
        "diagnosis": int(diagnosis),
        "severity_name": severity_name,
        "width": metrics["width"],
        "height": metrics["height"],
        "aspect_ratio": metrics["aspect_ratio"],
        "file_size_bytes": metrics["file_size_bytes"],
        "file_size_kb": metrics["file_size_kb"],
        "mean_brightness": metrics["mean_brightness"],
        "blur_score": metrics["blur_score"],
        "sim_encounter_id": sim_encounter_id,
        "sim_encounter_type": sim_encounter_type,
        "sim_eye_laterality": sim_eye_laterality,
        "sim_referral_needed": sim_referral_needed,
        "sim_triage_urgency": sim_triage_urgency,
    }


def build_ehr_table(max_workers: int = 12) -> pd.DataFrame:
    print("=" * 70)
    print(" Building Structured EHR Records Table from APTOS 2019 Dataset")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(train_df):,} records from {TRAIN_CSV.name}")

    rows_to_process = [
        (idx, row["id_code"], int(row["diagnosis"]))
        for idx, row in train_df.iterrows()
    ]

    print(f"Extracting image-derived metrics across {len(rows_to_process):,} images using {max_workers} worker threads...")
    t0 = time.time()

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_record, rows_to_process), total=len(rows_to_process), desc="Processing images"):
            records.append(res)

    elapsed = time.time() - t0
    print(f"\nFeature extraction completed in {elapsed:.1f}s ({elapsed / len(rows_to_process) * 1000:.1f} ms/image).")

    ehr_df = pd.DataFrame(records)

    # Save to data/ehr_records.csv
    ehr_df.to_csv(EHR_OUTPUT_CSV, index=False)
    print(f"[SAVED] Structured EHR records table saved to: {EHR_OUTPUT_CSV}")

    return ehr_df


def generate_markdown_report(ehr_df: pd.DataFrame) -> None:
    """Generates outputs/reports/t1_ehr_understanding.md with data dictionary and EHR concepts."""
    
    # Calculate summary statistics
    numeric_cols = ["width", "height", "aspect_ratio", "file_size_kb", "mean_brightness", "blur_score"]
    stats_df = ehr_df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).round(2)

    class_counts = ehr_df["severity_name"].value_counts()[["No DR", "Mild", "Moderate", "Severe", "Proliferative"]]
    class_pcts = (class_counts / len(ehr_df) * 100).round(2)

    sample_record = ehr_df.iloc[0].to_dict()

    md_content = f"""# RetinaScreen: EHR Understanding & Data Dictionary Report
**Dataset:** Kaggle APTOS 2019 Blindness Detection  
**Generated Table:** `data/ehr_records.csv` ({len(ehr_df):,} total records)  
**Report Artifact:** `outputs/reports/t1_ehr_understanding.md`  

---

## 1. Executive Summary & EHR Context

In modern clinical ophthalmology and tele-ophthalmology, retinal fundus photographs do not exist in isolation; they are diagnostic artifacts embedded within an **Electronic Health Record (EHR)** and **Picture Archiving and Communication System (PACS)** ecosystem. 

In this project, the primary table `train.csv` is operationalized as an EHR diagnostic registry where each record represents a distinct fundus imaging encounter for diabetic retinopathy (DR) screening. To bridge raw image files and structured clinical informatics, we derived physical and visual metadata (resolution, file size, mean illumination, and Laplacian edge-variance blur score) and paired them with clinical severity classifications and referral triage tiers.

### Clinical DR Grading (International Clinical Diabetic Retinopathy - ICDR Scale)
Diabetic retinopathy is progressive retinal microvascular damage caused by prolonged hyperglycemia. Clinical management relies on severity stratification:
1. **Grade 0 (No DR):** No visible microvascular abnormalities. Routine annual rescreening.
2. **Grade 1 (Mild Non-Proliferative DR - NPDR):** Microaneurysms only. Monitored annually.
3. **Grade 2 (Moderate NPDR):** More than just microaneurysms but less than severe NPDR (cotton wool spots, venous beading, hard exudates). Scheduled specialist referral (3–6 months).
4. **Grade 3 (Severe NPDR):** The "4-2-1" rule (severe hemorrhages in 4 quadrants, venous beading in 2 quadrants, or IRMA in 1 quadrant). High risk of progression; prompt retinal referral (2–4 weeks).
5. **Grade 4 (Proliferative DR - PDR):** Neovascularization of retina/disc or vitreous/preretinal hemorrhage. High risk of irreversible vision loss; urgent vitreoretinal intervention (within 1–2 weeks, panretinal photocoagulation or anti-VEGF injections).

---

## 2. Comprehensive Data Dictionary

The structured table `data/ehr_records.csv` contains {len(ehr_df.columns)} fields. Per clinical privacy protocols and task instructions, **no patient age or sex has been invented**. All simulated fields carry the explicit `sim_` prefix and are strictly documented as simulated encounter parameters.

| Column Name | Data Type | Source / Origin | Description & Clinical Meaning | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `record_id` | String | APTOS `id_code` | Unique anonymized accession/study identifier corresponding to the fundus image. | `"{sample_record['record_id']}"` |
| `diagnosis` | Integer (0-4) | APTOS `diagnosis` | ICDR diabetic retinopathy severity grade assigned by clinical experts. | `{sample_record['diagnosis']}` |
| `severity_name` | String | Clinical Mapping | Clinical label name corresponding to the numerical grade: *No DR, Mild, Moderate, Severe, Proliferative*. | `"{sample_record['severity_name']}"` |
| `width` | Integer (px) | Image Header | Horizontal resolution of the fundus photograph in pixels. | `{sample_record['width']}` |
| `height` | Integer (px) | Image Header | Vertical resolution of the fundus photograph in pixels. | `{sample_record['height']}` |
| `aspect_ratio` | Float | Image-Derived | Image width divided by height ($W / H$). Indicates sensor form factor and field cropping. | `{sample_record['aspect_ratio']}` |
| `file_size_bytes` | Integer | Filesystem | Exact raw image file size in bytes. Reflects uncompressed visual detail and storage footprint. | `{sample_record['file_size_bytes']}` |
| `file_size_kb` | Float | Filesystem | Image file size in kilobytes ($bytes / 1024$). | `{sample_record['file_size_kb']}` |
| `mean_brightness` | Float (0-255) | Image-Derived | Mean intensity of the grayscale image. Measures overall illumination, underexposure, or flash glare. | `{sample_record['mean_brightness']}` |
| `blur_score` | Float | Image-Derived | Variance of the Laplacian filter ($\sigma^2_{{\\nabla^2 I}}$). Higher values denote sharp focus; lower values denote motion blur or media opacity (e.g. cataracts). | `{sample_record['blur_score']}` |
| `sim_encounter_id` | String | **Simulated** (`sim_`) | Unique clinical screening encounter identifier simulating hospital encounter indexing. | `"{sample_record['sim_encounter_id']}"` |
| `sim_encounter_type` | String | **Simulated** (`sim_`) | Type of medical visit: *Diabetic Retinopathy Tele-Screening*. | `"{sample_record['sim_encounter_type']}"` |
| `sim_eye_laterality` | String | **Simulated** (`sim_`) | Anatomical eye side: `OD` (right eye / Oculus Dexter) or `OS` (left eye / Oculus Sinister). Tagged deterministically. | `"{sample_record['sim_eye_laterality']}"` |
| `sim_referral_needed`| Boolean | **Simulated** (`sim_`) | Clinical triage flag (`True` if diagnosis $\ge 2$, `False` otherwise). Distinguishes referable DR from non-referable cases. | `{sample_record['sim_referral_needed']}` |
| `sim_triage_urgency` | String | **Simulated** (`sim_`) | Standardized clinical care pathway recommendation and appointment scheduling timeframe. | `"{sample_record['sim_triage_urgency']}"` |

> [!NOTE]
> **Simulated Fields Notice:** `sim_encounter_id`, `sim_encounter_type`, `sim_eye_laterality`, `sim_referral_needed`, and `sim_triage_urgency` are synthetic demonstration attributes prefixed with `sim_` to illustrate full-stack EHR integration workflows. No demographic attributes (age, sex, ethnicity) were fabricated.

---

## 3. Example Record

Below is an extracted record from `data/ehr_records.csv`:

```json
{{
  "record_id": "{sample_record['record_id']}",
  "diagnosis": {sample_record['diagnosis']},
  "severity_name": "{sample_record['severity_name']}",
  "width": {sample_record['width']},
  "height": {sample_record['height']},
  "aspect_ratio": {sample_record['aspect_ratio']},
  "file_size_bytes": {sample_record['file_size_bytes']},
  "file_size_kb": {sample_record['file_size_kb']},
  "mean_brightness": {sample_record['mean_brightness']},
  "blur_score": {sample_record['blur_score']},
  "sim_encounter_id": "{sample_record['sim_encounter_id']}",
  "sim_encounter_type": "{sample_record['sim_encounter_type']}",
  "sim_eye_laterality": "{sample_record['sim_eye_laterality']}",
  "sim_referral_needed": {sample_record['sim_referral_needed']},
  "sim_triage_urgency": "{sample_record['sim_triage_urgency']}"
}}
```

---

## 4. Dataset Summary Statistics

### Numerical Image & Quality Metrics ({len(ehr_df):,} images)

| Metric | Width (px) | Height (px) | Aspect Ratio | File Size (KB) | Mean Brightness | Blur Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean** | {stats_df.loc['mean', 'width']} | {stats_df.loc['mean', 'height']} | {stats_df.loc['mean', 'aspect_ratio']} | {stats_df.loc['mean', 'file_size_kb']} | {stats_df.loc['mean', 'mean_brightness']} | {stats_df.loc['mean', 'blur_score']} |
| **Std Dev** | {stats_df.loc['std', 'width']} | {stats_df.loc['std', 'height']} | {stats_df.loc['std', 'aspect_ratio']} | {stats_df.loc['std', 'file_size_kb']} | {stats_df.loc['std', 'mean_brightness']} | {stats_df.loc['std', 'blur_score']} |
| **Min** | {stats_df.loc['min', 'width']} | {stats_df.loc['min', 'height']} | {stats_df.loc['min', 'aspect_ratio']} | {stats_df.loc['min', 'file_size_kb']} | {stats_df.loc['min', 'mean_brightness']} | {stats_df.loc['min', 'blur_score']} |
| **25% (Q1)**| {stats_df.loc['25%', 'width']} | {stats_df.loc['25%', 'height']} | {stats_df.loc['25%', 'aspect_ratio']} | {stats_df.loc['25%', 'file_size_kb']} | {stats_df.loc['25%', 'mean_brightness']} | {stats_df.loc['25%', 'blur_score']} |
| **50% (Median)**| {stats_df.loc['50%', 'width']} | {stats_df.loc['50%', 'height']} | {stats_df.loc['50%', 'aspect_ratio']} | {stats_df.loc['50%', 'file_size_kb']} | {stats_df.loc['50%', 'mean_brightness']} | {stats_df.loc['50%', 'blur_score']} |
| **75% (Q3)**| {stats_df.loc['75%', 'width']} | {stats_df.loc['75%', 'height']} | {stats_df.loc['75%', 'aspect_ratio']} | {stats_df.loc['75%', 'file_size_kb']} | {stats_df.loc['75%', 'mean_brightness']} | {stats_df.loc['75%', 'blur_score']} |
| **Max** | {stats_df.loc['max', 'width']} | {stats_df.loc['max', 'height']} | {stats_df.loc['max', 'aspect_ratio']} | {stats_df.loc['max', 'file_size_kb']} | {stats_df.loc['max', 'mean_brightness']} | {stats_df.loc['max', 'blur_score']} |

### Severity Class Breakdown

| Diagnosis Code | Clinical Severity | Total Records | Percentage | Referral Action |
| :---: | :--- | :---: | :---: | :--- |
| **0** | No DR | {class_counts['No DR']:,} | {class_pcts['No DR']}% | Routine Annual Rescreen |
| **1** | Mild | {class_counts['Mild']:,} | {class_pcts['Mild']}% | Routine Monitoring |
| **2** | Moderate | {class_counts['Moderate']:,} | {class_pcts['Moderate']}% | Ophthalmology Outpatient (3-6 mo) |
| **3** | Severe | {class_counts['Severe']:,} | {class_pcts['Severe']}% | Specialist Referral (2-4 wk) |
| **4** | Proliferative | {class_counts['Proliferative']:,} | {class_pcts['Proliferative']}% | Urgent Vitreoretinal (1-2 wk) |
| **Total** | | **{len(ehr_df):,}** | **100.00%** | |

* **Referable DR (Grades 2-4):** {class_counts['Moderate'] + class_counts['Severe'] + class_counts['Proliferative']:,} records ({(class_pcts['Moderate'] + class_pcts['Severe'] + class_pcts['Proliferative']):.2f}%)
* **Non-Referable DR (Grades 0-1):** {class_counts['No DR'] + class_counts['Mild']:,} records ({(class_pcts['No DR'] + class_pcts['Mild']):.2f}%)

---

## 5. EHR Concepts & Real-World Clinical Integration

### 1. Interoperability & Medical Standards (DICOM & HL7 FHIR)
In hospital and tele-ophthalmology networks, digital fundus cameras export images adhering to the **DICOM (Digital Imaging and Communications in Medicine)** standard (specifically the Ophthalmic Photography 8-Bit / 16-Bit Image Information Object Definition).
Each DICOM object carries header metadata linking the pixels to:
* **Patient Identity** (`PatientID`, `PatientBirthDate`, `PatientSex`)
* **Study & Series Context** (`StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID`)
* **Anatomical Laterality** (`ImageLaterality`: `R` or `L`)
* **Acquisition Device** (`Manufacturer`, `DeviceSerialNumber`, field of view angle: e.g. 45° or 60°)

When communicating with enterprise EHRs (e.g. Epic, Cerner), systems translate these imaging studies into **HL7 FHIR (Fast Healthcare Interoperability Resources)**:
- `DiagnosticReport`: Houses the overarching clinical interpretation and AI screening output.
- `Observation`: Stores structured observations (e.g., ICDR DR grade = 2, macular edema present = false, confidence = 0.94).
- `ServiceRequest`: Automates secondary ophthalmology referrals when referable DR is detected.
- `Media` / `ImagingStudy`: Links directly to the stored fundus imagery on PACS.

### 2. Dataset Limitations from an EHR Perspective
The APTOS 2019 dataset was curated for competitive machine learning and stripped of all protected health information (PHI) under HIPAA de-identification guidelines. As an EHR dataset, it exhibits several notable domain constraints:
1. **Absence of Patient Demographics & Systemic Covariates:** In clinical practice, DR risk and progression are strongly mediated by patient age, duration of diabetes, glycemic control (HbA1c %), systemic hypertension, lipid panels, and renal function (eGFR/microalbuminuria). In APTOS 2019, all such covariates are absent.
2. **Missing Longitudinal Tracking:** EHRs track patient eyes over time across multiple annual or biannual visits. The APTOS dataset provides isolated cross-sectional snapshots with independent record IDs, preventing intra-patient longitudinal modeling.
3. **Unspecified Eye Laterality and Multi-field Coverage:** Clinical guidelines recommend two-field photography (one macula-centered and one optic disc-centered) for each eye (OD and OS). In APTOS, laterality and field positioning are not annotated.
4. **Acquisition Artifacts & Domain Shift:** The high variance in image resolution (ranging from {stats_df.loc['min', 'width']}x{stats_df.loc['min', 'height']} to {stats_df.loc['max', 'width']}x{stats_df.loc['max', 'height']}) and blur scores (spanning {stats_df.loc['min', 'blur_score']} to {stats_df.loc['max', 'blur_score']}) reflects real-world multi-site hardware heterogeneity, varying operator proficiency, and patient non-compliance (pupillary constriction, uncooperative blinking, or ocular media opacities such as cataracts).
"""

    with open(REPORT_OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[REPORT] Comprehensive EHR report generated at: {REPORT_OUTPUT_MD}")


def main() -> None:
    ehr_df = build_ehr_table()
    generate_markdown_report(ehr_df)

    # Display requested console output:
    print("\n" + "=" * 70)
    print(" SAMPLE EHR RECORD (First Row):")
    print("=" * 70)
    for col, val in ehr_df.iloc[0].items():
        print(f"  {col:<22}: {val}")

    print("\n" + "=" * 70)
    print(" SUMMARY STATISTICS (Image-Derived Metrics):")
    print("=" * 70)
    numeric_cols = ["width", "height", "aspect_ratio", "file_size_kb", "mean_brightness", "blur_score"]
    stats_summary = ehr_df[numeric_cols].describe().round(2)
    print(stats_summary.to_string())

    print("\n" + "=" * 70)
    print(" CLASS DISTRIBUTION:")
    print("=" * 70)
    counts = ehr_df["severity_name"].value_counts()[["No DR", "Mild", "Moderate", "Severe", "Proliferative"]]
    for name, cnt in counts.items():
        pct = (cnt / len(ehr_df)) * 100
        print(f"  {name:<15}: {cnt:>5} records ({pct:>5.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
