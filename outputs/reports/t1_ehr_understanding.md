# RetinaScreen: EHR Understanding & Data Dictionary Report
**Dataset:** Kaggle APTOS 2019 Blindness Detection  
**Generated Table:** `data/ehr_records.csv` (3,662 total records)  
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

The structured table `data/ehr_records.csv` contains 15 fields. Per clinical privacy protocols and task instructions, **no patient age or sex has been invented**. All simulated fields carry the explicit `sim_` prefix and are strictly documented as simulated encounter parameters.

| Column Name | Data Type | Source / Origin | Description & Clinical Meaning | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `record_id` | String | APTOS `id_code` | Unique anonymized accession/study identifier corresponding to the fundus image. | `"000c1434d8d7"` |
| `diagnosis` | Integer (0-4) | APTOS `diagnosis` | ICDR diabetic retinopathy severity grade assigned by clinical experts. | `2` |
| `severity_name` | String | Clinical Mapping | Clinical label name corresponding to the numerical grade: *No DR, Mild, Moderate, Severe, Proliferative*. | `"Moderate"` |
| `width` | Integer (px) | Image Header | Horizontal resolution of the fundus photograph in pixels. | `3216` |
| `height` | Integer (px) | Image Header | Vertical resolution of the fundus photograph in pixels. | `2136` |
| `aspect_ratio` | Float | Image-Derived | Image width divided by height ($W / H$). Indicates sensor form factor and field cropping. | `1.5056` |
| `file_size_bytes` | Integer | Filesystem | Exact raw image file size in bytes. Reflects uncompressed visual detail and storage footprint. | `3218676` |
| `file_size_kb` | Float | Filesystem | Image file size in kilobytes ($bytes / 1024$). | `3143.24` |
| `mean_brightness` | Float (0-255) | Image-Derived | Mean intensity of the grayscale image. Measures overall illumination, underexposure, or flash glare. | `50.9` |
| `blur_score` | Float | Image-Derived | Variance of the Laplacian filter ($\sigma^2_{\nabla^2 I}$). Higher values denote sharp focus; lower values denote motion blur or media opacity (e.g. cataracts). | `6.86` |
| `sim_encounter_id` | String | **Simulated** (`sim_`) | Unique clinical screening encounter identifier simulating hospital encounter indexing. | `"ENC-2019-00001"` |
| `sim_encounter_type` | String | **Simulated** (`sim_`) | Type of medical visit: *Diabetic Retinopathy Tele-Screening*. | `"Diabetic Retinopathy Tele-Screening"` |
| `sim_eye_laterality` | String | **Simulated** (`sim_`) | Anatomical eye side: `OD` (right eye / Oculus Dexter) or `OS` (left eye / Oculus Sinister). Tagged deterministically. | `"OS"` |
| `sim_referral_needed`| Boolean | **Simulated** (`sim_`) | Clinical triage flag (`True` if diagnosis $\ge 2$, `False` otherwise). Distinguishes referable DR from non-referable cases. | `True` |
| `sim_triage_urgency` | String | **Simulated** (`sim_`) | Standardized clinical care pathway recommendation and appointment scheduling timeframe. | `"Ophthalmology Referral (3-6 months)"` |

> [!NOTE]
> **Simulated Fields Notice:** `sim_encounter_id`, `sim_encounter_type`, `sim_eye_laterality`, `sim_referral_needed`, and `sim_triage_urgency` are synthetic demonstration attributes prefixed with `sim_` to illustrate full-stack EHR integration workflows. No demographic attributes (age, sex, ethnicity) were fabricated.

---

## 3. Example Record

Below is an extracted record from `data/ehr_records.csv`:

```json
{
  "record_id": "000c1434d8d7",
  "diagnosis": 2,
  "severity_name": "Moderate",
  "width": 3216,
  "height": 2136,
  "aspect_ratio": 1.5056,
  "file_size_bytes": 3218676,
  "file_size_kb": 3143.24,
  "mean_brightness": 50.9,
  "blur_score": 6.86,
  "sim_encounter_id": "ENC-2019-00001",
  "sim_encounter_type": "Diabetic Retinopathy Tele-Screening",
  "sim_eye_laterality": "OS",
  "sim_referral_needed": True,
  "sim_triage_urgency": "Ophthalmology Referral (3-6 months)"
}
```

---

## 4. Dataset Summary Statistics

### Numerical Image & Quality Metrics (3,662 images)

| Metric | Width (px) | Height (px) | Aspect Ratio | File Size (KB) | Mean Brightness | Blur Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean** | 2015.18 | 1526.83 | 1.28 | 2294.22 | 66.34 | 25.19 |
| **Std Dev** | 884.3 | 542.66 | 0.18 | 1721.14 | 17.93 | 18.5 |
| **Min** | 474.0 | 358.0 | 1.0 | 216.54 | 14.73 | 3.07 |
| **25% (Q1)**| 1050.0 | 1050.0 | 1.0 | 980.46 | 51.65 | 10.6 |
| **50% (Median)**| 2144.0 | 1536.0 | 1.33 | 1889.6 | 68.44 | 18.07 |
| **75% (Q3)**| 2588.0 | 1958.0 | 1.39 | 2780.12 | 80.37 | 43.1 |
| **Max** | 4288.0 | 2848.0 | 1.51 | 7494.43 | 129.03 | 131.46 |

### Severity Class Breakdown

| Diagnosis Code | Clinical Severity | Total Records | Percentage | Referral Action |
| :---: | :--- | :---: | :---: | :--- |
| **0** | No DR | 1,805 | 49.29% | Routine Annual Rescreen |
| **1** | Mild | 370 | 10.1% | Routine Monitoring |
| **2** | Moderate | 999 | 27.28% | Ophthalmology Outpatient (3-6 mo) |
| **3** | Severe | 193 | 5.27% | Specialist Referral (2-4 wk) |
| **4** | Proliferative | 295 | 8.06% | Urgent Vitreoretinal (1-2 wk) |
| **Total** | | **3,662** | **100.00%** | |

* **Referable DR (Grades 2-4):** 1,487 records (40.61%)
* **Non-Referable DR (Grades 0-1):** 2,175 records (59.39%)

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
4. **Acquisition Artifacts & Domain Shift:** The high variance in image resolution (ranging from 474.0x358.0 to 4288.0x2848.0) and blur scores (spanning 3.07 to 131.46) reflects real-world multi-site hardware heterogeneity, varying operator proficiency, and patient non-compliance (pupillary constriction, uncooperative blinking, or ocular media opacities such as cataracts).
