# RetinaScreen

Retinal screening and a workspace for all 15 project tutorials. The app uses the existing trained ResNet-101, BMI regressor, behavioral classifier, scaler, K-Means model and embedding cache. Existing models are not retrained.

## Run

```powershell
.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:7860. Upload a raw fundus photograph or select a sample. Samples are already normalized; raw uploads receive the T2 crop and Ben Graham normalization used during training. Enable the optional behavioral experiment to use lifestyle inputs, including calories, stress and alcohol consumption.

Use the Workspace selector to visit each tutorial. Saved figures and reports are available within each tutorial; they are separate from the current upload. Clinical text controls work without an image.

## Tutorial responsibilities

| Tutorial | Purpose | Application behavior |
|---|---|---|
| T1 | EHR understanding | Cohort records and schema; uploaded image metadata. No inferred identity or laterality. |
| T2 | Standardization | Quality audit; real crop, normalization and resized CNN input. |
| T3 | Duplicates and missingness | Actual pixel MD5 / pHash comparisons against readable local processed files; saved duplicate log, missing-value audit and imputation experiment. Pixel MD5 differs from file-byte MD5 in the offline pipeline. |
| T4 | Image processing | Green channel, CLAHE, bilateral restoration, morphological candidate maps, interactive JPEG compression; saved NLM/deblurring benchmarks. |
| T5 | Image analysis | Separate SIFT, RANSAC and CNN sections: live keypoints, colored correspondence matches, inlier/error metrics, estimated homography, trained CNN input and five-grade probabilities. |
| T6 | Visualization | Four ordered sections: dataset overview, image preparation, evaluation and current-image explanation. Each figure has a reading guide; the CNN input and live Grad-CAM appear side by side. Illustrative charts are separate from saved evaluation metrics. |
| T7 | Demand forecasting | Saved 104-week simulated demand/held-out forecasts; workforce calculator using 175 screenings/grader/week and 75 consultations/specialist/week. |
| T8 | Descriptive planning | Observed deduplicated cohort proportions and saved workload scenarios. |
| T9 | Prescriptive planning | Threshold scenarios using actual label proportions and explicit cost assumptions. No invented sensitivity/specificity. |
| T10 | Text classification/clustering | Independent TF-IDF/logistic regression and K-Means predictions for entered text. Cluster IDs are not grade IDs. |
| T11 | Information extraction | Actual regex/spaCy Matcher spans from entered text. The original trained NER was not persisted; the live app labels its rule-based method accurately. |
| T12 | Ranking/disambiguation | BM25 and TF-IDF over all 3,504 synthetic notes; live Lesk and TF-IDF context sense comparison. |
| T13 | Dimensionality reduction | Cached reference PCA/t-SNE/UMAP; real PCA/UMAP transform for uploads. t-SNE shows reference points only. |
| T14 | Behavioral exploration | Simulated cohort descriptive statistics, cleaning, correlations and saved statistical analyses. |
| T15 | Behavioral modeling | Saved BMI, behavioral K-Means and risk classifier inference; saved SHAP experiment. Experimental fusion does not change the image grade/referral rule. |

T10 originally discarded its fitted estimators. The app creates and caches only these missing text estimators on first use in `outputs/models/clinical_text_models.joblib`, using the existing synthetic corpus and a stratified training split.

## Verification

```powershell
.venv\Scripts\python.exe src/test_all_tutorials.py
.venv\Scripts\python.exe -m unittest src.test_tutorial_integration -v
```

The first command checks the existing tutorial artifacts and selected algorithms. The second checks tutorial isolation, real retrieval/entity spans, all three screening samples, normalization, optional behavioral inference, input validation and PDF/JSON export.

`src/gradio_app.py` launches the same complete application. UI code is in `src/workspace_ui.py`; text/retrieval adapters are in `src/tutorial_services.py`. The 15 notebooks and offline tutorial pipelines remain available.

## Scope and data provenance

This is an educational research application. APTOS labels are real dataset labels; EHR note text, scheduling and behavioral records are synthetic/simulated. Model confidence is not diagnostic certainty. Grad-CAM indicates model attention; morphological bright regions are candidates, not confirmed lesions. Template referral actions require clinician review.

Saved historical reports contain original experiment claims. T6 includes illustrative hardcoded curves/confusion counts. T9 uses assumed performance adjustments and costs; its historical recommendation is not a validated optimal policy. These limitations are surfaced in the UI. Unreadable local comparison files are skipped; duplicate checks state how many images were actually compared. A sample label can differ from a model prediction.

## Presentation design

The presentation UI uses local Aptos/Segoe UI typography, a retinal image preview, grouped subject navigation and measured result panels. Responsive layout, keyboard focus and reduced-motion styles are included. See `src/presentation.py` and `outputs/reports/presentation_design.md` for the visual system.

Optional browser verification (requires Playwright and its Chromium browser):

```powershell
.venv\Scripts\python.exe src/test_browser_workspace.py
```

This checks all 15 tutorial navigation destinations, text classification, mobile column layout, sample screening and PDF/JSON download links. Desktop, mobile, text and result screenshots are saved in `outputs/figures/ui_tests/`.
