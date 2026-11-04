# Tutorial integration audit

All 15 notebooks were matched to their source pipelines, reports, datasets, figures and application workspaces. The responsibility map is in README.md.

## Corrections

- T1 no longer invents laterality, patient identity or an EHR link for an arbitrary upload.
- T2 now applies exactly the training path: crop, Ben Graham normalization at 512 pixels, then area resize to 224. Processed examples explicitly bypass normalization.
- T3 computes real decoded-pixel MD5 and imagehash pHash, reports the actual nearest readable local image and comparison count, and exposes original duplicate/missingness results. It skips corrupt comparison files instead of aborting screening. The original fake MD5, fixed distance and uniqueness statement were removed.
- T4 controls perform enhancement, bilateral restoration, segmentation-candidate visualization and JPEG encode/decode. Candidate maps are not labeled confirmed pathology.
- T5 performs real SIFT matching and RANSAC against an affine view instead of asserting 100% consensus. Existing ResNet-101 weights supply image predictions; missing weights now fail explicitly.
- T6 uses actual model gradients for uploaded Grad-CAM. Historical hardcoded curves and confusion counts are identified as illustrative in source titles, UI and report provenance.
- T7 uses the original workforce capacities: 175 screenings/grader/week and 75 consultations/specialist/week. Saved held-out forecasting results remain separate from an individual case.
- T8 shows actual deduplicated class counts rather than treating an image grade as a patient percentile.
- T9 computes referral shares from the cohort and costs from stated assumptions. Fixed sensitivity/specificity values and claims of proven optimality were removed from live results. Historical report assumptions are explicitly identified.
- T10 predicts entered text using real fitted estimators, rather than copying the CNN grade. K-Means cluster IDs are not severity IDs. The original pipeline had no serialized text estimators; only those missing estimators are fitted once and cached.
- T11 uses the existing regex/spaCy Matcher algorithm and returns actual text offsets. It no longer calls fabricated entities trained NER output. No persisted custom trained NER artifact was present. Mentions may be negated, so extraction is not asserted to confirm disease.
- T12 runs actual BM25 or TF-IDF over all 3,504 notes, plus the existing Lesk and TF-IDF sense comparison. The former first-300-row word-frequency proxy was removed.
- T13 transforms embeddings with the saved PCA/UMAP. It never displays invented UMAP coordinates or labels a nearest reference point as a new t-SNE projection.
- T14 separates simulated cohort exploration from patient inputs and displays actual age rather than the lifestyle-enabled boolean.
- T15 uses the saved regressor, scaler, K-Means and classifier with user-entered calories, stress and alcohol instead of hidden constants. Fusion remains an experimental score and does not alter image classification/referral rules. Missing behavioral artifacts fail explicitly.
- Empty image input produces a prompt; it never silently analyzes a substituted patient. Unique report identifiers prevent same-second overwrite. JSON includes input/model provenance.
- Both application entry points use the same complete UI. All tutorial workspaces have focused descriptions and separately identified saved experiments.

## Validation

- Existing artifact/algorithm suite: 15/15 passed.
- Added integration suite: independent text predictions, exact entity offsets, real ranking, workforce assumptions, 17 UI tabs (summary + 15 tutorials + report), three sample inference cases, normalization, optional behavioral inputs, invalid inputs and PDF/JSON exports.
- Headless Chromium: rendering, interactive text classification, sample screening, workspace navigation, JavaScript error collection and desktop/mobile screenshots.

## Limits

Existing model performance is unchanged. A sample ground-truth grade may differ from its prediction. Synthetic text performance does not establish performance on clinical notes. Behavioral and scheduling results are simulated. Historical T6 illustrative arrays and T9 scenario assumptions are retained with explicit provenance corrections; no new measured validation claims are made for them. Duplicate checks cover readable local processed images, not a guaranteed complete dataset. Existing trained image/behavior models were reused without retraining.
