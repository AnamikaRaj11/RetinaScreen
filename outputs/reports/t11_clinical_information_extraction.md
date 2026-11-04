# T11 Report: Clinical Information Extraction & Named Entity Recognition (NER)

> [!NOTE]
> This analysis extracts structured clinical entities from the SYNTHETIC clinical note corpus generated in T10. Two complementary paradigms are implemented and compared:
> 1. **Rule-Based Extraction:** Regex + token-level spaCy Matcher
> 2. **Machine Learned Custom NER:** spaCy statistical EntityRecognizer trained on annotated clinical notes.

---

## 1. Entity Recognition Evaluation (Held-Out Validation Set)

| Entity Type | Rule-Based Precision | Rule-Based Recall | Rule-Based F1 | Custom NER Precision | Custom NER Recall | Custom NER F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LATERALITY** | 100.0% | 100.0% | **1.000** | 100.0% | 100.0% | **1.000** |
| **LESION** | 100.0% | 100.0% | **1.000** | 100.0% | 100.0% | **1.000** |
| **SEVERITY** | 100.0% | 100.0% | **1.000** | 100.0% | 100.0% | **1.000** |
| **ACTION** | 100.0% | 100.0% | **1.000** | 100.0% | 100.0% | **1.000** |

- **Macro Average F1 (Rule-Based):** **1.000**
- **Macro Average F1 (Custom NER):** **1.000**

---

## 2. Key Clinical Extraction Findings
1. **Lesion Gradients with True Severity:** As expected, extracted lesion count monotonically climbs from **0.0** in Grade 0 (No DR) to an average of **2.4 distinct lesions** in Grade 4 (Proliferative DR, capturing NVD, NVE, and vitreous hemorrhage).
2. **Laterality Disambiguation:** Successfully separated right eye (`OD`), left eye (`OS`), and bilateral (`OU`) examinations across all 3,504 cases.
3. **Action Plan Extraction:** 100% of cases were linked to an operational action (annual screening, 6-month review, specialist referral, or emergency vitrectomy consultation).

---

## 3. Figures & Linked Data Artifacts
- **Extracted Entities Linked to Record ID:** [`data/extracted_clinical_entities.csv`](file:///c:/Users/anami/OneDrive/Desktop/Health/data/extracted_clinical_entities.csv)
- **Evaluation & Clinical Distribution Figure:** [`outputs/figures/t11_information_extraction_ner.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t11_information_extraction_ner.png)
- **Interactive Jupyter Notebook:** [`notebooks/t11_clinical_information_extraction.ipynb`](file:///c:/Users/anami/OneDrive/Desktop/Health/notebooks/t11_clinical_information_extraction.ipynb)
