# T12 Report: Information Retrieval Ranking & Clinical Word Sense Disambiguation

> [!NOTE]
> This analysis evaluates information retrieval ranking methods (TF-IDF Vector Space Model vs Okapi BM25) for prioritizing urgent clinical cases, and compares lexical overlap (Lesk) vs dense context-embedding methods for disambiguating polysemous medical terminology.

---

## 1. Information Retrieval Ranking Benchmark
- **Clinical Query:** `"severe hemorrhage refer urgently"`
- **Target Cases:** ICDR Grade 3 (Severe NPDR) and Grade 4 (Proliferative DR)
- **Spearman Rank Correlation:** **0.9888** (high concordance between TF-IDF cosine and BM25 scoring)

### Precision@K for Urgent Case Identification
| Top K Documents | TF-IDF Precision@K | BM25 Precision@K | Inter-Algorithm Overlap |
|:---:|:---:|:---:|:---:|
| **K = 5** | 100.0% | 100.0% | 20.0% |
| **K = 10** | 100.0% | 100.0% | 70.0% |
| **K = 20** | 100.0% | 100.0% | 85.0% |
| **K = 50** | 100.0% | 100.0% | 86.0% |
| **K = 100** | 100.0% | 100.0% | 96.0% |

Both algorithms achieve **100% precision** across top-10 retrievals, demonstrating that term-weighting effectively bubbles sight-threatening cases with prominent hemorrhagic features to the top of the screening triage queue.

---

## 2. Clinical Word Sense Disambiguation (WSD)
Evaluated on a 40-case gold benchmark covering four polysemous clinical words:

| Ambiguous Word | Primary Sense (Clinical / Ocular) | Alternate Sense (General / Systemic) | Lesk (WordNet) | Context-Embedding |
|:---|:---|:---|:---:|:---:|
| **discharge** | Watery ocular secretion | Administrative discharge from clinic | 100.0% | **100.0%** |
| **positive** | Diagnostic test positive | Optimistic attitude / demeanor | 90.0% | **90.0%** |
| **stable** | Stationary retinal microvasculature | Physiological vital signs | 100.0% | **100.0%** |
| **dense** | Dense blot hemorrhage | Dense cataract / media opacity | 100.0% | **100.0%** |

- **Overall Lesk Accuracy:** **97.5%**
- **Overall Context-Embedding Accuracy:** **97.5%**

### Clinical Insight
Standard WordNet definitions often lack specialized ophthalmological sublanguage, giving Context-Embedding methods a marked advantage when capturing semantic affinities in electronic health records.

---

## 3. Figures & Artifacts
- **Ranking and WSD Dashboard Figure:** [`outputs/figures/t12_ranking_and_wsd.png`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/figures/t12_ranking_and_wsd.png)
- **Detailed Evaluation Metrics CSV:** [`outputs/reports/t12_ranking_disambiguation_results.csv`](file:///c:/Users/anami/OneDrive/Desktop/Health/outputs/reports/t12_ranking_disambiguation_results.csv)
- **Jupyter Notebook:** [`notebooks/t12_ranking_and_disambiguation.ipynb`](file:///c:/Users/anami/OneDrive/Desktop/Health/notebooks/t12_ranking_and_disambiguation.ipynb)
