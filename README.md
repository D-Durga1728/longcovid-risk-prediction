# Predictive Modelling of COVID-19 In-Hospital Mortality from Admission-Time Variables

**A Responsible Machine-Learning Pipeline · MSc Computing (Data Analytics), Dublin City University**

| | |
|---|---|
| **Student** | Durga Prasad Narsing (A00050350) |
| **Supervisors** | Dr Martin Crane · Dr Tai Tan Mai |
| **School** | School of Computing, Dublin City University |

**🔗 Live demo:** https://longcovid-risk-prediction-dcu-2026.streamlit.app/

---

## Overview

This project builds an end-to-end, **responsible machine-learning pipeline that predicts COVID-19 in-hospital mortality from nine admission-time clinical features**. Five model families are benchmarked, the deployed model is selected by a formal statistical test (DeLong), its probabilities are isotonic-calibrated, and the full system — including a subgroup fairness audit and multi-layer explanations — is exposed through a Streamlit research prototype.

> **Scope & framing.** The prediction target is **in-hospital mortality**, a directly labelled severe outcome in the dataset. Long COVID (a distinct post-acute condition) is the project's *motivation* and *intended future extension* — predicting it directly would require PASC-labelled follow-up data, which no large-scale public dataset currently provides. This is a **research and education prototype, not a validated clinical device.**

---

## Key Results

| Metric | Value |
|---|---|
| Dataset | 220,218 confirmed COVID-19 cases (Mexican Govt. surveillance) |
| Features | 9 admission-time (age, sex, 7 binary comorbidities) |
| Outcome / prevalence | In-hospital mortality / 12.31% (≈1:7 class imbalance) |
| Deployed model | Calibrated Logistic Regression (C = 0.01, isotonic) |
| Test AUC | **0.888** (95% CI 0.884–0.892) |
| Temporal (out-of-time) AUC | 0.891 |
| Brier score | 0.135 → **0.078** (post-calibration) |
| ECE (count-weighted) | 0.195 → **0.002** (post-calibration) |
| Model selection | DeLong test: no ensemble shows a significant AUC difference vs LR (p > 0.05) |
| Fairness | ECE < 0.015 across 8 subgroups; TPR disparity 0.603 → **0.045** |
| Top drivers (SHAP, deployed LR) | Pneumonia 39.9% · Age 33.1% (together 73.0%) |

---

## Deliverables

- **`Practicum_Final_Report.pdf`** — final IEEE-format research paper.
- **`streamlit_covid_predictor.py`** — 2-page Streamlit clinical research prototype.
- **`covid_analysis_full.py`** + **`advanced_methods.py`** — the 13-phase analysis pipeline.
- **`analysis_output/`** — all exported results (CSVs, calibrated model artefacts, figures, Model Card, Datasheet).

---

## Repository Structure

```
├── Practicum_Final_Report.pdf       Final IEEE research paper
├── streamlit_covid_predictor.py     Streamlit clinical research prototype
├── covid_analysis_full.py           Main 13-phase responsible-ML pipeline
├── advanced_methods.py              XGBoost, Stacking, DeLong, DCA, temporal, Tree-SHAP
├── test_analysis.py                 44 unit tests for the statistical helpers
├── requirements.txt                 Lean app runtime (Streamlit Cloud)
├── requirements-analysis.txt        Full analysis-pipeline dependencies
│
├── analysis_output/
│   ├── models/
│   │   ├── model_primary_deployed.joblib   Calibrated Logistic Regression (deployed)
│   │   ├── imputer.joblib                  Median imputer (fit on train only)
│   │   └── scaler.joblib                   StandardScaler (fit on train only)
│   ├── advanced/                    DeLong, AUC CIs, calibration, thresholds, SHAP,
│   │                                nomogram, temporal, MODEL_CARD.md, DATASHEET.md
│   ├── fairness_calibration/        Fairness audit + calibration analysis
│   └── visualizations/              EDA, ROC, calibration, SHAP, fairness figures
│
├── assets/                          DCU logos
└── .streamlit/config.toml           Streamlit theme
```

---

## Models Benchmarked

Five families, all trained cost-sensitively for the ≈1:7 class imbalance:

1. **Logistic Regression** (L2, `C = 0.01`) — **deployed** (calibrated).
2. **Random Forest** (`max_depth = 6`, `n_estimators = 200`).
3. **Gradient Boosting** (`sample_weight` balanced).
4. **XGBoost** (`scale_pos_weight = 7`).
5. **Stacking ensemble** — Logistic Regression, Random Forest and Gradient Boosting as base learners with a Logistic Regression meta-learner.

Selection is by **DeLong's test** (2,000-resample bootstrap CIs). Because no complex model shows a statistically significant AUC difference, the simpler, calibrated, interpretable Logistic Regression is deployed on parsimony, calibration and transparency grounds.

---

## Pipeline Phases

| Phase | Description |
|---|---|
| 1 | Exploratory Data Analysis |
| 2 | Preprocessing — sentinel decoding, leakage-free (split-then-fit) imputation & scaling |
| 3 | Statistical feature validation — t-test, chi-square, Benjamini–Hochberg FDR, multicollinearity |
| 4 | Model training — 5 families, cost-sensitive, GridSearchCV |
| 5 | SHAP explainability (deployed Logistic Regression) |
| 6 | Subgroup fairness audit — 8 groups (AUC + ECE), threshold mitigation |
| 7 | Probability calibration — isotonic, Brier, ECE |
| 8 | Cost-benefit / decision-curve analysis |
| 9 | Clinical explanations — odds ratios, nomogram |
| 10 | Feature interactions (secondary tree-model check) |
| 11 | Counterfactual sensitivity analysis |
| 12 | Export — CSVs, joblib artefacts, Model Card, Datasheet |
| 13 | Advanced methods — XGBoost, Stacking, DeLong, temporal out-of-time validation |

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run streamlit_covid_predictor.py
```

The app loads the three pre-fitted artefacts from `analysis_output/models/` and applies the **identical `impute → scale → predict_proba` chain used during training**, so its outputs match the deployed model. SHAP explanations, odds ratios and the nomogram all describe that same Logistic Regression.

## Running the Analysis Pipeline

The dataset (`covid.csv`, Mexican Government COVID-19 surveillance) is **not committed** (size / data governance). Download it from
[Kaggle](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset), then:

```bash
pip install -r requirements-analysis.txt
python covid_analysis_full.py --data covid.csv
```

All experiments use `random_state = 42` for reproducibility.

## Continuous Integration (GitLab CI)

The repository ships a `.gitlab-ci.yml` that installs dependencies and runs the unit-test suite on every push and merge request:

```yaml
# .gitlab-ci.yml
stages:
  - test

unit-tests:
  stage: test
  image: python:3.11-slim
  cache:
    key: "$CI_COMMIT_REF_SLUG"
    paths:
      - .cache/pip
  variables:
    PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  before_script:
    - python -m pip install --upgrade pip
    - pip install -r requirements-analysis.txt
    - pip install pytest
  script:
    - pytest test_analysis.py -v
  rules:
    - if: '$CI_PIPELINE_SOURCE == "push"'
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
```

The pipeline validates the 44 statistical-helper unit tests only — it does **not** retrain models or require the dataset (`covid.csv` is not committed), so it runs quickly on the shared GitLab runners.

---

## Responsible AI

- **Model Card** — `analysis_output/advanced/MODEL_CARD.md` (Mitchell et al., 2019)
- **Datasheet for Datasets** — `analysis_output/advanced/DATASHEET.md` (Gebru et al., 2021)
- Leakage-free design (post-admission variables such as intubation/ICU deliberately excluded)
- Eight-subgroup fairness audit (age, sex, diabetes) with group-specific threshold mitigation
- In-app disclaimers framing every output as a mortality-risk estimate, not a validated clinical decision
- Public, de-identified dataset

---

## Limitations

Single-country (Mexico), single early-pandemic window (Mar–Nov 2020, pre-vaccine and pre-major-variants); nine admission-time features only (no labs, vitals or variant data); outcome is in-hospital mortality, **not** Long COVID. External, multi-site validation is required before any clinical use.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for full terms.

The **code** (analysis pipeline, Streamlit app) is MIT-licensed and free to reuse with attribution. The **dataset** (Mexican Government COVID-19 surveillance) is **not** covered by this license and is not distributed here; obtain it from [Kaggle](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset) under its own terms. This software is provided for research and education only and is **not** a validated medical device.

```
MIT License

Copyright (c) 2026 Durga Prasad Narsing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
