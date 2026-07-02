# Predictive Modelling for Personalised Long COVID Risk Assessment and Prevention

**MSc Computing (Data Analytics) — Dublin City University**

| | |
|---|---|
| **Student** | Durga Prasad Narsing (A00050350) |
| **Supervisors** | Dr Martin Crane · Dr Tai Tan Mai |
| **School** | School of Computing, Dublin City University |

---

## Overview

This project builds an end-to-end, responsible-ML pipeline that predicts **COVID-19 in-hospital mortality from nine admission-time clinical features**, used as a clinically justified proxy for severe Long COVID outcome risk. Five model families are benchmarked, the winner is selected by formal statistical testing (DeLong), its probabilities are calibrated, and the full system — including fairness audit and multi-layer explanations — is exposed through a Streamlit web prototype.

> **Disclaimer.** The dataset (Mexican Government COVID-19 surveillance, Kaggle) contains acute-phase clinical data with in-hospital mortality labels but no PASC / Long COVID follow-up labels. Mortality is therefore a proxy for severe outcome. This is a research and education prototype — not a validated clinical device.

---

## Key Results

| Metric | Value |
|---|---|
| Dataset | 220,218 confirmed COVID-19 cases |
| Features | 9 admission-time (age, sex, 7 comorbidities) |
| Mortality rate | 12.31% (1:7 class imbalance) |
| Deployed model | Calibrated Logistic Regression (C=0.01, isotonic) |
| Test AUC | **0.888** (95% CI 0.884–0.892) |
| Temporal AUC | 0.891 (out-of-time validation) |
| Brier Score | 0.078 (post-calibration) |
| ECE (count-weighted) | 0.002 (post-calibration) |
| DeLong test | No ensemble significantly beats LR (p > 0.05) |
| TPR disparity | 0.603 → **0.045** (group-specific threshold mitigation) |
| Top risk drivers | Pneumonia 44.2% SHAP · Age 33.0% SHAP |

---

## Repository Structure

```
longcovid-risk-prediction/
│
├── streamlit_covid_predictor.py   Streamlit clinical web prototype (2,967 lines)
├── covid_analysis_full.py         Main 13-phase responsible-ML analysis pipeline
├── requirements.txt               App dependencies (Streamlit Cloud)
│
├── analysis_output/
│   ├── models/
│   │   ├── model_primary_deployed.joblib   Calibrated Logistic Regression
│   │   ├── imputer.joblib                  Median imputer (fit on train only)
│   │   └── scaler.joblib                   StandardScaler (fit on train only)
│   ├── advanced/                   CSVs: AUC CIs, calibration, DeLong, SHAP,
│   │                               fairness, nomogram, thresholds, model card
│   ├── fairness_calibration/       Fairness audit + calibration analysis CSVs
│   ├── visualizations/             35+ PNGs (EDA, ROC, calibration, SHAP, fairness)
│   └── ...                         Clinical explanations, feature analysis, cost-benefit
│
├── assets/
│   ├── dcu_logo.png
│   └── dcu_logo_white.png
│
└── .streamlit/config.toml         Streamlit theme config
```

---

## Pipeline Phases (`covid_analysis_full.py`)

| Phase | Description |
|---|---|
| 1 | Exploratory Data Analysis — 8 visualisations |
| 2 | Preprocessing — sentinel decoding, leakage-free imputation |
| 3 | Statistical Feature Validation — t-test, chi-square, Benjamini-Hochberg FDR |
| 4 | Model Training — LR, RF, GB, Stacking; cost-sensitive; GridSearchCV |
| 5 | SHAP Analysis — 5 visualisations, global + per-patient |
| 6 | Fairness Audit — 8 subgroups, AUC + ECE, threshold mitigation |
| 7 | Calibration — isotonic correction, Brier, ECE |
| 8 | Cost-Benefit Analysis |
| 9 | Clinical Explanations — nomogram, odds ratios |
| 10 | Feature Interactions — SHAP interaction values |
| 11 | Sensitivity Analysis — counterfactual what-if |
| 12 | Export — all CSVs, joblib models, MODEL_CARD, DATASHEET |
| 13 | Advanced Methods — XGBoost, Stacking, DeLong, DCA, temporal validation |

---

## Running the App Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_covid_predictor.py
```

The app loads the three pre-fitted artefacts from `analysis_output/models/` and applies the identical `impute → scale → predict_proba` chain used during training.

## Running the Analysis Pipeline

The analysis requires `covid.csv` (Mexican Government COVID-19 dataset, available on [Kaggle](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset)) — not included in this repo due to size.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn shap xgboost statsmodels joblib streamlit
python covid_analysis_full.py --data covid.csv
```

---

## Responsible AI

- **Model Card** — `analysis_output/advanced/MODEL_CARD.md` (Mitchell et al. 2019)
- **Datasheet for Datasets** — `analysis_output/advanced/DATASHEET.md` (Gebru et al. 2021)
- Leakage-free design (split-then-fit pipelines)
- Eight-subgroup fairness audit (age, sex, diabetes) with group-specific threshold mitigation
- Four-location disclaimer in the Streamlit prototype
- DCU F-REC ethics approved; public de-identified dataset; GDPR compliant
