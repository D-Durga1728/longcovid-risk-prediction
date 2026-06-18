# Predictive Modelling for Personalized Long COVID Risk Assessment

MSc Computing (Data Analytics) practicum, Dublin City University.

**Student:** Durga Prasad Narsing (A00050350)
**Supervisors:** Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)

---

## What this project is
An end-to-end, responsible-ML pipeline that predicts **COVID-19 in-hospital mortality from
acute-phase clinical data**, used as a **proxy for severe outcome**. It benchmarks an
interpretable model against complex alternatives and adds calibration, fairness, explainability
and decision-analysis on top.

> **Important caveat.** The dataset (Mexican Government COVID-19, Kaggle) has acute-phase data
> and in-hospital mortality, but **no Long COVID / PASC follow-up labels**. Mortality is therefore
> a proxy for severe outcome; sequela-specific Long COVID prediction is future work. This is a
> research/education prototype, not a validated clinical device.

## Headline results
- Deployed model: **calibrated, tuned Logistic Regression** (C=0.01).
- Test AUC **0.888** (95% CI 0.884-0.892); temporal out-of-time AUC **0.891**.
- Well-calibrated after isotonic calibration: Brier 0.078, ECE 0.002 (count-weighted).
- DeLong test: no model (RF, GB, XGBoost, Stacking) significantly beats LR, so the interpretable
  model is deployed.
- Fairness audited across age, sex and comorbidity (discrimination + calibration), with
  group-specific threshold mitigation (TPR disparity 0.60 to 0.045).
- Trained on 220,218 confirmed cases; 9 acute-phase features; mortality rate 12.31%.

## Repository structure
```
covid_analysis_full.py        Main 13-phase analysis pipeline
advanced_methods.py           Advanced methods (DeLong, calibration, fairness, SHAP, nomogram, ...)
test_analysis.py              Unit tests (pytest)
streamlit_covid_predictor.py  Streamlit risk-calculator web app
covid.csv                     Dataset (Mexican Government COVID-19, Kaggle)
requirements-analysis.txt     Dependencies for the analysis pipeline
requirements.txt              Lean dependencies for the deployed app
analysis_output/              Generated results: CSVs, plots, models, MODEL_CARD.md, DATASHEET.md
RESEARCH_AND_REPORT_PACKAGE.md   Consolidated research + report handoff
RESEARCH_FRAMING.md           Research questions, methodology justification, critical analysis
RELATED_WORK_COMPARISON.md    Prior-work comparison (what they did, limits, how this advances)
REPORT_HANDOFF.md             Methodology + results + numbers for the report
REFERENCES_TABLE.md           The 27 references with links and where each is used
```

## How to run

### Analysis pipeline
```bash
pip install -r requirements-analysis.txt
python covid_analysis_full.py --data covid.csv
```
Runs all phases and writes results to `analysis_output/`. Runtime is a few minutes.

### Unit tests
```bash
python -m pytest test_analysis.py -v
```

### Streamlit app
```bash
pip install -r requirements.txt
python -m streamlit run streamlit_covid_predictor.py
```
The app loads the persisted model + scaler + imputer from `analysis_output/models/`.

## Methods (summary)
Leakage-controlled preprocessing (sentinel decoding, median imputation, train-only fit),
cost-sensitive learning for class imbalance (SMOTE evaluated and rejected), hyperparameter
tuning, AUC with bootstrap confidence intervals and DeLong tests, isotonic calibration,
threshold optimisation, decision-curve analysis, SHAP (global + per-patient), Odds Ratios with
confidence intervals, a points-based nomogram, fairness auditing with mitigation, temporal
validation, and responsible-AI documentation (Model Card and Datasheet).

## Responsible AI
- `analysis_output/advanced/MODEL_CARD.md` (Mitchell et al. 2019)
- `analysis_output/advanced/DATASHEET.md` (Gebru et al. 2021)

## References
27 references (11 clinical + 16 methodology). See `REFERENCES_TABLE.md` for the full Harvard list
with links and where each is used in the project.

## Disclaimer
Research and educational prototype only. Predicts a mortality proxy, not a clinical diagnosis.
Not a validated medical device and not for individual treatment decisions.
