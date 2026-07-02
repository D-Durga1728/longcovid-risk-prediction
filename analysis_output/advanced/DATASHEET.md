# Datasheet - COVID-19 Dataset (Mexican Government, via Kaggle)

*Format: Gebru et al. (2021), "Datasheets for Datasets".*
**Compiled by:** Durga Prasad Narsing (A00050350), MSc Computing (Data Analytics), DCU
**Supervisors:** Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)
**Date:** 2026-07-02

## Motivation
- **Purpose:** Epidemiological surveillance of COVID-19 cases reported by the Mexican
  Federal Government. Re-used here to model mortality risk from acute-phase data as a
  proxy for severe adverse outcomes.
- **Created by:** Mexican Government (Secretaría de Salud); redistributed on Kaggle.

## Composition
- **Instances:** 566,602 patient records; **220,218** confirmed-positive used for modelling.
- **Each instance:** one patient with demographics, comorbidities, severity markers, and outcome (`date_died`).
- **Features used (9):** age, sex, diabetes, hypertension, cardiovascular, pneumonia, obesity, asthma, copd.
- **Label:** mortality (derived from `date_died`; `9999-99-99` = survived). Positive rate 12.31%.
- **Events-per-variable (EPV):** 3,011 (27,102 events / 9 predictors) - far above the ≥10 guideline (Riley et al. 2020), indicating ample data and low overfitting risk.
- **Missing data:** binary fields use 97/98/99 as "unknown" sentinels (~0.35–0.39% per comorbidity); decoded to NaN and median-imputed.

## Collection process
- Routinely collected clinical/administrative surveillance data at point of care; not collected for ML.
- Acute-phase variables recorded at/around initial COVID-19 presentation.

## Preprocessing / cleaning
- Removed invalid ages (≤0 or >120). Encoded 1=Yes/2=No → 1/0; sentinels → NaN → median imputation.
- Excluded `intubed` and `icu` (post-admission treatment outcomes → target leakage).
- Standardised features; imputer + scaler fit on the training split only.

## Uses
- **Used for:** mortality-risk prediction, fairness/calibration auditing, explainability.
- **Should NOT be used for:** inferring Long COVID/PASC sequelae (no follow-up labels), or
  individual clinical decisions without oversight.

## Distribution & licensing
- Publicly available via Kaggle (Mexican Government open data). Verify the dataset licence
  before redistribution.

## Maintenance
- Static snapshot; not actively maintained by this project. No personally identifying
  information is used in modelling.

## Ethical considerations
- Single-country data → limited generalisability; older age groups (50+) show weaker AUC
  (audited and partially mitigated). Mortality is a proxy for severe outcomes.
