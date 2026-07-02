# Model Card - COVID-19 Mortality Risk Model

**Author:** Durga Prasad Narsing (A00050350) - MSc Computing (Data Analytics), DCU
**Supervisors:** Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)
**Date generated:** 2026-07-02

## Model details
- **Algorithm:** Primary deployed model: calibrated, tuned Logistic Regression (LR/RF/GB/XGBoost/Stacking benchmarked alongside) on standardised, median-imputed acute-phase features.
- **Features (9):** age, sex, diabetes, hypertension, cardiovascular, pneumonia, obesity, asthma, copd
- **Training / test split:** 176,174 / 44,044 (stratified, leakage-free: imputer + scaler fit on train only).

## Intended use
- **Purpose:** Stratify confirmed COVID-19 patients by risk of in-hospital mortality from acute-phase data, to support earlier clinical review.
- **Users:** Researchers / clinical decision-support prototyping. **Not** a validated clinical device.
- **Out of scope:** Individual treatment decisions without clinician oversight.

## ⚠️ Important caveat - target is a proxy
The dataset (Mexican Government, Kaggle) contains acute-phase data and in-hospital **mortality**, but **no Long COVID / PASC follow-up labels**. Mortality is therefore used as a *proxy* for severe adverse outcomes. Sequelae-specific Long COVID prediction is future work requiring EHR datasets with post-acute follow-up.

## Performance (held-out test set)
- **Ensemble AUC:** 0.8878 (95% CI 0.8838–0.8916)
- **Best single model (DeLong-tested):** no model significantly beats Logistic Regression - interpretable LR preferred
- **Temporal (out-of-time) AUC:** 0.8909047150814184

## Calibration
- **Before:** Brier 0.1353, ECE 0.1950
- **After isotonic calibration:** Brier 0.0778, ECE 0.0022

## Fairness
- Audited across age, sex and comorbidity subgroups (AUC) with a 5% disparity threshold.
- Age 50–65 and 65+ flagged on AUC; mitigated via group-specific thresholds, reducing TPR disparity from 0.603 to 0.045.

## Ethical considerations & limitations
- Single-country data; external/multi-site validation not performed.
- Mortality proxy (see caveat); unknown sentinels (97/98/99) treated as missing and median-imputed (~0.3% of comorbidity fields).
- Risk scores must be interpreted alongside clinical judgement.

## Top risk factors (Odds Ratios, per +1 SD)
| feature_name   |   odds_ratio |   or_ci_lower |   or_ci_upper |     p_value |
|:---------------|-------------:|--------------:|--------------:|------------:|
| pneumonia      |      2.64811 |       2.6077  |       2.68915 | 0           |
| age            |      2.32721 |       2.2804  |       2.37499 | 0           |
| diabetes       |      1.15012 |       1.13364 |       1.16683 | 1.77402e-80 |
| obesity        |      1.09734 |       1.07989 |       1.11507 | 6.78093e-30 |
| hypertension   |      1.08532 |       1.06837 |       1.10254 | 2.10527e-24 |

---
*Generated automatically by advanced_methods.generate_model_card(). Format: Mitchell et al. (2019), "Model Cards for Model Reporting".*
