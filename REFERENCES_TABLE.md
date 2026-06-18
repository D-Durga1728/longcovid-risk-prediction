# References Used in the Project (27)

11 clinical/applied + 16 methodology. Every entry is genuinely used: implemented in the code,
or cited to justify a design/feature decision. Verify each paper's specific figures against the
original before final submission.

## A. Clinical / applied ML - problem framing & feature selection (11)

| # | Paper | What it is | Official link | Where & why used |
|---|---|---|---|---|
| 1 | Bjerre et al. (2024) | AI/ML vs classical regression on COVID health databases | https://doi.org/10.1016/j.gloepi.2024.100168 | Justifies LR-vs-ensemble comparison; supports the "interpretable wins" conclusion |
| 2 | Cordelli et al. (2024) | ML predicts pulmonary long-COVID sequelae | https://doi.org/10.1186/s12911-024-03204-w | Precedent for ML on COVID sequelae |
| 3 | Dozier et al. (2024) | Super Learner ensemble for Long COVID | https://doi.org/10.2196/53322 | Justifies ensemble + stacking design |
| 4 | Guzman-Esquivel et al. (2023) | Acute-phase predictors of Long COVID | https://doi.org/10.3390/healthcare11020197 | Feature selection; core proposal paper |
| 5 | Jeffrey et al. (2024) | Regression risk model for long COVID (Scotland) | https://doi.org/10.1177/01410768241297833 | LR model + validation emphasis |
| 6 | Lee et al. (2025) | ML for PASC prediction in primary care | https://doi.org/10.1186/s12916-025-04050-w | Closest precedent for ML risk prediction |
| 7 | Mahmud et al. (2021) | Post-COVID syndrome predictors (cohort) | https://doi.org/10.1371/journal.pone.0249644 | Feature selection (comorbidity/severity) |
| 8 | Menezes et al. (2022) | Acute severity predicts severe Long COVID | https://doi.org/10.7759/cureus.29826 | Severity/mortality-proxy rationale |
| 9 | Shakhovska et al. (2022) | Hybrid ensemble for post-COVID prediction | https://doi.org/10.3934/mbe.2022285 | Justifies hybrid/stacking ensemble |
| 10 | Teeuw et al. (2023) | Post-COVID cardiovascular risk model | https://doi.org/10.1093/ehjopen/oead101 | Feature selection + prediction-model framing |
| 11 | Zang et al. (2024) | EHR predictors of Long COVID (RECOVER) | Communications Medicine - verify DOI | Feature selection from large EHR study |

## B. Methodology - implemented in the code (16)

| # | Paper | What it is | Official link | Where & why used |
|---|---|---|---|---|
| 12 | Antonini et al. (2024) | Applied SHAP interpretability | https://doi.org/10.1016/j.acags.2024.100178 | SHAP analysis (Phase 5) |
| 13 | Cetin & Yildiz (2022) | Review of data-preprocessing techniques | https://doi.org/10.5505/pajes.2021.62687 | Missing-data handling, imputation, scaling |
| 14 | Chen & Guestrin (2016) | XGBoost algorithm | https://doi.org/10.1145/2939672.2939785 | XGBoost benchmark model |
| 15 | DeLong et al. (1988) | Test comparing correlated AUCs | https://doi.org/10.2307/2531595 | DeLong test (model selection) |
| 16 | Efron & Tibshirani (1993) | The bootstrap | https://doi.org/10.1201/9780429246593 | Bootstrap 95% CIs for AUC |
| 17 | Guo et al. (2017) | Calibration of classifiers / ECE | https://proceedings.mlr.press/v70/guo17a.html | Expected Calibration Error |
| 18 | Hardt et al. (2016) | Equalized-odds fairness | https://arxiv.org/abs/1610.02413 | Fairness mitigation (group thresholds) |
| 19 | Lundberg & Lee (2017) | SHAP framework | https://arxiv.org/abs/1705.07874 | Global + local SHAP |
| 20 | Lundberg et al. (2020) | Tree SHAP | https://doi.org/10.1038/s42256-019-0138-9 | Tree SHAP + local SHAP export |
| 21 | Mitchell et al. (2019) | Model Cards | https://doi.org/10.1145/3287560.3287596 | Auto-generated MODEL_CARD.md |
| 22 | Niculescu-Mizil & Caruana (2005) | Predicting good probabilities | https://doi.org/10.1145/1102351.1102430 | Calibration approach |
| 23 | Platt (1999) | Platt scaling | https://www.microsoft.com/en-us/research/publication/probabilistic-outputs-for-support-vector-machines-and-comparisons-to-regularized-likelihood-methods/ | Calibration method |
| 24 | Steyerberg (2019) | Clinical Prediction Models (textbook) | https://doi.org/10.1007/978-3-030-16399-0 | Nomogram, calibration, temporal validation |
| 25 | Sun & Xu (2014) | Fast DeLong algorithm | https://doi.org/10.1109/LSP.2014.2337313 | Fast DeLong implementation |
| 26 | Vickers & Elkin (2006) | Decision Curve Analysis | https://doi.org/10.1177/0272989X06295361 | Decision-curve analysis |
| 27 | Wolpert (1992) | Stacked generalization | https://doi.org/10.1016/S0893-6080(05)80023-1 | Stacking ensemble |

**Total: 27** (11 clinical + 16 methodology).
Note: confirm the Zang (2024) DOI; standardise "Guzman-Esquivel" (not "Esquivel") throughout.
