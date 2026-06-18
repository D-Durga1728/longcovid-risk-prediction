# Related Work — What Prior Studies Did, Their Limitations, and How This Work Advances

A critical comparison for the literature-review / discussion chapters. Honest in both
directions: where this work improves on prior studies, and where prior studies remain
stronger. Verify each paper's specific figures against the original before final submission.

**Project:** Predictive Modelling for Personalized Long COVID Risk Assessment
**Student:** Durga Prasad Narsing (A00050350) — MSc Computing (Data Analytics), DCU

---

## 1. The honest top-line (state this up front)
Several prior studies predict **actual** Long COVID / PASC or specific sequelae because they
had follow-up labels this dataset lacks. This work predicts **mortality as a proxy** for
severe outcome. So the advance here is **NOT** "better Long COVID prediction" — it is a
**methodological rigour layer** (calibration, fairness, leakage control, interpretability,
reproducibility) that most of those studies under-report, plus an empirical
interpretability-vs-complexity finding. Frame the comparison that way and it is defensible.

---

## 2. Comparison table — predictive-modelling studies (direct comparators)

| Study | What they did | Key limitations | How THIS work advances on it |
|---|---|---|---|
| **Jeffrey et al. (2024)** — JRSM | Derived & validated a regression risk model for long COVID, population cohort (Scotland) | Only "moderate" discrimination; calls for external validation; limited interpretability/fairness reporting | Adds calibration before/after, per-subgroup fairness + mitigation, SHAP/odds-ratios/nomogram, decision-curve, temporal validation; DeLong-tested model selection |
| **Bjerre et al. (2024)** — Global Epidemiology | Compared AI/ML vs classical regression on large COVID health databases | Reports the comparison but limited calibration/fairness/explainability depth | Independently **confirms their finding** (ML ≈ regression) with a formal **DeLong significance test**, and adds the rigour layer they don't |
| **Dozier et al. (2024)** — JMIR | Super Learner ensemble for Long COVID (national NC3 cohort) | Ensemble is a **black box**; interpretability and calibration under-emphasised | Shows a transparent LR is statistically non-inferior here; adds SHAP, odds ratios, nomogram and calibration so the model is explainable, not just accurate |
| **Shakhovska et al. (2022)** — Math. Biosci. Eng. | Hybrid ensemble for severity / post-COVID prediction | Interpretability, fairness, and calibration largely absent | Provides explainability + fairness audit + calibration; benchmarks ensemble/stacking but deploys the interpretable model |
| **Lee et al. (2025)** — BMC Medicine | ML using diagnoses + medications to predict PASC in primary care | Richer features (meds/diagnoses) but limited fairness/calibration reporting; different care setting | Adds the responsible-ML layer (fairness + calibration + leakage control) and a clinician-facing nomogram; honest about its narrower feature set as a limitation |
| **Cordelli et al. (2024)** — BMC MIDM | ML predicts pulmonary long-COVID sequelae; high **accuracy (~94%)** | **Accuracy** is a weak metric under class imbalance; single sequela; calibration/fairness not foregrounded | Uses imbalance-appropriate metrics (AUC, PR, Brier, ECE) instead of accuracy; audits fairness/calibration; explains why accuracy misleads |
| **Teeuw et al. (2023)** — EHJ Open | Multivariable model for post-COVID cardiovascular disease | Single outcome; calibration/fairness reporting limited | Broader evaluation framework (calibration + fairness + decision curve + temporal validation) on the mortality target |

## 3. Comparison table — risk-factor / epidemiological studies (not deployable models)

| Study | What they did | Key limitations | How THIS work advances on it |
|---|---|---|---|
| **Mahmud et al. (2021)** — PLOS ONE | Prospective cohort; severe acute infection → lasting fatigue/respiratory issues (Bangladesh) | Single-centre, **descriptive**; no predictive model or validation | Operationalises identified risk factors into a validated, calibrated predictive model |
| **Menezes et al. (2022)** — Cureus | Observational; acute severity predicts severe Long COVID | Observational; **no ML model**; smaller sample | Builds and validates a model; quantifies each factor with odds ratios + CIs |
| **Guzman-Esquivel et al. (2023)** — Healthcare | Cohort; tachycardia, myalgia, antibiotics as predictors | Identifies predictors; **no deployable model / calibration** | Turns predictors into a deployable, explainable, calibrated tool |
| **Zang et al. (2024)** — Comms Medicine | Large EHR (RECOVER) predictors of Long COVID (obesity, diabetes, pulmonary) | Predictor identification at scale; less focus on calibration/fairness of a deployed model | Provides a deployed model with fairness + calibration auditing on those predictors |

---

## 4. The recurring field-wide limitations (your literature review's own "gaps")
These appear across the studies above and are the strongest part of your positioning:
1. **Calibration under-reported** — most report discrimination (AUC/accuracy) but not whether
   predicted probabilities are accurate.
2. **Little fairness / subgroup analysis** — few audit performance across age, sex, comorbidity.
3. **Feature leakage not controlled** — treatment/outcome variables sometimes included.
4. **Black-box ML without interpretability** — ensembles/Super Learners deployed without
   patient-level explanation.
5. **Inconsistent Long COVID definitions / weak metrics** — e.g., accuracy under imbalance.

### How this work closes each (the table that wins marks)
| Field gap | This work |
|---|---|
| Calibration under-reported | Measured (ECE 0.195) and corrected (→0.002); audited per subgroup |
| Scant fairness analysis | 8-subgroup audit (AUC **and** calibration) + threshold mitigation (TPR disparity 0.60→0.045) |
| Feature leakage | intubed/icu excluded; split-then-fit leakage-free pipeline |
| No interpretability | SHAP (global + per-patient), odds ratios with CIs, nomogram, DeLong-justified model choice |
| Weak/imbalanced metrics | AUC + bootstrap CIs + Brier + ECE + decision-curve; not accuracy |
| Reproducibility/governance | Unit tests, persisted models, model card + datasheet |

---

## 5. Be honest — where prior work is STRONGER (say this; examiners test for it)
- **Target fidelity:** Lee (2025), Cordelli (2024), Dozier (2024), Zang (2024) predict **actual
  Long COVID / sequelae**; this work uses a **mortality proxy** (data limitation).
- **Scale / generalisability:** RECOVER (Zang) and the NC3 cohort (Dozier) are large multi-site
  US collaboratives; this is a **single-country** dataset.
- **External validation:** some prior models pursue external/temporal-geographic validation; this
  work has **temporal (out-of-time) validation only**, not multi-site external.
- **Feature richness:** Lee (2025) uses medications + diagnoses; this work has **9 acute-phase
  features** only.

Stating these makes your "how we're better" claim credible rather than promotional.

---

## 6. Positioning statement (drop into the discussion)
> *"Prior Long COVID prediction studies fall into two groups: epidemiological works that
> identify risk factors without a deployable model (Mahmud 2021; Menezes 2022; Guzman-Esquivel
> 2023), and ML works that achieve strong discrimination but under-report calibration, fairness
> and interpretability, sometimes as black boxes (Dozier 2024; Shakhovska 2022; Cordelli 2024).
> This study contributes a methodological rigour layer largely missing from both: leakage
> control, probability calibration, subgroup fairness and calibration auditing with mitigation,
> patient-level explainability, decision-analytic evaluation, and reproducible responsible-AI
> documentation. Its empirical finding — that an interpretable, calibrated logistic regression is
> statistically non-inferior to RF, gradient boosting, XGBoost and stacking (DeLong, p > 0.05) —
> independently corroborates Bjerre et al. (2024) on a separate dataset. The principal limitation
> is the use of in-hospital mortality as a proxy for severe outcome, since the dataset lacks
> post-acute follow-up labels; direct sequelae prediction on PASC-labelled data is identified as
> future work."*

---

## 7. To do before submission
- Verify each paper's **specific numbers** (AUC/accuracy, sample sizes, exact limitations) from
  the original PDFs — the rows above are at the level your literature review supports.
- Add **Zang (2024)** and **Cordelli (2024)** to the reference list (flagged earlier as missing).
- Keep the "where prior work is stronger" section (§5) in the report — do not delete it.
