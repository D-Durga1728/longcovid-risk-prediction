# Research & Report Master Package

One self-contained handoff for the research / report-writing chat. Three parts:
(1) what the project set out to do, (2) what was actually delivered and the honest gap,
(3) what the supervisor asked for as a research project (prior work, limitations, how this
advances, contribution, critical analysis). All numbers are verified from the final pipeline run.

**Project:** Predictive Modelling for Personalized Long COVID Risk Assessment and Prevention
**Student:** Durga Prasad Narsing (A00050350), MSc Computing (Data Analytics), DCU
**Supervisors:** Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)
**Dataset:** Mexican Government COVID-19 dataset (Kaggle): 566,602 records, 23 columns

---

## PART 1 - WHAT I SET OUT TO DO (the proposal)

**Aim (as proposed):** build highly interpretable ML models to predict an individual's risk of
specific Long COVID sequelae (Cardiac, Fatigue, Brain Fog) from acute-phase clinical data, and
translate the prediction into preventative advice.

**Proposal research questions:**
- Main: how effective are interpretable ML models at predicting specific Long COVID sequelae
  from acute-phase data?
- Secondary 1: how much does each acute-phase factor increase future risk (risk quantification)?
- Secondary 2: how do simple, transparent models compare in accuracy to complex black-box models?

**Promised deliverables:** three separate Logistic Regression models (one per sequela), Odds
Ratios (e^beta) for risk quantification, and a personalised advice engine including medication
suggestions.

**Proposal reference set (5):** Guzman-Esquivel 2023, Menezes 2022, Mahmud 2021, Lee 2025,
Cordelli 2024.

---

## PART 2 - WHAT WAS ACTUALLY DELIVERED (and the honest gap)

### 2.1 The data reality (must be stated up front in the report)
The Mexican Government dataset contains acute-phase variables and in-hospital **mortality**,
but **no Long COVID / PASC follow-up labels** and no per-sequela outcomes. So the proposal's
sequela-specific models cannot be trained on this data. The project therefore predicts
**mortality as a proxy for severe outcome**, and sequela-specific prediction becomes future work.

### 2.2 Proposal vs delivery (be transparent)
| Promised | Delivered | Why |
|---|---|---|
| 3 sequela models (Cardiac/Fatigue/Brain Fog) | 1 mortality-proxy model | No sequela labels in the data |
| Odds Ratios (e^beta) | Delivered: OR with 95% CI + p-values (statsmodels) | Fulfilled, and strengthened |
| Medication advice engine | Not built; replaced by risk tiers + referral urgency | Clinical-liability + data scope |
| Interpretable vs black-box comparison | Delivered: LR vs RF/GB/XGBoost/Stacking + DeLong test | Fulfilled |

### 2.3 What got built (final state)
- A 13-phase analysis pipeline (`covid_analysis_full.py` + `advanced_methods.py`), 14 unit tests.
- A deployed model: **calibrated, tuned Logistic Regression**, persisted to joblib.
- A Streamlit web app (risk calculator + explanations + nomogram).
- Responsible-AI artifacts: Model Card and Datasheet.
- Reframed research questions (Part 3.1).

---

## PART 3 - WHAT WE GOT (verified results)

### 3.1 Reframed research questions
- RQ1 (Predictability): can acute-phase variables predict COVID-19 mortality with useful
  discrimination AND calibration?
- RQ2 (Interpretability vs complexity): do complex models significantly beat an interpretable LR?
- RQ3 (Fairness & reliability): is the model equitable and well-calibrated across subgroups?
- RQ4 (Actionability): can the predictions be made interpretable and clinically usable?

### 3.2 Data & preprocessing
- Modelling set: 220,218 confirmed cases; mortality 12.31% (1:7 imbalance).
- 9 acute-phase features: age, sex, diabetes, hypertension, cardiovascular, pneumonia, obesity,
  asthma, copd. `intubed` and `icu` excluded as target leakage (post-admission outcomes).
- Sentinels 97/98/99 decoded to NaN then median-imputed; leakage-free (split first, fit
  imputer + scaler on train only; CV uses a Pipeline). EPV approximately 3,011 (>> 10).

### 3.3 Model performance (test set, n=44,044; AUC with bootstrap 95% CI)
| Model | AUC | 95% CI |
|---|---|---|
| Logistic Regression | 0.888 | 0.884-0.892 |
| Random Forest | 0.866 | 0.861-0.871 |
| Gradient Boosting | 0.888 | 0.884-0.892 |
| Ensemble | 0.887 | 0.883-0.891 |
| XGBoost | 0.888 | 0.884-0.892 |
| Stacking | 0.888 | 0.884-0.892 |
| **Primary (calibrated tuned LR, deployed)** | **0.888** | **0.884-0.892** |

- DeLong test vs LR: RF significantly worse (p approximately 3e-42); GB, XGBoost, Stacking,
  Primary show no significant difference. Conclusion: the interpretable LR is non-inferior.
- 5-fold CV AUC: 0.891 +/- 0.002. Temporal (out-of-time) AUC: 0.891.
- Operating point (F1-optimal threshold 0.26): sensitivity 0.70, specificity 0.88, precision
  0.45, F1 0.55. (Youden's J threshold 0.10: sensitivity 0.88, specificity 0.76.)

### 3.4 Calibration
- Uncalibrated: ECE 0.195, Brier 0.135 (poorly calibrated).
- After isotonic calibration: ECE 0.002 (count-weighted), Brier 0.078 (well-calibrated).

### 3.5 Fairness (two lenses: discrimination AND calibration)
| Group | n | AUC | auc_status | ECE | Calibration |
|---|---|---|---|---|---|
| Age 18-30 | 6,290 | 0.891 | OK | 0.002 | well-calibrated |
| Age 30-50 | 19,713 | 0.874 | OK | 0.004 | well-calibrated |
| Age 50-65 | 11,189 | 0.799 | FLAG | 0.008 | well-calibrated |
| Age 65+ | 5,926 | 0.733 | FLAG | 0.015 | well-calibrated |
| Sex: Female | 19,961 | 0.901 | OK | 0.005 | well-calibrated |
| Sex: Male | 24,083 | 0.873 | OK | 0.003 | well-calibrated |
| With Diabetes | 7,160 | 0.776 | FLAG | 0.012 | well-calibrated |
| Without Diabetes | 36,704 | 0.899 | OK | 0.004 | well-calibrated |

- Three subgroups flagged on AUC (lower discrimination = restriction of range in high-risk
  groups), but every subgroup is well-calibrated (ECE < 0.015) so risk estimates stay reliable.
- Mitigation via group-specific thresholds: TPR disparity reduced 0.60 to 0.045.
- Gender gap approximately 2.8% (within the 5% threshold).

### 3.6 Imbalance handling
- Cost-sensitive learning: class_weight='balanced' (LR, RF), sample_weight (GB), scale_pos_weight
  (XGBoost). SMOTE was evaluated and rejected: identical AUC (0.888 vs 0.888) while fabricating
  132,810 synthetic patients.

### 3.7 Interpretability
- Odds Ratios (statsmodels, per +1 SD, 95% CI): pneumonia 2.65, age 2.33, diabetes 1.15,
  obesity 1.10, hypertension 1.09, copd 1.03 (all significant); cardiovascular 1.00 and asthma
  0.98 not significant; sex (female) 0.80 (protective).
- SHAP (label which view): Linear-LR pneumonia 39.9% / age 33.1% / sex 10.9%; Tree-XGBoost
  pneumonia 44.2% / age 33.0% / sex 8.5%.
- Per-patient (local) SHAP exported. Nomogram (points): pneumonia +24, age +6/decade,
  diabetes +5, copd +4, hypertension +3, obesity +3, cardiovascular +1, asthma -1, sex (female) -5.
- Decision Curve Analysis: model net benefit exceeds treat-all / treat-none.

### 3.8 Clinical economics (model-driven, calibrated)
- Flagged high-risk approximately 6.4%, precision approximately 56%; net savings approximately
  $102M to $310M across 30-90% adoption; ROI approximately 1,600-2,100%; payback approximately
  0.3 years. (Grounded in the model's actual precision, not a flat multiplier.)

---

## PART 4 - WHAT THE SUPERVISOR WANTS MORE (research dimension)

### 4.1 Contribution (state honestly)
The methods are established, so the contribution is NOT a new algorithm. It is:
1. Applied rigour: a complete, leakage-controlled, responsible-ML pipeline on a large real
   clinical dataset, integrating discrimination, calibration, fairness, explainability and
   decision-analysis in one workflow.
2. An empirical finding: a transparent, calibrated LR is statistically non-inferior to the
   black-box models here (DeLong p > 0.05), so interpretability costs nothing on this problem.
3. Gap-closing: it implements the exact practices the literature review found missing.

One-line: "A rigorous, responsible-ML benchmarking study showing an interpretable, calibrated
logistic regression matches black-box models for acute-phase COVID-19 mortality risk, with a
fairness and explainability framework that addresses gaps in the Long COVID prediction literature."

### 4.2 Prior work: what they did, limitations, how this advances
**Predictive-modelling comparators**
| Study | What they did | Limitation | How this advances |
|---|---|---|---|
| Jeffrey 2024 | Regression risk model for long COVID (Scotland) | Moderate discrimination; needs external validation | Adds calibration, per-subgroup fairness + mitigation, SHAP/OR/nomogram, decision curve, temporal validation |
| Bjerre 2024 | ML vs classical regression on COVID databases | Limited calibration/fairness depth | Confirms the finding with a formal DeLong test; adds the rigour layer |
| Dozier 2024 | Super Learner ensemble for Long COVID | Black box; calibration/interpretability under-emphasised | Shows transparent LR is non-inferior; adds explainability + calibration |
| Shakhovska 2022 | Hybrid ensemble for post-COVID prediction | Interpretability/fairness/calibration absent | Provides all three; benchmarks but deploys the interpretable model |
| Lee 2025 | ML (diagnoses + meds) for PASC, primary care | Limited fairness/calibration reporting | Adds responsible-ML layer + nomogram; honest about narrower features |
| Cordelli 2024 | ML for pulmonary sequelae, ~94% accuracy | Accuracy misleads under imbalance; single sequela | Uses AUC/PR/Brier/ECE; audits fairness/calibration |
| Teeuw 2023 | Post-COVID cardiovascular model | Single outcome; limited calibration/fairness | Broader evaluation framework |

**Risk-factor studies (not deployable models)**
| Study | What they did | Limitation | How this advances |
|---|---|---|---|
| Mahmud 2021 | Cohort; severity to fatigue/respiratory | Descriptive; no model | Operationalises predictors into a validated model |
| Menezes 2022 | Acute severity predicts severe Long COVID | Observational; no ML | Builds + validates a model; OR with CIs |
| Guzman-Esquivel 2023 | Acute-phase predictors | No deployable model | Deployable, explainable, calibrated tool |
| Zang 2024 | Large EHR predictors (RECOVER) | Less focus on deployed-model calibration/fairness | Deployed model with fairness + calibration |

**Field-wide gaps this work closes:** under-reported calibration; scant fairness/subgroup
analysis; feature leakage; black-box ML without interpretability; weak metrics (accuracy under
imbalance).

### 4.3 Where prior work is STRONGER (keep this; examiners test for it)
- Target fidelity: Lee, Cordelli, Dozier, Zang predict actual Long COVID/sequelae; this uses a
  mortality proxy (data limitation).
- Scale/generalisability: RECOVER (Zang) and NC3 (Dozier) are large multi-site US cohorts; this
  is single-country.
- External validation: only temporal here, not multi-site external.
- Feature richness: Lee uses medications + diagnoses; this has 9 acute-phase features.

### 4.4 Critical analysis (interpret WHY, do not just report)
- Why LR ties the ensembles: with 9 mostly-binary features the signal is largely linear and
  additive, so tree flexibility adds little. Consistent with Bjerre 2024.
- Why older/diabetic subgroups have lower AUC: restriction of range (homogeneous high-risk
  groups), not miscalibration; calibration holds (ECE < 0.015).
- Why calibration matters more than AUC for deployment: clinicians act on the probability, so a
  well-calibrated moderate-AUC model is more trustworthy than an over-confident high-AUC one.
- Why low precision is acceptable: at 12% prevalence, precision 0.45 is a 3.6x enrichment;
  the screening operating point favours recall.

### 4.5 Threats to validity (address head-on)
- Construct: mortality is a proxy for Long COVID (stated; future work on PASC-labelled data).
- External: single country/period; temporal validation done, multi-site is future work.
- Internal: strong (leakage controlled, stratified CV, fixed seeds, unit tests).
- Statistical: DeLong for AUC differences; Benjamini-Hochberg FDR on chi-square; EPV ~3,011.

---

## PART 5 - REFERENCES (27 used: 11 clinical + 16 methodology)
Clinical/applied: Bjerre 2024; Cordelli 2024; Dozier 2024; Guzman-Esquivel 2023; Jeffrey 2024;
Lee 2025; Mahmud 2021; Menezes 2022; Shakhovska 2022; Teeuw 2023; Zang 2024.
Methodology: Antonini 2024; Cetin & Yildiz 2022; Chen & Guestrin 2016; DeLong 1988;
Efron & Tibshirani 1993; Guo 2017; Hardt 2016; Lundberg & Lee 2017; Lundberg 2020; Mitchell 2019;
Niculescu-Mizil & Caruana 2005; Platt 1999; Steyerberg 2019; Sun & Xu 2014; Vickers & Elkin 2006;
Wolpert 1992. (Full Harvard list + links: see REFERENCES_TABLE.md.)

---

## PART 6 - FIXES / TO-DO BEFORE SUBMISSION
- Citation fixes: standardise "Guzman-Esquivel" (not "Esquivel"); add Zang 2024 and Cordelli 2024
  to the reference list (cited but currently missing).
- Cite Cetin & Yildiz 2022 (preprocessing) and Antonini 2024 (SHAP) in-text.
- App / model-card: correct gender claim from "<1%" to approximately 2.8% (within 5% threshold);
  add the diabetes subgroup flag to the fairness note.
- Verify each prior paper's exact figures against the original PDFs before final submission.

---

## PART 7 - DELIVERABLE FILES (where the detail lives)
- `RESEARCH_FRAMING.md` - RQs, methodology justification, critical analysis, viva Q&A.
- `RELATED_WORK_COMPARISON.md` - full prior-work comparison tables.
- `REPORT_HANDOFF.md` - methodology + results + numbers for the report chapters.
- `REFERENCES_TABLE.md` - the 27 references with links and where each is used.
- `STREAMLIT_HANDOFF.md` - app build spec.
- `analysis_output/advanced/MODEL_CARD.md` and `DATASHEET.md` - responsible-AI artifacts.
- `analysis_output/` - all result CSVs and plots (figures for the report).
