# Research Framing, Methodology Justification & Critical Analysis

The academic backbone of the practicum: research questions, contribution, justification of
every methodological choice, critical discussion, literature positioning, and viva prep.
Grounded in the built pipeline and the 27 cited references. Share with the report chat /
supervisor.

**Project:** Predictive Modelling for Personalized Long COVID Risk Assessment and Prevention
**Student:** Durga Prasad Narsing (A00050350) — MSc Computing (Data Analytics), DCU
**Supervisors:** Dr Martin Crane & Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)

---

## 1. RESEARCH QUESTIONS & CONTRIBUTION

### 1.1 Honest reframe (essential)
The proposal asked about predicting Long COVID **sequelae** (cardiac, fatigue, brain fog).
The dataset (Mexican Government COVID-19) contains acute-phase data and **in-hospital
mortality only** — no post-acute follow-up labels. The research questions are therefore
restated around mortality as a **proxy** for severe outcome. State this transparently; it
is a data constraint, not a flaw in the work.

### 1.2 Research questions
- **RQ1 (Predictability).** Can acute-phase clinical variables predict COVID-19 in-hospital
  mortality with clinically useful *discrimination and calibration*?
- **RQ2 (Interpretability vs complexity).** Do complex models (RF, GB, XGBoost, stacking)
  significantly outperform an interpretable logistic regression for this task?
- **RQ3 (Fairness & reliability).** Is performance equitable and well-calibrated across
  demographic and comorbidity subgroups?
- **RQ4 (Actionability).** How can predictions be made interpretable and clinically usable?

### 1.3 Contribution (state it honestly)
The methods used are all established — the contribution is **not a new algorithm**. It is:
1. **Applied rigour:** a complete, leakage-controlled, responsible-ML pipeline on a large
   (220k) real clinical dataset, integrating discrimination, calibration, fairness,
   explainability and decision-analytic evaluation in one workflow.
2. **A concrete empirical finding:** on these acute-phase features, a **transparent logistic
   regression is statistically non-inferior** to RF/GB/XGBoost/stacking (DeLong p > 0.05) —
   so interpretability costs nothing here. This is the headline research result and matters
   for clinical adoption.
3. **Gap-closing:** the pipeline addresses the exact weaknesses the literature review itself
   identified (under-reported calibration, scant fairness auditing, feature leakage, missing
   explainability). It demonstrates *how* those gaps should be closed in practice.

> One-line contribution statement: *"A rigorous, responsible-ML benchmarking study showing
> that an interpretable, calibrated logistic regression matches black-box models for
> acute-phase COVID-19 mortality risk, with a fairness and explainability framework that
> addresses gaps in the existing Long COVID prediction literature."*

---

## 2. METHODOLOGY JUSTIFICATION (choice → why → citation)

| Choice | Justification | Reference |
|---|---|---|
| Mortality as target/proxy | Only acute-phase + mortality available; no PASC labels | (data limitation; future work cites Lee 2025, Dozier 2024) |
| 9 acute-phase features; exclude intubed/icu | intubed/icu are post-admission treatment outcomes → target leakage; not available at decision time | Bjerre 2024; general leakage principle |
| Sentinel decoding (97/98/99→NaN) + median imputation | Avoids mislabelling "unknown" as "No"; documented missingness | Cetin & Yıldız 2022 |
| Split-then-fit (imputer/scaler on train only); Pipeline CV | Prevents information leakage from test/validation folds | Steyerberg 2019 |
| class_weight / sample_weight / scale_pos_weight (not SMOTE) | Cost-sensitive learning rebalances the 1:7 ratio without fabricating patients; SMOTE tested, gave identical AUC (0.888) + 132,810 synthetic rows | Chawla 2002 (SMOTE, considered); empirical comparison |
| Logistic Regression as deployed model | Interpretable (odds ratios), and DeLong shows no model significantly beats it | Bjerre 2024; DeLong 1988 |
| RF / GB / XGBoost / stacking as benchmarks | Establish whether complexity buys discrimination | Chen & Guestrin 2016; Wolpert 1992; Dozier 2024; Shakhovska 2022 |
| Hyperparameter tuning via GridSearchCV (leakage-free) | Avoids over-fitting the test set; tuned C used in deployment | standard practice |
| AUC + bootstrap CI + DeLong (not accuracy) | Accuracy is misleading at 12% prevalence; AUC unaffected by base rate; DeLong tests significance of differences | DeLong 1988; Efron & Tibshirani 1993 |
| Isotonic calibration | Measured poor calibration (ECE 0.195) and corrected it (→0.002); calibrated probabilities are clinically interpretable | Platt 1999; Niculescu-Mizil & Caruana 2005; Guo 2017 |
| Threshold optimisation (Youden, F1) | 0.5 is wrong under imbalance; operating point chosen for screening | Youden 1950 |
| Decision Curve Analysis | Clinical net-benefit test beyond accuracy metrics | Vickers & Elkin 2006 |
| Fairness audit (AUC + calibration per subgroup) + mitigation | Detect *and* mitigate inequities; calibration is the right lens where AUC drops | Hardt et al. 2016 |
| Temporal (out-of-time) validation | Simulates prospective deployment; stronger than random hold-out | Steyerberg 2019; Riley 2020 (TRIPOD) |
| SHAP (linear + tree) + odds ratios + nomogram | Global + per-patient explanations; clinician-usable risk score | Lundberg & Lee 2017; Lundberg 2020; Steyerberg 2019 |
| Model card + datasheet | Responsible-AI documentation of model and data | Mitchell 2019; Gebru 2021 |

---

## 3. CRITICAL ANALYSIS & DISCUSSION (interpret WHY, don't just report)

### 3.1 Why does interpretable LR tie the black-box models?
With only **9 features**, most of them binary comorbidities plus age, the mortality signal is
largely **linear and additive** — there is little high-order interaction structure for tree
ensembles to exploit. Pneumonia and age dominate (SHAP ~40% + ~33%), and their effect is
monotonic. So the flexibility of RF/GB/XGBoost has nothing extra to model, and LR captures
essentially all the available signal. This is consistent with **Bjerre et al. (2024)**, who
found ML offered only marginal gains over regression on large COVID datasets. **Implication:**
for low-dimensional, mostly-linear clinical problems, complexity is not justified — favour the
interpretable model.

### 3.2 Why is subgroup AUC lower for older / diabetic patients?
**Restriction of range.** These are high-risk, homogeneous groups: within "65+" almost everyone
is elevated-risk, so there is less variation to rank, and AUC (a ranking metric) drops
mechanically — even for a good model. Crucially, **calibration holds** within every subgroup
(ECE < 0.015), so the risk *estimates* remain accurate. This separates two distinct fairness
questions (ranking vs probability accuracy) that are often conflated.

### 3.3 Why calibration matters more than AUC here
For a deployed risk tool, a clinician acts on the *probability* ("this patient is 30% risk"),
not the ranking. A well-calibrated model with modest AUC is more trustworthy than a high-AUC
model that is over/under-confident. This is why calibration was measured, corrected, and
audited per subgroup — a step most COVID prediction papers omit.

### 3.4 Why low precision is acceptable
At 12% prevalence, precision is structurally limited. Precision 0.45 still means a 3.6×
enrichment over base rate. As a screening tool the operating point favours recall (missing a
death is worse than an extra review). The precision/F1 figures are *correct framing*, not a
weakness — see REPORT_HANDOFF §2.3b.

### 3.5 Positioning vs prior work
- **Discrimination** (AUC 0.89) is comparable to or above the regression models in Jeffrey
  (2024) and the ML models in Bjerre (2024) and Dozier (2024).
- **What this work adds** that those papers under-report: calibration before/after, per-subgroup
  fairness *and* calibration, leakage control, decision-curve analysis, and a reproducible
  responsible-AI artifact set (model card + datasheet).

### 3.6 Threats to validity (address head-on)
- **Construct validity:** mortality ≠ Long COVID (proxy). Mitigation: state explicitly; future
  work on PASC-labelled EHR data.
- **External validity:** single country, single period. Mitigation: temporal validation done;
  external/multi-site validation is future work.
- **Internal validity:** strong — leakage controlled, stratified CV, deterministic seeds, tests.
- **Statistical:** AUC differences tested with DeLong; multiple chi-square tests FDR-corrected;
  EPV ≈ 3,011 (>> 10) so ample data, low overfitting risk.

---

## 4. LITERATURE POSITIONING

### 4.1 The gaps your own review identified → what you did
| Gap named in the literature review | How this work addresses it |
|---|---|
| Calibration metrics under-reported | Phase 7 + calibration improvement (ECE 0.195 → 0.002) |
| Scant fairness / subgroup analysis | 8-group audit (AUC + calibration) + threshold mitigation |
| Feature leakage not controlled | intubed/icu excluded; leakage-free pipeline |
| Limited explainability | SHAP (linear + tree), odds ratios, nomogram |
| No standardised outcome / heterogeneous definitions | single clearly-defined target (mortality), stated as proxy |

### 4.2 Where you sit among the clinical papers
- **Risk-factor papers** (Mahmud 2021, Menezes 2022, Guzman-Esquivel 2023, Teeuw 2023, Zang
  2024): you operationalise their identified predictors into a validated model.
- **ML papers** (Lee 2025, Cordelli 2024, Dozier 2024, Shakhovska 2022): you benchmark similar
  model families but add the rigour layer (calibration/fairness/explainability) they omit.
- **Methodology papers** (Bjerre 2024): your DeLong result empirically confirms their claim that
  ML gives little over regression — on an independent dataset.

### 4.3 Honest novelty statement
This is **applied/benchmarking research with responsible-ML rigour**, not methodological
invention. That is an appropriate and defensible contribution for an MSc Data Analytics
practicum — examiners reward correct, critical application over superficial novelty.

---

## 5. VIVA PREPARATION — likely questions & crisp answers

- **"Why predict mortality, not Long COVID?"** Data has no PASC labels; mortality is a
  validated proxy for severe outcome; sequelae prediction is future work needing follow-up data.
- **"Why is your best model the simplest?"** DeLong shows no model significantly beats LR; the
  signal is low-dimensional and additive, so complexity adds nothing — and LR is interpretable.
- **"Your precision is only 45% — isn't the model weak?"** No — precision is base-rate-bounded
  at 12% prevalence; 0.45 is a 3.6× enrichment; the screening operating point favours recall;
  AUC (0.89) is the unbiased discrimination measure.
- **"Three subgroups are flagged — is the model unfair?"** Lower *discrimination* (restriction of
  range), but well-*calibrated* in every subgroup (ECE < 0.015) and TPR disparity mitigated
  0.60 → 0.045. The risk estimates remain reliable.
- **"What's novel?"** Applied rigour and a responsible-ML framework that closes gaps the
  literature itself identifies; the empirical interpretability-vs-complexity result.
- **"Did you handle imbalance / leakage / overfitting?"** Cost-sensitive learning (+ SMOTE
  comparison); leakage-free split-then-fit pipeline; EPV ≈ 3,011; stratified CV + temporal
  validation + unit tests.
- **"Why should a clinician trust it?"** Calibrated probabilities, transparent odds ratios and
  nomogram, per-patient SHAP, decision-curve net benefit, and an honest model card + datasheet.

---

## 6. Suggested next research-writing steps
1. Lock the reframed RQs (§1.2) into the introduction.
2. Write the contribution statement (§1.3) into the abstract + conclusion.
3. Build the methodology chapter from the §2 table (one paragraph per row).
4. Make §3 the spine of Discussion (interpretation, not repetition of Results).
5. Use §4 to upgrade the literature review's "gaps" into "gaps this work closes."
6. Rehearse §5 before the viva.
