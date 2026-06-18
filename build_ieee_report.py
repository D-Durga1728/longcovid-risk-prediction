"""
Builds IEEE-formatted final paper as a .docx file.
IEEE A4, double-column, Times New Roman, max 10 pages.
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

OUT = r'C:\Users\durga\OneDrive\Desktop\Practicum\DUR#12839\Report_IEEE.docx'

doc = Document()

# ─────────────────────────────────────────────
# PAGE SETUP  (A4, IEEE margins ~1.78 cm all)
# ─────────────────────────────────────────────
section = doc.sections[0]
section.page_height = Cm(29.7)
section.page_width  = Cm(21.0)
section.top_margin    = Cm(1.9)
section.bottom_margin = Cm(2.54)
section.left_margin   = Cm(1.78)
section.right_margin  = Cm(1.78)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def set_two_columns(section):
    """Add two-column layout to a section via raw XML."""
    sectPr = section._sectPr
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '284')   # ~0.5 cm gap in twentieths of a pt
    cols.set(qn('w:equalWidth'), '1')
    sectPr.insert(0, cols)

def set_one_column(section):
    """Single-column section."""
    sectPr = section._sectPr
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '1')
    sectPr.insert(0, cols)

def para(doc, text, font='Times New Roman', size=10, bold=False, italic=False,
         align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=3,
         first_indent=0, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after  = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Pt(first_indent)
    run = p.add_run(text)
    run.font.name  = font
    run.font.size  = Pt(size)
    run.bold       = bold
    run.italic     = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    return p

def heading(doc, number, title, size=10):
    """IEEE-style numbered section heading."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after  = Pt(3)
    run = p.add_run(f"{number}. {title.upper()}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.bold = True
    return p

def subheading(doc, letter, title):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after  = Pt(2)
    run = p.add_run(f"{letter}. {title}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    run.bold  = True
    run.italic = True
    return p

def body(doc, text, indent=True, space_after=3):
    return para(doc, text, size=10, first_indent=14 if indent else 0, space_after=space_after)

def ref_entry(doc, number, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after  = Pt(1)
    pf.left_indent  = Pt(18)
    pf.first_line_indent = Pt(-18)
    run = p.add_run(f"[{number}]  {text}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(8)
    return p

# ══════════════════════════════════════════════════════════════
#  SINGLE-COLUMN HEADER SECTION  (title, authors, abstract)
# ══════════════════════════════════════════════════════════════
set_one_column(section)

# Title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after  = Pt(6)
r = p.add_run("Predictive Modelling for Personalised Long COVID Risk\nAssessment and Prevention")
r.font.name = 'Times New Roman'
r.font.size = Pt(24)
r.bold = True

# Authors
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Durga Prasad Narsing")
r.font.name = 'Times New Roman'; r.font.size = Pt(11); r.bold = True

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("MSc Computing (Data Analytics), Dublin City University")
r.font.name = 'Times New Roman'; r.font.size = Pt(9)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Student No: A00050350  |  durgaprasadnarsing2@gmail.com")
r.font.name = 'Times New Roman'; r.font.size = Pt(9)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Supervisors: Dr Martin Crane & Dr Tai Tan Mai, School of Computing, DCU")
r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.italic = True

# Abstract label
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after  = Pt(0)
r = p.add_run("Abstract")
r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True; r.italic = True

# Abstract body
abstract_text = (
    "Post-COVID-19 Condition (Long COVID) affects millions globally, yet no validated "
    "predictive tool exists for personalised risk stratification at the point of acute "
    "presentation. This paper presents a rigorous, responsible-ML pipeline trained on "
    "220,218 confirmed COVID-19 cases from the Mexican Government surveillance dataset, "
    "predicting in-hospital mortality as a proxy for severe outcome risk. Nine "
    "admission-time features — age, sex, diabetes, hypertension, cardiovascular disease, "
    "pneumonia, obesity, asthma, and COPD — are modelled with Logistic Regression, "
    "Random Forest, Gradient Boosting, XGBoost, and a Stacking ensemble. A calibrated, "
    "tuned Logistic Regression achieves AUC 0.888 (95% CI 0.884–0.892) and isotonic "
    "calibration reduces Brier score from 0.135 to 0.078. DeLong tests confirm that no "
    "complex model significantly outperforms Logistic Regression (p > 0.05), supporting "
    "deployment of the interpretable model. Fairness audits across age, sex, and "
    "comorbidity subgroups reveal well-calibrated predictions in all groups despite "
    "lower discrimination in older and diabetic patients, attributable to restriction of "
    "range. SHAP analysis identifies pneumonia (44.2%) and age (33.0%) as dominant risk "
    "drivers. A Streamlit web prototype delivers per-patient risk scores, odds-ratio "
    "explanations, and a nomogram. The pipeline addresses gaps in the Long COVID "
    "prediction literature — calibration, fairness auditing, leakage control, and "
    "explainability — in a single reproducible workflow."
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after  = Pt(4)
r = p.add_run(abstract_text)
r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.italic = True

# Index Terms
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
p.paragraph_format.space_after = Pt(10)
r1 = p.add_run("Index Terms — ")
r1.font.name = 'Times New Roman'; r1.font.size = Pt(9); r1.bold = True; r1.italic = True
r2 = p.add_run(
    "Long COVID, PASC, COVID-19 mortality prediction, Logistic Regression, SHAP, "
    "calibration, fairness audit, responsible machine learning, clinical decision support."
)
r2.font.name = 'Times New Roman'; r2.font.size = Pt(9); r2.italic = True

# ══════════════════════════════════════════════════════════════
#  Switch to TWO-COLUMN section for body
# ══════════════════════════════════════════════════════════════
new_section = doc.add_section()
new_section.page_height    = Cm(29.7)
new_section.page_width     = Cm(21.0)
new_section.top_margin     = Cm(1.9)
new_section.bottom_margin  = Cm(2.54)
new_section.left_margin    = Cm(1.78)
new_section.right_margin   = Cm(1.78)
new_section.start_type     = 1   # continuous
set_two_columns(new_section)

# ══════════════════════════════════════════════════════════════
#  I. INTRODUCTION
# ══════════════════════════════════════════════════════════════
heading(doc, 'I', 'Introduction')

body(doc, (
    "Post-COVID-19 Condition (PASC), or Long COVID, is a major public health challenge "
    "affecting an estimated 6% of infected individuals, with higher susceptibility in "
    "older adults, females, and those with chronic comorbidities [1]. US surveillance "
    "data report 6.9% of adults affected, with 3.4% experiencing ongoing symptoms [2]. "
    "The heterogeneity of PASC and lack of validated, deployable risk tools limit "
    "clinicians' ability to identify high-risk patients at the point of acute presentation."
))

body(doc, (
    "Existing predictive models have improved risk stratification for Long COVID, yet "
    "critical gaps persist: calibration is rarely reported, fairness across demographic "
    "subgroups is seldom audited, leakage from post-admission features inflates performance, "
    "and most models lack per-patient explainability [3][4]. Furthermore, whether complex "
    "ensemble models provide meaningful gains over interpretable logistic regression "
    "on large, mostly binary clinical datasets remains an open empirical question [5]."
))

body(doc, (
    "This paper addresses these gaps with a 13-phase, leakage-free responsible-ML "
    "pipeline applied to 220,218 confirmed COVID-19 cases from the Mexican Government "
    "surveillance dataset. Because no Long COVID / PASC follow-up labels exist in the "
    "data, in-hospital mortality is used as a clinically justified proxy for severe "
    "adverse outcome, consistent with the approach of Menezes et al. [6] and Dozier "
    "et al. [7]. The pipeline delivers: (i) a calibrated Logistic Regression model "
    "deployed as a web prototype; (ii) a DeLong-based model comparison showing "
    "interpretable LR is statistically non-inferior to black-box ensembles; "
    "(iii) a full fairness audit with subgroup mitigation; and (iv) SHAP, "
    "odds-ratio, and nomogram explainability layers."
))

body(doc, (
    "The remainder of this paper is organised as follows. Section II reviews "
    "related work. Section III describes the dataset and methodology. Section IV "
    "presents performance evaluation. Section V concludes with limitations and "
    "future work directions."
))

# ══════════════════════════════════════════════════════════════
#  II. RELATED WORK
# ══════════════════════════════════════════════════════════════
heading(doc, 'II', 'Related Work')

body(doc, (
    "Predictive modelling for Long COVID risk has grown rapidly, with studies "
    "reporting moderate to high discrimination (AUROC 0.75–0.80) on large EHR cohorts "
    "[8][9]. Jeffrey et al. [4] developed a Scottish population logistic regression "
    "model identifying female sex, BMI, age, deprivation, and severe acute infection as "
    "predictors. However, calibration is not reported, subgroup fairness is not "
    "audited, and the model is not publicly deployable."
))

body(doc, (
    "Shakhovska et al. [10] proposed a stacking ensemble achieving AUC = 0.978 for "
    "post-COVID severity, but on only 122 patients from a single centre, severely "
    "limiting generalisability. Dozier et al. [7] applied a Super Learner on the "
    "National COVID Cohort Collaborative, demonstrating high discrimination but "
    "limited calibration and interpretability reporting. Swinnerton et al. [11] "
    "incorporated near-real-time population risk into logistic regression, improving "
    "prospective AUC to 0.85, but the focus is acute severity rather than PASC."
))

body(doc, (
    "Teeuw et al. [12] modelled cardiovascular risk post-COVID-19, achieving "
    "C-statistic 0.78, but noted that high discrimination does not translate to "
    "clinically meaningful stratification when event prevalence is low — motivating "
    "this work's emphasis on calibration and decision-curve analysis."
))

body(doc, (
    "Bjerre et al. [5] conducted a systematic comparison of ML versus classical "
    "regression on large COVID databases and found that ML offered only marginal gains, "
    "consistent with the finding presented here that no model significantly beats "
    "Logistic Regression on the 9-feature Mexican dataset. The present work "
    "operationalises the predictors identified in risk-factor studies [6][13] into "
    "a validated, calibrated, fairness-audited deployable model — addressing the "
    "exact gaps the existing literature identifies."
))

# ══════════════════════════════════════════════════════════════
#  III. METHODOLOGY
# ══════════════════════════════════════════════════════════════
heading(doc, 'III', 'Methodology')

subheading(doc, 'A', 'Dataset')
body(doc, (
    "The Mexican Government COVID-19 dataset (Kaggle, 2020 [14]) contains 566,602 "
    "patient records from the acute phase of SARS-CoV-2 infection, including "
    "demographic variables (age, sex), binary comorbidities (diabetes, hypertension, "
    "cardiovascular disease, pneumonia, obesity, asthma, COPD), and admission outcomes. "
    "Records were filtered to confirmed COVID-19 cases (covid_res == 1), yielding "
    "220,218 patients with 12.31% in-hospital mortality (27,102 deaths). The outcome "
    "variable was derived from the presence of a recorded death date."
))

subheading(doc, 'B', 'Preprocessing and Feature Engineering')
body(doc, (
    "Sentinel values (97, 98, 99) encoding unknown comorbidity status were decoded to "
    "NaN and median-imputed, affecting approximately 0.3% of comorbidity fields. "
    "Binary features were recoded from the raw 1 = Yes / 2 = No encoding to 1.0 / 0.0. "
    "Intubation and ICU admission were deliberately excluded: both are post-admission "
    "treatment outcomes unavailable at the decision point, and their inclusion "
    "would constitute target leakage inflating performance estimates."
))

body(doc, (
    "A leakage-free pipeline was enforced throughout: a SimpleImputer (median) and "
    "StandardScaler were fit exclusively on the training partition (176,174 patients) "
    "and applied to the held-out test set (44,044 patients). Scikit-learn Pipelines "
    "ensured the same split-then-fit discipline within all cross-validation folds. "
    "The resulting feature matrix comprises nine admission-time predictors: age, sex, "
    "diabetes, hypertension, cardiovascular disease, pneumonia, obesity, asthma, COPD. "
    "Events-per-variable (EPV ≈ 3,011) far exceeds the threshold of 10, confirming "
    "ample data for stable estimation."
))

subheading(doc, 'C', 'Model Training and Selection')
body(doc, (
    "Five model families were trained: Logistic Regression (L2, C = 0.01 via 5-fold "
    "GridSearchCV), Random Forest (max_depth = 6), Gradient Boosting, XGBoost, and "
    "a Stacking ensemble. Class imbalance (1:7 ratio) was addressed via "
    "class_weight='balanced' (LR, RF), sample_weight (GB), and scale_pos_weight "
    "(XGBoost). SMOTE was evaluated and rejected: it produced an identical test AUC "
    "(0.888) while fabricating 132,810 synthetic patients, introducing unnecessary "
    "distributional distortion."
))

body(doc, (
    "The deployed model is a CalibratedClassifierCV (isotonic, cv = 5) wrapping the "
    "tuned Logistic Regression, persisted as a joblib pipeline. Model selection was "
    "based on DeLong's [15] statistical test of AUC differences: no model "
    "significantly outperformed LR (p > 0.05 for GB, XGBoost, Stacking), supporting "
    "the choice of the interpretable model on parsimony grounds."
))

subheading(doc, 'D', 'Explainability and Fairness')
body(doc, (
    "Global feature importance was assessed with Tree-SHAP applied to the XGBoost "
    "benchmark model, and cross-checked with linear-SHAP on the deployed LR. "
    "Per-patient explanation uses odds ratios from an unregularised statsmodels logit "
    "(per +1 SD, 95% CI). A nomogram translates log-odds coefficients into "
    "integer points for clinical usability."
))

body(doc, (
    "Fairness was audited across eight subgroups (four age bands, sex, diabetes "
    "presence/absence) using both AUC (discrimination) and Expected Calibration "
    "Error (ECE, count-weighted). Group-specific thresholds were applied as a "
    "mitigation strategy. Decision Curve Analysis quantified clinical net benefit "
    "relative to treat-all and treat-none baselines."
))

# ══════════════════════════════════════════════════════════════
#  IV. PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════
heading(doc, 'IV', 'Performance Evaluation')

subheading(doc, 'A', 'Model Discrimination')
body(doc, (
    "Table I summarises test-set AUC with 95% bootstrap confidence intervals "
    "and DeLong test results versus Logistic Regression."
))

# Table I
tbl = doc.add_table(rows=8, cols=4)
tbl.style = 'Table Grid'
hdr = tbl.rows[0].cells
for c, txt in zip(hdr, ['Model', 'AUC', '95% CI', 'DeLong vs LR']):
    c.text = txt
    for run in c.paragraphs[0].runs:
        run.font.name = 'Times New Roman'; run.font.size = Pt(8); run.bold = True

rows_data = [
    ('Logistic Regression',    '0.888', '0.884–0.892', '—'),
    ('Random Forest',          '0.866', '0.861–0.871', 'p ≈ 3×10⁻⁴² (worse)'),
    ('Gradient Boosting',      '0.888', '0.884–0.892', 'p > 0.05 (n.s.)'),
    ('XGBoost',                '0.888', '0.884–0.892', 'p > 0.05 (n.s.)'),
    ('Stacking',               '0.888', '0.884–0.892', 'p > 0.05 (n.s.)'),
    ('Ensemble',               '0.887', '0.883–0.891', 'p < 0.05 (worse)'),
    ('Primary (deployed LR)',  '0.888', '0.884–0.892', 'chosen'),
]
for row, (m, auc, ci, dl) in zip(tbl.rows[1:], rows_data):
    row.cells[0].text = m
    row.cells[1].text = auc
    row.cells[2].text = ci
    row.cells[3].text = dl
    for cell in row.cells:
        for run in cell.paragraphs[0].runs:
            run.font.name = 'Times New Roman'; run.font.size = Pt(8)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after  = Pt(6)
r = p.add_run("TABLE I.   Model Comparison — Test-Set AUC (n = 44,044)")
r.font.name = 'Times New Roman'; r.font.size = Pt(8); r.bold = True

body(doc, (
    "Random Forest is the only model significantly inferior to LR (p ≈ 3×10⁻⁴²). "
    "All other models are statistically indistinguishable. This confirms the "
    "finding of Bjerre et al. [5]: on low-dimensional, mostly-binary acute-phase "
    "clinical features, ML complexity yields no discriminative advantage. The signal "
    "is largely linear and additive (pneumonia and age dominate); tree flexibility "
    "has no additional structure to exploit. 5-fold CV AUC: 0.891 ± 0.002; "
    "out-of-time (temporal) AUC: 0.891, confirming stability over the data collection period."
))

subheading(doc, 'B', 'Calibration')
body(doc, (
    "Before isotonic calibration the deployed LR was poorly calibrated "
    "(Brier = 0.135, count-weighted ECE = 0.195). After CalibratedClassifierCV "
    "(isotonic, cv = 5) calibration improved substantially: Brier = 0.078, "
    "count-weighted ECE = 0.002. The bin-averaged ECE (stricter definition, "
    "fewer samples per high-probability bin) is 0.064, also confirming well-calibrated "
    "output probabilities. The deployed probability can therefore be interpreted "
    "directly as a mortality risk estimate (e.g., a score of 0.30 ≈ 30% observed mortality)."
))

subheading(doc, 'C', 'Operating Threshold')
body(doc, (
    "At the default 0.50 threshold: sensitivity 0.296, specificity 0.968, precision 0.564. "
    "At the F1-optimal threshold (0.26): sensitivity 0.702, specificity 0.878, "
    "precision 0.447, F1 = 0.546. At Youden's J threshold (0.10): "
    "sensitivity 0.877, specificity 0.761. Precision of 0.447 at 0.26 reflects the "
    "12.3% base rate; it represents a 3.6× enrichment over random screening. "
    "The screening operating point deliberately favours recall over precision — "
    "missing a high-risk patient is clinically more costly than an extra review."
))

subheading(doc, 'D', 'Explainability')
body(doc, (
    "Tree-SHAP global importance (XGBoost benchmark): pneumonia 44.2%, age 33.0%, "
    "sex 8.5%, diabetes 5.6%, hypertension 4.5%, obesity 3.0%, COPD 0.6%, "
    "asthma 0.3%, cardiovascular 0.3%. This ranking aligns with odds ratios "
    "from the deployed LR (statsmodels, per +1 SD): pneumonia OR = 2.65 "
    "(95% CI 2.61–2.69), age OR = 2.33 (2.28–2.37), diabetes OR = 1.15 "
    "(1.13–1.17), obesity OR = 1.10 (1.08–1.12), hypertension OR = 1.09 "
    "(1.07–1.10). Cardiovascular (OR 1.00) and asthma (OR 0.98) are not statistically "
    "significant. Sex (female) OR = 0.80 (protective). The dominant pneumonia effect "
    "is consistent with acute respiratory involvement as the primary mortality driver [6][13]. "
    "A points-based nomogram was derived from log-odds coefficients: pneumonia +24 points, "
    "age +6 per decade, diabetes +5, COPD +4, hypertension +3, obesity +3, asthma −1, "
    "sex (female) −5."
))

subheading(doc, 'E', 'Fairness Audit')
body(doc, (
    "Table II reports AUC and ECE per subgroup."
))

# Table II
tbl2 = doc.add_table(rows=9, cols=4)
tbl2.style = 'Table Grid'
hdr2 = tbl2.rows[0].cells
for c, txt in zip(hdr2, ['Subgroup', 'n', 'AUC', 'ECE']):
    c.text = txt
    for run in c.paragraphs[0].runs:
        run.font.name = 'Times New Roman'; run.font.size = Pt(8); run.bold = True

rows2 = [
    ('Age 18–30',       '6,290',  '0.891', '0.002'),
    ('Age 30–50',       '19,713', '0.874', '0.004'),
    ('Age 50–65 ★',     '11,189', '0.799', '0.008'),
    ('Age 65+ ★',       '5,926',  '0.733', '0.015'),
    ('Sex: Female',     '19,961', '0.901', '0.005'),
    ('Sex: Male',       '24,083', '0.873', '0.003'),
    ('Diabetes ★',      '7,160',  '0.776', '0.012'),
    ('No Diabetes',     '36,704', '0.899', '0.004'),
]
for row, (g, n, auc, ece) in zip(tbl2.rows[1:], rows2):
    row.cells[0].text = g
    row.cells[1].text = n
    row.cells[2].text = auc
    row.cells[3].text = ece
    for cell in row.cells:
        for run in cell.paragraphs[0].runs:
            run.font.name = 'Times New Roman'; run.font.size = Pt(8)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after  = Pt(6)
r = p.add_run("TABLE II.   Subgroup Fairness Audit  (★ = AUC flagged)")
r.font.name = 'Times New Roman'; r.font.size = Pt(8); r.bold = True

body(doc, (
    "Three subgroups show lower AUC (age 50–65: 0.799; 65+: 0.733; diabetes: 0.776). "
    "This is attributable to restriction of range: within high-prevalence, "
    "homogeneous risk groups, there is less variation for a ranking metric to exploit. "
    "Crucially, ECE remains below 0.015 in every subgroup, confirming that "
    "calibration — the clinically relevant property for deployed risk tools — "
    "is maintained equitably. Gender gap in AUC is 2.8% (female 0.901 vs male 0.873), "
    "within the accepted 5% fairness threshold. Group-specific classification thresholds "
    "reduced TPR disparity from 0.603 to 0.045."
))

subheading(doc, 'F', 'Decision Curve Analysis')
body(doc, (
    "Decision Curve Analysis confirms positive net benefit over treat-all and treat-none "
    "across the clinically relevant threshold range (0.05–0.50). At threshold 0.10, "
    "model net benefit = 0.085 versus treat-all = 0.026, demonstrating that using "
    "the model to guide intervention is substantially preferable to blanket strategies."
))

# ══════════════════════════════════════════════════════════════
#  V. CONCLUSION
# ══════════════════════════════════════════════════════════════
heading(doc, 'V', 'Conclusion')

body(doc, (
    "This paper presents a rigorous, responsible-ML benchmarking study demonstrating "
    "that an interpretable, calibrated Logistic Regression matches black-box ensembles "
    "(AUC 0.888, DeLong p > 0.05) for acute-phase COVID-19 mortality risk prediction "
    "on a 220,218-patient dataset. Isotonic calibration substantially improves "
    "probability reliability (Brier 0.135 → 0.078), and fairness auditing confirms "
    "well-calibrated predictions across all demographic and comorbidity subgroups. "
    "SHAP analysis identifies pneumonia and age as dominant drivers, consistent with "
    "clinical understanding of acute respiratory involvement."
))

body(doc, (
    "The work makes three contributions. First, it operationalises an empirical "
    "finding: on low-dimensional, mostly binary clinical features, interpretable LR is "
    "statistically non-inferior to RF, GB, XGBoost, and stacking — so interpretability "
    "costs nothing here. Second, it addresses the exact gaps the Long COVID prediction "
    "literature identifies: calibration, fairness auditing, leakage control, and "
    "explainability, integrated in a single reproducible pipeline. Third, it delivers a "
    "functioning Streamlit prototype providing per-patient risk scores, odds-ratio "
    "explanations, a nomogram, and an auditable model card and datasheet."
))

body(doc, (
    "Limitations include: (i) mortality as a proxy — no PASC labels exist in the "
    "dataset, so sequela-specific prediction is future work; (ii) single-country data "
    "(Mexico), limiting external generalisability; (iii) nine acute-phase features only — "
    "no labs, vitals, vaccination history, or variant information; "
    "(iv) no multi-site external validation (temporal validation performed). "
    "Future work should apply the pipeline to PASC-labelled EHR cohorts, "
    "incorporate longitudinal immunological data, and validate externally across "
    "healthcare systems."
))

# ══════════════════════════════════════════════════════════════
#  GENERATIVE AI DISCLOSURE
# ══════════════════════════════════════════════════════════════
heading(doc, '', 'Generative AI Disclosure')

body(doc, (
    "Generative AI tools (Claude, Anthropic) were used in this project for the "
    "following purposes: (i) code review and debugging of the Python analysis pipeline; "
    "(ii) drafting and editing sections of this report based on verified results "
    "produced by the pipeline; (iii) assisting with IEEE formatting. All quantitative "
    "results, model training, statistical tests, and data analysis were performed "
    "exclusively by the author's own code (covid_analysis_full.py, advanced_methods.py). "
    "All AI-assisted text was reviewed, verified against the actual pipeline outputs, "
    "and edited by the author before inclusion."
))

# ══════════════════════════════════════════════════════════════
#  REFERENCES  [1]–[15]
# ══════════════════════════════════════════════════════════════
heading(doc, '', 'References')

refs = [
    "World Health Organisation, \"Post COVID-19 condition (long COVID),\" WHO Fact Sheet, 2025. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/post-covid-19-condition",
    "D. A. Gbewonyo et al., \"Long COVID in Adults: United States, 2022,\" National Center for Health Statistics, 2023. doi: 10.15620/cdc:132417.",
    "A. Noroozi, A. Orooji, and L. Erfannia, \"Analyzing the impact of feature selection methods on machine learning algorithms for heart disease prediction,\" Scientific Reports, vol. 13, no. 1, 2023. doi: 10.1038/s41598-023-49962-w.",
    "K. Jeffrey et al., \"Deriving and validating a risk prediction model for long COVID: a population-based, retrospective cohort study in Scotland,\" Journal of the Royal Society of Medicine, vol. 117, no. 12, pp. 402–414, 2024. doi: 10.1177/01410768241297833.",
    "L. M. Bjerre et al., \"Comparing AI/ML approaches and classical regression for predictive modeling using large population health databases,\" Global Epidemiology, vol. 8, p. 100168, 2024. doi: 10.1016/j.gloepi.2024.100168.",
    "A. S. Menezes et al., \"Acute COVID-19 Syndrome Predicts Severe Long COVID-19: An Observational Study,\" Cureus, 2022. doi: 10.7759/cureus.29826.",
    "Z. B. Dozier et al., \"Predicting Long COVID in the National COVID Cohort Collaborative Using Super Learner,\" JMIR Public Health and Surveillance, vol. 10, p. e53322, 2024. doi: 10.2196/53322.",
    "B. Antony et al., \"Predictive models of long COVID,\" EBioMedicine, vol. 96, p. 104777, 2023. doi: 10.1016/j.ebiom.2023.104777.",
    "C. Zang et al., \"Identification of risk factors of Long COVID and predictive modeling in the RECOVER EHR cohorts,\" Communications Medicine, vol. 4, no. 1, 2024. doi: 10.1038/s43856-024-00549-0.",
    "N. Shakhovska, V. Yakovyna, and V. Chopyak, \"A new hybrid ensemble machine-learning model for severity risk assessment and post-COVID prediction system,\" Mathematical Biosciences and Engineering, vol. 19, no. 6, pp. 6102–6123, 2022. doi: 10.3934/mbe.2022285.",
    "K. Swinnerton et al., \"Leveraging near-real-time patient and population data to incorporate fluctuating risk of severe COVID-19,\" eClinicalMedicine, vol. 81, p. 103114, 2025. doi: 10.1016/j.eclinm.2025.103114.",
    "H. M. la R. Teeuw et al., \"Incidence and individual risk prediction of post-COVID-19 cardiovascular disease,\" European Society of Cardiology Open Heart, vol. 3, pp. 1–12, 2023. doi: 10.1093/ehjopen/oead101.",
    "J. G. Guzman-Esquivel et al., \"Clinical Characteristics in the Acute Phase of COVID-19 That Predict Long COVID,\" Healthcare, vol. 11, no. 2, pp. 197–197, 2023. doi: 10.3390/healthcare11020197.",
    "Kaggle, \"COVID-19 Mexico Patient Health Dataset,\" 2020. [Online]. Available: https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset",
    "E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, \"Comparing the areas under two or more correlated receiver operating characteristic curves,\" Biometrics, vol. 44, no. 3, pp. 837–845, 1988.",
]

for i, r in enumerate(refs, 1):
    ref_entry(doc, i, r)

doc.save(OUT)
print(f"Saved: {OUT}")
