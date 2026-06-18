"""
===============================================================================
Long COVID Risk Assessment - FastAPI Web Application
===============================================================================
Student: Durga Prasad Narsing (A00050350)
Program: MSc Computing (Data Analytics), Dublin City University
Supervisors: Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)

Three sections:
  1. Dashboard     - load pre-generated CSVs/plots from powerbi_export/
  2. Pipeline      - upload CSV → run analysis pipeline → display results
  3. Risk Predictor- enter patient data → get risk score from deployed model

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
===============================================================================
"""

import os, sys, json, uuid, shutil, logging, traceback
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
EXPORT_DIR    = BASE_DIR / "powerbi_export"          # pre-generated results
UPLOADS_DIR   = BASE_DIR / "uploads"
OUTPUTS_DIR   = BASE_DIR / "outputs"
MODELS_DIR    = EXPORT_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"

for d in [UPLOADS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Job registry (in-memory; swap for Redis/DB in production) ─────────────────
jobs: Dict[str, Dict[str, Any]] = {}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Long COVID Risk Assessment",
    description="Predictive Modelling for Personalized Long COVID Risk Assessment - DCU MSc",
    version="1.0.0",
)

app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Serve pre-generated plots if the export folder exists
if EXPORT_DIR.exists():
    app.mount("/export", StaticFiles(directory=str(EXPORT_DIR)), name="export")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 - Dashboard helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None

def _collect_plots(subdir: str) -> list:
    """Return /export-relative URLs for all PNGs in a visualisations subfolder."""
    folder = EXPORT_DIR / "visualizations" / subdir
    if not folder.exists():
        return []
    return sorted(
        f"/export/visualizations/{subdir}/{p.name}"
        for p in folder.glob("*.png")
    )

@app.get("/api/dashboard", response_class=JSONResponse)
async def api_dashboard():
    """Return summary metrics + plot URLs from pre-generated export folder."""
    if not EXPORT_DIR.exists():
        return {"status": "no_export", "message": "Run the pipeline first to generate results."}

    # ── Key metrics ───────────────────────────────────────────────────────────
    metrics: Dict[str, Any] = {}

    # AUC confidence intervals
    auc_df = _read_csv_safe(EXPORT_DIR / "advanced" / "covid_auc_confidence_intervals.csv")
    if auc_df is not None:
        metrics["auc_table"] = auc_df.to_dict(orient="records")

    # Fairness audit
    fair_df = _read_csv_safe(EXPORT_DIR / "fairness_calibration" / "covid_fairness_audit.csv")
    if fair_df is not None and len(fair_df):
        metrics["fairness"] = fair_df.to_dict(orient="records")

    # Calibration
    cal_df = _read_csv_safe(EXPORT_DIR / "fairness_calibration" / "covid_calibration_analysis.csv")
    if cal_df is not None and len(cal_df):
        metrics["calibration_ece"] = float(cal_df["calibration_error"].mean())

    # Cost-benefit
    cb_df = _read_csv_safe(EXPORT_DIR / "cost_benefit" / "covid_cost_benefit_analysis.csv")
    if cb_df is not None and len(cb_df):
        metrics["cost_benefit"] = cb_df.to_dict(orient="records")

    # SHAP importance (top 9)
    shap_df = _read_csv_safe(EXPORT_DIR / "analysis" / "covid_shap_values.csv")
    if shap_df is not None and "mean_abs_shap" in shap_df.columns:
        top = shap_df.nlargest(9, "mean_abs_shap")[["feature_name", "mean_abs_shap"]]
        metrics["shap_top"] = top.to_dict(orient="records")

    # Prediction stats
    pred_df = _read_csv_safe(EXPORT_DIR / "core_data" / "covid_predictions.csv")
    if pred_df is not None and len(pred_df):
        metrics["n_patients"] = len(pred_df)
        metrics["risk_distribution"] = pred_df["risk_level"].value_counts().to_dict() if "risk_level" in pred_df.columns else {}
        metrics["avg_risk"] = float(pred_df["risk_score"].mean()) if "risk_score" in pred_df.columns else None

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots = {
        "eda":        _collect_plots("eda_plots"),
        "model":      _collect_plots("model_comparison_plots"),
        "shap":       _collect_plots("shap_plots"),
        "fairness":   _collect_plots("fairness_plots"),
        "calibration":_collect_plots("calibration_plots"),
        "advanced":   _collect_plots("advanced_plots"),
    }

    return {"status": "ok", "metrics": metrics, "plots": plots}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(job_id: str, csv_path: Path, output_dir: Path):
    """Background task: runs covid_analysis_full.py and streams progress via jobs[]."""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["log"] = []

    try:
        # Add the project root to path so we can import the analysis module
        sys.path.insert(0, str(BASE_DIR))
        import importlib, types

        # ── Redirect logger so we capture progress ────────────────────────────
        class JobHandler(logging.Handler):
            def emit(self, record):
                jobs[job_id]["log"].append(self.format(record))

        jh = JobHandler()
        jh.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))

        # ── Import & run each phase ───────────────────────────────────────────
        import covid_analysis_full as caf

        # Attach our handler to the module's logger
        caf.logger.addHandler(jh)

        # Override output folder to job-specific directory
        caf.CONFIG["output_folder"] = str(output_dir)
        caf.create_folders()

        df = caf.load_data(str(csv_path))
        jobs[job_id]["log"].append("✅ Data loaded")

        df_full, df_confirmed = caf.phase2_preprocessing(df)
        jobs[job_id]["log"].append("✅ Phase 1-2 complete")

        caf.phase1_eda(df)
        caf.phase3_statistical_tests(df_confirmed)
        jobs[job_id]["log"].append("✅ Phase 3 complete")

        model_results = caf.phase4_model_training(df_confirmed)
        jobs[job_id]["log"].append("✅ Phase 4 (models) complete")

        shap_results  = caf.phase5_shap_analysis(model_results["models"], model_results["X_test"], model_results["feature_columns"])
        fair_results  = caf.phase6_fairness_analysis(model_results["df_model"], model_results["predictions"], model_results["y_test"], model_results["idx_test"])
        cal_results   = caf.phase7_calibration_analysis(model_results["y_test"], model_results["predictions"])
        cb_results    = caf.phase8_cost_benefit_analysis(df_confirmed, model_results["predictions"], model_results["y_test"])
        jobs[job_id]["log"].append("✅ Phases 5-8 complete")

        exp_results   = caf.phase9_clinical_explanations(df_confirmed, model_results["predictions"], model_results["df_model"], model_results["idx_test"])
        int_results   = caf.phase10_feature_interactions(model_results["models"], model_results["feature_columns"])
        sens_results  = caf.phase11_sensitivity_analysis(df_confirmed, model_results["predictions"], model_results["models"], model_results["scaler"], model_results["feature_columns"], model_results["imputer"], model_results["deployed_model"])
        jobs[job_id]["log"].append("✅ Phases 9-11 complete")

        caf.phase12_export_results(df_full, df_confirmed, model_results, shap_results, fair_results, cal_results, cb_results, exp_results, int_results, sens_results)
        jobs[job_id]["log"].append("✅ Phase 12 (export) complete")

        # Collect output plots
        plot_urls = []
        for png in sorted(output_dir.rglob("*.png")):
            rel = png.relative_to(output_dir)
            plot_urls.append(f"/outputs/{job_id}/{rel.as_posix()}")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["plots"]  = plot_urls
        jobs[job_id]["log"].append(f"🎉 Pipeline complete - {len(plot_urls)} plots generated")

        caf.logger.removeHandler(jh)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        jobs[job_id]["log"].append(f"❌ ERROR: {e}")
        logger.error(traceback.format_exc())


@app.post("/api/pipeline/start")
async def pipeline_start(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a CSV and kick off the analysis pipeline in the background."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    job_id   = str(uuid.uuid4())[:8]
    csv_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    out_dir  = OUTPUTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save upload
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "log": [], "plots": [], "csv": str(csv_path)}
    background_tasks.add_task(_run_pipeline, job_id, csv_path, out_dir)

    return {"job_id": job_id}


@app.get("/api/pipeline/status/{job_id}")
async def pipeline_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 - Risk predictor
# ══════════════════════════════════════════════════════════════════════════════

class PatientInput(BaseModel):
    age:            float = Field(..., ge=0, le=120, description="Patient age in years")
    sex:            int   = Field(..., ge=1, le=2,   description="1=Female, 2=Male")
    diabetes:       int   = Field(..., ge=0, le=1)
    hypertension:   int   = Field(..., ge=0, le=1)
    cardiovascular: int   = Field(..., ge=0, le=1)
    pneumonia:      int   = Field(..., ge=0, le=1)
    obesity:        int   = Field(..., ge=0, le=1)
    asthma:         int   = Field(..., ge=0, le=1)
    copd:           int   = Field(..., ge=0, le=1)


def _load_deployed_model():
    """Load the persisted primary deployed model + preprocessing artefacts."""
    import joblib
    model_path = MODELS_DIR / "model_primary_deployed.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    imputer_path = MODELS_DIR / "imputer.joblib"

    if not model_path.exists():
        return None, None, None

    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)  if scaler_path.exists()  else None
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None
    return model, scaler, imputer


# Cache artefacts at startup
_model, _scaler, _imputer = None, None, None

@app.on_event("startup")
async def load_artefacts():
    global _model, _scaler, _imputer
    _model, _scaler, _imputer = _load_deployed_model()
    if _model:
        logger.info("✅ Deployed model loaded from %s", MODELS_DIR)
    else:
        logger.warning("⚠️  No deployed model found - predictor will use a demo model")


def _demo_predict(patient: PatientInput) -> float:
    """
    Simple logistic regression approximation used when no trained model exists.
    Weights estimated from literature (age + comorbidities → mortality risk).
    FOR DEMONSTRATION ONLY - not clinically validated.
    """
    age_norm = (patient.age - 50) / 20
    score = (
        -3.5
        + 0.8  * age_norm
        + 0.5  * (1 if patient.sex == 2 else 0)   # Male slightly higher
        + 0.6  * patient.pneumonia
        + 0.4  * patient.diabetes
        + 0.35 * patient.hypertension
        + 0.35 * patient.cardiovascular
        + 0.25 * patient.obesity
        + 0.15 * patient.copd
        + 0.10 * patient.asthma
    )
    return float(1 / (1 + np.exp(-score)))


FEATURE_COLS = ["age", "sex", "diabetes", "hypertension",
                "cardiovascular", "pneumonia", "obesity", "asthma", "copd"]

RISK_FACTORS = {
    "pneumonia":      {"label": "Pneumonia",       "weight": "Very High"},
    "age":            {"label": "Age",              "weight": "High"},
    "diabetes":       {"label": "Diabetes",         "weight": "High"},
    "hypertension":   {"label": "Hypertension",     "weight": "Moderate"},
    "cardiovascular": {"label": "Cardiovascular",   "weight": "Moderate"},
    "obesity":        {"label": "Obesity",          "weight": "Moderate"},
    "copd":           {"label": "COPD",             "weight": "Low-Moderate"},
    "asthma":         {"label": "Asthma",           "weight": "Low"},
}


@app.post("/api/predict")
async def predict(patient: PatientInput):
    """Return risk score and clinical interpretation for a patient."""
    global _model, _scaler, _imputer
    used_demo = False

    if _model is None:
        prob = _demo_predict(patient)
        used_demo = True
    else:
        row = np.array([[getattr(patient, c) for c in FEATURE_COLS]], dtype=float)
        # Convert sex 1/2 → 1/0 (Female=1, Male=0) as in training encoding
        row[0, 1] = 1.0 if patient.sex == 1 else 0.0
        if _imputer:
            row = _imputer.transform(row)
        if _scaler:
            row = _scaler.transform(row)
        prob = float(_model.predict_proba(row)[0, 1])

    risk_score = round(prob * 100, 1)

    if risk_score < 30:
        risk_level, colour = "LOW", "#2ecc71"
    elif risk_score < 50:
        risk_level, colour = "MODERATE", "#f39c12"
    elif risk_score < 70:
        risk_level, colour = "HIGH", "#e67e22"
    else:
        risk_level, colour = "CRITICAL", "#e74c3c"

    # Build contributing factors
    factors = []
    if patient.age > 60:
        factors.append({"factor": "Age > 60", "impact": "High"})
    if patient.pneumonia:
        factors.append({"factor": "Pneumonia present", "impact": "Very High"})
    if patient.diabetes:
        factors.append({"factor": "Diabetes", "impact": "High"})
    if patient.hypertension:
        factors.append({"factor": "Hypertension", "impact": "Moderate"})
    if patient.cardiovascular:
        factors.append({"factor": "Cardiovascular disease", "impact": "Moderate"})
    if patient.obesity:
        factors.append({"factor": "Obesity", "impact": "Moderate"})
    if patient.copd:
        factors.append({"factor": "COPD", "impact": "Low-Moderate"})
    if patient.asthma:
        factors.append({"factor": "Asthma", "impact": "Low"})

    recommendations = []
    if risk_level in ("HIGH", "CRITICAL"):
        recommendations = [
            "Immediate clinical review recommended",
            "Consider early intervention protocol",
            "Monitor oxygen saturation closely",
            "Assess for hospital admission criteria",
        ]
    elif risk_level == "MODERATE":
        recommendations = [
            "Schedule follow-up within 48 hours",
            "Monitor symptoms and vital signs",
            "Educate patient on warning signs",
        ]
    else:
        recommendations = [
            "Standard outpatient monitoring",
            "Advise patient on Long COVID symptoms to watch for",
        ]

    return {
        "risk_score":      risk_score,
        "risk_level":      risk_level,
        "colour":          colour,
        "probability":     round(prob, 4),
        "factors":         factors,
        "recommendations": recommendations,
        "model_used":      "Demo approximation" if used_demo else "Calibrated LR (deployed)",
        "disclaimer":      "For research and educational purposes only. Not a clinical decision tool.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTML pages
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline(request: Request):
    return templates.TemplateResponse(request, "pipeline.html")

@app.get("/predictor", response_class=HTMLResponse)
async def predictor(request: Request):
    return templates.TemplateResponse(request, "predictor.html")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}
