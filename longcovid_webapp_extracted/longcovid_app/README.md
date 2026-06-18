# Long COVID Risk Assessment — FastAPI Web App

**Student:** Durga Prasad Narsing (A00050350)  
**Program:** MSc Computing (Data Analytics), Dublin City University  
**Supervisor:** Martin Crane  

---

## Project Structure

```
longcovid_app/
├── main.py                    # FastAPI application (all routes + API endpoints)
├── requirements.txt
├── Procfile                   # For Render / Railway deployment
├── templates/
│   ├── base.html              # Shared nav, styles, lightbox
│   ├── index.html             # Landing page
│   ├── dashboard.html         # Pre-generated results dashboard
│   ├── pipeline.html          # CSV upload + live pipeline runner
│   └── predictor.html         # Interactive patient risk predictor
├── static/                    # (optional CSS/JS overrides)
├── uploads/                   # Uploaded CSV files (auto-created)
├── outputs/                   # Per-job pipeline outputs (auto-created)
└── powerbi_export/            # Pre-generated results (from covid_analysis_full.py)
```

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Copy your analysis outputs so the dashboard works immediately
cp -r /path/to/powerbi_export ./powerbi_export

# 3. (Optional) Place your trained models for the live predictor
#    They should be in powerbi_export/models/
#    - model_primary_deployed.joblib
#    - scaler.joblib
#    - imputer.joblib

# 4. Run the app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

---

## Three Sections

### 1. Dashboard (`/dashboard`)
- Loads pre-generated CSVs and plot PNGs from `powerbi_export/`
- Shows AUC confidence intervals, fairness audit, cost-benefit, SHAP importance
- Tabbed plot viewer with lightbox zoom

### 2. Pipeline Runner (`/pipeline`)
- Upload a COVID CSV → runs `covid_analysis_full.py` phases 1–12 in the background
- Live log console + phase indicator dots
- Results appear as a zoomable plot grid when complete

### 3. Risk Predictor (`/predictor`)
- Patient form: age, sex, 7 comorbidity toggles
- Calls `/api/predict` → returns risk score 0–100, level, contributing factors, recommendations
- Uses the deployed `model_primary_deployed.joblib` if available; falls back to a demo approximation
- 4 example profiles (Low / Moderate / High / Critical)

---

## Cloud Deployment

### Render (recommended — free tier)
1. Push this folder to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add any large files (models, CSVs) via environment variables or a persistent disk

### Railway
1. Push to GitHub
2. Connect repo → Railway auto-detects the `Procfile`
3. Set `PORT` environment variable if not auto-set

### Google Cloud Run
```bash
gcloud run deploy longcovid-app \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Landing page |
| GET | `/dashboard` | Dashboard page |
| GET | `/pipeline` | Pipeline runner page |
| GET | `/predictor` | Risk predictor page |
| GET | `/api/dashboard` | Dashboard data (JSON) |
| POST | `/api/pipeline/start` | Upload CSV + start pipeline |
| GET | `/api/pipeline/status/{job_id}` | Poll pipeline status |
| POST | `/api/predict` | Predict risk for a patient |
| GET | `/health` | Health check |
| GET | `/docs` | Auto-generated FastAPI docs |

---

## Notes
- The pipeline runner runs `covid_analysis_full.py` as a module. Both files must be in the same directory (or adjust the `sys.path` in `main.py`).
- For cloud deployments with long pipelines, consider adding a task queue (Celery + Redis) to replace the in-memory `jobs` dict.
- The demo predictor approximation is for illustration only; load real `.joblib` artefacts for production use.
