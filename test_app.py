#!/usr/bin/env python3
"""
Tests for the Streamlit app logic (streamlit_covid_predictor.py).

Run:  python -m pytest test_app.py -v   (or run the whole suite: pytest -v)

Streamlit is replaced by a minimal stub BEFORE the app is imported, and the app's
router is guarded by `if __name__ == "__main__"`, so importing the module loads
only its helper functions — no Streamlit server, no page rendering.

Covered:
  * risk banding (rlevel), urgency mapping, band colour (bclr) and their mutual
    consistency at the 30/50/70 cut-offs
  * recovery-time heuristic (score / age / comorbidity burden)
  * Long-COVID sequelae heuristic (lc_risks) — ranges & directionality
  * the FEATURE-COLUMN ORDER contract predict_risk depends on
  * predict_risk / patient_factors against the REAL deployed artifacts
    (imputer + scaler + calibrated LR) — range and monotonic sanity
  * nomogram_score shape (skips gracefully if the nomogram CSVs are absent)
"""

import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal Streamlit stub, installed BEFORE importing the app.
# ---------------------------------------------------------------------------
class _Null:
    """Universal no-op: callable, context manager, attribute/iterator transparent."""
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return self
    def __iter__(self): return iter(())
    def __bool__(self): return False


class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _make_streamlit_stub():
    st = types.ModuleType("streamlit")
    null = _Null()

    def _cache(*dargs, **dkwargs):
        # supports both @st.cache_data(show_spinner=...) and bare @st.cache_data
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        def wrap(fn):
            return fn
        return wrap

    st.cache_data = _cache
    st.cache_resource = _cache
    st.session_state = _SessionState()

    class _QP:
        def get(self, k, d=None):
            return d
    st.query_params = _QP()

    # markdown / set_page_config / error / stop / columns / ... -> no-op
    st.__getattr__ = lambda name: null
    return st


sys.modules["streamlit"] = _make_streamlit_stub()

import streamlit_covid_predictor as app   # noqa: E402


HERE = Path(__file__).resolve().parent
MODELS = HERE / "analysis_output" / "models"


def _patient(**over):
    """A patient dict in the app's contract (sex: Female=1, Male=0; conditions 0/1)."""
    base = dict(age=50, sex=0, diabetes=0, hypertension=0, cardiovascular=0,
                pneumonia=0, obesity=0, asthma=0, copd=0)
    base.update(over)
    return base


@pytest.fixture(scope="module")
def artifacts():
    joblib = pytest.importorskip("joblib")
    if not MODELS.exists():
        pytest.skip("deployed model artifacts not present in analysis_output/models/")
    imp = joblib.load(MODELS / "imputer.joblib")
    sc = joblib.load(MODELS / "scaler.joblib")
    model = joblib.load(MODELS / "model_primary_deployed.joblib")
    return imp, sc, model


# ===========================================================================
# Risk banding / urgency / colour
# ===========================================================================

class TestRiskBanding:
    @pytest.mark.parametrize("score,band", [
        (0, "LOW"), (29.9, "LOW"), (30, "MEDIUM"), (49.9, "MEDIUM"),
        (50, "HIGH"), (69.9, "HIGH"), (70, "CRITICAL"), (100, "CRITICAL"),
    ])
    def test_rlevel_thresholds(self, score, band):
        assert app.rlevel(score) == band

    def test_urgency_covers_all_levels(self):
        for lvl in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            assert isinstance(app.urgency(lvl), str) and app.urgency(lvl)

    @pytest.mark.parametrize("score,color", [
        (10, "#059669"), (35, "#D97706"), (55, "#EA580C"), (85, "#DC2626"),
    ])
    def test_bclr_colors(self, score, color):
        assert app.bclr(score) == color

    @pytest.mark.parametrize("s", [0, 15, 30, 45, 50, 65, 70, 90, 100])
    def test_band_and_color_are_consistent(self, s):
        # rlevel and bclr must agree at the shared 30/50/70 cut-offs
        mapping = {"LOW": "#059669", "MEDIUM": "#D97706",
                   "HIGH": "#EA580C", "CRITICAL": "#DC2626"}
        assert app.bclr(s) == mapping[app.rlevel(s)]


# ===========================================================================
# Recovery-time heuristic
# ===========================================================================

class TestRecovery:
    @pytest.mark.parametrize("score,expected_prefix", [
        (20, "3-4 months"), (40, "4-6 months"),
        (60, "6-9 months"), (80, "9-12 months"),
    ])
    def test_recovery_bands_by_score(self, score, expected_prefix):
        assert app.recovery(score, 40).startswith(expected_prefix)

    def test_elderly_high_risk_extended(self):
        # age >= 65 AND score >= 50 -> "(extended)"
        assert "(extended)" in app.recovery(60, 70)

    def test_high_comorbidity_burden_extended(self):
        pt = _patient(diabetes=1, hypertension=1, obesity=1)  # 3 comorbidities
        out = app.recovery(40, 40, pt)   # not elderly/high, so the comorbidity clause fires
        assert "extended" in out

    def test_no_extension_for_low_risk_healthy(self):
        assert "extended" not in app.recovery(20, 40, _patient())


# ===========================================================================
# Long-COVID sequelae heuristic
# ===========================================================================

class TestLongCovidHeuristic:
    def test_five_categories_in_range(self):
        r = app.lc_risks(_patient(age=50), base=10.0)
        assert set(r) == {"Respiratory", "Cardiac", "Neurological", "Systemic", "Metabolic"}
        assert all(0.0 <= v <= 100.0 for v in r.values())

    def test_pneumonia_raises_respiratory(self):
        lo = app.lc_risks(_patient(pneumonia=0), 10.0)["Respiratory"]
        hi = app.lc_risks(_patient(pneumonia=1), 10.0)["Respiratory"]
        assert hi > lo

    def test_diabetes_raises_metabolic(self):
        lo = app.lc_risks(_patient(diabetes=0), 10.0)["Metabolic"]
        hi = app.lc_risks(_patient(diabetes=1), 10.0)["Metabolic"]
        assert hi > lo

    def test_capped_at_100(self):
        r = app.lc_risks(_patient(age=90, pneumonia=1, cardiovascular=1, diabetes=1,
                                  hypertension=1, obesity=1, asthma=1, copd=1), base=100.0)
        assert all(v <= 100.0 for v in r.values())


# ===========================================================================
# Feature-column contract  (predict_risk / patient_factors depend on this order)
# ===========================================================================

def test_feature_columns_exact_order():
    assert app.FEAT_COLS == ["age", "sex", "diabetes", "hypertension",
                             "cardiovascular", "pneumonia", "obesity", "asthma", "copd"]


# ===========================================================================
# predict_risk against the REAL deployed artifacts
# ===========================================================================

class TestPredictRisk:
    def test_returns_percentage_in_range(self, artifacts):
        imp, sc, model = artifacts
        r = app.predict_risk(imp, sc, model, _patient(age=50))
        assert isinstance(r, float) and 0.0 <= r <= 100.0

    def test_pneumonia_increases_risk(self, artifacts):
        imp, sc, model = artifacts
        lo = app.predict_risk(imp, sc, model, _patient(age=60, pneumonia=0))
        hi = app.predict_risk(imp, sc, model, _patient(age=60, pneumonia=1))
        assert hi > lo          # pneumonia is the model's top risk driver

    def test_older_age_increases_risk(self, artifacts):
        imp, sc, model = artifacts
        young = app.predict_risk(imp, sc, model, _patient(age=25))
        old = app.predict_risk(imp, sc, model, _patient(age=85))
        assert old > young

    def test_deterministic(self, artifacts):
        imp, sc, model = artifacts
        pt = _patient(age=63, diabetes=1, pneumonia=1)
        assert app.predict_risk(imp, sc, model, pt) == app.predict_risk(imp, sc, model, pt)


# ===========================================================================
# patient_factors (local explanation) & nomogram
# ===========================================================================

def test_patient_factors_signs(artifacts):
    imp, sc, model = artifacts
    raises, lowers = app.patient_factors(imp, sc, _patient(age=80, pneumonia=1, diabetes=1))
    assert isinstance(raises, list) and isinstance(lowers, list)
    assert all(v > 0 for _, v in raises)     # raisers push risk up
    assert all(v < 0 for _, v in lowers)     # lowerers push risk down


def test_nomogram_score_shape():
    res = app.nomogram_score(_patient(age=70, pneumonia=1))
    # None when the nomogram CSVs are absent; otherwise (points, risk_pct, breakdown)
    assert res is None or (len(res) == 3 and 0.0 <= res[1] <= 100.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
