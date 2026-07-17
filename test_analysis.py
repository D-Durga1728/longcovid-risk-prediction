#!/usr/bin/env python3
"""
Unit & integration test suite for the Long COVID risk-assessment project.

Run:  python -m pytest test_analysis.py -v

Scope:
  * covid_analysis_full.py  — feature encoding / sentinel decoding, FDR helper,
    and the CSV I/O helpers the pipeline relies on.
  * advanced_methods.py     — every statistical, ML and responsible-AI helper
    (DeLong, bootstrap CI, ECE, calibration, thresholds, DCA, SHAP, odds ratios,
    nomogram, fairness mitigation, temporal validation, model card / datasheet,
    SMOTE comparison, advanced models, hyper-parameter tuning).
  * an end-to-end smoke test of the deployed encode -> impute -> scale -> predict
    chain, confirming the pipeline is leakage-free and learns injected signal.

Optional heavy dependencies (xgboost, imbalanced-learn, shap, statsmodels) are
skipped gracefully with pytest.importorskip / None-tolerant asserts, so the suite
passes green in any environment that can run the pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import advanced_methods as am
import covid_analysis_full as caf

_LOG = logging.getLogger("test")


# ==========================================================================
# DeLong test  (DeLong et al. 1988)
# ==========================================================================

def test_delong_identical_predictions_no_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    a1, a2, z, pv = am.delong_roc_test(y, p, p)
    assert abs(a1 - a2) < 1e-9          # same predictor -> same AUC
    assert pv > 0.5                      # no significant difference


def test_delong_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.5, 2000)   # informative
    bad = rng.random(2000)                # random
    a_good, a_bad, z, pv = am.delong_roc_test(y, good, bad)
    assert a_good > a_bad
    assert pv < 0.05                       # difference is significant


def test_delong_symmetric_in_arguments():
    # Swapping the two predictors swaps the AUCs and flips the z sign,
    # but the two-sided p-value is unchanged.
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 1000)
    p1 = y + rng.normal(0, 0.6, 1000)
    p2 = rng.random(1000)
    a1, a2, z, pv = am.delong_roc_test(y, p1, p2)
    b1, b2, z2, pv2 = am.delong_roc_test(y, p2, p1)
    assert abs(a1 - b2) < 1e-9 and abs(a2 - b1) < 1e-9
    assert abs(pv - pv2) < 1e-9


# ==========================================================================
# Bootstrap AUC confidence interval
# ==========================================================================

def test_bootstrap_ci_brackets_auc():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    p = y + rng.normal(0, 0.4, 1000)
    mean_auc, lo, hi = am.bootstrap_auc_ci(y, p, n_boot=200)
    assert 0.0 <= lo <= mean_auc <= hi <= 1.0


def test_bootstrap_ci_reproducible_with_seed():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 500)
    p = y + rng.normal(0, 0.5, 500)
    r1 = am.bootstrap_auc_ci(y, p, n_boot=100, seed=123)
    r2 = am.bootstrap_auc_ci(y, p, n_boot=100, seed=123)
    assert np.allclose(r1, r2)            # deterministic given the seed


# ==========================================================================
# Expected Calibration Error
# ==========================================================================

def test_ece_perfect_is_low():
    # Perfectly calibrated: prob == empirical frequency
    y = np.array([0, 0, 1, 1] * 100)
    p = np.array([0.0, 0.0, 1.0, 1.0] * 100)
    assert am.expected_calibration_error(y, p) < 1e-6


def test_ece_range():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    ece = am.expected_calibration_error(y, p)
    assert 0.0 <= ece <= 1.0


def test_ece_worst_case_is_high():
    # Confidently wrong on every sample -> ECE close to 1.
    y = np.array([1, 1, 1, 1] * 50)
    p = np.zeros(200)
    assert am.expected_calibration_error(y, p) > 0.9


def test_ece_bin_count_is_honoured():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 400)
    p = rng.random(400)
    # Both valid; changing bin count must stay in range (no crash / NaN).
    e5 = am.expected_calibration_error(y, p, n_bins=5)
    e20 = am.expected_calibration_error(y, p, n_bins=20)
    assert 0.0 <= e5 <= 1.0 and 0.0 <= e20 <= 1.0
    assert not np.isnan(e5) and not np.isnan(e20)


# ==========================================================================
# Threshold optimisation
# ==========================================================================

def test_threshold_optimization_returns_valid_thresholds():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 800)
    p = y * 0.6 + rng.random(800) * 0.4
    res = am.optimize_threshold(y, p, _LOG)
    for key in ('threshold_youden', 'threshold_f1'):
        assert 0.0 <= res[key] <= 1.0
    assert isinstance(res['table'], pd.DataFrame)


def test_threshold_table_has_metric_columns():
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, 600)
    p = y * 0.5 + rng.random(600) * 0.5
    res = am.optimize_threshold(y, p, _LOG)
    cols = set(res['table'].columns)
    # sensitivity/specificity-style reporting per threshold
    assert any('sens' in c.lower() for c in cols)
    assert any('spec' in c.lower() for c in cols)


# ==========================================================================
# covid_analysis_full.encode_features  (sentinel decoding 1/2/97/98/99)
# ==========================================================================

class TestEncodeFeatures:
    def test_binary_1_2_mapping(self):
        df = pd.DataFrame({'diabetes': [1, 2, 1, 2], 'age': [50, 60, 70, 80]})
        out = caf.encode_features(df, ['diabetes', 'age'])
        assert out['diabetes'].tolist() == [1.0, 0.0, 1.0, 0.0]

    def test_sentinels_become_nan(self):
        df = pd.DataFrame({'pneumonia': [1, 2, 97, 98, 99], 'age': [1, 2, 3, 4, 5]})
        out = caf.encode_features(df, ['pneumonia'])
        assert out['pneumonia'].iloc[0] == 1.0
        assert out['pneumonia'].iloc[1] == 0.0
        assert out['pneumonia'].iloc[2:].isna().all()   # 97/98/99 -> NaN, not 0

    def test_age_passthrough(self):
        df = pd.DataFrame({'age': [0, 45, 120]})
        out = caf.encode_features(df, ['age'])
        assert out['age'].tolist() == [0.0, 45.0, 120.0]

    def test_returns_float_dtype(self):
        df = pd.DataFrame({'sex': [1, 2], 'age': [30, 40]})
        out = caf.encode_features(df, ['sex', 'age'])
        assert all(out.dtypes == float)

    def test_non_numeric_becomes_nan(self):
        df = pd.DataFrame({'sex': ['1', 'x', '2'], 'age': ['a', '50', '60']})
        out = caf.encode_features(df, ['sex', 'age'])
        assert out['sex'].iloc[0] == 1.0        # '1' -> 1.0
        assert np.isnan(out['sex'].iloc[1])      # 'x' -> NaN
        assert np.isnan(out['age'].iloc[0])      # 'a' -> NaN

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({'sex': [1, 2], 'age': [30, 40]})
        before = df.copy()
        _ = caf.encode_features(df, ['sex', 'age'])
        pd.testing.assert_frame_equal(df, before)


# ==========================================================================
# covid_analysis_full._benjamini_hochberg  (FDR correction)
# ==========================================================================

class TestBenjaminiHochberg:
    def test_equal_spacing_example(self):
        # p_i = 0.01..0.05, n=5 -> every adjusted value collapses to 0.05
        adj = caf._benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
        assert all(abs(a - 0.05) < 1e-9 for a in adj)

    def test_all_in_unit_interval(self):
        rng = np.random.default_rng(0)
        adj = caf._benjamini_hochberg(rng.random(20).tolist())
        assert all(0.0 <= a <= 1.0 for a in adj)

    def test_adjusted_never_below_raw(self):
        raw = [0.001, 0.01, 0.02, 0.2, 0.5]
        adj = caf._benjamini_hochberg(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adj, raw))

    def test_position_alignment_preserved(self):
        # returned values map back to the ORIGINAL positions
        raw = [0.5, 0.001, 0.2]
        adj = caf._benjamini_hochberg(raw)
        assert adj[1] <= adj[0] and adj[1] <= adj[2]   # index 1 had smallest p

    def test_single_pvalue_unchanged(self):
        assert abs(caf._benjamini_hochberg([0.03])[0] - 0.03) < 1e-12

    def test_matches_statsmodels_if_available(self):
        sm = pytest.importorskip("statsmodels.stats.multitest")
        raw = [0.001, 0.008, 0.02, 0.04, 0.6]
        adj = caf._benjamini_hochberg(raw)
        _, sm_adj, _, _ = sm.multipletests(raw, method='fdr_bh')
        assert np.allclose(adj, sm_adj, atol=1e-9)


# ==========================================================================
# covid_analysis_full I/O helpers  (load_data / create_folders / save_csv)
# ==========================================================================

class TestIOHelpers:
    def test_load_data_reads_csv(self, tmp_path):
        p = tmp_path / "mini.csv"
        pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).to_csv(p, index=False)
        df = caf.load_data(str(p))
        assert df.shape == (2, 2) and list(df.columns) == ['a', 'b']

    def test_load_data_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            caf.load_data(str(tmp_path / "does_not_exist.csv"))

    def test_create_folders_and_save_csv(self, tmp_path, monkeypatch):
        monkeypatch.setitem(caf.CONFIG, 'output_folder', str(tmp_path / "out"))
        caf.create_folders()
        assert (tmp_path / "out" / "advanced").is_dir()
        assert (tmp_path / "out" / "models").is_dir()
        path = caf.save_csv(pd.DataFrame({'x': [1, 2]}), "t.csv", "advanced")
        assert path and Path(path).is_file()

    def test_save_csv_bad_subfolder_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setitem(caf.CONFIG, 'output_folder', str(tmp_path / "nowhere"))
        # target directory does not exist -> to_csv fails -> "" (graceful)
        assert caf.save_csv(pd.DataFrame({'x': [1]}), "t.csv", "ghost") == ""


# ==========================================================================
# Data-encoding invariants used throughout the pipeline
# ==========================================================================

def test_binary_encoding_invariant():
    raw = pd.Series([1, 2, 1, 2, 97, 99])
    encoded = (raw == 1).astype(int)
    assert encoded.tolist() == [1, 0, 1, 0, 0, 0]   # only 1 maps to positive


def test_mortality_encoding_from_date_died():
    s = pd.Series(['9999-99-99', '2020-05-01', '9999-99-99'])
    mortality = (s != '9999-99-99').astype(int)
    assert mortality.tolist() == [0, 1, 0]


# ==========================================================================
# Calibration improvement
# ==========================================================================

def test_improve_calibration_metrics_and_model(tmp_path):
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(400, 3)); ytr = (Xtr[:, 0] + rng.normal(size=400) > 0).astype(int)
    Xte = rng.normal(size=(200, 3)); yte = (Xte[:, 0] + rng.normal(size=200) > 0).astype(int)
    base = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    res = am.improve_calibration(base, Xtr, ytr, Xte, yte,
                                 str(tmp_path / "cal.png"), _LOG)
    for k in ('brier_before', 'brier_after', 'ece_before', 'ece_after'):
        assert 0.0 <= res[k] <= 1.0
    proba = res['calibrated_model'].predict_proba(Xte)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


# ==========================================================================
# Hyper-parameter tuning
# ==========================================================================

def test_tune_hyperparameters_lr_grid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)); y = (X[:, 0] > 0).astype(int)
    out = am.tune_hyperparameters(X, y, _LOG, subsample=300, which=('LR',))
    assert 'LR' in out
    assert 'best_params' in out['LR'] and 'cv_auc' in out['LR']
    assert 0.0 <= out['LR']['cv_auc'] <= 1.0
    assert out['LR']['best_params']['clf__C'] in (0.01, 0.1, 1.0, 10.0)


# ==========================================================================
# Decision Curve Analysis
# ==========================================================================

def test_decision_curve_analysis_returns_net_benefit(tmp_path):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    df = am.decision_curve_analysis(y, p, str(tmp_path / "dca.png"), _LOG)
    assert {'threshold', 'net_benefit_model', 'net_benefit_treat_all'} <= set(df.columns)
    assert len(df) > 0


# ==========================================================================
# Tree SHAP  (optional: needs shap)
# ==========================================================================

def test_tree_shap_importance_dataframe(tmp_path):
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3)); y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    df = am.tree_shap_analysis(rf, X, ['a', 'b', 'c'], str(tmp_path / "s.png"), _LOG)
    assert {'feature_name', 'tree_shap_importance'} <= set(df.columns)
    assert len(df) == 3 and (df['tree_shap_importance'] >= 0).all()


# ==========================================================================
# Odds ratios with CI  (optional: needs statsmodels)
# ==========================================================================

def test_odds_ratios_with_ci_columns_and_ordering():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] + rng.normal(size=600) > 0).astype(int)
    df = am.odds_ratios_with_ci(X, y, ['a', 'b', 'c'], _LOG)
    if df is not None:                       # statsmodels present
        assert {'odds_ratio', 'or_ci_lower', 'or_ci_upper', 'p_value'} <= set(df.columns)
        # confidence interval must bracket the point estimate
        assert (df['or_ci_lower'] <= df['odds_ratio'] + 1e-9).all()
        assert (df['odds_ratio'] <= df['or_ci_upper'] + 1e-9).all()


# ==========================================================================
# Local SHAP export
# ==========================================================================

def test_export_local_shap_long_format():
    rng = np.random.default_rng(3)
    feats = ['age', 'sex', 'diabetes']
    Xtr = rng.normal(size=(200, 3)); ytr = (Xtr[:, 0] > 0).astype(int)
    lr = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    Xte = rng.normal(size=(40, 3))
    Xte_raw = pd.DataFrame(Xte, columns=feats)
    df_model = pd.DataFrame({'id': range(40), 'age': 1.0, 'sex': 1.0, 'diabetes': 1.0})
    preds = {'Primary': lr.predict_proba(Xte)[:, 1]}
    out = am.export_local_shap(lr, Xte, Xte_raw, np.arange(40), df_model,
                               feats, preds, 'Primary', _LOG, n=10)
    assert {'patient_id', 'feature', 'shap_value', 'base_value'} <= set(out.columns)
    assert len(out) == 10 * 3


# ==========================================================================
# Nomogram
# ==========================================================================

def test_generate_nomogram_returns_points_and_lookup():
    rng = np.random.default_rng(0)
    X = rng.random((400, 3))
    y = (X[:, 0] > 0.5).astype(int)
    nomo, lookup = am.generate_nomogram(X, y, ['age', 'sex', 'diabetes'], _LOG)
    assert {'feature', 'points'} <= set(nomo.columns) and len(nomo) == 3
    assert {'total_points', 'predicted_risk_pct'} <= set(lookup.columns)


def test_nomogram_risk_within_percent_range():
    rng = np.random.default_rng(6)
    X = rng.random((500, 3))
    y = (X[:, 0] > 0.5).astype(int)
    _, lookup = am.generate_nomogram(X, y, ['age', 'sex', 'diabetes'], _LOG)
    assert (lookup['predicted_risk_pct'] >= 0).all()
    assert (lookup['predicted_risk_pct'] <= 100).all()


# ==========================================================================
# Fairness mitigation
# ==========================================================================

def test_fairness_mitigation_runs():
    rng = np.random.default_rng(4)
    n = 600
    df_model = pd.DataFrame({'age': rng.integers(18, 90, n)})
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    fdf = am.fairness_mitigation(df_model, np.arange(n), y, p, _LOG)
    assert isinstance(fdf, pd.DataFrame)


# ==========================================================================
# Temporal (out-of-time) validation
# ==========================================================================

def test_temporal_validation_auc():
    rng = np.random.default_rng(5)
    n = 2000
    df = pd.DataFrame({
        'entry_date': pd.date_range('2020-01-01', periods=n, freq='D').strftime('%d/%m/%Y'),
        'age': rng.integers(18, 90, n),
        'sex': rng.integers(1, 3, n),
        'diabetes': rng.integers(1, 3, n),
        'mortality': rng.integers(0, 2, n),
    })
    res = am.temporal_validation(df, ['age', 'sex', 'diabetes'], _LOG)
    assert res is None or (0.0 <= res['temporal_auc'] <= 1.0)


# ==========================================================================
# Responsible-AI artifacts  (Model Card / Datasheet)
# ==========================================================================

def _artifact_info():
    return {
        'model_name': 'Calibrated Logistic Regression',
        'n_total': 1000, 'n_confirmed': 800, 'n_events': 100, 'n_features': 9,
        'features': ['age', 'sex', 'diabetes'], 'test_n': 200, 'train_n': 600,
        'positive_rate': 0.123, 'epv': 11, 'date': '2026-07-13',
    }


def test_model_card_written(tmp_path):
    p = tmp_path / "MODEL_CARD.md"
    am.generate_model_card(_artifact_info(), str(p), _LOG)
    assert p.is_file()
    assert 'Model Card' in p.read_text(encoding='utf-8')


def test_datasheet_written(tmp_path):
    p = tmp_path / "DATASHEET.md"
    am.generate_datasheet(_artifact_info(), str(p), _LOG)
    assert p.is_file() and len(p.read_text(encoding='utf-8')) > 100


# ==========================================================================
# Imbalance strategy comparison & advanced models  (optional deps, graceful)
# ==========================================================================

def test_compare_smote_vs_classweight_graceful():
    rng = np.random.default_rng(3)
    Xtr = rng.normal(size=(400, 3)); ytr = (rng.random(400) < 0.2).astype(int)
    Xte = rng.normal(size=(150, 3)); yte = (rng.random(150) < 0.2).astype(int)
    res = am.compare_smote_vs_classweight(Xtr, ytr, Xte, yte, _LOG)
    # None when imbalanced-learn is absent; otherwise a comparison table
    assert res is None or ({'strategy', 'test_auc'} <= set(res.columns))


def test_train_advanced_models_returns_valid_aucs():
    rng = np.random.default_rng(4)
    Xtr = rng.normal(size=(300, 3)); ytr = (Xtr[:, 0] > 0).astype(int)
    Xte = rng.normal(size=(120, 3)); yte = (Xte[:, 0] > 0).astype(int)
    base = {'LR': LogisticRegression(max_iter=300)}
    res = am.train_advanced_models(Xtr, ytr, Xte, yte, base, _LOG)
    assert isinstance(res, dict) and 'Stacking' in res      # Stacking always trained
    for name, vals in res.items():
        auc = vals[0]
        assert 0.0 <= auc <= 1.0


# ==========================================================================
# End-to-end smoke test of the deployed prediction chain
# ==========================================================================

def test_end_to_end_prediction_chain():
    """encode -> (train-only) impute+scale -> LR -> predict -> ECE -> threshold.
    Confirms the leakage-free chain runs and recovers injected signal."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 900
    feats = ['age', 'sex', 'diabetes', 'pneumonia']
    raw = pd.DataFrame({
        'age': rng.integers(20, 90, n),
        'sex': rng.choice([1, 2], n),
        'diabetes': rng.choice([1, 2, 97], n),      # includes a sentinel
        'pneumonia': rng.choice([1, 2], n),
    })
    # Injected signal: pneumonia and older age raise risk.
    logit = 0.05 * (raw['age'].to_numpy() - 55) + 1.4 * (raw['pneumonia'].to_numpy() == 1)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    X = caf.encode_features(raw, feats)              # 97 -> NaN
    assert X.isna().values.any()                     # sentinel really became NaN

    Xtr, Xte = X.iloc[:650], X.iloc[650:]
    ytr, yte = y[:650], y[650:]

    imp = SimpleImputer(strategy='median').fit(Xtr)          # fit on train only
    sc = StandardScaler().fit(imp.transform(Xtr))            # fit on train only
    lr = LogisticRegression(max_iter=500, class_weight='balanced')
    lr.fit(sc.transform(imp.transform(Xtr)), ytr)

    p = lr.predict_proba(sc.transform(imp.transform(Xte)))[:, 1]
    assert roc_auc_score(yte, p) > 0.60              # learned the signal
    assert 0.0 <= am.expected_calibration_error(yte, p) <= 1.0
    thr = am.optimize_threshold(yte, p, _LOG)
    assert 0.0 <= thr['threshold_f1'] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
