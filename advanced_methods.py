#!/usr/bin/env python3
"""
===============================================================================
ADVANCED METHODOLOGY MODULE - COVID-19 Risk Assessment
===============================================================================
Student: Durga Prasad Narsing (A00050350)
Program: MSc Computing (Data Analytics), Dublin City University

This module implements MSc-level methodological extensions to the main analysis
script (covid_analysis_full.py). Each technique is grounded in the published
literature:

- DeLong test for comparing correlated ROC-AUCs ......... DeLong et al. (1988);
  fast algorithm: Sun & Xu (2014)
- Bootstrap confidence intervals for AUC ................ Efron & Tibshirani (1993)
- Probability calibration (Platt / isotonic) ........... Platt (1999);
  Niculescu-Mizil & Caruana (2005)
- Expected Calibration Error ........................... Guo et al. (2017)
- Decision Curve Analysis (net benefit) ................ Vickers & Elkin (2006)
- Gradient-boosted trees (XGBoost) ..................... Chen & Guestrin (2016)
- Stacked generalization (stacking ensemble) ........... Wolpert (1992)
- Tree SHAP for non-linear explainability .............. Lundberg & Lee (2017);
  Lundberg et al. (2020)
- Equalized-odds style fairness mitigation ............. Hardt et al. (2016)
- Temporal (out-of-time) validation .................... Steyerberg (2019),
  Riley et al. (2021) TRIPOD guidance
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                             confusion_matrix, roc_curve)


# ===========================================================================
# 1. STATISTICAL AUC COMPARISON  (DeLong et al. 1988; Sun & Xu 2014)
# ===========================================================================

def _compute_midrank(x):
    """Mid-ranks used by the fast DeLong algorithm (Sun & Xu 2014)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """Vectorised DeLong covariance for k predictors. `m` = #positives."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, p1, p2):
    """
    Two-sided DeLong test that two correlated ROC-AUCs are equal.

    Returns (auc1, auc2, z, p_value). Used to test whether one model's AUC is
    *significantly* higher than another's on the same patients.
    """
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort(kind='mergesort')   # positives first
    m = int(y_true.sum())
    preds = np.vstack((np.asarray(p1), np.asarray(p2)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]])
    var = np.atleast_2d(l.dot(cov).dot(l.T))[0, 0].item()
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * scipy.stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ===========================================================================
# 2. BOOTSTRAP CONFIDENCE INTERVALS FOR AUC  (Efron & Tibshirani 1993)
# ===========================================================================

def bootstrap_auc_ci(y_true, y_pred, n_boot=1000, seed=42):
    """Percentile bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)


# ===========================================================================
# 3. EXPECTED CALIBRATION ERROR  (Guo et al. 2017)
# ===========================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Count-weighted ECE over uniform probability bins."""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


# ===========================================================================
# 4. PROBABILITY CALIBRATION  (Platt 1999; Niculescu-Mizil & Caruana 2005)
# ===========================================================================

def improve_calibration(base_estimator, X_train, y_train, X_test, y_test,
                        out_path, logger, method='isotonic'):
    """
    Fit a post-hoc calibrator on top of the base model and quantify the
    improvement in Brier score and ECE (before vs after).

    The main script found the ensemble was poorly calibrated (ECE ~0.19).
    This addresses that directly. Returns a results dict.
    """
    logger.info(f"Calibrating with method='{method}' ...")
    # Uncalibrated reference
    base_estimator.fit(X_train, y_train)
    p_before = base_estimator.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, p_before)
    ece_before = expected_calibration_error(y_test, p_before)

    # Calibrated (cv on the training set only - no test leakage)
    calibrated = CalibratedClassifierCV(base_estimator, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    p_after = calibrated.predict_proba(X_test)[:, 1]
    brier_after = brier_score_loss(y_test, p_after)
    ece_after = expected_calibration_error(y_test, p_after)

    logger.info(f"  Brier: {brier_before:.4f} -> {brier_after:.4f}")
    logger.info(f"  ECE:   {ece_before:.4f} -> {ece_after:.4f}")

    # Before/after reliability plot
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    for p, lbl, c in [(p_before, f'Before (ECE={ece_before:.3f})', 'coral'),
                       (p_after, f'After {method} (ECE={ece_after:.3f})', 'steelblue')]:
        pt, pp = calibration_curve(y_test, p, n_bins=10)
        plt.plot(pp, pt, 'o-', color=c, label=lbl)
    plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
    plt.title('Calibration Improvement (Platt/Isotonic)', fontweight='bold')
    plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()

    return {
        'method': method,
        'brier_before': brier_before, 'brier_after': brier_after,
        'ece_before': ece_before, 'ece_after': ece_after,
        'calibrated_model': calibrated,
        'p_after': p_after,
    }


# ===========================================================================
# 5. THRESHOLD OPTIMISATION  (Youden 1950; F1)
# ===========================================================================

def optimize_threshold(y_true, y_pred, logger):
    """
    Find clinically meaningful operating points instead of the default 0.5:
    - Youden's J (maximises sensitivity + specificity - 1)
    - F1-optimal threshold
    Returns a results dict and logs a metrics table at each threshold.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    j_idx = int(np.argmax(youden_j))
    thr_youden = float(thr[j_idx])

    # F1-optimal over a fine grid
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (y_pred >= t).astype(int), zero_division=0) for t in grid]
    thr_f1 = float(grid[int(np.argmax(f1s))])

    results = {'threshold_default': 0.5,
               'threshold_youden': thr_youden,
               'threshold_f1': thr_f1}

    logger.info("Threshold optimisation (vs default 0.50):")
    rows = []
    for name, t in [('Default 0.50', 0.5),
                    ('Youden J', thr_youden),
                    ('F1-optimal', thr_f1)]:
        pred = (y_pred >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = f1_score(y_true, pred, zero_division=0)
        logger.info(f"  {name} (t={t:.3f}): Sens={sens:.3f} Spec={spec:.3f} "
                    f"Prec={prec:.3f} F1={f1:.3f}")
        rows.append({'strategy': name, 'threshold': round(t, 3),
                     'sensitivity': round(sens, 3), 'specificity': round(spec, 3),
                     'precision': round(prec, 3), 'f1': round(f1, 3)})
    results['table'] = pd.DataFrame(rows)
    return results


# ===========================================================================
# 6. DECISION CURVE ANALYSIS  (Vickers & Elkin 2006)
# ===========================================================================

def decision_curve_analysis(y_true, y_pred, out_path, logger):
    """
    Net-benefit decision curve: compares the model against 'treat all' and
    'treat none' across the full range of clinical threshold probabilities.
    This is the clinically-correct counterpart to the cost-benefit phase.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    N = len(y_true)
    prevalence = y_true.mean()
    thresholds = np.linspace(0.01, 0.60, 60)

    nb_model, nb_all = [], []
    for pt in thresholds:
        pred_pos = y_pred >= pt
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        w = pt / (1 - pt)
        nb_model.append(tp / N - (fp / N) * w)
        nb_all.append(prevalence - (1 - prevalence) * w)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, color='steelblue', linewidth=2, label='Model')
    plt.plot(thresholds, nb_all, color='gray', linewidth=1.5, linestyle='--', label='Treat all')
    plt.axhline(0, color='black', linewidth=1, label='Treat none')
    plt.xlabel('Threshold probability'); plt.ylabel('Net benefit')
    plt.title('Decision Curve Analysis (Vickers & Elkin 2006)', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(bottom=min(0, min(nb_model)) - 0.01)
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info("Decision curve analysis saved (model vs treat-all vs treat-none)")

    return pd.DataFrame({'threshold': thresholds,
                         'net_benefit_model': nb_model,
                         'net_benefit_treat_all': nb_all})


# ===========================================================================
# 7. ADVANCED MODELS: XGBoost + Stacking  (Chen & Guestrin 2016; Wolpert 1992)
# ===========================================================================

def _try_import_xgboost(logger):
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        try:
            import subprocess, sys
            logger.info("Installing xgboost ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import xgboost as xgb
            return xgb
        except Exception:
            logger.warning("⚠️ xgboost unavailable - skipping XGBoost model")
            return None


def train_advanced_models(X_train, y_train, X_test, y_test, base_models, logger):
    """
    Train XGBoost and a stacked ensemble, benchmark AUC against the existing
    base models. Returns dict of {name: (auc, pred)}.
    """
    results = {}
    pos = int((np.asarray(y_train) == 1).sum())
    neg = int((np.asarray(y_train) == 0).sum())
    spw = (neg / pos) if pos else 1.0   # scale_pos_weight for imbalance

    xgb = _try_import_xgboost(logger)
    if xgb is not None:
        logger.info("Training XGBoost ...")
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
            eval_metric='logloss', random_state=42, n_jobs=-1, tree_method='hist'
        )
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        results['XGBoost'] = (auc, p, clf)
        logger.info(f"  ✅ XGBoost AUC: {auc:.4f}")

    # Stacked generalization: base learners -> logistic meta-learner
    logger.info("Training Stacking ensemble (Wolpert 1992) ...")
    estimators = [(name, base_models[name]) for name in ('LR', 'RF', 'GB') if name in base_models]
    # cv=2 keeps this benchmark tractable (it refits every base learner per fold);
    # Stacking is a comparison model only, not the deployed one.
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=2, n_jobs=-1, passthrough=False
    )
    stack.fit(X_train, y_train)
    p_stack = stack.predict_proba(X_test)[:, 1]
    auc_stack = roc_auc_score(y_test, p_stack)
    results['Stacking'] = (auc_stack, p_stack, stack)
    logger.info(f"  ✅ Stacking AUC: {auc_stack:.4f}")
    return results


# ===========================================================================
# 8. HYPERPARAMETER TUNING  (leakage-free pipelines, small grids)
# ===========================================================================

def tune_hyperparameters(X_train_raw, y_train, logger, subsample=40000, seed=42,
                         which=('LR', 'RF')):
    """
    GridSearchCV over leakage-free pipelines (scaler refit per fold). Subsampled
    for tractable runtime. Returns best params + CV AUC per model.

    `which` selects which models to tune. Phase 4 already tunes LR's C for the
    deployed model, so the main script passes which=('RF',) here to avoid
    re-searching LR (the LR row is injected by the caller from Phase 4's result).
    """
    X = np.asarray(X_train_raw); y = np.asarray(y_train)
    if len(X) > subsample:
        Xs, _, ys, _ = train_test_split(X, y, train_size=subsample,
                                        stratify=y, random_state=seed)
    else:
        Xs, ys = X, y

    out = {}

    if 'LR' in which:
        logger.info("Tuning Logistic Regression (C) ...")
        lr_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(max_iter=1000,
                                                       class_weight='balanced', random_state=seed))])
        lr_gs = GridSearchCV(lr_pipe, {'clf__C': [0.01, 0.1, 1.0, 10.0]},
                             cv=3, scoring='roc_auc', n_jobs=-1)
        lr_gs.fit(Xs, ys)
        out['LR'] = {'best_params': lr_gs.best_params_, 'cv_auc': lr_gs.best_score_}
        logger.info(f"  Best LR: {lr_gs.best_params_} CV AUC={lr_gs.best_score_:.4f}")

    if 'RF' not in which:
        return out

    logger.info("Tuning Random Forest (depth, n_estimators) ...")
    rf_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('clf', RandomForestClassifier(class_weight='balanced',
                                                       n_jobs=-1, random_state=seed))])
    rf_gs = GridSearchCV(rf_pipe,
                         {'clf__n_estimators': [200], 'clf__max_depth': [6, 10, None]},
                         cv=3, scoring='roc_auc', n_jobs=-1)
    rf_gs.fit(Xs, ys)
    out['RF'] = {'best_params': rf_gs.best_params_, 'cv_auc': rf_gs.best_score_}
    logger.info(f"  Best RF: {rf_gs.best_params_} CV AUC={rf_gs.best_score_:.4f}")

    return out


# ===========================================================================
# 9. TREE SHAP  (Lundberg & Lee 2017; Lundberg et al. 2020)
# ===========================================================================

def tree_shap_analysis(tree_model, X_test, feature_columns, out_path, logger, max_n=2000):
    """Exact Tree SHAP on a fitted tree model (RF/GB/XGB), summarised globally."""
    import shap
    X = np.asarray(X_test)
    if len(X) > max_n:
        rng = np.random.default_rng(42)
        X = X[rng.choice(len(X), max_n, replace=False)]
    logger.info("Computing Tree SHAP values ...")
    explainer = shap.TreeExplainer(tree_model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):           # binary classifiers may return [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                   # (n, features, classes)
        sv = sv[:, :, 1]
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_columns)[order], importance[order],
             color='seagreen', alpha=0.8, edgecolor='black')
    plt.gca().invert_yaxis()
    plt.title('Tree SHAP Feature Importance (non-linear model)', fontweight='bold')
    plt.xlabel('Mean |SHAP value|'); plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()

    df = pd.DataFrame({'feature_name': np.array(feature_columns)[order],
                       'tree_shap_importance': importance[order]})
    logger.info("Tree SHAP top features: " +
                ", ".join(f"{r.feature_name}={r.tree_shap_importance:.3f}"
                          for r in df.head(3).itertuples()))
    return df


# ===========================================================================
# 10. FAIRNESS MITIGATION  (Hardt et al. 2016 - group-specific thresholds)
# ===========================================================================

def fairness_mitigation(df_model, idx_test, y_test, y_pred, logger,
                        target_recall=0.75):
    """
    Detection + mitigation: instead of one global 0.5 cut, choose a per-age-group
    threshold that achieves a common target recall. Reports TPR disparity before
    vs after, demonstrating the gap shrinks (equalized-odds-style mitigation).
    """
    y_test = np.asarray(y_test); y_pred = np.asarray(y_pred)
    test_demo = df_model.iloc[idx_test].reset_index(drop=True)
    ages = test_demo['age'].values
    groups = [('18-30', 18, 30), ('30-50', 30, 50), ('50-65', 50, 65), ('65+', 65, 120)]

    def tpr_at(mask, t):
        yp = (y_pred[mask] >= t).astype(int); yt = y_test[mask]
        tp = np.sum((yp == 1) & (yt == 1)); fn = np.sum((yp == 0) & (yt == 1))
        return tp / (tp + fn) if (tp + fn) else 0.0

    rows = []
    for label, lo, hi in groups:
        m = (ages >= lo) & (ages < hi)
        if m.sum() == 0 or len(np.unique(y_test[m])) < 2:
            continue
        tpr_before = tpr_at(m, 0.5)
        # find smallest threshold reaching the target recall in this group
        grid = np.linspace(0.05, 0.95, 91)
        feasible = [t for t in grid if tpr_at(m, t) >= target_recall]
        t_star = max(feasible) if feasible else 0.05
        tpr_after = tpr_at(m, t_star)
        rows.append({'group': label, 'n': int(m.sum()),
                     'threshold': round(float(t_star), 3),
                     'tpr_before(0.5)': round(tpr_before, 3),
                     'tpr_after': round(tpr_after, 3)})

    fdf = pd.DataFrame(rows)
    if len(fdf):
        disp_before = fdf['tpr_before(0.5)'].max() - fdf['tpr_before(0.5)'].min()
        disp_after = fdf['tpr_after'].max() - fdf['tpr_after'].min()
        logger.info(f"Fairness mitigation (target recall={target_recall}):")
        for r in fdf.itertuples():
            logger.info(f"  Age {r.group} (n={r.n}): TPR {getattr(r,'_4'):.3f} "
                        f"-> {r.tpr_after:.3f} @ t={r.threshold}")
        logger.info(f"  TPR disparity: {disp_before:.3f} -> {disp_after:.3f}")
        fdf.attrs['disparity_before'] = disp_before
        fdf.attrs['disparity_after'] = disp_after
    return fdf


# ===========================================================================
# 11. TEMPORAL (OUT-OF-TIME) VALIDATION  (Steyerberg 2019; TRIPOD)
# ===========================================================================

def temporal_validation(df_confirmed, feature_columns, logger, date_col='entry_date'):
    """
    Train on earlier patients, test on later ones (chronological split) to
    simulate real prospective deployment - far stronger than a random split.
    """
    if date_col not in df_confirmed.columns:
        logger.warning(f"⚠️ '{date_col}' not found - skipping temporal validation")
        return None

    df = df_confirmed.copy()
    df['_date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['_date']).sort_values('_date')
    if len(df) < 1000 or 'mortality' not in df.columns:
        logger.warning("⚠️ Not enough dated rows for temporal validation")
        return None

    cutoff = df['_date'].quantile(0.8)
    train = df[df['_date'] <= cutoff]; test = df[df['_date'] > cutoff]
    if len(test) < 100 or test['mortality'].nunique() < 2:
        logger.warning("⚠️ Temporal test split too small/degenerate - skipping")
        return None

    def prep(frame):
        # Sentinel-aware encoding (1=Yes, 2=No, 97/98/99->NaN); NaNs imputed in pipeline.
        X = frame[feature_columns].copy()
        for c in feature_columns:
            if c == 'age':
                X[c] = pd.to_numeric(X[c], errors='coerce')
            else:
                X[c] = pd.to_numeric(X[c], errors='coerce').map({1: 1.0, 2: 0.0})
        return X.astype(float)

    Xtr, ytr = prep(train), train['mortality']
    Xte, yte = prep(test), test['mortality']

    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                                random_state=42))])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    logger.info(f"Temporal validation: train n={len(train):,} (≤{cutoff.date()}), "
                f"test n={len(test):,} (>{cutoff.date()}) → AUC={auc:.4f}")
    return {'cutoff': str(cutoff.date()), 'train_n': len(train),
            'test_n': len(test), 'temporal_auc': float(auc)}


# ===========================================================================
# 12. ODDS RATIOS WITH 95% CI + p-VALUES  (statsmodels Logit; Wald CIs)
# ===========================================================================

def odds_ratios_with_ci(X_train_scaled, y_train, feature_columns, logger):
    """
    Re-fit the logistic model with statsmodels to obtain, for each feature, the
    Odds Ratio with a 95% Wald confidence interval and a p-value - the proper,
    publishable form of the proposal's e^β interpretability deliverable.

    Inputs are the imputed+scaled training features (so ORs are 'per +1 SD').
    Falls back gracefully if statsmodels is unavailable or the fit fails.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        try:
            import subprocess, sys
            logger.info("Installing statsmodels ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels", "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import statsmodels.api as sm
        except Exception:
            logger.warning("⚠️ statsmodels unavailable - skipping OR confidence intervals")
            return None

    X = np.asarray(X_train_scaled)
    y = np.asarray(y_train).astype(int)
    Xc = sm.add_constant(X)
    try:
        res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    except Exception as e:
        logger.warning(f"⚠️ statsmodels Logit failed ({e}) - skipping OR CIs")
        return None

    params = res.params[1:]            # drop intercept
    conf = res.conf_int()[1:]
    pvals = res.pvalues[1:]
    rows = []
    for i, feat in enumerate(feature_columns):
        rows.append({
            'feature_name': feat,
            'odds_ratio': float(np.exp(params[i])),
            'or_ci_lower': float(np.exp(conf[i, 0])),
            'or_ci_upper': float(np.exp(conf[i, 1])),
            'p_value': float(pvals[i]),
        })
    df = pd.DataFrame(rows).sort_values('odds_ratio', ascending=False).reset_index(drop=True)
    logger.info("Odds Ratios with 95% CI (statsmodels, per +1 SD):")
    for r in df.itertuples():
        star = '***' if r.p_value < 0.001 else '**' if r.p_value < 0.01 else \
               '*' if r.p_value < 0.05 else 'ns'
        logger.info(f"  {r.feature_name}: OR={r.odds_ratio:.3f} "
                    f"[{r.or_ci_lower:.3f}–{r.or_ci_upper:.3f}], p={r.p_value:.2e} {star}")
    return df


# ===========================================================================
# 13. MODEL CARD  (Mitchell et al. 2019)  + DATASHEET (Gebru et al. 2021)
# ===========================================================================

def generate_model_card(info: dict, out_path, logger):
    """
    Write a one-page Model Card (Mitchell et al. 2019) capturing intended use,
    performance, fairness, calibration and limitations - including the explicit
    mortality-proxy caveat. Saved as Markdown for the thesis appendix.
    """
    md = f"""# Model Card - COVID-19 Mortality Risk Model

**Author:** Durga Prasad Narsing (A00050350) - MSc Computing (Data Analytics), DCU
**Supervisors:** Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)
**Date generated:** {info.get('date', '')}

## Model details
- **Algorithm:** {info.get('model_name', 'Ensemble (LR + RF + GB)')} on standardised, median-imputed acute-phase features.
- **Features ({len(info.get('features', []))}):** {', '.join(info.get('features', []))}
- **Training / test split:** {info.get('train_n', '?')} / {info.get('test_n', '?')} (stratified, leakage-free: imputer + scaler fit on train only).

## Intended use
- **Purpose:** Stratify confirmed COVID-19 patients by risk of in-hospital mortality from acute-phase data, to support earlier clinical review.
- **Users:** Researchers / clinical decision-support prototyping. **Not** a validated clinical device.
- **Out of scope:** Individual treatment decisions without clinician oversight.

## ⚠️ Important caveat - target is a proxy
The dataset (Mexican Government, Kaggle) contains acute-phase data and in-hospital **mortality**, but **no Long COVID / PASC follow-up labels**. Mortality is therefore used as a *proxy* for severe adverse outcomes. Sequelae-specific Long COVID prediction is future work requiring EHR datasets with post-acute follow-up.

## Performance (held-out test set)
- **Ensemble AUC:** {info.get('ensemble_auc', '?')} (95% CI {info.get('ensemble_ci', '?')})
- **Best single model (DeLong-tested):** {info.get('delong_summary', 'no model significantly beats Logistic Regression - interpretable LR preferred')}
- **Temporal (out-of-time) AUC:** {info.get('temporal_auc', 'n/a')}

## Calibration
- **Before:** Brier {info.get('brier_before', '?')}, ECE {info.get('ece_before', '?')}
- **After isotonic calibration:** Brier {info.get('brier_after', '?')}, ECE {info.get('ece_after', '?')}

## Fairness
- Audited across age, sex and comorbidity subgroups (AUC) with a 5% disparity threshold.
- Age 50–65 and 65+ flagged on AUC; mitigated via group-specific thresholds, reducing TPR disparity from {info.get('disp_before', '?')} to {info.get('disp_after', '?')}.

## Ethical considerations & limitations
- Single-country data; external/multi-site validation not performed.
- Mortality proxy (see caveat); unknown sentinels (97/98/99) treated as missing and median-imputed (~0.3% of comorbidity fields).
- Risk scores must be interpreted alongside clinical judgement.

## Top risk factors (Odds Ratios, per +1 SD)
{info.get('or_table', '_see covid_odds_ratios_ci.csv_')}

---
*Generated automatically by advanced_methods.generate_model_card(). Format: Mitchell et al. (2019), "Model Cards for Model Reporting".*
"""
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md)
        logger.info(f"✅ Model card written to {out_path}")
    except Exception as e:
        logger.error(f"❌ Model card write failed: {e}")
    return out_path


# ===========================================================================
# 14. PER-PATIENT (LOCAL) SHAP  (Lundberg & Lee 2017)
# ===========================================================================

def export_local_shap(lr_model, X_test, X_test_raw, idx_test, df_model,
                      feature_columns, predictions, primary_key, logger,
                      n=100, seed=42):
    """
    Per-patient SHAP contributions for a sample of test patients - lets the
    app explain *why THIS patient's* risk is what it is (local explanation),
    not just global importance.

    Returns a long-format DataFrame:
      patient_id, risk_score, feature, feature_value(raw), shap_value, base_value
    """
    import shap
    X_test = np.asarray(X_test)
    explainer = shap.LinearExplainer(lr_model, X_test)
    base = float(np.array(explainer.expected_value).ravel()[0])

    rng = np.random.default_rng(seed)
    n = min(n, len(X_test))
    sel = rng.choice(len(X_test), n, replace=False)
    sv = np.asarray(explainer.shap_values(X_test[sel]))

    test_rows = df_model.iloc[idx_test].reset_index(drop=True)
    ids = (test_rows['id'].values if 'id' in test_rows.columns
           else test_rows.index.values)
    Xr = (X_test_raw.reset_index(drop=True) if hasattr(X_test_raw, 'reset_index')
          else pd.DataFrame(np.asarray(X_test_raw), columns=feature_columns))

    rows = []
    for i, s in enumerate(sel):
        pid = ids[s]
        risk = float(predictions[primary_key][s]) * 100
        for j, feat in enumerate(feature_columns):
            rows.append({
                'patient_id': pid,
                'risk_score': round(risk, 1),
                'feature': feat,
                'feature_value': float(Xr.iloc[s][feat]),
                'shap_value': float(sv[i, j]),
                'base_value': base,
            })
    df = pd.DataFrame(rows)
    logger.info(f"Per-patient SHAP exported for {n} patients "
                f"({len(df)} rows, base value={base:.4f})")
    return df


# ===========================================================================
# 15. NOMOGRAM / POINTS-BASED RISK SCORE  (Steyerberg 2019)
# ===========================================================================

def generate_nomogram(X_train_raw_imputed, y_train, feature_columns, logger,
                      points_per_logit=10):
    """
    Build a clinician-friendly points-based risk score (nomogram) from a logistic
    regression fit on RAW (unscaled) units - 0/1 for conditions, years for age.

    For each feature:
      log-odds contribution per unit = coefficient (raw)
      points per unit                = round(coef * unit * points_per_logit)
    'unit' is 1 for binary features (points awarded if condition present) and 10
    for age (points per decade). A patient's TOTAL points map back to risk via
        logit = intercept_logodds + total_points / points_per_logit
        risk  = 1 / (1 + exp(-logit))

    Returns (nomogram_df, lookup_df).
    """
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(np.asarray(X_train_raw_imputed), np.asarray(y_train))
    coefs = lr.coef_[0]
    intercept = float(lr.intercept_[0])

    rows = []
    for j, feat in enumerate(feature_columns):
        unit = 10.0 if feat == 'age' else 1.0
        unit_label = 'per 10 years' if feat == 'age' else 'if present (Yes)'
        log_odds = coefs[j] * unit
        rows.append({
            'feature': feat,
            'unit': unit_label,
            'coefficient_raw': round(float(coefs[j]), 5),
            'odds_ratio_per_unit': round(float(np.exp(log_odds)), 3),
            'points': int(round(log_odds * points_per_logit)),
        })
    nomo = pd.DataFrame(rows).sort_values('points', ascending=False, key=abs).reset_index(drop=True)

    baseline_points = int(round(intercept * points_per_logit))
    logger.info(f"Nomogram (points-based score, intercept={baseline_points} pts):")
    for r in nomo.itertuples():
        logger.info(f"  {r.feature} ({r.unit}): {r.points:+d} pts (OR={r.odds_ratio_per_unit})")

    # Total-points -> risk lookup
    pts_range = np.arange(-20, 81, 5)
    lookup = []
    for tp in pts_range:
        logit = intercept + (tp / points_per_logit)
        risk = 1.0 / (1.0 + np.exp(-logit))
        lookup.append({'total_points': int(tp), 'predicted_risk_pct': round(risk * 100, 1)})
    lookup_df = pd.DataFrame(lookup)

    # Stash metadata for the app / downstream consumers
    nomo.attrs['intercept_logodds'] = intercept
    nomo.attrs['baseline_points'] = baseline_points
    nomo.attrs['points_per_logit'] = points_per_logit
    return nomo, lookup_df


# ===========================================================================
# 16. DATASHEET FOR THE DATASET  (Gebru et al. 2021)
# ===========================================================================

def generate_datasheet(info: dict, out_path, logger):
    """
    Write a 'Datasheet for the dataset' (Gebru et al. 2021) - a structured record
    of the dataset's motivation, composition, collection, preprocessing, uses,
    distribution and maintenance. Complements the Model Card (which documents the
    *model*). Saved as Markdown for the thesis appendix.
    """
    md = f"""# Datasheet - COVID-19 Dataset (Mexican Government, via Kaggle)

*Format: Gebru et al. (2021), "Datasheets for Datasets".*
**Compiled by:** Durga Prasad Narsing (A00050350), MSc Computing (Data Analytics), DCU
**Supervisors:** Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)
**Date:** {info.get('date', '')}

## Motivation
- **Purpose:** Epidemiological surveillance of COVID-19 cases reported by the Mexican
  Federal Government. Re-used here to model mortality risk from acute-phase data as a
  proxy for severe adverse outcomes.
- **Created by:** Mexican Government (Secretaría de Salud); redistributed on Kaggle.

## Composition
- **Instances:** {info.get('n_total', '566,602')} patient records; **{info.get('n_confirmed', '220,218')}** confirmed-positive used for modelling.
- **Each instance:** one patient with demographics, comorbidities, severity markers, and outcome (`date_died`).
- **Features used (9):** {', '.join(info.get('features', []))}.
- **Label:** mortality (derived from `date_died`; `9999-99-99` = survived). Positive rate {info.get('positive_rate', '12.31%')}.
- **Events-per-variable (EPV):** {info.get('epv', '?')} ({info.get('n_events', '?')} events / {info.get('n_features', '9')} predictors) - far above the ≥10 guideline (Riley et al. 2020), indicating ample data and low overfitting risk.
- **Missing data:** binary fields use 97/98/99 as "unknown" sentinels (~0.35–0.39% per comorbidity); decoded to NaN and median-imputed.

## Collection process
- Routinely collected clinical/administrative surveillance data at point of care; not collected for ML.
- Acute-phase variables recorded at/around initial COVID-19 presentation.

## Preprocessing / cleaning
- Removed invalid ages (≤0 or >120). Encoded 1=Yes/2=No → 1/0; sentinels → NaN → median imputation.
- Excluded `intubed` and `icu` (post-admission treatment outcomes → target leakage).
- Standardised features; imputer + scaler fit on the training split only.

## Uses
- **Used for:** mortality-risk prediction, fairness/calibration auditing, explainability.
- **Should NOT be used for:** inferring Long COVID/PASC sequelae (no follow-up labels), or
  individual clinical decisions without oversight.

## Distribution & licensing
- Publicly available via Kaggle (Mexican Government open data). Verify the dataset licence
  before redistribution.

## Maintenance
- Static snapshot; not actively maintained by this project. No personally identifying
  information is used in modelling.

## Ethical considerations
- Single-country data → limited generalisability; older age groups (50+) show weaker AUC
  (audited and partially mitigated). Mortality is a proxy for severe outcomes.
"""
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md)
        logger.info(f"✅ Datasheet written to {out_path}")
    except Exception as e:
        logger.error(f"❌ Datasheet write failed: {e}")
    return out_path


# ===========================================================================
# 17. IMBALANCE STRATEGY COMPARISON: SMOTE vs class_weight
#     (Chawla et al. 2002 SMOTE; supports an evidence-based design choice)
# ===========================================================================

def compare_smote_vs_classweight(X_train, y_train, X_test, y_test, logger):
    """
    Compare two ways of handling the 1:7 class imbalance on the same LR:
      (a) cost-sensitive reweighting via class_weight='balanced' (what we deploy)
      (b) SMOTE oversampling of the minority class (no class weighting)

    Reports test AUC for each so the imbalance strategy is an evidence-based
    choice rather than an assertion. Returns a comparison DataFrame, or None if
    imbalanced-learn is unavailable.
    """
    X_train = np.asarray(X_train); y_train = np.asarray(y_train)

    # (a) class_weight baseline
    lr_cw = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_cw.fit(X_train, y_train)
    auc_cw = roc_auc_score(y_test, lr_cw.predict_proba(X_test)[:, 1])

    # (b) SMOTE
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        try:
            import subprocess, sys
            logger.info("Installing imbalanced-learn for the SMOTE comparison ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from imblearn.over_sampling import SMOTE
        except Exception:
            logger.warning("⚠️ imbalanced-learn unavailable - skipping SMOTE comparison")
            return None

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    lr_sm = LogisticRegression(max_iter=1000, random_state=42)   # SMOTE already balances classes
    lr_sm.fit(X_res, y_res)
    auc_sm = roc_auc_score(y_test, lr_sm.predict_proba(X_test)[:, 1])

    synth = int(len(y_res) - len(y_train))
    logger.info("Imbalance strategy comparison (Logistic Regression):")
    logger.info(f"  class_weight='balanced': AUC={auc_cw:.4f} (no synthetic data)")
    logger.info(f"  SMOTE oversampling:      AUC={auc_sm:.4f} (+{synth:,} synthetic patients)")
    chosen = "class_weight" if auc_cw >= auc_sm - 0.002 else "SMOTE"
    logger.info(f"  → chosen: {chosen} "
                f"(class_weight preferred unless SMOTE is clearly better, as it adds no synthetic patients)")

    return pd.DataFrame([
        {'strategy': "class_weight='balanced'", 'test_auc': round(float(auc_cw), 4),
         'n_train': int(len(y_train)), 'synthetic_samples': 0},
        {'strategy': 'SMOTE oversampling', 'test_auc': round(float(auc_sm), 4),
         'n_train': int(len(y_res)), 'synthetic_samples': synth},
    ])
