#!/usr/bin/env python3
"""
Full model benchmark: every model gets the SAME treatment so the comparison table
has no empty cells and is fair.

For each model (LR, RF, GB, XGBoost, Stacking) we:
  1. train on the same stratified 80/20 split (leakage-free: impute + scale on train),
  2. isotonic-calibrate it (CalibratedClassifierCV),
  3. report AUC (threshold-free), and at the model's OWN F1-optimal threshold the
     sensitivity / specificity / precision / F1, plus Brier and ECE (after calibration).

Writes analysis_output/advanced/covid_model_benchmark_full.csv.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

FEATURES = ['age','sex','diabetes','hypertension','cardiovascular','pneumonia','obesity','asthma','copd']
SEED = 42

def ece(y, p, n_bins=10):
    y = np.asarray(y); p = np.asarray(p); bins = np.linspace(0,1,n_bins+1); N=len(y); e=0.0
    for i in range(n_bins):
        m = (p>bins[i])&(p<=bins[i+1]) if i>0 else (p>=bins[0])&(p<=bins[1])
        if m.sum()>0: e += (m.sum()/N)*abs(y[m].mean()-p[m].mean())
    return float(e)

def f1_opt_metrics(y, p):
    grid = np.linspace(0.01,0.99,99)
    f1s = [f1_score(y,(p>=t).astype(int),zero_division=0) for t in grid]
    t = float(grid[int(np.argmax(f1s))])
    tn,fp,fn,tp = confusion_matrix(y,(p>=t).astype(int)).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    return t, sens, spec, prec, f1_score(y,(p>=t).astype(int),zero_division=0)

print("Loading + preprocessing ...")
df = pd.read_csv("covid.csv")
df = df[df["covid_res"]==1].copy()
df = df[(df["age"]>0)&(df["age"]<=120)]
df["mortality"] = (df["date_died"]!="9999-99-99").astype(int)
X = df[FEATURES].copy()
for c in FEATURES:
    X[c] = pd.to_numeric(X[c],errors="coerce") if c=="age" else pd.to_numeric(X[c],errors="coerce").map({1:1.0,2:0.0})
X = X.astype(float); y = df["mortality"].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=SEED)
imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
Xtr_s = sc.transform(imp.transform(Xtr)); Xte_s = sc.transform(imp.transform(Xte))
spw = (ytr==0).sum()/(ytr==1).sum()

import xgboost as xgb
base = {
    "Logistic Regression (deployed)": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.01, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1, tree_method="hist"),
}
base["Stacking Ensemble"] = StackingClassifier(
    estimators=[("lr",base["Logistic Regression (deployed)"]),("rf",base["Random Forest"]),("gb",base["Gradient Boosting"])],
    final_estimator=LogisticRegression(max_iter=1000), cv=2, n_jobs=-1)

rows=[]
for name, est in base.items():
    print(f"  calibrating + evaluating {name} ...")
    # Identical calibration treatment for every model so the comparison is fair.
    cv = 2 if name=="Stacking Ensemble" else 3
    cal = CalibratedClassifierCV(est, method="isotonic", cv=cv)
    cal.fit(Xtr_s, ytr)
    p = cal.predict_proba(Xte_s)[:,1]
    auc = roc_auc_score(yte,p)
    t,sens,spec,prec,f1 = f1_opt_metrics(yte,p)
    rows.append({"Model":name,"AUC":round(auc,4),"Threshold":round(t,2),
                 "Sensitivity":round(sens,3),"Specificity":round(spec,3),
                 "Precision":round(prec,3),"F1":round(f1,3),
                 "Brier":round(brier_score_loss(yte,p),4),"ECE":round(ece(yte,p),4),
                 "Deployed":"Yes" if name.startswith("Logistic") else "No"})

out = pd.DataFrame(rows)
out.to_csv("analysis_output/advanced/covid_model_benchmark_full.csv", index=False)
print("\n", out.to_string(index=False))
print("\nSaved analysis_output/advanced/covid_model_benchmark_full.csv")
