#!/usr/bin/env python3
"""
===============================================================================
COMPLETE COVID-19 LONG COVID RISK ASSESSMENT ANALYSIS - MAIN SCRIPT
===============================================================================
All Phases 1-12 with 35+ Visualizations
Student: Durga Prasad Narsing (A00050350)
Program: MSc Computing (Data Analytics), Dublin City University
Supervisors: Dr Martin Crane; Dr Tai Tan Mai (Assistant Professor, School of Computing, DCU)
Project: Predictive Modelling for Personalized Long COVID Risk Assessment

DESCRIPTION:
This is the MAIN script containing:
- Phase 1:  EDA (visualizations)
- Phase 2:  Preprocessing (sentinel-aware missing-data handling)
- Phase 3:  Statistical Tests (t-test, chi-square + Benjamini-Hochberg FDR)
- Phase 4:  Model Training (LR/RF/GB/Ensemble) + PRIMARY deployed model
            (calibrated, tuned Logistic Regression used by all later phases)
- Phase 5:  SHAP Analysis (linear)
- Phase 6:  Fairness Analysis
- Phase 7:  Calibration
- Phase 8:  Cost-Benefit (model-driven)
- Phase 9:  Clinical Explanations
- Phase 10: Feature Interactions
- Phase 11: Sensitivity (what-if on the deployed model)
- Phase 13: Advanced methodology (tuning, XGBoost/Stacking, AUC CIs, DeLong,
            calibration, threshold opt, decision-curve, Tree SHAP, fairness
            mitigation, temporal validation, statsmodels ORs, model card)
- Phase 12: Export results (runs last)

Companion module: advanced_methods.py   Tests: test_analysis.py

USAGE:
    python covid_analysis_full.py --data covid.csv
    python covid_analysis_full.py --data /path/to/covid.csv
    python covid_analysis_full.py          # uses DATA_FILE fallback below
===============================================================================
"""

import os, sys, warnings, logging, argparse
from pathlib import Path
from typing import Tuple, Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend - fixes tkinter thread crash on Windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                            brier_score_loss, precision_recall_curve, accuracy_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import joblib
import shap

warnings.filterwarnings('ignore')

# Force UTF-8 on the console as early as possible (before any emoji print).
# Windows default stdout is cp1252 and raises UnicodeEncodeError on emoji.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

def install_missing_libraries():
    """
    Auto-install required packages if not already installed.

    Uses subprocess.check_call (not os.system) for secure, reliable installs.
    Also writes a requirements.txt to the working directory so the environment
    is fully reproducible.

    Packages installed:
    - pandas: Data manipulation
    - numpy: Numerical computing
    - matplotlib: Visualization
    - seaborn: Statistical visualization
    - scipy: Scientific computing
    - scikit-learn: Machine learning
    - shap: Model interpretability
    """
    import subprocess

    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'shap': 'shap'
    }

    for name, pip_name in packages.items():
        try:
            __import__(name)
        except ImportError:
            print(f"Installing missing package: {pip_name}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    # Write requirements.txt so the environment is reproducible
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        reqs = [
            "pandas>=1.5",
            "numpy>=1.23",
            "matplotlib>=3.6",
            "seaborn>=0.12",
            "scipy>=1.9",
            "scikit-learn>=1.1",
            "shap>=0.41",
            "xgboost>=1.7",
            "statsmodels>=0.13",
            "joblib>=1.2",
            "streamlit>=1.28",
        ]
        with open(req_path, "w") as f:
            f.write("\n".join(reqs) + "\n")
        print(f"✅ requirements.txt written to {os.path.abspath(req_path)}")

install_missing_libraries()

CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'output_folder': 'analysis_output',
    'log_file': 'covid_analysis.log',
    'bias_threshold': 0.05,
    'primary_key': 'Primary',   # the deployed model used by all downstream phases
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file'], encoding='utf-8'),  # UTF-8 for emoji in log file
        logging.StreamHandler(sys.stdout)                           # stdout now UTF-8 (see above)
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load COVID-19 dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing COVID data
        
    Returns:
        pd.DataFrame: Loaded dataframe with patient records
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: Any other CSV loading errors
    """
    try:
        logger.info(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"✅ Successfully loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise

def create_folders():
    """
    Create organized output folder structure for all analysis outputs.
    
    Creates folders for:
    - Core data (raw, processed, predictions)
    - Analysis results (SHAP, statistics)
    - Fairness and calibration outputs
    - Cost-benefit analysis
    - Clinical explanations
    - Feature analysis
    - All visualization categories
    """
    folders = [
        f"{CONFIG['output_folder']}/core_data",
        f"{CONFIG['output_folder']}/analysis",
        f"{CONFIG['output_folder']}/fairness_calibration",
        f"{CONFIG['output_folder']}/cost_benefit",
        f"{CONFIG['output_folder']}/clinical_explanations",
        f"{CONFIG['output_folder']}/feature_analysis",
        f"{CONFIG['output_folder']}/visualizations/eda_plots",
        f"{CONFIG['output_folder']}/visualizations/calibration_plots",
        f"{CONFIG['output_folder']}/visualizations/shap_plots",
        f"{CONFIG['output_folder']}/visualizations/fairness_plots",
        f"{CONFIG['output_folder']}/visualizations/model_comparison_plots",
        f"{CONFIG['output_folder']}/visualizations/advanced_plots",
        f"{CONFIG['output_folder']}/advanced",
        f"{CONFIG['output_folder']}/models",
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    logger.info("✅ Created output folder structure")

def save_csv(df: pd.DataFrame, filename: str, subfolder: str) -> str:
    """
    Save DataFrame to CSV file in specified subfolder.

    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of output file
        subfolder (str): Subfolder within output_folder

    Returns:
        str: Full path to saved file or empty string on error
    """
    try:
        filepath = f"{CONFIG['output_folder']}/{subfolder}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Saved: {filename} ({len(df):,} rows)")
        return filepath
    except Exception as e:
        logger.error(f"❌ Error saving {filename}: {str(e)}")
        return ""

def encode_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Encode model features with PROPER missing-data handling.

    The Mexican Government dataset codes binary fields as 1=Yes, 2=No and uses
    97/98/99 as "unknown / not applicable" sentinels. The earlier pipeline used
    (x == 1) which silently mapped those sentinels to 0 ("No"), conflating
    *unknown* with *negative*. This helper instead maps:
        1            -> 1.0  (Yes)
        2            -> 0.0  (No)
        anything else (97/98/99/blank) -> NaN  (genuinely missing)
    'age' is passed through unchanged. NaNs are imputed downstream inside a
    leakage-free pipeline (median strategy; see Cetin & Yildiz 2022 on
    preprocessing). Returns a float DataFrame.
    """
    X = df[feature_columns].copy()
    for col in feature_columns:
        if col == 'age':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        else:
            mapped = pd.to_numeric(X[col], errors='coerce')
            X[col] = mapped.map({1: 1.0, 2: 0.0})   # 97/98/99/other -> NaN
    return X.astype(float)

def _benjamini_hochberg(pvals: List[float]) -> List[float]:
    """
    Benjamini–Hochberg FDR-adjusted p-values for a list of raw p-values.
    Controls the false discovery rate across simultaneous tests (here, the
    5 condition-vs-mortality chi-square tests).
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj.tolist()

# ============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS - 8 VISUALIZATIONS
# ============================================================================

def phase1_eda(df: pd.DataFrame) -> None:
    """
    Exploratory data analysis with 8 visualizations.
    
    Creates visualizations showing:
    1. Age distribution
    2. Sex distribution
    3. Condition prevalence
    4. Mortality overview
    5. Mortality by age group
    6. Missing data analysis
    7. Comorbidity distribution
    8. Patient type breakdown
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: EXPLORATORY DATA ANALYSIS (8 VISUALIZATIONS)")
    logger.info("="*80)
    logger.info(f"Dataset contains {len(df):,} records with {len(df.columns)} columns")
    
    try:
        # VIS 1: Age distribution - shows age range of patient population
        if 'age' in df.columns:
            logger.info(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
            plt.figure(figsize=(10, 5))
            plt.hist(df['age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            plt.title('Age Distribution - All Patients', fontsize=12, fontweight='bold')
            plt.xlabel('Age (years)')
            plt.ylabel('Number of Patients')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/01_age_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 1: Age distribution created")

        # VIS 2: Sex distribution - gender breakdown
        if 'sex' in df.columns:
            sex_counts = df['sex'].value_counts()
            logger.info(f"Sex distribution: {dict(sex_counts)}")
            plt.figure(figsize=(8, 5))
            sex_labels = {1: 'Female', 2: 'Male'}
            colors = ['#FF69B4', '#4169E1']
            plt.bar(range(len(sex_counts)), sex_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            plt.title('Sex Distribution - Patient Population', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Patients')
            plt.xticks(range(len(sex_counts)), [sex_labels.get(x, f'Unknown-{x}') for x in sex_counts.index])
            plt.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(sex_counts.values):
                plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/02_sex_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 2: Sex distribution created")

        # VIS 3: Condition prevalence - how common are comorbidities
        conditions = ['diabetes', 'hypertension', 'cardiovascular', 'pneumonia', 'obesity', 'asthma', 'copd']
        cond_prev = {}
        for cond in conditions:
            if cond in df.columns:
                cond_prev[cond] = (df[cond] == 1).sum() / len(df) * 100
        
        if cond_prev:
            logger.info("Condition Prevalence:")
            for cond, prev in sorted(cond_prev.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {cond.upper()}: {prev:.1f}%")
            
            plt.figure(figsize=(10, 6))
            sorted_conds = sorted(cond_prev.items(), key=lambda x: x[1], reverse=True)
            cond_names = [c[0] for c in sorted_conds]
            cond_values = [c[1] for c in sorted_conds]
            plt.barh(cond_names, cond_values, color='coral', alpha=0.7, edgecolor='black', linewidth=2)
            plt.title('Condition Prevalence in Patient Population', fontsize=12, fontweight='bold')
            plt.xlabel('Percentage (%)')
            plt.grid(True, alpha=0.3, axis='x')
            for i, v in enumerate(cond_values):
                plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/03_condition_prevalence.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 3: Condition prevalence created")

        # VIS 4: Mortality overview - overall survival rates
        if 'date_died' in df.columns:
            total = len(df)
            died = (df['date_died'] != '9999-99-99').sum()
            mortality_pct = (died / total * 100) if total > 0 else 0
            logger.info(f"Mortality: {died:,} deaths out of {total:,} ({mortality_pct:.2f}%)")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].bar(['Overall Population'], [mortality_pct], color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=2)
            axes[0].set_title('Overall Mortality Rate', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Mortality Rate (%)')
            axes[0].set_ylim([0, 100])
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].text(0, mortality_pct + 2, f'{mortality_pct:.2f}%', ha='center', fontweight='bold')
            
            axes[1].pie([mortality_pct, 100-mortality_pct], labels=['Died', 'Survived'], 
                       autopct='%1.1f%%', colors=['#FF6B6B', '#95E1D3'], startangle=90)
            axes[1].set_title('Patient Outcome Distribution', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/04_mortality_overview.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 4: Mortality overview created")

        # VIS 5: Mortality by age group - how mortality changes with age
        if 'age' in df.columns and 'date_died' in df.columns:
            df_temp = df.copy()
            df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 120])
            mort_by_age = df_temp.groupby('age_group', observed=True)['date_died'].apply(
                lambda x: (x != '9999-99-99').sum() / len(x) * 100 if len(x) > 0 else 0)
            
            plt.figure(figsize=(10, 5))
            mort_by_age.plot(kind='bar', color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
            plt.title('Mortality Rate by Age Group', fontsize=12, fontweight='bold')
            plt.xlabel('Age Group')
            plt.ylabel('Mortality Rate (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/05_mortality_by_age.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 5: Mortality by age created")

        # VIS 6: Missing data - data quality assessment
        missing_data = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        if missing_data.sum() > 0:
            logger.info(f"Total missing values: {missing_data.sum():.2f}%")
            plt.figure(figsize=(10, 5))
            plt.barh(missing_data[missing_data > 0].index, missing_data[missing_data > 0].values, 
                    color='#FF9999', alpha=0.7, edgecolor='black', linewidth=2)
            plt.title('Missing Data Percentage by Column', fontsize=12, fontweight='bold')
            plt.xlabel('Missing Data (%)')
            plt.grid(True, alpha=0.3, axis='x')
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/06_missing_data.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 6: Missing data created")

        # VIS 7: Comorbidity distribution - how many conditions per patient
        if all(c in df.columns for c in ['diabetes', 'hypertension', 'cardiovascular']):
            comorbidity_counts = df[['diabetes', 'hypertension', 'cardiovascular']].apply(
                lambda x: (x == 1).sum(), axis=1).value_counts().sort_index()
            logger.info(f"Comorbidity distribution: {dict(comorbidity_counts)}")
            
            plt.figure(figsize=(10, 5))
            plt.bar(comorbidity_counts.index, comorbidity_counts.values, 
                   color='#90EE90', alpha=0.7, edgecolor='black', linewidth=2)
            plt.title('Comorbidity Count Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Number of Comorbidities (from 3 major conditions)')
            plt.ylabel('Number of Patients')
            plt.grid(True, alpha=0.3, axis='y')
            for i, (idx, val) in enumerate(zip(comorbidity_counts.index, comorbidity_counts.values)):
                plt.text(idx, val + 1000, str(val), ha='center', fontweight='bold')
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/07_comorbidity_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 7: Comorbidity distribution created")

        # VIS 8: Patient type breakdown - inpatient vs outpatient
        if 'patient_type' in df.columns:
            pt_counts = df['patient_type'].value_counts()
            logger.info(f"Patient type distribution: {dict(pt_counts)}")
            
            plt.figure(figsize=(8, 5))
            colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#FFD700']
            plt.pie(pt_counts.values, labels=[f'Type {x}' for x in pt_counts.index], 
                   autopct='%1.1f%%', colors=colors[:len(pt_counts)], startangle=90)
            plt.title('Patient Type Distribution', fontsize=12, fontweight='bold')
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/eda_plots/08_patient_type.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 8: Patient type created")
        
        # Note: VIS 6 (missing-data plot) only renders if the dataset actually
        # contains NaNs. This dataset uses sentinel codes (9999-99-99, 97/98/99)
        # rather than NaN, so VIS 6 is normally skipped and 7 plots are produced.
        logger.info("✅ PHASE 1 COMPLETE: EDA visualizations created")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 Error: {str(e)}")

# ============================================================================
# PHASE 2: DATA PREPROCESSING
# ============================================================================

def phase2_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data into two datasets for analysis.
    
    Creates:
    1. Full dataset: All 566K records (for understanding data quality)
    2. Confirmed dataset: Only COVID-confirmed cases (for model training)
    
    Cleans:
    - Age outliers (removes age < 0 or > 120)
    - Creates mortality indicator from date_died column
    
    Args:
        df (pd.DataFrame): Raw input dataframe
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (full_dataset, confirmed_dataset)
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: DATA PREPROCESSING (DUAL DATASET APPROACH)")
    logger.info("="*80)
    
    try:
        # Dataset 1: Full data for quality assessment
        df_full = df.copy()
        if 'age' in df_full.columns:
            initial_len = len(df_full)
            df_full = df_full[(df_full['age'] > 0) & (df_full['age'] <= 120)]
            removed = initial_len - len(df_full)
            if removed > 0:
                logger.info(f"Removed {removed} records with invalid age")
        
        logger.info(f"✅ Full dataset: {len(df_full):,} records")
        
        # Dataset 2: Confirmed cases only
        df_confirmed = df[df['covid_res'] == 1].copy() if 'covid_res' in df.columns else df.copy()
        if 'age' in df_confirmed.columns:
            initial_len = len(df_confirmed)
            df_confirmed = df_confirmed[(df_confirmed['age'] > 0) & (df_confirmed['age'] <= 120)]
            removed = initial_len - len(df_confirmed)
            if removed > 0:
                logger.info(f"Removed {removed} confirmed cases with invalid age")
        
        logger.info(f"✅ Confirmed dataset: {len(df_confirmed):,} records")
        confirmation_pct = (len(df_confirmed) / len(df_full) * 100) if len(df_full) > 0 else 0
        logger.info(f"📊 Confirmation rate: {confirmation_pct:.1f}%")
        
        # Create mortality indicator - YOUR DATA USES '9999-99-99' FOR NO DEATH
        if 'date_died' in df_confirmed.columns:
            df_confirmed['mortality'] = (df_confirmed['date_died'] != '9999-99-99').astype(int)
            df_full['mortality'] = (df_full['date_died'] != '9999-99-99').astype(int)
            
            mortality_full = (df_full['mortality'].sum() / len(df_full) * 100) if len(df_full) > 0 else 0
            mortality_confirmed = (df_confirmed['mortality'].sum() / len(df_confirmed) * 100) if len(df_confirmed) > 0 else 0
            
            logger.info(f"💀 Mortality (full dataset): {mortality_full:.2f}%")
            logger.info(f"💀 Mortality (confirmed cases): {mortality_confirmed:.2f}%")
        
        return df_full, df_confirmed
        
    except Exception as e:
        logger.error(f"❌ Phase 2 Error: {str(e)}")
        raise

# ============================================================================
# PHASE 3: STATISTICAL TESTS
# ============================================================================

def phase3_statistical_tests(df_confirmed: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical tests to validate findings.
    
    Tests:
    - T-test: Age difference between died and survived patients
    - Chi-square: Association of conditions with mortality
    
    Args:
        df_confirmed (pd.DataFrame): Confirmed COVID cases dataset
        
    Returns:
        Dict[str, Any]: Statistical test results
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: STATISTICAL TESTS")
    logger.info("="*80)
    
    try:
        stats_results = {}
        
        if 'mortality' not in df_confirmed.columns:
            logger.warning("⚠️ Mortality column not found, skipping tests")
            return stats_results
        
        # T-test: Age differences
        if 'age' in df_confirmed.columns:
            age_died = df_confirmed[df_confirmed['mortality'] == 1]['age']
            age_survived = df_confirmed[df_confirmed['mortality'] == 0]['age']

            if len(age_died) > 1 and len(age_survived) > 1:
                t_stat, p_value = stats.ttest_ind(age_died, age_survived)
                logger.info("T-Test Results:")
                logger.info(f"  Age (died): {age_died.mean():.1f} ± {age_died.std():.1f} years")
                logger.info(f"  Age (survived): {age_survived.mean():.1f} ± {age_survived.std():.1f} years")
                logger.info(f"  T-statistic: {t_stat:.4f}")
                logger.info(f"  P-value: {p_value:.2e}")
                if p_value < 0.05:
                    logger.info("  ✅ Statistically significant difference (p < 0.05)")
                else:
                    logger.info("  ⚠️ Not statistically significant")
                stats_results['t_test'] = {'t_stat': t_stat, 'p_value': p_value}

        # Chi-square: association of each condition with mortality
        chi_sq_conditions = ['diabetes', 'hypertension', 'cardiovascular', 'pneumonia', 'obesity']
        logger.info("\nChi-Square Test Results (condition vs mortality):")
        chi_results = {}
        raw_pvals, tested_conds, chi_stats, dofs = [], [], [], []
        for cond in chi_sq_conditions:
            if cond not in df_confirmed.columns:
                continue
            # Binarise to 1=Yes / 0=otherwise so the table is always 2x2.
            # Previously crosstab ran on raw codes (1=Yes, 2=No, 97/98/99=unknown),
            # producing a 3+ row table, so shape == (2,2) was never True and every
            # condition was silently skipped (header logged, but zero results).
            cond_binary = (df_confirmed[cond] == 1).astype(int)
            contingency = pd.crosstab(cond_binary, df_confirmed['mortality'])
            if contingency.shape == (2, 2):
                chi2, p, dof, _ = stats.chi2_contingency(contingency)
                tested_conds.append(cond); raw_pvals.append(p)
                chi_stats.append(chi2); dofs.append(dof)

        # Multiple-comparison correction (Benjamini–Hochberg FDR) across the
        # 5 simultaneous tests, so significance is not over-claimed.
        adj_pvals = _benjamini_hochberg(raw_pvals) if raw_pvals else []
        for cond, chi2, p, p_adj, dof in zip(tested_conds, chi_stats, raw_pvals, adj_pvals, dofs):
            sig = "✅ Significant" if p_adj < 0.05 else "⚠️ Not significant"
            logger.info(f"  {cond.upper()}: chi2={chi2:.2f}, p={p:.2e}, "
                        f"FDR-adj p={p_adj:.2e}, dof={dof} → {sig}")
            chi_results[cond] = {'chi2': chi2, 'p_value': p,
                                 'p_value_fdr': p_adj, 'dof': dof}
        stats_results['chi_square'] = chi_results

        return stats_results

    except Exception as e:
        logger.error(f"❌ Phase 3 Error: {str(e)}")
        return {}

# ============================================================================
# PHASE 4: MODEL TRAINING - 8 VISUALIZATIONS
# ============================================================================

def phase4_model_training(df_confirmed: pd.DataFrame) -> Dict[str, Any]:
    """
    Train machine learning models for mortality prediction.
    
    Models trained:
    1. Logistic Regression - baseline linear model
    2. Random Forest - ensemble tree-based model
    3. Gradient Boosting - boosted tree model
    4. Ensemble - average of all 3 models
    
    Creates 8 visualizations showing model performance.
    
    Args:
        df_confirmed (pd.DataFrame): Confirmed COVID cases with mortality labels
        
    Returns:
        Dict with models, predictions, feature columns, and test data
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: MODEL TRAINING (8 VISUALIZATIONS)")
    logger.info("="*80)
    
    try:
        df_model = df_confirmed.copy()
        
        if 'mortality' not in df_model.columns:
            raise ValueError("Mortality column required for model training")
        
        # Select features.
        # NOTE: 'intubed' and 'icu' were removed - they are recorded DURING severe
        # hospitalization (treatment outcomes), so they leak the target and are not
        # available at an early-intervention decision point. Keeping them inflated AUC
        # and contradicted the early-intervention premise of this project.
        feature_columns = ['age', 'sex', 'diabetes', 'hypertension', 'cardiovascular',
                          'pneumonia', 'obesity', 'asthma', 'copd']
        feature_columns = [col for col in feature_columns if col in df_model.columns]
        
        logger.info(f"Features selected: {feature_columns}")
        
        # Prepare features with proper missing-data handling (1/0/NaN).
        # 97/98/99 sentinels become NaN instead of being mislabelled as "No".
        X = encode_features(df_model, feature_columns)
        y = df_model['mortality']

        # Report missingness so it is documented, not hidden.
        miss = X.isnull().sum()
        miss = miss[miss > 0]
        if len(miss) > 0:
            logger.info("Missing values after sentinel decoding (to be imputed):")
            for col, n in miss.items():
                logger.info(f"  {col}: {n:,} ({n/len(X)*100:.2f}%)")
        else:
            logger.info("No missing values after sentinel decoding")
        
        # Check class balance
        pos_count = (y == 1).sum()
        neg_count = (y == 0).sum()
        logger.info("Class distribution:")
        logger.info(f"  Positive (mortality=1): {pos_count:,} ({(y == 1).mean()*100:.2f}%)")
        logger.info(f"  Negative (mortality=0): {neg_count:,} ({(y == 0).mean()*100:.2f}%)")

        # Events-per-variable (EPV) - sample-size adequacy for a prediction model
        # (Riley et al. 2020). EPV >> 10 indicates ample data and low overfitting risk.
        epv = int(pos_count) / max(len(feature_columns), 1)
        logger.info(f"  Events-per-variable (EPV): {epv:,.0f} "
                    f"({int(pos_count):,} events / {len(feature_columns)} predictors) "
                    f"- {'adequate (>=10)' if epv >= 10 else 'LOW (<10)'}")

        if len(np.unique(y)) < 2:
            raise ValueError("Only one class in target variable!")

        # Leakage fix: SPLIT FIRST, then fit imputer + scaler on TRAIN ONLY.
        # Both the median-imputer and the scaler are fit only on the training rows
        # and then applied to the test rows, so no test/CV statistics leak in.
        indices = np.arange(len(X))
        X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices,
            test_size=CONFIG['test_size'],
            stratify=y, random_state=CONFIG['random_state']
        )

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train_raw)   # fit on train only
        X_test_imp = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imp)   # fit on train only
        X_test = scaler.transform(X_test_imp)         # apply train stats to test

        logger.info("Data split:")
        logger.info(f"  Training: {len(X_train):,} samples")
        logger.info(f"  Testing: {len(X_test):,} samples")

        models = {}
        predictions = {}
        scores = {}
        fpr_tpr = {}

        # ===== LOGISTIC REGRESSION (with quick C tuning) =====
        # Tune the L2 strength C on the training folds (leakage-free pipeline) and
        # USE the result for the final model - so the deployed model reflects the
        # tuning rather than ignoring it.
        logger.info("Tuning + training Logistic Regression...")
        tune_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'],
                                       class_weight='balanced'))
        ])
        c_search = GridSearchCV(tune_pipe, {'clf__C': [0.01, 0.1, 1.0, 10.0]},
                                cv=3, scoring='roc_auc', n_jobs=-1)
        c_search.fit(X_train_raw, y_train)
        best_C = c_search.best_params_['clf__C']
        best_C_cv_auc = c_search.best_score_
        logger.info(f"  Best LR C={best_C} (CV AUC={best_C_cv_auc:.4f})")

        lr = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'],
                                class_weight='balanced', C=best_C)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred)
        scores['LR'] = lr_auc
        predictions['LR'] = lr_pred
        models['LR'] = lr
        fpr_tpr['LR'] = (fpr_lr, tpr_lr)
        logger.info(f"  ✅ Logistic Regression AUC: {lr_auc:.4f}")

        # Cross-validation on LR - leakage-free via a Pipeline so the scaler is
        # re-fit inside every fold (StandardScaler never sees validation rows).
        # CV runs on the RAW training features (X_train_raw), never the held-out test set.
        lr_cv_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'],
                                       class_weight='balanced'))
        ])
        lr_cv = cross_val_score(
            lr_cv_pipe, X_train_raw, y_train,
            cv=CONFIG['cv_folds'], scoring='roc_auc', n_jobs=-1
        )
        logger.info(f"  📊 LR CV AUC (leakage-free): {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

        # ===== RANDOM FOREST =====
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'],
                                    n_jobs=-1, class_weight='balanced')
        rf.fit(X_train, y_train)
        rf_pred = rf.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred)
        scores['RF'] = rf_auc
        predictions['RF'] = rf_pred
        models['RF'] = rf
        fpr_tpr['RF'] = (fpr_rf, tpr_rf)
        logger.info(f"  ✅ Random Forest AUC: {rf_auc:.4f}")
        
        # ===== GRADIENT BOOSTING =====
        # GradientBoostingClassifier has no class_weight argument, so the same
        # cost-sensitive reweighting is applied through sample_weight instead -
        # giving the minority (death) class ~7x the weight, consistent with the
        # other models.
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=CONFIG['random_state'])
        gb_sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        gb.fit(X_train, y_train, sample_weight=gb_sample_weight)
        gb_pred = gb.predict_proba(X_test)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_pred)
        fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_pred)
        scores['GB'] = gb_auc
        predictions['GB'] = gb_pred
        models['GB'] = gb
        fpr_tpr['GB'] = (fpr_gb, tpr_gb)
        logger.info(f"  ✅ Gradient Boosting AUC: {gb_auc:.4f}")
        
        # ===== ENSEMBLE =====
        logger.info("Creating Ensemble (Average of 3 models)...")
        ens_pred = (lr_pred + rf_pred + gb_pred) / 3
        ens_auc = roc_auc_score(y_test, ens_pred)
        fpr_ens, tpr_ens, _ = roc_curve(y_test, ens_pred)
        scores['Ensemble'] = ens_auc
        predictions['Ensemble'] = ens_pred
        fpr_tpr['Ensemble'] = (fpr_ens, tpr_ens)
        logger.info(f"  ✅ Ensemble AUC: {ens_auc:.4f}")

        # ===== PRIMARY DEPLOYED MODEL: calibrated, tuned Logistic Regression =====
        # This is the model that ALL downstream phases (fairness, calibration,
        # cost-benefit, clinical, export) actually use - so the tuning and the
        # calibration are deployed, not merely measured. LR is chosen because the
        # DeLong test shows no other model significantly beats it, and it is the
        # most interpretable (Odds Ratios). Isotonic calibration is fit with
        # internal CV on the training set only (no test leakage).
        logger.info("Building PRIMARY deployed model (calibrated tuned LR)...")
        deployed_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'],
                               class_weight='balanced', C=best_C),
            method='isotonic', cv=5)
        deployed_model.fit(X_train, y_train)
        primary_pred = deployed_model.predict_proba(X_test)[:, 1]
        primary_auc = roc_auc_score(y_test, primary_pred)
        predictions[CONFIG['primary_key']] = primary_pred
        logger.info(f"  ✅ Primary (calibrated tuned LR) AUC: {primary_auc:.4f}")

        # ===== VIS 1: AUC COMPARISON =====
        plt.figure(figsize=(10, 5))
        model_names = list(scores.keys())
        auc_scores = list(scores.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        plt.bar(model_names, auc_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        plt.title('Model Performance Comparison (AUC Scores)', fontsize=12, fontweight='bold')
        plt.ylabel('AUC Score')
        plt.ylim([0.75, 1.0])
        for i, v in enumerate(auc_scores):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/01_auc_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 1: AUC comparison")
        
        # ===== VIS 2: ROC CURVES =====
        plt.figure(figsize=(10, 8))
        colors_roc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for (model_name, (fpr, tpr)), color in zip(fpr_tpr.items(), colors_roc):
            auc_score = scores[model_name]
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2, color=color)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontsize=12, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/02_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 2: ROC curves")
        
        # ===== VIS 3: CONFUSION MATRIX =====
        ens_pred_binary = (ens_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, ens_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix - Ensemble Model', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/03_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 3: Confusion matrix")
        
        # ===== VIS 4: PERFORMANCE METRICS =====
        accuracy = accuracy_score(y_test, ens_pred_binary)
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info("Ensemble Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        plt.figure(figsize=(10, 5))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors_metrics = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        plt.bar(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black', linewidth=2)
        plt.title('Ensemble Model Performance Metrics', fontweight='bold')
        plt.ylabel('Score')
        plt.ylim([0, 1])
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/04_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 4: Performance metrics")
        
        # ===== VIS 5: PRECISION-RECALL =====
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, ens_pred)
        plt.figure(figsize=(10, 5))
        plt.plot(recall_curve, precision_curve, linewidth=2, color='steelblue', label='Ensemble')
        plt.fill_between(recall_curve, precision_curve, alpha=0.2, color='steelblue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Ensemble Model', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/05_precision_recall.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 5: Precision-Recall curve")
        
        # ===== VIS 6: RISK DISTRIBUTION =====
        plt.figure(figsize=(10, 5))
        plt.hist(ens_pred * 100, bins=30, edgecolor='black', alpha=0.7, color='coral')
        plt.xlabel('Risk Score (0-100)')
        plt.ylabel('Number of Patients')
        plt.title('Risk Score Distribution - Ensemble Predictions', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/06_risk_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 6: Risk distribution")
        
        # ===== VIS 7: LR FEATURE IMPORTANCE =====
        lr_coef = np.abs(lr.coef_[0])
        sorted_idx = np.argsort(lr_coef)[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feature_columns)[sorted_idx], lr_coef[sorted_idx], 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
        plt.title('Feature Importance - Logistic Regression', fontweight='bold')
        plt.xlabel('Coefficient Magnitude')
        plt.grid(True, alpha=0.3, axis='x')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/07_feature_importance_lr.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 7: LR feature importance")

        # ===== LR COEFFICIENTS (descriptive) =====
        # Descriptive coefficients from the REGULARISED, class-weighted model used
        # for prediction. The CANONICAL inferential Odds Ratios - with 95% CIs and
        # p-values from an unregularised statsmodels Logit - are produced in Phase 13
        # (covid_odds_ratios_ci.csv). The two differ by design (regularisation +
        # class weighting shrink these coefficients); the statsmodels table is the
        # one to cite for inference.
        odds_ratios = np.exp(lr.coef_[0])
        or_df = pd.DataFrame({
            'feature_name': feature_columns,
            'coefficient': lr.coef_[0],
            'odds_ratio_regularised': odds_ratios,
        }).sort_values('odds_ratio_regularised', ascending=False).reset_index(drop=True)
        logger.info("LR coefficients (regularised model, per +1 SD - descriptive only; "
                    "see covid_odds_ratios_ci.csv for canonical ORs):")
        for _, row in or_df.iterrows():
            logger.info(f"  {row['feature_name']}: OR(reg)={row['odds_ratio_regularised']:.3f} "
                        f"(β={row['coefficient']:+.3f})")

        # ===== VIS 8: RF FEATURE IMPORTANCE =====
        rf_coef = rf.feature_importances_
        sorted_idx = np.argsort(rf_coef)[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feature_columns)[sorted_idx], rf_coef[sorted_idx], 
                color='coral', alpha=0.7, edgecolor='black', linewidth=2)
        plt.title('Feature Importance - Random Forest', fontweight='bold')
        plt.xlabel('Importance Score')
        plt.grid(True, alpha=0.3, axis='x')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/08_feature_importance_rf.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 8: RF feature importance")
        
        logger.info("✅ PHASE 4 COMPLETE: All 8 visualizations created")

        return {
            'models': models,
            'predictions': predictions,
            'scores': scores,
            'X_test': X_test,
            'y_test': y_test,
            'X_train': X_train,
            'y_train': y_train,
            'idx_test': idx_test,       # test-row indices, so fairness maps to the right patients
            'feature_columns': feature_columns,
            'scaler': scaler,
            'imputer': imputer,
            'deployed_model': deployed_model,   # calibrated tuned LR (primary)
            'best_C': best_C,
            'best_C_cv_auc': best_C_cv_auc,
            'epv': epv,
            'n_events': int(pos_count),
            'df_model': df_model,
            'fpr_tpr': fpr_tpr,
            'odds_ratios': or_df,      # proposal deliverable: e^β risk quantification
            'X_train_raw': X_train_raw,   # unscaled splits for leakage-free pipelines
            'X_test_raw': X_test_raw,
        }
        
    except Exception as e:
        logger.error(f"❌ Phase 4 Error: {str(e)}")
        raise


# ============================================================================
# PHASE 5: SHAP ANALYSIS - 5 VISUALIZATIONS
# ============================================================================

def phase5_shap_analysis(models: dict, X_test: np.ndarray, feature_columns: list) -> pd.DataFrame:
    """
    Perform SHAP (SHapley Additive exPlanations) analysis for model interpretability.
    
    SHAP is a game theory approach to explaining machine learning predictions.
    It shows which features have the most impact on each prediction, enabling
    clinicians to understand WHY the model makes its decisions.
    
    Creates 5 visualizations:
    1. Feature importance (mean |SHAP|)
    2. Dependence plots (how features affect predictions)
    3. Force plot (sample-level explanations)
    4. Waterfall plot (prediction breakdown)
    5. Bar plot (summary of feature impacts)
    
    Args:
        models (dict): Dictionary of trained models
        X_test (np.ndarray): Test features
        feature_columns (list): Feature names
        
    Returns:
        pd.DataFrame: SHAP importance rankings
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: SHAP ANALYSIS (5 VISUALIZATIONS)")
    logger.info("Explainability: Understanding individual predictions")
    logger.info("="*80)
    
    try:
        logger.info("Computing SHAP values from Logistic Regression...")
        explainer = shap.LinearExplainer(models['LR'], X_test)
        shap_values = explainer.shap_values(X_test)
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # VIS 1: SHAP Feature Importance
        logger.info("Creating SHAP importance visualization...")
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)[::-1]
        plt.barh(np.array(feature_columns)[sorted_idx], feature_importance[sorted_idx], 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
        plt.title('SHAP Feature Importance - Global Explanation', fontweight='bold')
        plt.xlabel('Mean |SHAP Value| - Impact on Model Output')
        plt.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(feature_importance[sorted_idx]):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/shap_plots/01_shap_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 1: SHAP feature importance")
        
        # VIS 2: SHAP Dependence Plots
        logger.info("Creating SHAP dependence plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        top_features = np.argsort(-feature_importance)[:4]
        
        for idx, feat_idx in enumerate(top_features):
            feature_name = feature_columns[feat_idx]
            feature_vals = X_test[:, feat_idx]
            shap_vals = shap_values[:, feat_idx]
            
            scatter = axes[idx].scatter(feature_vals, shap_vals, alpha=0.5, 
                                       c=feature_vals, cmap='viridis', s=30)
            axes[idx].set_xlabel(f'{feature_name} Value')
            axes[idx].set_ylabel('SHAP Value')
            axes[idx].set_title(f'{feature_name} Dependence Plot', fontweight='bold')
            axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Value')
        
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/shap_plots/02_shap_dependence.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 2: SHAP dependence plots")
        
        # VIS 3: SHAP Force Plot
        logger.info("Creating SHAP force plot...")
        plt.figure(figsize=(12, 5))
        base_value = np.mean(shap_values[:100])
        
        for i in range(min(5, X_test.shape[0])):
            color = 'steelblue' if shap_values[i].sum() > 0 else 'coral'
            plt.barh(i, shap_values[i].sum(), left=base_value, color=color, alpha=0.7, edgecolor='black')
        
        plt.title('SHAP Force Plot (Sample Patient Explanations)', fontweight='bold')
        plt.xlabel('Model Output Value (Risk Score)')
        plt.ylabel('Sample Index')
        plt.grid(True, alpha=0.3, axis='x')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/shap_plots/03_shap_force_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 3: SHAP force plot")
        
        # VIS 4: SHAP Waterfall Plot
        logger.info("Creating SHAP waterfall plot...")
        plt.figure(figsize=(10, 6))
        sample_idx = 0
        sample_shap = shap_values[sample_idx]
        sorted_shap = sorted(enumerate(sample_shap), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        labels = [feature_columns[i] for i, _ in sorted_shap]
        values = [v for _, v in sorted_shap]
        cumsum = np.cumsum([0] + values[:-1])
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        plt.barh(labels, values, left=cumsum, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        plt.title('SHAP Waterfall Plot - Individual Patient Explanation', fontweight='bold')
        plt.xlabel('SHAP Value Contribution')
        plt.grid(True, alpha=0.3, axis='x')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/shap_plots/04_shap_waterfall.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 4: SHAP waterfall plot")
        
        # VIS 5: SHAP Bar Plot Summary
        logger.info("Creating SHAP summary bar plot...")
        plt.figure(figsize=(10, 6))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        plt.bar(range(len(sorted_idx)), mean_abs_shap[sorted_idx], 
               color='coral', alpha=0.7, edgecolor='black', linewidth=2)
        plt.xticks(range(len(sorted_idx)), np.array(feature_columns)[sorted_idx], rotation=45)
        plt.title('SHAP Bar Plot - Feature Impact Summary', fontweight='bold')
        plt.ylabel('Mean |SHAP Value|')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/shap_plots/05_shap_bar_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 5: SHAP bar plot")
        
        # SHAP results, ranked by importance descending
        sorted_idx = np.argsort(-feature_importance)
        shap_df = pd.DataFrame({
            'feature_name': np.array(feature_columns)[sorted_idx],
            'shap_importance': feature_importance[sorted_idx],
            'importance_percent': (feature_importance[sorted_idx] / np.sum(feature_importance) * 100)
        })
        
        logger.info("Top 5 Most Important Features:")
        for idx, row in shap_df.head(5).iterrows():
            logger.info(f"  {row['feature_name']}: {row['importance_percent']:.1f}%")
        
        logger.info("✅ PHASE 5 COMPLETE: All 5 SHAP visualizations created")
        return shap_df
        
    except Exception as e:
        logger.error(f"❌ Phase 5 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 6: FAIRNESS & BIAS AUDIT - 4 VISUALIZATIONS
# ============================================================================

def phase6_fairness_analysis(df_model: pd.DataFrame, predictions: dict,
                            y_test: np.ndarray, idx_test: np.ndarray) -> pd.DataFrame:
    """
    Audit model fairness across demographic groups.

    Ensures the model works equally well for:
    - All age groups (18-30, 30-50, 50-65, 65+)
    - Both genders
    - Different comorbidity levels

    A model is fair if accuracy differences between groups < 5%.
    Unfair models discriminate against minority groups.

    Creates 4 visualizations showing fairness across demographics.

    Args:
        df_model (pd.DataFrame): Dataset with demographics
        predictions (dict): Model predictions
        y_test (np.ndarray): True labels
        idx_test (np.ndarray): Original row indices of the test set
            (returned by phase4) so demographics are mapped correctly.

    Returns:
        pd.DataFrame: Fairness audit results
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: FAIRNESS & BIAS AUDIT (4 VISUALIZATIONS)")
    logger.info("Ensuring equitable performance across demographic groups")
    logger.info("="*80)
    
    try:
        fairness_results = []
        
        if 'age' not in df_model.columns:
            logger.warning("⚠️ Age column not found")
            return pd.DataFrame()

        # Map test predictions back to demographics by their original row index.
        # A positional slice like iloc[-test_size:] would be wrong here, since
        # stratified splitting does not leave the test rows at the end.
        test_demo = df_model.iloc[idx_test].reset_index(drop=True)

        pkey = CONFIG['primary_key']   # audit the deployed model, not the raw ensemble

        def _subgroup_calibration(mask, n_bins=10):
            """Brier score and count-weighted ECE within a subgroup. Tells us
            whether predicted risks stay accurate even where AUC (ranking) drops."""
            yt = np.asarray(y_test)[mask]; pp = np.asarray(predictions[pkey])[mask]
            brier = brier_score_loss(yt, pp)
            bins = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0; N = len(yt)
            for i in range(n_bins):
                m = (pp > bins[i]) & (pp <= bins[i + 1]) if i > 0 else (pp >= bins[0]) & (pp <= bins[1])
                if m.sum() > 0:
                    ece += (m.sum() / N) * abs(yt[m].mean() - pp[m].mean())
            return round(float(brier), 4), round(float(ece), 4)

        age_groups = [(18, 30), (30, 50), (50, 65), (65, 120)]
        age_labels = ['18-30', '30-50', '50-65', '65+']

        logger.info("Fairness Audit - Age Groups:")

        for (min_age, max_age), label in zip(age_groups, age_labels):
            test_ages = test_demo['age'].values
            mask = (test_ages >= min_age) & (test_ages < max_age)

            if mask.sum() > 0:
                y_subset = y_test[mask]
                pred_subset = predictions[pkey][mask]

                if len(np.unique(y_subset)) > 1:
                    auc_score = roc_auc_score(y_subset, pred_subset)
                    baseline = roc_auc_score(y_test, predictions[pkey])
                    difference = auc_score - baseline
                    brier, ece = _subgroup_calibration(mask)

                    threshold = CONFIG['bias_threshold']
                    fairness_results.append({
                        'demographic_group': f'Age {label}',
                        'group_size': mask.sum(),
                        'model_auc': auc_score,
                        'difference_from_baseline': difference,
                        'auc_status': 'OK' if abs(difference) < threshold else 'FLAG',
                        'brier': brier,
                        'ece': ece,
                        'calibration': 'well-calibrated' if ece < 0.10 else 'check',
                    })

                    status = "✅ OK" if abs(difference) < threshold else "⚠️ FLAG"
                    logger.info(f"  Age {label} (n={mask.sum()}): AUC={auc_score:.4f} "
                                f"Brier={brier} ECE={ece} {status}")

        fair_df = pd.DataFrame(fairness_results)

        if len(fair_df) > 0:
            # VIS 1: Age accuracy
            logger.info("Creating age fairness visualization...")
            plt.figure(figsize=(10, 5))
            colors = ['#2ca02c' if row['auc_status'] == 'OK' else '#ff7f0e' for _, row in fair_df.iterrows()]
            plt.bar(fair_df['demographic_group'], fair_df['model_auc'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            plt.title('Model Accuracy by Age Group (Fairness Audit)', fontweight='bold')
            plt.ylabel('AUC Score')
            baseline = roc_auc_score(y_test, predictions[pkey])
            plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='Baseline')
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/fairness_plots/01_fairness_age.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 1: Fairness by age")

            # VIS 2: Disparity
            logger.info("Creating disparity visualization...")
            plt.figure(figsize=(10, 5))
            disparities = abs(fair_df['difference_from_baseline']) * 100
            colors_disp = ['#2ca02c' if x < 5 else '#ff7f0e' for x in disparities]
            plt.bar(fair_df['demographic_group'], disparities,
                   color=colors_disp, alpha=0.7, edgecolor='black', linewidth=2)
            plt.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Fairness Threshold')
            plt.title('Accuracy Disparity from Baseline (Fairness Threshold: 5%)', fontweight='bold')
            plt.ylabel('Disparity (%)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()
            plt.savefig(f"{CONFIG['output_folder']}/visualizations/fairness_plots/02_fairness_disparity.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ VIS 2: Fairness disparity")

        baseline = roc_auc_score(y_test, predictions[pkey])
        threshold = CONFIG['bias_threshold']

        # VIS 3: Gender fairness
        if 'sex' in df_model.columns:
            logger.info("Analyzing gender fairness...")
            gender_results = []
            test_sex = test_demo['sex'].values

            for sex_val, label in [(1, 'Female'), (2, 'Male')]:
                mask = (test_sex == sex_val)
                if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
                    auc = roc_auc_score(y_test[mask], predictions[pkey][mask])
                    gender_results.append({'sex': label, 'auc': auc})
                    diff = auc - baseline
                    brier, ece = _subgroup_calibration(mask)
                    fairness_results.append({
                        'demographic_group': f'Sex: {label}',
                        'group_size': int(mask.sum()),
                        'model_auc': auc,
                        'difference_from_baseline': diff,
                        'auc_status': 'OK' if abs(diff) < threshold else 'FLAG',
                        'brier': brier,
                        'ece': ece,
                        'calibration': 'well-calibrated' if ece < 0.10 else 'check',
                    })
                    logger.info(f"  Sex {label} (n={mask.sum()}): AUC={auc:.4f} "
                                f"Brier={brier} ECE={ece}")

            if gender_results:
                gdf = pd.DataFrame(gender_results)
                plt.figure(figsize=(8, 5))
                plt.bar(gdf['sex'], gdf['auc'], color=['#FF69B4', '#4169E1'],
                       alpha=0.7, edgecolor='black', linewidth=2)
                plt.title('Model Accuracy by Gender', fontweight='bold')
                plt.ylabel('AUC Score')
                plt.grid(True, alpha=0.3, axis='y')
                plt.savefig(f"{CONFIG['output_folder']}/visualizations/fairness_plots/03_fairness_gender.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("✅ VIS 3: Gender fairness")

        # VIS 4: Comorbidity fairness
        # df_model holds raw codes (1=Yes, 2=No, 98=unknown), so "Without Diabetes"
        # is value 2, not 0. A previous version compared against 0 and never matched.
        if all(c in df_model.columns for c in ['diabetes', 'hypertension']):
            logger.info("Analyzing comorbidity fairness...")
            comorb_results = []

            for diab_val, grp_label in [(1, 'With Diabetes'), (2, 'Without Diabetes')]:
                mask = (test_demo['diabetes'].values == diab_val)
                if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
                    auc = roc_auc_score(y_test[mask], predictions[pkey][mask])
                    comorb_results.append({'group': grp_label, 'auc': auc})
                    diff = auc - baseline
                    brier, ece = _subgroup_calibration(mask)
                    fairness_results.append({
                        'demographic_group': grp_label,
                        'group_size': int(mask.sum()),
                        'model_auc': auc,
                        'difference_from_baseline': diff,
                        'auc_status': 'OK' if abs(diff) < threshold else 'FLAG',
                        'brier': brier,
                        'ece': ece,
                        'calibration': 'well-calibrated' if ece < 0.10 else 'check',
                    })
                    logger.info(f"  {grp_label} (n={mask.sum()}): AUC={auc:.4f} "
                                f"Brier={brier} ECE={ece}")

            if comorb_results:
                cdf = pd.DataFrame(comorb_results)
                plt.figure(figsize=(8, 5))
                plt.bar(cdf['group'], cdf['auc'], color=['#90EE90', '#FFB6C1'],
                       alpha=0.7, edgecolor='black', linewidth=2)
                plt.title('Model Accuracy by Diabetes Status', fontweight='bold')
                plt.ylabel('AUC Score')
                plt.grid(True, alpha=0.3, axis='y')
                plt.savefig(f"{CONFIG['output_folder']}/visualizations/fairness_plots/04_fairness_comorbidity.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("✅ VIS 4: Comorbidity fairness")

        # Return the full audit (age + gender + comorbidity), so every fairness
        # number in the app/report is backed by covid_fairness_audit.csv.
        logger.info("✅ PHASE 6 COMPLETE: Fairness audit finished")
        return pd.DataFrame(fairness_results) if fairness_results else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"❌ Phase 6 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 7: CALIBRATION ANALYSIS - 3 VISUALIZATIONS
# ============================================================================

def phase7_calibration_analysis(y_test: np.ndarray, predictions: dict) -> pd.DataFrame:
    """
    Analyze model calibration - do predicted probabilities match reality?
    
    A calibrated model's 70% risk predictions should have ~70% mortality.
    Uncalibrated models are overconfident or underconfident.
    
    Creates 3 visualizations showing prediction vs actual outcomes.
    
    Args:
        y_test (np.ndarray): True outcomes
        predictions (dict): Model predictions
        
    Returns:
        pd.DataFrame: Calibration analysis results
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 7: CALIBRATION ANALYSIS (3 VISUALIZATIONS)")
    logger.info("Verifying predicted probabilities match actual outcomes")
    logger.info("="*80)
    
    try:
        ensemble_pred = predictions[CONFIG['primary_key']]   # deployed (calibrated) model
        prob_true, prob_pred = calibration_curve(y_test, ensemble_pred, n_bins=10)
        
        cal_df = pd.DataFrame({
            'predicted_probability': prob_pred,
            'actual_probability': prob_true,
            'calibration_error': np.abs(prob_pred - prob_true)
        })
        
        brier = brier_score_loss(y_test, ensemble_pred)
        ece = np.mean(cal_df['calibration_error'])
        
        logger.info("Calibration Metrics:")
        logger.info(f"  Brier Score: {brier:.4f} (lower is better, 0=perfect)")
        logger.info(f"  Expected Calibration Error: {ece:.4f}")
        logger.info(f"  Interpretation: Model predictions are {'well-calibrated' if ece < 0.1 else 'poorly calibrated'}")
        
        # VIS 1: Calibration plot
        logger.info("Creating calibration curve...")
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, 
                color='steelblue', label='Ensemble Model')
        plt.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='steelblue')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Probability (from data)')
        plt.title('Calibration Plot: Do Predictions Match Reality?', fontweight='bold')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/calibration_plots/01_calibration_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 1: Calibration plot")
        
        # VIS 2: Reliability diagram
        logger.info("Creating reliability diagram...")
        plt.figure(figsize=(10, 5))
        plt.bar(prob_pred, prob_true - prob_pred, width=0.08, alpha=0.7, 
               edgecolor='black', linewidth=2, color='coral')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Calibration Error (Actual - Predicted)')
        plt.title('Reliability Diagram', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/calibration_plots/02_reliability_diagram.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 2: Reliability diagram")
        
        # VIS 3: Calibration error by bin
        logger.info("Creating calibration error by bin...")
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(prob_pred)), cal_df['calibration_error'].values, 
               color='coral', alpha=0.7, edgecolor='black', linewidth=2)
        plt.xlabel('Probability Bin')
        plt.ylabel('Calibration Error')
        plt.title('Expected Calibration Error by Probability Bin', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/calibration_plots/03_calibration_error_by_bin.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 3: Calibration error")
        
        logger.info("✅ PHASE 7 COMPLETE: Calibration analysis finished")
        return cal_df
        
    except Exception as e:
        logger.error(f"❌ Phase 7 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 8: COST-BENEFIT ANALYSIS - 4 VISUALIZATIONS
# ============================================================================

def phase8_cost_benefit_analysis(df_confirmed: pd.DataFrame,
                                 predictions: dict = None,
                                 y_test: np.ndarray = None,
                                 risk_threshold: float = 0.5) -> pd.DataFrame:
    """
    Calculate healthcare savings from predictive model deployment.

    Realistic model-driven version (replaces the old "intervene on everyone"
    assumption that produced absurd $B savings / 100,000%+ ROI):

    Only patients the model FLAGS as high-risk (predicted probability >=
    risk_threshold) are put forward for early intervention. Of those flagged,
    only a fraction are genuine future complications (the model's precision at
    that threshold). Intervention then prevents a success-rate fraction of those
    genuine cases. Every intervened patient - true or false positive - still
    incurs the intervention cost. This grounds ROI in the model's actual
    operating point rather than a flat multiplier.

    Assumptions:
    - Early intervention cost: $1,000 per intervened patient
    - Late complication cost (avoided): $75,000 per prevented case
    - Prevention success rate: 60%
    - Deployment cost: $2,000,000 (one-off)
    - Benefits annualised over a 5-year horizon for payback

    Args:
        df_confirmed (pd.DataFrame): Confirmed COVID cases (for total population size)
        predictions (dict): Model predictions; uses the 'Ensemble' test-set scores
        y_test (np.ndarray): True outcomes aligned with predictions['Ensemble']
        risk_threshold (float): Probability cut-off for flagging a patient high-risk

    Returns:
        pd.DataFrame: Cost-benefit analysis for 3 adoption scenarios
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 8: COST-BENEFIT ANALYSIS (4 VISUALIZATIONS)")
    logger.info("Calculating healthcare savings from model deployment")
    logger.info("="*80)

    try:
        total_patients = len(df_confirmed)
        early_intervention_cost = 1000
        late_complication_cost = 75000
        deployment_cost = 2_000_000
        prevention_success_rate = 0.60
        horizon_years = 5

        # Derive the model's operating point from the test set.
        if predictions is not None and y_test is not None and 'Ensemble' in predictions:
            ens = np.asarray(predictions[CONFIG['primary_key']])
            y_arr = np.asarray(y_test)
            flagged = ens >= risk_threshold
            n_flagged = int(flagged.sum())
            high_risk_rate = flagged.mean() if len(ens) > 0 else 0.0
            precision = (y_arr[flagged].mean() if n_flagged > 0 else 0.0)
        else:
            # Fallback to dataset prevalence if predictions weren't supplied.
            high_risk_rate = 0.12
            precision = 0.45
            logger.warning("⚠️ No predictions passed - using fallback prevalence/precision")

        flagged_population = int(total_patients * high_risk_rate)

        logger.info("Parameters:")
        logger.info(f"  Total patients: {total_patients:,}")
        logger.info(f"  Risk threshold: {risk_threshold:.2f}")
        logger.info(f"  High-risk rate (model-flagged): {high_risk_rate*100:.1f}%")
        logger.info(f"  Precision at threshold: {precision*100:.1f}%")
        logger.info(f"  Flagged population: {flagged_population:,}")
        logger.info(f"  Early intervention cost: ${early_intervention_cost:,}")
        logger.info(f"  Late complication cost (avoided): ${late_complication_cost:,}")
        logger.info(f"  Deployment cost: ${deployment_cost:,}")

        scenarios = []

        for adoption_rate in [0.30, 0.60, 0.90]:
            # Intervene only on flagged patients, scaled by adoption.
            patients_intervened = int(flagged_population * adoption_rate)
            true_positives = int(patients_intervened * precision)
            patients_prevented = int(true_positives * prevention_success_rate)

            gross_benefit = patients_prevented * late_complication_cost
            intervention_spend = patients_intervened * early_intervention_cost
            total_cost = intervention_spend + deployment_cost
            net_savings = gross_benefit - total_cost
            roi = (net_savings / total_cost) * 100 if total_cost > 0 else 0
            annual_net = net_savings / horizon_years
            payback_period = (total_cost / annual_net) if annual_net > 0 else float('inf')

            scenario_name = "Conservative" if adoption_rate == 0.30 else \
                          "Moderate" if adoption_rate == 0.60 else "Optimistic"

            scenarios.append({
                'scenario': f'{int(adoption_rate*100)}% Adoption ({scenario_name})',
                'adoption_rate': adoption_rate,
                'patients_intervened': patients_intervened,
                'patients_prevented': patients_prevented,
                'total_savings': net_savings,
                'roi_percent': roi,
                'payback_years': payback_period
            })

            logger.info(f"\n{scenario_name} Scenario ({int(adoption_rate*100)}% adoption):")
            logger.info(f"  Patients intervened: {patients_intervened:,}")
            logger.info(f"  True positives caught: {true_positives:,}")
            logger.info(f"  Patients prevented (60% success): {patients_prevented:,}")
            logger.info(f"  Net savings: ${net_savings:,}")
            logger.info(f"  ROI: {roi:.0f}%")
            payback_str = f"{payback_period:.1f} years" if payback_period != float('inf') else "N/A (no net benefit)"
            logger.info(f"  Payback period: {payback_str}")
        
        cb_df = pd.DataFrame(scenarios)
        
        # 4-Panel Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Patients Intervened
        axes[0,0].bar(range(len(cb_df)), cb_df['patients_intervened']/1000,
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black', linewidth=2)
        axes[0,0].set_xticks(range(len(cb_df)))
        axes[0,0].set_xticklabels([s.split('(')[1][:-1] for s in cb_df['scenario']])
        axes[0,0].set_title('Patients Intervened', fontweight='bold')
        axes[0,0].set_ylabel('Count (thousands)')
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Net Savings
        axes[0,1].bar(range(len(cb_df)), cb_df['total_savings']/1e6,
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black', linewidth=2)
        axes[0,1].set_xticks(range(len(cb_df)))
        axes[0,1].set_xticklabels([s.split('(')[1][:-1] for s in cb_df['scenario']])
        axes[0,1].set_title('Net Savings', fontweight='bold')
        axes[0,1].set_ylabel('Net Savings ($ Millions)')
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Panel 3: ROI
        axes[1,0].bar(range(len(cb_df)), cb_df['roi_percent'],
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black', linewidth=2)
        axes[1,0].set_xticks(range(len(cb_df)))
        axes[1,0].set_xticklabels([s.split('(')[1][:-1] for s in cb_df['scenario']])
        axes[1,0].set_title('Return on Investment (ROI)', fontweight='bold')
        axes[1,0].set_ylabel('ROI (%)')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Intervention vs Prevention
        x_pos = np.arange(len(cb_df))
        width = 0.35
        axes[1,1].bar(x_pos - width/2, cb_df['patients_intervened']/1000, width,
                     label='Intervened', alpha=0.7, edgecolor='black', linewidth=2)
        axes[1,1].bar(x_pos + width/2, cb_df['patients_prevented']/1000, width,
                     label='Prevented', alpha=0.7, edgecolor='black', linewidth=2)
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels([s.split('(')[1][:-1] for s in cb_df['scenario']])
        axes[1,1].set_title('Intervention vs Prevention Outcomes', fontweight='bold')
        axes[1,1].set_ylabel('Count (thousands)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_folder']}/visualizations/model_comparison_plots/09_cost_benefit_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ VIS 1-4: Cost-benefit analysis complete")
        
        logger.info("✅ PHASE 8 COMPLETE: Cost-benefit analysis finished")
        return cb_df
        
    except Exception as e:
        logger.error(f"❌ Phase 8 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 9: CLINICAL EXPLANATIONS
# ============================================================================

def phase9_clinical_explanations(df_confirmed: pd.DataFrame, predictions: dict,
                                 df_model: pd.DataFrame = None,
                                 idx_test: np.ndarray = None) -> pd.DataFrame:
    """
    Generate plain-language risk assessments for clinicians.
    
    Converts model predictions into actionable clinical recommendations:
    - CRITICAL: Risk >= 70%, urgency: within 3 days
    - HIGH: Risk 50-70%, urgency: within 1 week
    - MEDIUM: Risk 30-50%, urgency: within 2 weeks
    - LOW: Risk < 30%, urgency: as scheduled
    
    Args:
        df_confirmed (pd.DataFrame): Patient dataset
        predictions (dict): Model predictions
        
    Returns:
        pd.DataFrame: Clinical explanations for sample patients
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 9: CLINICAL EXPLANATIONS")
    logger.info("="*80)
    
    try:
        explanations = []
        sample_size = min(100, len(df_confirmed))
        # Seed for reproducibility - without this the sampled patients (and the
        # CRITICAL/HIGH/MEDIUM/LOW counts) differ on every run.
        np.random.seed(CONFIG['random_state'])
        pkey = CONFIG['primary_key']   # use the deployed (calibrated) risk scores
        positions = np.random.choice(len(predictions[pkey]), sample_size, replace=False)

        # Resolve a real, traceable patient id for each test-set position.
        # predictions[pkey] is over the test rows (idx_test) of df_model.
        if df_model is not None and idx_test is not None:
            test_rows = df_model.iloc[idx_test]
            id_lookup = (test_rows['id'].values if 'id' in test_rows.columns
                         else test_rows.index.values)
        else:
            id_lookup = None

        logger.info(f"Generating explanations for {sample_size} sample patients...")

        for pos in positions:
            risk_score = predictions[pkey][pos] * 100

            if risk_score >= 70:
                risk_level = "CRITICAL"
                urgency = "Within 3 days"
            elif risk_score >= 50:
                risk_level = "HIGH"
                urgency = "Within 1 week"
            elif risk_score >= 30:
                risk_level = "MEDIUM"
                urgency = "Within 2 weeks"
            else:
                risk_level = "LOW"
                urgency = "As scheduled"
            
            explanation = f"Mortality-proxy risk: {risk_score:.0f}/100 - {risk_level} | Clinical Urgency: {urgency}"
            
            explanations.append({
                'patient_id': id_lookup[pos] if id_lookup is not None else pos,
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'recommendation': f"Specialist referral urgency: {urgency}",
                'explanation': explanation
            })
        
        exp_df = pd.DataFrame(explanations)
        logger.info(f"✅ Generated {len(exp_df)} clinical explanations")
        logger.info(f"   CRITICAL: {(exp_df['risk_level']=='CRITICAL').sum()} patients")
        logger.info(f"   HIGH: {(exp_df['risk_level']=='HIGH').sum()} patients")
        logger.info(f"   MEDIUM: {(exp_df['risk_level']=='MEDIUM').sum()} patients")
        logger.info(f"   LOW: {(exp_df['risk_level']=='LOW').sum()} patients")
        
        logger.info("✅ PHASE 9 COMPLETE")
        return exp_df
        
    except Exception as e:
        logger.error(f"❌ Phase 9 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 10: FEATURE INTERACTIONS
# ============================================================================

def phase10_feature_interactions(models: dict, feature_columns: list) -> pd.DataFrame:
    """
    Identify feature interactions - combinations that matter.
    
    Finds top feature pairs that together strongly influence predictions.
    Example: Diabetes + Age might have stronger effect than either alone.
    
    Args:
        models (dict): Trained models
        feature_columns (list): Feature names
        
    Returns:
        pd.DataFrame: Top feature interaction pairs
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 10: FEATURE INTERACTIONS")
    logger.info("="*80)
    
    try:
        rf_model = models['RF']
        feature_imp = rf_model.feature_importances_
        
        top_features_idx = np.argsort(-feature_imp)[:min(5, len(feature_columns))]
        
        interactions = []
        for i, feat1_idx in enumerate(top_features_idx):
            for feat2_idx in top_features_idx[i+1:]:
                feat1_name = feature_columns[feat1_idx]
                feat2_name = feature_columns[feat2_idx]
                interaction_strength = feature_imp[feat1_idx] * feature_imp[feat2_idx]
                
                interactions.append({
                    'feature_1': feat1_name,
                    'feature_2': feat2_name,
                    'interaction_strength': interaction_strength,
                })
        
        int_df = pd.DataFrame(interactions).sort_values('interaction_strength', ascending=False)
        logger.info(f"✅ Identified {len(int_df)} feature interactions")
        logger.info("Top interactions:")
        for idx, row in int_df.head(5).iterrows():
            logger.info(f"  {row['feature_1']} + {row['feature_2']}: {row['interaction_strength']:.4f}")
        
        logger.info("✅ PHASE 10 COMPLETE")
        return int_df
        
    except Exception as e:
        logger.error(f"❌ Phase 10 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 11: SENSITIVITY ANALYSIS
# ============================================================================

def phase11_sensitivity_analysis(df_confirmed: pd.DataFrame, predictions: dict,
                                  models: dict, scaler, feature_columns: list,
                                  imputer=None, deployed_model=None) -> pd.DataFrame:
    """
    What-if analysis: How would patient risk change if a modifiable condition were resolved?

    For each sampled patient the model is re-run with one modifiable risk factor
    switched off at a time (diabetes=0, hypertension=0, pneumonia=0, obesity=0).
    This gives a data-backed estimate of how much removing each comorbidity could
    reduce the predicted mortality risk - a clinically meaningful sensitivity rather
    than a flat percentage multiplier.

    Each scenario re-runs the model with the modified feature vector rather than
    scaling the score by a flat multiplier, so the estimated reduction reflects
    what the model actually learned.

    Args:
        df_confirmed (pd.DataFrame): Patient dataset (confirmed COVID cases)
        predictions (dict): Existing ensemble predictions for reference
        models (dict): Trained LR, RF, GB models
        scaler: Fitted StandardScaler from Phase 4
        feature_columns (list): Ordered list of feature names used by the models

    Returns:
        pd.DataFrame: Sensitivity scenarios for sample patients
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 11: SENSITIVITY ANALYSIS (feature-removal what-if)")
    logger.info("="*80)

    try:
        modifiable = ['diabetes', 'hypertension', 'pneumonia', 'obesity']
        modifiable = [f for f in modifiable if f in feature_columns]

        # Build feature matrix for the confirmed set (mirrors Phase 4 preprocessing:
        # sentinel-aware encoding + the SAME fitted median imputer).
        X_enc = encode_features(df_confirmed, feature_columns)
        if imputer is not None:
            X_raw = pd.DataFrame(imputer.transform(X_enc), columns=feature_columns)
        else:
            X_raw = X_enc.fillna(X_enc.median())

        sample_size = min(100, len(X_raw))
        np.random.seed(CONFIG['random_state'])
        positions = np.random.choice(len(X_raw), sample_size, replace=False)

        sensitivity_results = []
        logger.info(f"Running what-if scenarios for {sample_size} patients "
                    f"across {len(modifiable)} modifiable risk factors...")

        def _ensemble_prob(feature_row: np.ndarray) -> float:
            """Risk for a single feature row using the deployed (calibrated) model.
            Falls back to the 3-model average if no deployed model is supplied."""
            rs = scaler.transform(feature_row.reshape(1, -1))
            if deployed_model is not None:
                return float(deployed_model.predict_proba(rs)[0, 1])
            p_lr = models['LR'].predict_proba(rs)[0, 1]
            p_rf = models['RF'].predict_proba(rs)[0, 1]
            p_gb = models['GB'].predict_proba(rs)[0, 1]
            return float((p_lr + p_rf + p_gb) / 3)

        for pos in positions:
            row = X_raw.iloc[pos].values.copy()
            current_risk = _ensemble_prob(row) * 100

            record = {
                'patient_id': int(pos),
                'current_risk': round(current_risk, 1),
            }

            for feat in modifiable:
                feat_idx = feature_columns.index(feat)
                original_val = row[feat_idx]
                if original_val == 0:
                    # Condition already absent - no change
                    record[f'risk_without_{feat}'] = round(current_risk, 1)
                    record[f'risk_reduction_{feat}'] = 0.0
                else:
                    row_modified = row.copy()
                    row_modified[feat_idx] = 0
                    new_risk = _ensemble_prob(row_modified) * 100
                    record[f'risk_without_{feat}'] = round(new_risk, 1)
                    record[f'risk_reduction_{feat}'] = round(current_risk - new_risk, 1)

            sensitivity_results.append(record)

        sens_df = pd.DataFrame(sensitivity_results)
        logger.info(f"✅ Generated {len(sens_df)} sensitivity scenarios")
        logger.info(f"   Average current risk: {sens_df['current_risk'].mean():.1f}/100")
        for feat in modifiable:
            avg_red = sens_df[f'risk_reduction_{feat}'].mean()
            logger.info(f"   Avg risk reduction from removing {feat}: {avg_red:.1f} points")

        logger.info("✅ PHASE 11 COMPLETE")
        return sens_df

    except Exception as e:
        logger.error(f"❌ Phase 11 Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# PHASE 13: ADVANCED METHODOLOGY (MSc-level extensions)
# ============================================================================

def phase13_advanced_methods(model_results: Dict[str, Any],
                             df_confirmed: pd.DataFrame) -> Dict[str, Any]:
    """
    MSc-level methodological extensions, each grounded in the literature
    (see advanced_methods.py for citations):

    1.  Hyperparameter tuning (leakage-free GridSearchCV pipelines)
    2.  XGBoost + Stacking ensemble benchmarks
    3.  AUC 95% bootstrap confidence intervals
    4.  DeLong test for statistically comparing model AUCs
    5.  Probability calibration (isotonic) with before/after Brier & ECE
    6.  Threshold optimisation (Youden's J, F1)
    7.  Decision Curve Analysis (net benefit)
    8.  Tree SHAP on the best non-linear model
    9.  Fairness mitigation via group-specific thresholds
    10. Temporal (out-of-time) validation

    Each step is individually guarded so one failure cannot abort the rest.
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 13: ADVANCED METHODOLOGY (MSc EXTENSIONS)")
    logger.info("="*80)

    import advanced_methods as am

    vp = f"{CONFIG['output_folder']}/visualizations/advanced_plots"
    out: Dict[str, Any] = {}

    models = model_results['models']
    preds = model_results['predictions']
    X_train = model_results['X_train']; y_train = model_results['y_train']
    X_test = model_results['X_test']; y_test = model_results['y_test']
    feats = model_results['feature_columns']

    # 1. Hyperparameter tuning - LR was already tuned in Phase 4 (reused here to
    #    avoid a duplicate search); only RF is tuned in this step.
    try:
        tune = {'LR': {'best_params': {'clf__C': model_results['best_C']},
                       'cv_auc': model_results['best_C_cv_auc']}}
        logger.info(f"  Reusing Phase-4 LR tuning: C={model_results['best_C']} "
                    f"CV AUC={model_results['best_C_cv_auc']:.4f}")
        tune.update(am.tune_hyperparameters(model_results['X_train_raw'], y_train,
                                            logger, which=('RF',)))
        out['tuning'] = tune
        save_csv(pd.DataFrame([
            {'model': k, 'cv_auc': round(v['cv_auc'], 4), 'best_params': str(v['best_params'])}
            for k, v in tune.items()]), 'covid_hyperparameter_tuning.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Tuning failed: {e}")

    # 2. Advanced models (XGBoost + Stacking)
    adv_models = {}
    try:
        adv = am.train_advanced_models(X_train, y_train, X_test, y_test, models, logger)
        for name, (auc, p, mdl) in adv.items():
            preds[name] = p
            model_results['scores'][name] = auc
            adv_models[name] = mdl
        out['advanced_models'] = {k: v[0] for k, v in adv.items()}
    except Exception as e:
        logger.error(f"❌ Advanced models failed: {e}")

    # 3 + 4. Bootstrap CIs for every model + DeLong comparison
    try:
        rows = []
        for name, p in preds.items():
            mean_auc, lo, hi = am.bootstrap_auc_ci(y_test, p, n_boot=300)
            rows.append({'model': name, 'auc': round(roc_auc_score(y_test, p), 4),
                         'ci_lower': round(lo, 4), 'ci_upper': round(hi, 4)})
            logger.info(f"  {name}: AUC={roc_auc_score(y_test, p):.4f} "
                        f"[95% CI {lo:.4f}–{hi:.4f}]")
        out['auc_ci'] = pd.DataFrame(rows)
        save_csv(out['auc_ci'], 'covid_auc_confidence_intervals.csv', 'advanced')

        # DeLong: compare each model against LR (interpretable baseline)
        logger.info("DeLong tests vs Logistic Regression:")
        delong_rows = []
        for name, p in preds.items():
            if name == 'LR':
                continue
            a1, a2, z, pv = am.delong_roc_test(y_test, p, preds['LR'])
            verdict = "sig. better" if (pv < 0.05 and a1 > a2) else \
                      "sig. worse" if (pv < 0.05 and a1 < a2) else "no sig. difference"
            logger.info(f"  {name} vs LR: ΔAUC={a1-a2:+.4f}, p={pv:.3e} → {verdict}")
            delong_rows.append({'model_vs_LR': name, 'auc_model': round(a1, 4),
                                'auc_LR': round(a2, 4), 'delta_auc': round(a1-a2, 4),
                                'p_value': pv, 'verdict': verdict})
        save_csv(pd.DataFrame(delong_rows), 'covid_delong_tests.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Bootstrap/DeLong failed: {e}")

    # 5. Probability calibration improvement
    try:
        from sklearn.linear_model import LogisticRegression as _LR
        cal = am.improve_calibration(
            _LR(max_iter=1000, class_weight='balanced', random_state=CONFIG['random_state']),
            X_train, y_train, X_test, y_test,
            f"{vp}/01_calibration_improvement.png", logger, method='isotonic')
        out['calibration'] = cal
        save_csv(pd.DataFrame([{
            'method': cal['method'],
            'brier_before': round(cal['brier_before'], 4), 'brier_after': round(cal['brier_after'], 4),
            'ece_before': round(cal['ece_before'], 4), 'ece_after': round(cal['ece_after'], 4),
        }]), 'covid_calibration_improvement.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Calibration improvement failed: {e}")

    pkey = CONFIG['primary_key']

    # 6. Threshold optimisation (on the deployed model)
    try:
        thr = am.optimize_threshold(y_test, preds[pkey], logger)
        out['thresholds'] = thr
        save_csv(thr['table'], 'covid_threshold_optimization.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Threshold optimisation failed: {e}")

    # 7. Decision curve analysis (on the deployed model)
    try:
        dca = am.decision_curve_analysis(y_test, preds[pkey],
                                         f"{vp}/02_decision_curve.png", logger)
        save_csv(dca, 'covid_decision_curve.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Decision curve failed: {e}")

    # 8. Tree SHAP on best non-linear model (prefer XGBoost, else GB)
    try:
        tree_model = adv_models.get('XGBoost', models.get('GB'))
        if tree_model is not None:
            tshap = am.tree_shap_analysis(tree_model, X_test, feats,
                                          f"{vp}/03_tree_shap.png", logger)
            save_csv(tshap, 'covid_tree_shap_importance.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Tree SHAP failed: {e}")

    # 9. Fairness mitigation
    try:
        fm = am.fairness_mitigation(model_results['df_model'], model_results['idx_test'],
                                    y_test, preds[pkey], logger)
        if fm is not None and len(fm):
            save_csv(fm, 'covid_fairness_mitigation.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Fairness mitigation failed: {e}")

    # 10. Temporal validation
    try:
        tv = am.temporal_validation(df_confirmed, feats, logger)
        if tv:
            out['temporal'] = tv
            save_csv(pd.DataFrame([tv]), 'covid_temporal_validation.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Temporal validation failed: {e}")

    # 11. Odds Ratios with 95% CI + p-values (statsmodels) - publishable form
    try:
        or_ci = am.odds_ratios_with_ci(X_train, y_train, feats, logger)
        if or_ci is not None:
            out['or_ci'] = or_ci
            save_csv(or_ci, 'covid_odds_ratios_ci.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Odds-ratio CI failed: {e}")

    # 11b. Per-patient (local) SHAP - local explanations for the app
    try:
        local_shap = am.export_local_shap(
            models['LR'], X_test, model_results['X_test_raw'],
            model_results['idx_test'], model_results['df_model'],
            feats, preds, pkey, logger, n=100)
        save_csv(local_shap, 'covid_local_shap.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Local SHAP export failed: {e}")

    # 11c. Nomogram / points-based risk score (Steyerberg 2019) - interpretable calculator
    try:
        X_tr_imp = model_results['imputer'].transform(model_results['X_train_raw'])
        nomo, nomo_lookup = am.generate_nomogram(X_tr_imp, y_train, feats, logger)
        save_csv(nomo, 'covid_nomogram_points.csv', 'advanced')
        save_csv(nomo_lookup, 'covid_nomogram_risk_lookup.csv', 'advanced')
        out['nomogram'] = nomo
    except Exception as e:
        logger.error(f"❌ Nomogram failed: {e}")

    # 11d. Imbalance strategy: SMOTE vs class_weight (evidence for the design choice)
    try:
        smote_cmp = am.compare_smote_vs_classweight(X_train, y_train, X_test, y_test, logger)
        if smote_cmp is not None:
            out['imbalance_comparison'] = smote_cmp
            save_csv(smote_cmp, 'covid_imbalance_strategy_comparison.csv', 'advanced')
    except Exception as e:
        logger.error(f"❌ Imbalance comparison failed: {e}")

    # 12. Model card (Mitchell et al. 2019) - responsible-AI documentation
    try:
        from datetime import datetime as _dt
        ci_df = out.get('auc_ci')
        ens_row = ci_df[ci_df['model'] == pkey].iloc[0] if ci_df is not None else None
        cal = out.get('calibration', {})
        fm_attrs = fm.attrs if ('fm' in dir() and fm is not None and len(fm)) else {}
        or_table_md = ''
        if out.get('or_ci') is not None:
            or_table_md = out['or_ci'].head(5).to_markdown(index=False)
        info = {
            'date': _dt.now().strftime('%Y-%m-%d'),
            'model_name': 'Primary deployed model: calibrated, tuned Logistic Regression '
                          '(LR/RF/GB/XGBoost/Stacking benchmarked alongside)',
            'features': feats,
            'train_n': f"{len(y_train):,}", 'test_n': f"{len(y_test):,}",
            'ensemble_auc': f"{roc_auc_score(y_test, preds[pkey]):.4f}",
            'ensemble_ci': f"{ens_row['ci_lower']:.4f}–{ens_row['ci_upper']:.4f}" if ens_row is not None else '?',
            'temporal_auc': out.get('temporal', {}).get('temporal_auc', 'n/a'),
            'brier_before': f"{cal.get('brier_before', float('nan')):.4f}" if cal else '?',
            'brier_after': f"{cal.get('brier_after', float('nan')):.4f}" if cal else '?',
            'ece_before': f"{cal.get('ece_before', float('nan')):.4f}" if cal else '?',
            'ece_after': f"{cal.get('ece_after', float('nan')):.4f}" if cal else '?',
            'disp_before': f"{fm_attrs.get('disparity_before', float('nan')):.3f}" if fm_attrs else '?',
            'disp_after': f"{fm_attrs.get('disparity_after', float('nan')):.3f}" if fm_attrs else '?',
            'or_table': or_table_md,
        }
        am.generate_model_card(info, f"{CONFIG['output_folder']}/advanced/MODEL_CARD.md", logger)
    except Exception as e:
        logger.error(f"❌ Model card failed: {e}")

    # 13. Datasheet for the dataset (Gebru et al. 2021)
    try:
        from datetime import datetime as _dt2
        ds_info = {
            'date': _dt2.now().strftime('%Y-%m-%d'),
            'features': feats,
            'n_features': len(feats),
            'epv': f"{model_results.get('epv', 0):,.0f}",
            'n_events': f"{model_results.get('n_events', 0):,}",
            'n_total': '566,602',
            'n_confirmed': f"{len(df_confirmed):,}",
            'positive_rate': '12.31%',
        }
        am.generate_datasheet(ds_info, f"{CONFIG['output_folder']}/advanced/DATASHEET.md", logger)
    except Exception as e:
        logger.error(f"❌ Datasheet failed: {e}")

    # Persist trained models for the Streamlit app (no retraining needed)
    try:
        mp = f"{CONFIG['output_folder']}/models"
        joblib.dump(model_results['scaler'], f"{mp}/scaler.joblib")
        joblib.dump(model_results['imputer'], f"{mp}/imputer.joblib")
        for name, mdl in {**models, **adv_models}.items():
            joblib.dump(mdl, f"{mp}/model_{name}.joblib")
        # The PRIMARY deployed model (calibrated tuned LR) - what Streamlit should load.
        joblib.dump(model_results['deployed_model'], f"{mp}/model_primary_deployed.joblib")
        logger.info(f"✅ Persisted {len(models)+len(adv_models)+1} models + scaler + imputer to {mp}")
    except Exception as e:
        logger.error(f"❌ Model persistence failed: {e}")

    logger.info("✅ PHASE 13 COMPLETE: Advanced methodology finished")
    return out


# ============================================================================
# PHASE 12: EXPORT RESULTS
# ============================================================================

def phase12_export_results(df_full, df_confirmed, model_results, shap_results,
                          fair_results, cal_results, cb_results, exp_results, int_results, sens_results):
    """
    Export all analysis results to CSV files.
    
    Exports:
    - Core data (raw, processed, predictions)
    - Analysis (SHAP importance)
    - Fairness (bias audit)
    - Calibration (probability validation)
    - Cost-benefit (ROI scenarios)
    - Clinical explanations
    - Feature interactions
    - Sensitivity analyses
    
    Args:
        All analysis results from previous phases
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 12: EXPORTING RESULTS")
    logger.info("="*80)
    logger.info("Saving all analysis outputs to CSV files...")

    try:
        # Core data
        save_csv(df_full, 'covid_full_raw_analysis.csv', 'core_data')
        save_csv(df_confirmed, 'covid_confirmed_cleaned.csv', 'core_data')
        
        # Predictions
        # Map each prediction back to a real, traceable patient identifier.
        # Exports the PRIMARY deployed (calibrated, tuned LR) risk scores over the
        # TEST rows (idx_test) of df_model, recovering the dataset's 'id' column
        # instead of emitting a meaningless 0..N positional index.
        ens_pred = model_results['predictions'][CONFIG['primary_key']]
        test_rows = model_results['df_model'].iloc[model_results['idx_test']]
        if 'id' in test_rows.columns:
            patient_ids = test_rows['id'].values
        else:
            patient_ids = test_rows.index.values
        predictions_export = pd.DataFrame({
            'patient_id': patient_ids,
            'risk_score': (ens_pred * 100).round(1),
            'risk_level': pd.cut(ens_pred * 100,
                               bins=[0, 30, 50, 70, 100],
                               labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                               include_lowest=True),
            'mortality_probability': ens_pred.round(4)
        })
        save_csv(predictions_export, 'covid_predictions.csv', 'core_data')
        
        # Analysis
        if len(shap_results) > 0:
            save_csv(shap_results, 'covid_shap_values.csv', 'analysis')
        if 'odds_ratios' in model_results and len(model_results['odds_ratios']) > 0:
            # Descriptive regularised coefficients (canonical ORs are covid_odds_ratios_ci.csv)
            save_csv(model_results['odds_ratios'], 'covid_lr_coefficients.csv', 'analysis')
        
        # Fairness & Calibration
        if len(fair_results) > 0:
            save_csv(fair_results, 'covid_fairness_audit.csv', 'fairness_calibration')
        if len(cal_results) > 0:
            save_csv(cal_results, 'covid_calibration_analysis.csv', 'fairness_calibration')
        
        # Cost-Benefit
        if len(cb_results) > 0:
            save_csv(cb_results, 'covid_cost_benefit_analysis.csv', 'cost_benefit')
        
        # Clinical
        if len(exp_results) > 0:
            save_csv(exp_results, 'covid_clinical_explanations.csv', 'clinical_explanations')
        
        # Features
        if len(int_results) > 0:
            save_csv(int_results, 'covid_feature_interactions.csv', 'feature_analysis')
        if len(sens_results) > 0:
            save_csv(sens_results, 'covid_sensitivity_analysis.csv', 'feature_analysis')
        
        logger.info("✅ All exports complete!")
        logger.info("✅ PHASE 12 COMPLETE")
        
    except Exception as e:
        logger.error(f"❌ Phase 12 Error: {str(e)}")


if __name__ == "__main__":
    # Data file path comes from --data (defaults to covid.csv in the cwd)
    parser = argparse.ArgumentParser(
        description="COVID-19 Long COVID Risk Assessment Analysis"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="covid.csv",
        help="Path to the COVID-19 CSV dataset (default: covid.csv in current directory)",
    )
    args = parser.parse_args()
    DATA_FILE = args.data

    if not os.path.isfile(DATA_FILE):
        logger.error(f"❌ Dataset not found: {DATA_FILE}")
        logger.error("   Usage: python covid_analysis_full.py --data /path/to/covid.csv")
        sys.exit(1)

    create_folders()

    # Phase 1-2
    df = load_data(DATA_FILE)
    phase1_eda(df)
    df_full, df_confirmed = phase2_preprocessing(df)

    # Phase 3-4
    phase3_statistical_tests(df_confirmed)
    model_results = phase4_model_training(df_confirmed)

    # Phase 5-8
    shap_results = phase5_shap_analysis(
        model_results['models'], model_results['X_test'], model_results['feature_columns'])
    fair_results = phase6_fairness_analysis(
        model_results['df_model'], model_results['predictions'],
        model_results['y_test'], model_results['idx_test'])
    cal_results  = phase7_calibration_analysis(model_results['y_test'], model_results['predictions'])
    cb_results   = phase8_cost_benefit_analysis(
        df_confirmed, model_results['predictions'], model_results['y_test'])

    # Phase 9-12
    exp_results  = phase9_clinical_explanations(
        df_confirmed, model_results['predictions'],
        model_results['df_model'], model_results['idx_test'])
    int_results  = phase10_feature_interactions(model_results['models'], model_results['feature_columns'])
    sens_results = phase11_sensitivity_analysis(
        df_confirmed, model_results['predictions'],
        model_results['models'], model_results['scaler'],
        model_results['feature_columns'], model_results['imputer'],
        model_results['deployed_model'])

    # Phase 12: export core results (runs before the advanced extensions so the
    # phase banners read in ascending order 11 → 12 → 13).
    phase12_export_results(
        df_full, df_confirmed, model_results, shap_results,
        fair_results, cal_results, cb_results, exp_results,
        int_results, sens_results)

    # Phase 13: advanced methodology (MSc-level extensions) - saves its own CSVs
    phase13_advanced_methods(model_results, df_confirmed)

    logger.info("✅ ALL PHASES COMPLETE")

# ============================================================================
# END OF FILE (Extended Analysis Phases 1-12)
# ============================================================================
