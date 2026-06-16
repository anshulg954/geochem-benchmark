#!/usr/bin/env python3
"""
Combined Classification and Regression Benchmark

Evaluates all datasets in the all_real_datasets directory.
Includes sanity checks to ensure identical data splits across models.

For regression, we use negative RMSE so higher is better (compatible with classification metrics).
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch, gc
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Regression models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Optional backends
HAVE_XGB = True
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as e:
    HAVE_XGB = False
    print("XGBoost not available:", e, file=sys.stderr)

HAVE_LIGHTGBM = True
try:
    import lightgbm as lgb
except Exception as e:
    HAVE_LIGHTGBM = False
    print("LightGBM not available:", e, file=sys.stderr)

HAVE_CATBOOST = True
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception as e:
    HAVE_CATBOOST = False
    print("CatBoost not available:", e, file=sys.stderr)

HAVE_TABPFNv2 = False
HAVE_TABPFN = True
HAVE_REAL_TABPFN = False
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions.many_class import ManyClassClassifier
    from tabpfn.constants import ModelVersion
except Exception as e:
    HAVE_TABPFNv2 = False
    HAVE_TABPFN = False
    HAVE_REAL_TABPFN = False
    print("TabPFN2.5 not available:", e, file=sys.stderr)

# Configuration
DATA_DIR = Path("/home/guptaa/anshul/geochem-benchmark/data/")
OUT_DIR = Path("/home/guptaa/anshul/geochem-benchmark/results/")

# -----------------------
# Estimator factories
# -----------------------

def make_classifier(name: str, multiclass: bool = False):
    if name == "LogReg":
        return LogisticRegression(random_state=42, n_jobs=-1)
    if name == "KNN":
        return KNeighborsClassifier(n_jobs=-1)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    if name == "SVM":
        return SVC(probability=True, random_state=42)
    if name == "MLP":
        return MLPClassifier(random_state=42)
    if name == "XGBoost" and HAVE_XGB:
        return XGBClassifier(random_state=42, verbose=-1, n_jobs=-1)
    if name == "LightGBM" and HAVE_LIGHTGBM:
        return lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    if name == "CatBoost" and HAVE_CATBOOST:
        return CatBoostClassifier(random_state=42, verbose=False, thread_count=-1)
    if name == "TabPFNv2" and HAVE_TABPFNv2:
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, ignore_pretraining_limits=True)
    if name == "TabPFN2.5" and HAVE_TABPFN:
        clf = TabPFNClassifier(
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            random_state=42,
            ignore_pretraining_limits=True,
            fit_mode="low_memory",
            memory_saving_mode=True
        )
        if multiclass:
            return ManyClassClassifier(
                    estimator=clf,
                    alphabet_size=10,  # TabPFN supports up to 10 classes by default
                    n_estimators_redundancy=4,  # Increase redundancy for better stability
                    random_state=42,
                )
        return clf
    return None

def make_regressor(name: str):
    if name == "LinReg":
        return LinearRegression(n_jobs=-1)
    if name == "KNN":
        return KNeighborsRegressor(n_jobs=-1)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=-1)
    if name == "SVR":
        return SVR()
    if name == "MLP":
        return MLPRegressor(random_state=42)
    if name == "XGBoost" and HAVE_XGB:
        return XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    if name == "LightGBM" and HAVE_LIGHTGBM:
        return lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    if name == "CatBoost" and HAVE_CATBOOST:
        return CatBoostRegressor(random_state=42, verbose=False, thread_count=-1)
    if name == "TabPFNv2" and HAVE_TABPFNv2:
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2, ignore_pretraining_limits=True)
    if name == "TabPFN2.5" and HAVE_TABPFN:
        return TabPFNRegressor(
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            random_state=42,
            ignore_pretraining_limits=True,
            fit_mode="low_memory",
            memory_saving_mode=True
        )
    return None

def available_classification_models() -> List[str]:
    names = ["LogReg", "RandomForest", "SVM", "MLP"]
    if HAVE_XGB: names.append("XGBoost")
    if HAVE_LIGHTGBM: names.append("LightGBM")
    if HAVE_CATBOOST: names.append("CatBoost")
    if HAVE_TABPFNv2: names.append("TabPFNv2")
    if HAVE_TABPFN: names.append("TabPFN2.5")
    if HAVE_REAL_TABPFN: names.append("RealTabPFN")
    return names

def available_regression_models() -> List[str]:
    names = ["LinReg", "RandomForest", "MLP", "SVR"]
    if HAVE_XGB: names.append("XGBoost")
    if HAVE_LIGHTGBM: names.append("LightGBM")
    if HAVE_CATBOOST: names.append("CatBoost")
    if HAVE_TABPFNv2: names.append("TabPFNv2")
    if HAVE_TABPFN: names.append("TabPFN2.5")
    if HAVE_REAL_TABPFN: names.append("RealTabPFN")
    return names

# -----------------------
# Preprocessor builders
# -----------------------

def build_preprocessor_for_tabpfn():
    """TabPFN2.5 preprocessing: categorical OrdinalEncode only, numeric passthrough."""
    categorical = Pipeline(steps=[
        ("ord", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32
        ))
    ])
    categorical_selector = make_column_selector(
        dtype_include=["object", "category", "bool", "boolean", "string", pd.StringDtype]
    )
    return make_column_transformer(
        (categorical, categorical_selector),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

def build_preprocessor_for_others(
    numeric_strategy: str = "mean",
    numeric_constant: float = 0.0,
    cat_missing_token: str = "__MISSING__",
):
    """Others: categorical impute + OrdinalEncode, numeric impute."""
    if numeric_strategy not in {"mean", "constant"}:
        raise ValueError("numeric_strategy must be 'mean' or 'constant'.")
    
    numeric_imputer = (
        SimpleImputer(strategy="mean")
        if numeric_strategy == "mean"
        else SimpleImputer(strategy="constant", fill_value=numeric_constant)
    )
    
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=cat_missing_token)),
        ("ord", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32
        )),
    ])
    
    numeric = Pipeline(steps=[("imputer", numeric_imputer)])
    
    categorical_selector = make_column_selector(
        dtype_include=["object", "category", "bool", "boolean", "string", pd.StringDtype]
    )
    
    return make_column_transformer(
        (categorical, categorical_selector),
        (numeric, make_column_selector(dtype_include=[np.number])),
        remainder="drop",
        verbose_feature_names_out=False,
    )

# -----------------------
# Metrics
# -----------------------

def roc_auc_safe(y_true, proba) -> float:
    y_true = np.asarray(y_true)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))

def predict_proba_safe(clf: Pipeline, X: pd.DataFrame) -> np.ndarray:
    est = clf.named_steps["clf"]
    if hasattr(est, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(est, "decision_function"):
        dec = clf.decision_function(X)
        if dec.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-dec))
            return np.vstack([1 - p1, p1]).T
        e = np.exp(dec - dec.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    raise RuntimeError(f"{est.__class__.__name__} cannot produce probabilities.")

def rmse_safe(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))

# -----------------------
# Evaluation functions
# -----------------------

def detect_task_type_from_filename(filename: str) -> str | None:
    """
    Detect task type from CSV filename.
    Returns: "classification", "regression", or None if neither found.
    """
    filename_lower = filename.lower()
    if "classification" in filename_lower:
        return "classification"
    elif "regression" in filename_lower:
        return "regression"
    return None

def evaluate_classification(
    data_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    max_rows: int | None = 50000,
    max_columns: int | None = 2000,
    models: List[str] | None = None,
    numeric_strategy: str = "mean",
    numeric_constant: float = 0.0,
    cat_missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    """Evaluate classification datasets."""
    csvs = sorted(data_dir.glob("*.csv"))
    if models is None:
        models = available_classification_models()
    
    rows: List[Dict] = []
    
    # -------------------------------------------------------------
    # CRITICAL: We define StratifiedKFold HERE with a fixed seed.
    # This generator will produce the EXACT same split indices 
    # every time we call .split(X, y) as long as X and y are unchanged.
    # -------------------------------------------------------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for i, p in enumerate(csvs, 1):
        # Detect task type from filename - skip if not classification
        task_type = detect_task_type_from_filename(p.name)
        if task_type != "classification":
            if task_type == "regression":
                print(f"Skipping {p.name}: detected as regression (not classification)")
            else:
                print(f"Skipping {p.name}: no task type found in filename")
            continue
        
        print(f"\n[{i}/{len(csvs)}] {p.name} (Classification)")
        df = pd.read_csv(p)
        df.replace({"": np.nan, "NA": np.nan, "N/A": np.nan}, inplace=True)
        
        target_col = df.columns[-1]
        
        if max_rows is not None and df.shape[0] > max_rows:
            y_temp = df[target_col]
            try:
                df, _ = train_test_split(
                    df, 
                    train_size=max_rows, 
                    stratify=y_temp, 
                    random_state=seed
                )
                print(f"Sampled {max_rows} rows (Stratified).")
            except ValueError:
                # Fallback if a class has only 1 member and cannot be stratified
                df = df.sample(n=max_rows, random_state=seed)
                print(f"Sampled {max_rows} rows (Random as stratification failed)")

        # Drop Rare Classes (Prevents StratifiedKFold & CatBoost crashes)
        y_temp_c = df[target_col]
        class_counts = y_temp_c.value_counts()
        valid_classes = class_counts[class_counts >= n_splits].index
        
        if len(valid_classes) < len(class_counts):
            dropped_count = len(class_counts) - len(valid_classes)
            print(f"  Dropping {dropped_count} rare classes (fewer than {n_splits} samples).")
            df = df[df[target_col].isin(valid_classes)].reset_index(drop=True)

        if df.empty or df[target_col].nunique() < 2:
            print(f"Skipping {p.name}: fewer than 2 classes remaining after filtering.")
            continue

        y_raw = df[target_col].values
        y = LabelEncoder().fit_transform(y_raw)
        
        if len(np.unique(y)) < 2:
            print(f"Skipping {p.name}: less than 2 unique values in target.")
            continue
        
        large_classes = False
        if len(np.unique(y)) > 10:
            print(f" Warning: More than 10 classes ({len(np.unique(y))}). TabPFN2.5 will use ManyClassClassifier wrapper.")
            large_classes = True
        
        X = df.drop(columns=[target_col])
        if max_columns is not None and X.shape[1] > max_columns:
            print(f"Skipping {p.name}: more than {max_columns} columns.")
            continue
        
        X = X.loc[:, X.nunique(dropna=False) > 1]
        print(f"  X: {X.shape}, y: {y.shape}")


        for name in models:
            est = make_classifier(name, multiclass=large_classes)
            if est is None:
                print(f"Skipping {name}: not available.")
                continue
            
            if name in ["TabPFNv2", "TabPFN2.5", "RealTabPFN"]:
                pre = build_preprocessor_for_tabpfn()
            else:
                pre = build_preprocessor_for_others(
                    numeric_strategy=numeric_strategy,
                    numeric_constant=numeric_constant,
                    cat_missing_token=cat_missing_token,
                )
            
            pipe = Pipeline([("pre", pre), ("clf", est)])
            
            for split_id, (tr, te) in enumerate(skf.split(X, y), 1):
                # --- SANITY CHECK: ENSURE IDENTICAL DATA SPLITS ---
                train_sum = tr.sum()
                test_sum = te.sum()
                # print(f"DEBUG: {name} | Split {split_id} | Train Sum: {train_sum} | Test Sum: {test_sum}")
                # Uncomment the above line if you want to see the checksums in your terminal
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipe.fit(X.iloc[tr], y[tr])
                except Exception as e:
                    print(f"Fit failed [{name}] split {split_id}: {e}", file=sys.stderr)
                    continue
                
                try:
                    proba = predict_proba_safe(pipe, X.iloc[te])
                    roc = roc_auc_safe(y[te], proba)
                    rows.append({
                        "ds": p.stem,
                        "task": "classification",
                        "method": name,
                        "split_number": split_id,
                        "roc": roc,
                        "score": roc,  # For compatibility
                        "split_checksum": train_sum # Store checksum for later verification
                    })
                    print(f"{name:12s} | split {split_id:02d} | ROC AUC: {roc:.4f} | Train Sum: {train_sum} | Test Sum: {test_sum}")
                except Exception as e:
                    print(f"Eval failed [{name}] split {split_id}: {e}", file=sys.stderr)
    
    return pd.DataFrame(rows)

def evaluate_regression(
    data_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    max_rows: int | None = 50000,
    max_columns: int | None = 2000,
    models: List[str] | None = None,
    numeric_strategy: str = "mean",
    numeric_constant: float = 0.0,
    cat_missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    """Evaluate regression datasets."""
    csvs = sorted(data_dir.glob("*.csv"))
    if models is None:
        models = available_regression_models()
    
    rows: List[Dict] = []
    
    # -------------------------------------------------------------
    # CRITICAL: We define KFold HERE with a fixed seed.
    # This generator will produce the EXACT same split indices 
    # every time we call .split(X, y) as long as X and y are unchanged.
    # -------------------------------------------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for i, p in enumerate(csvs, 1):
        # Detect task type from filename - skip if not regression
        task_type = detect_task_type_from_filename(p.name)
        if task_type != "regression":
            if task_type == "classification":
                print(f"Skipping {p.name}: detected as classification (not regression)")
            else:
                print(f"Skipping {p.name}: no task type found in filename")
            continue
        
        print(f"\n[{i}/{len(csvs)}] {p.name} (Regression)")
        df = pd.read_csv(p)
        df.replace({"": np.nan, "NA": np.nan, "N/A": np.nan}, inplace=True)
        
        target_col = df.columns[-1]
        
        # Drop rows with missing target
        df = df.dropna(subset=[target_col])
        
        if max_rows is not None and df.shape[0] > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
            print(f"Sampled {max_rows} rows.")
        
        y_raw = df[target_col]
        y = pd.to_numeric(y_raw, errors="coerce")
        
        nan_mask = y.isna()
        if nan_mask.any():
            print(f"  Dropping {nan_mask.sum()} rows where target couldn't be converted to numeric")
            df = df[~nan_mask].reset_index(drop=True)
            y = y[~nan_mask].reset_index(drop=True)
        
        if y.isna().all() or y.nunique(dropna=True) < 2:
            print(f"Skipping {p.name}: non-numeric or near-constant target.")
            continue
        
        X = df.drop(columns=[target_col])
        if max_columns is not None and X.shape[1] > max_columns:
            print(f"Skipping {p.name}: more than {max_columns} columns.")
            continue
        
        X = X.loc[:, X.nunique(dropna=False) > 1]
        print(f"  X: {X.shape}, y: {y.shape}")
        
        for name in models:
            est = make_regressor(name)
            if est is None:
                print(f"Skipping {name}: not available.")
                continue
            
            if name in ["TabPFNv2", "TabPFN2.5", "RealTabPFN"]:
                pre = build_preprocessor_for_tabpfn()
            else:
                pre = build_preprocessor_for_others(
                    numeric_strategy=numeric_strategy,
                    numeric_constant=numeric_constant,
                    cat_missing_token=cat_missing_token,
                )
            
            pipe = Pipeline([("pre", pre), ("reg", est)])
            
            for split_id, (tr, te) in enumerate(kf.split(X, y), 1):
                # --- SANITY CHECK: ENSURE IDENTICAL DATA SPLITS ---
                train_sum = tr.sum()
                test_sum = te.sum()
                # print(f"DEBUG: {name} | Split {split_id} | Train Sum: {train_sum} | Test Sum: {test_sum}")
                # Uncomment the above line if you want to see the checksums in your terminal
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipe.fit(X.iloc[tr], y.iloc[tr])
                except Exception as e:
                    print(f"Fit failed [{name}] split {split_id}: {e}", file=sys.stderr)
                    continue
                
                try:
                    preds = pipe.predict(X.iloc[te])
                    rmse = rmse_safe(y.iloc[te].values, preds)
                    neg_rmse = -rmse  # Higher is better
                    rows.append({
                        "ds": p.stem,
                        "task": "regression",
                        "method": name,
                        "split_number": split_id,
                        "rmse": rmse,
                        "neg_rmse": neg_rmse,
                        "score": neg_rmse,  # For compatibility
                        "split_checksum": train_sum # Store checksum for later verification
                    })
                    print(f"{name:12s} | split {split_id:02d} | RMSE: {rmse:.6f}")
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Eval failed [{name}] split {split_id}: {e}", file=sys.stderr)
    
    return pd.DataFrame(rows)

# -----------------------
# Normalization
# -----------------------

def normalized_per_dataset_split(df: pd.DataFrame, metric="score") -> pd.DataFrame:
    """Normalize scores per dataset and split (min-max normalization)."""
    tmp = df.copy()
    mins = tmp.groupby(["ds", "split_number"])[metric].min()
    maxs = tmp.groupby(["ds", "split_number"])[metric].max()
    tmp["normalized"] = tmp.apply(
        lambda r: (r[metric] - mins[(r["ds"], r["split_number"])]) /
                  (maxs[(r["ds"], r["split_number"])] - mins[(r["ds"], r["split_number"])] + 1e-12),
        axis=1
    )
    return tmp

# -----------------------
# Plotting
# -----------------------
def _sanitize_filename(name: str) -> str:
    """Replace characters that are unsafe in filenames with underscores."""
    import re
    return re.sub(r'(?u)[^-\w.]', '_', name)


def create_aggregate_barplot(df_normalized: pd.DataFrame, out_png: Path):
    """
    1) After normalization, average across splits within each dataset → one value per (ds, method).
    2) Compute overall mean across datasets per method.
    3) Error bars = SEM across those dataset-level means.
    
    Combines LogReg (classification) and LinReg (regression) into a single "Linear" method.
    """
    # Combine LogReg (classification) and LinReg (regression) into a single "Linear" method
    df_combined = df_normalized.copy()
    # Map LogReg -> Linear for classification, LinReg -> Linear for regression
    df_combined["method"] = df_combined.apply(
        lambda row: "Linear" if row["method"] in ["LogReg", "LinReg"] else row["method"],
        axis=1
    )
    
    # 1) Collapse splits within datasets
    per_ds = (
        df_combined
        .groupby(["ds", "method"], as_index=False)["normalized"]
        .mean()
        .rename(columns={"normalized": "score_per_dataset"})
    )

    # 2) Aggregate across datasets
    summary = (
        per_ds
        .groupby("method", as_index=False)["score_per_dataset"]
        .agg(mean="mean", std="std", count="count")
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))
    summary = summary.sort_values("mean", ascending=False)

    # Display names (optional)
    rename_map = {
        "Linear": "Linear",  # Combined LogReg (classification) + LinReg (regression)
        "SVM": "SVM",
        "SVR": "SVR",
        "XGBoost": "XGB",
        "RandomForest": "RF",
        "LightGBM": "LGBM",
        "CatBoost": "CB",
        "TabPFNv2": "TabPFNv2",
        "TabPFN2.5": "TabPFN2.5",
        "RealTabPFN": "Real-TabPFN-2.5",
        "KNN": "KNN",
        "MLP": "MLP",
    }
    summary["method_display"] = summary["method"].map(rename_map).fillna(summary["method"])

    # Plot (bars = mean; error bars = SEM)
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    methods_disp = summary["method_display"].tolist()
    methods_raw = summary["method"].tolist()
    means = summary["mean"].values
    errs = summary["sem"].values  # SEM

    x = np.arange(len(methods_disp))

    # Highlight TabPFN / RealTabPFN by raw method name
    highlight = {"TabPFNv2", "TabPFN2.5", "RealTabPFN"}
    colors = ["#101075" if m in highlight else "#1f77b4" for m in methods_raw]

    bars = ax.bar(
        x, means, width=0.6,
        color=colors, edgecolor="black", linewidth=1.5
    )
    ax.errorbar(
        x, means, yerr=errs, fmt="none",
        ecolor="black", elinewidth=1.2, capsize=3
    )

    ax.set_ylabel("Mean Normalized Score (higher is better)", fontsize=14)
    # ax.set_xlabel("Method", fontsize=14)
    # ax.set_title("Aggregate Performance Across Datasets (error bars = SEM)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_disp, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Numeric labels above bars (placed above mean + SEM)
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, means, errs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + (sem_val if np.isfinite(sem_val) else 0.0) + 0.01,
            f"{mean_val:.3f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

    # Return concise summary
    return summary[["method", "method_display", "mean", "sem", "count"]]

def create_per_dataset_barplots(df_normalized: pd.DataFrame, out_dir: Path):
    """
    Create individual bar plots for each dataset.
    For each dataset:
    1) Compute mean and SEM across splits per method
    2) Create bar plot with same styling as aggregate plot
    """
    # Combine LogReg (classification) and LinReg (regression) into a single "Linear" method
    df_combined = df_normalized.copy()
    df_combined["method"] = df_combined.apply(
        lambda row: "Linear" if row["method"] in ["LogReg", "LinReg"] else row["method"],
        axis=1
    )
    
    # Display names mapping
    rename_map = {
        "Linear": "Linear",
        "SVM": "SVM",
        "SVR": "SVR",
        "XGBoost": "XGB",
        "RandomForest": "RF",
        "LightGBM": "LGBM",
        "CatBoost": "CB",
        "TabPFNv2": "TabPFNv2",
        "TabPFN2.5": "TabPFN2.5",
        "RealTabPFN": "Real-TabPFN-2.5",
        "KNN": "KNN",
        "MLP": "MLP",
    }
    
    # Plot settings
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    
    # Get unique datasets
    datasets = sorted(df_combined["ds"].unique())
    
    for dataset_name in datasets:
        # Filter data for this dataset
        df_ds = df_combined[df_combined["ds"] == dataset_name].copy()
        
        # Compute mean and SEM across splits for each method
        summary = (
            df_ds
            .groupby("method", as_index=False)["normalized"]
            .agg(mean="mean", std="std", count="count")
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))
        summary = summary.sort_values("mean", ascending=False)
        summary["method_display"] = summary["method"].map(rename_map).fillna(summary["method"])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        methods_disp = summary["method_display"].tolist()
        methods_raw = summary["method"].tolist()
        means = summary["mean"].values
        errs = summary["sem"].values
        
        x = np.arange(len(methods_disp))
        
        # Highlight TabPFN / RealTabPFN
        highlight = {"TabPFNv2", "TabPFN2.5", "RealTabPFN"}
        colors = ["#101075" if m in highlight else "#1f77b4" for m in methods_raw]
        
        bars = ax.bar(
            x, means, width=0.6,
            color=colors, edgecolor="black", linewidth=1.5
        )
        ax.errorbar(
            x, means, yerr=errs, fmt="none",
            ecolor="black", elinewidth=1.2, capsize=3
        )
        
        # Get task type for this dataset
        task_type = df_ds["task"].iloc[0]
        ax.set_ylabel("Normalized Score (higher is better)", fontsize=14)
        # ax.set_xlabel("Method", fontsize=14)
        # ax.set_title(f"{dataset_name} ({task_type}) - Error bars = SEM", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(methods_disp, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        
        # Numeric labels above bars
        for i, (bar, mean_val, sem_val) in enumerate(zip(bars, means, errs)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + (sem_val if np.isfinite(sem_val) else 0.0) + 0.01,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=9
            )
        
        plt.tight_layout()
        
        # Save plot with sanitized filename
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")
        out_file = out_dir / f"barplot_{safe_name}.pdf"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")
    
    print(f"\nCreated {len(datasets)} per-dataset bar plots")


def create_raw_metric_barplots(df_raw: pd.DataFrame, out_dir: Path) -> None:
    """
    For every dataset in df_raw, create a bar plot of the task-appropriate raw
    metric (ROC-AUC for classification, RMSE for regression) with 95 % CI error
    bars across CV folds.  Bars are sorted best → worst.
 
    Per-task behaviour is controlled by a small config dict so both tasks share
    one code path:
 
    * Classification — metric column ``score`` (ROC-AUC), sorted descending,
      labels centred **inside** each bar (white, bold).
    * Regression — metric column ``rmse``, sorted ascending (lower is better),
      labels placed **above** each error whisker.
 
    One PDF per dataset is written to *out_dir* as ``barplot_{ds_name}.pdf``.
 
    Parameters
    ----------
    df_raw:
        Combined raw results DataFrame (output of ``pd.concat([df_class, df_reg])``).
        Must contain columns: ``ds``, ``task``, ``method``, ``score``, ``rmse``.
    out_dir:
        Directory where PDFs are written (must already exist).
    """
    # Per-task configuration — the only things that actually differ between tasks
    TASK_CFG = {
        "classification": dict(
            metric="score",
            ylabel="Mean ROC-AUC across 5-fold CV (Higher is Better)",
            ascending=False,          # best = highest
            label_inside=True,        # white labels centred inside bar
        ),
        "regression": dict(
            metric="rmse",
            ylabel="Mean RMSE across 5-fold CV (Lower is Better)",
            ascending=True,           # best = lowest
            label_inside=False,       # labels above the error whisker
        ),
    }
 
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
 
    datasets = sorted(df_raw["ds"].unique())
    n_saved = 0
 
    for ds_name in datasets:
        ds_df = df_raw[df_raw["ds"] == ds_name].copy()
        task = ds_df["task"].iloc[0]
 
        cfg = TASK_CFG.get(task)
        if cfg is None:
            print(f"Skipping {ds_name}: unknown task type '{task}'.")
            continue
 
        metric = cfg["metric"]
 
        method_order = (
            ds_df.groupby("method")[metric]
            .mean()
            .sort_values(ascending=cfg["ascending"])
            .index.tolist()
        )
 
        fig, ax = plt.subplots(figsize=(12, 7))
 
        sns.barplot(
            data=ds_df,
            x="method",
            y=metric,
            order=method_order,
            palette="viridis",
            errorbar=("ci", 95),
            capsize=0.1,
            ax=ax,
        )
 
        if cfg["label_inside"]:
            # Labels centred inside the bar (classification)
            for container in ax.containers:
                ax.bar_label(
                    container,
                    fmt="%.3f",
                    label_type="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )
        else:
            # Labels just above the 95 % CI whisker (regression)
            for i, method in enumerate(method_order):
                vals = ds_df[ds_df["method"] == method][metric].dropna()
                if vals.empty:
                    continue
                mean_val = vals.mean()
                sem_val = vals.sem() if len(vals) > 1 else 0.0
                error_top = mean_val + 1.96 * sem_val
                ax.text(
                    i,
                    error_top + abs(error_top) * 0.02 + ax.get_ylim()[1] * 0.005,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            # Extra headroom so labels are never clipped
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max * 1.12)
 
        ax.set_ylabel(cfg["ylabel"], fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
 
        plt.tight_layout()
        out_path = out_dir / f"barplot_{_sanitize_filename(ds_name)}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved ({task}): {out_path}")
        n_saved += 1
 
    print(f"\nCreated {n_saved} raw-metric bar plots.")

# -----------------------
# Big Table Creation
# -----------------------

def extract_domain_from_dataset_name(dataset_name: str, task: str) -> str:
    """
    Extract domain from dataset name.
    Format: {domain}_{task_type}_{dataset_name}
    """
    # Remove task type prefix to get domain
    task_prefix = f"_{task}_"
    if task_prefix in dataset_name:
        domain = dataset_name.split(task_prefix)[0]
        return domain
    # If pattern doesn't match, return "unknown"
    return "unknown"

def get_dataset_shape(data_dir: Path, dataset_name: str, task: str) -> Tuple[int | None, int | None]:
    """Get the number of rows and columns for a dataset."""
    # Try to find the CSV file - need to reconstruct filename
    # Format: {domain}_{task}_{dataset_name}.csv
    # But we only have dataset_name which is the stem
    # So we need to search for files matching the pattern
    csv_files = list(data_dir.glob(f"*{task}*.csv"))
    for csv_file in csv_files:
        if csv_file.stem == dataset_name:
            try:
                df = pd.read_csv(csv_file)
                return len(df), len(df.columns)
            except Exception as e:
                print(f"Warning: Could not read dataset {csv_file}: {e}", file=sys.stderr)
                return None, None
    return None, None

def compute_ranks(wide_df: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """Rank within each dataset (row). Method columns are ranked; smaller rank number is better."""
    ascending = not higher_is_better
    ranks = wide_df.rank(axis=1, method="min", ascending=ascending)
    ranks = ranks.astype("Int64")
    return ranks

def build_big_table(
    df_combined: pd.DataFrame,
    df_normalized: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Build a big table with all datasets, domains, tasks, methods, raw scores, normalized scores, and ranks.
    
    Format:
    - Each row is a dataset
    - Columns: dataset, domain, task, num_rows, num_columns
    - Then for each method: {method}_raw, {method}_normalized, {method}_rank
    """
    all_rows = []
    
    # Get unique datasets
    datasets = sorted(df_normalized["ds"].unique())
    
    for ds_name in datasets:
        # Get task type for this dataset
        ds_data = df_normalized[df_normalized["ds"] == ds_name]
        task = ds_data["task"].iloc[0] if len(ds_data) > 0 else "unknown"
        
        # Extract domain from dataset name
        domain = extract_domain_from_dataset_name(ds_name, task)
        
        # Get dataset shape
        num_rows, num_columns = get_dataset_shape(data_dir, ds_name, task)
        
        # Average across splits for raw scores
        raw_ds = df_combined[df_combined["ds"] == ds_name]
        raw_mean = raw_ds.groupby("method")["score"].mean()
        
        # Average across splits for normalized scores
        norm_ds = df_normalized[df_normalized["ds"] == ds_name]
        norm_mean = norm_ds.groupby("method")["normalized"].mean()
        
        # Get all methods
        all_methods = sorted(set(raw_mean.index.tolist() + norm_mean.index.tolist()))
        
        # Create row dictionary
        row_dict = {
            "dataset": ds_name,
            "domain": domain,
            "task": task,
            "num_rows": num_rows,
            "num_columns": num_columns,
        }
        
        # Add raw scores
        for method in all_methods:
            raw_val = raw_mean.get(method)
            row_dict[f"{method}_raw"] = float(raw_val) if raw_val is not None and not pd.isna(raw_val) else None
        
        # Add normalized scores
        for method in all_methods:
            norm_val = norm_mean.get(method)
            row_dict[f"{method}_normalized"] = float(norm_val) if norm_val is not None and not pd.isna(norm_val) else None
        
        all_rows.append(row_dict)
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    
    # Compute ranks based on normalized scores
    # Create pivot table for normalized scores
    norm_cols = [col for col in df.columns if col.endswith("_normalized")]
    methods_for_rank = [col.replace("_normalized", "") for col in norm_cols]
    
    if methods_for_rank:
        norm_wide = df.set_index("dataset")[norm_cols]
        norm_wide.columns = methods_for_rank  # Remove _normalized suffix
        
        # Compute ranks (higher normalized score is better)
        ranks = compute_ranks(norm_wide, higher_is_better=True)
        
        # Add rank columns
        for method in methods_for_rank:
            if method in ranks.columns:
                df[f"{method}_rank"] = df["dataset"].map(ranks[method])
            else:
                df[f"{method}_rank"] = None
    
    # Reorder columns: dataset, domain, task, num_rows, num_columns, then for each method: raw, normalized, rank
    method_set = set()
    for col in df.columns:
        if col.endswith("_raw") or col.endswith("_normalized") or col.endswith("_rank"):
            method = col.replace("_raw", "").replace("_normalized", "").replace("_rank", "")
            method_set.add(method)
    
    method_list = sorted(method_set)
    ordered_cols = ["dataset", "domain", "task", "num_rows", "num_columns"]
    for method in method_list:
        ordered_cols.extend([f"{method}_raw", f"{method}_normalized", f"{method}_rank"])
    
    # Only keep columns that exist
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols]
    
    return df

# -----------------------
# Main
# -----------------------

def main():
    # Configuration
    data_dir = DATA_DIR
    out_dir = OUT_DIR / "benchmark_outputs_combined"
    out_dir.mkdir(exist_ok=True)

    raw_plots_dir = out_dir / "raw_metric_plots"
    raw_plots_dir.mkdir(exist_ok=True)

    n_splits = 5
    seed = 42
    max_rows = 50000
    max_columns = 2000

    print("=" * 60)
    print("COMBINED CLASSIFICATION AND REGRESSION BENCHMARK")
    print("=" * 60)

    # Classification
    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION")
    print("=" * 60)
    df_class = evaluate_classification(
        data_dir=data_dir,
        n_splits=n_splits,
        seed=seed,
        max_rows=max_rows,
        max_columns=max_columns,
    )
    print(f"\nClassification results: {df_class.shape[0]} rows")

    # Regression
    print("\n" + "=" * 60)
    print("REGRESSION EVALUATION")
    print("=" * 60)
    df_reg = evaluate_regression(
        data_dir=data_dir,
        n_splits=n_splits,
        seed=seed,
        max_rows=max_rows,
        max_columns=max_columns,
    )
    print(f"\nRegression results: {df_reg.shape[0]} rows")

    # Combine and save raw
    df_combined = pd.concat([df_class, df_reg], ignore_index=True)
    raw_csv = out_dir / "raw_combined_results.csv"
    df_combined.to_csv(raw_csv, index=False)
    print(f"\nSaved raw results: {raw_csv}")
    print(f"Total results: {df_combined.shape[0]} rows across {df_combined['ds'].nunique()} datasets")
    print(f"Tasks: {df_combined['task'].value_counts().to_dict()}")

    # Raw-metric bar plots (before normalization)
    print("\n" + "=" * 60)
    print("CREATING RAW-METRIC BAR PLOTS (ROC-AUC / RMSE per dataset)")
    print("=" * 60)
    create_raw_metric_barplots(df_combined, raw_plots_dir)

    # Normalize per dataset & split
    print("\n" + "=" * 60)
    print("NORMALIZING RESULTS (per dataset, per split)")
    print("=" * 60)
    df_normalized = normalized_per_dataset_split(df_combined, metric="score")

    # Save normalized
    norm_csv = out_dir / "normalized_combined_results.csv"
    df_normalized.to_csv(norm_csv, index=False)
    print(f"Saved normalized results: {norm_csv}")

    # Bar plot with SEM error bars (after per-dataset averaging)
    print("\n" + "=" * 60)
    print("CREATING AGGREGATE BAR PLOT (mean across datasets; error bars = SEM)")
    print("=" * 60)
    summary_df = create_aggregate_barplot(
        df_normalized,
        out_dir / "aggregate_barplot_ranked.pdf"
    )

    print("\nMethod Rankings (mean ± SEM):")
    print(summary_df[["method_display", "mean", "sem", "count"]])

    # Create per-dataset bar plots
    print("\n" + "=" * 60)
    print("CREATING PER-DATASET BAR PLOTS (mean across splits; error bars = SEM)")
    print("=" * 60)
    create_per_dataset_barplots(df_normalized, out_dir)

    # Also print per-task summaries following the same rule (collapse splits → SEM across datasets)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS BY TASK (mean ± SEM across datasets)")
    print("=" * 60)
    per_ds_task = (
        df_normalized
        .groupby(["ds", "task", "method"], as_index=False)["normalized"]
        .mean()
        .rename(columns={"normalized": "score_per_dataset"})
    )
    task_summary = (
        per_ds_task
        .groupby(["task", "method"], as_index=False)["score_per_dataset"]
        .agg(mean="mean", std="std", count="count")
    )
    task_summary["sem"] = task_summary["std"] / np.sqrt(task_summary["count"].replace(0, np.nan))
    print(task_summary.sort_values(["task", "mean"], ascending=[True, False]))

    # Create big table
    print("\n" + "=" * 60)
    print("CREATING BIG TABLE (dataset-level summary with ranks)")
    print("=" * 60)
    big_table = build_big_table(df_combined, df_normalized, data_dir)
    
    if not big_table.empty:
        big_table_csv = out_dir / "big_table_all_datasets.csv"
        big_table.to_csv(big_table_csv, index=False)
        print(f"Saved big table: {big_table_csv}")
        print(f"  Shape: {big_table.shape}")
        print(f"  Columns: {len(big_table.columns)}")
        print(f"  Datasets: {len(big_table)}")
    else:
        print("Warning: Big table is empty!")

    print(f"\nDone. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()