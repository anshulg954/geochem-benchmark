# Unified Geochemistry Tabular Benchmark

A unified benchmarking pipeline that evaluates multiple ML models across tabular datasets for both classification and regression tasks in geochemistry. Produces normalized performance comparisons, per-dataset bar plots, and an aggregate summary table.

---

## Features

- Supports **classification** (ROC-AUC) and **regression** (RMSE) tasks, auto-detected from filenames
- Evaluates 10+ models including tree ensembles, neural networks, and TabPFN2.5
- 5-fold cross-validation with fixed seeds for reproducibility
- Per-dataset and aggregate normalized score plots (PDF)
- Raw metric bar plots (ROC-AUC / RMSE) per dataset
- Big summary table with raw scores, normalized scores, and per-dataset ranks

---

## Requirements

### Core dependencies (required)
```
numpy pandas torch matplotlib seaborn scikit-learn 
```

### Optional backends
| Package | Models enabled |
|---|---|
| `xgboost` | XGBoost classifier/regressor |
| `lightgbm` | LightGBM classifier/regressor |
| `catboost` | CatBoost classifier/regressor |
| `tabpfn` + `tabpfn_extensions` | TabPFN2.5 classifier/regressor |

The script runs gracefully without any optional packages — missing backends are skipped automatically.

---

## Directory Structure

```
geochem-benchmark/
├── datasets/                          # Input CSV files (set as DATA_DIR)
│   ├── AutomaticRock_classification.csv
│   ├── Jorgenson_Liq_Regression.csv
│   ├── Jorgenson_Pressure_Cpx_Regression.csv
│   ├── Jorgenson_Temp_Cpx_Regression.csv
│   ├── Jorgenson_Temp_Liq_Regression.csv
│   └── TE_Data_classification.csv
├── logs/                              # SLURM job logs
├── results/                           # Benchmark outputs (set as OUT_DIR)
│   └── benchmark_outputs_combined/
│       ├── raw_combined_results.csv
│       ├── normalized_combined_results.csv
│       ├── aggregate_barplot_ranked.pdf
│       ├── big_table_all_datasets.csv
│       ├── barplot_{dataset}.pdf      # One per dataset
│       └── raw_metric_plots/
│           └── barplot_{dataset}.pdf  # One per dataset
├── README.md                          # This file
├── requirements.txt
├── run_combined_benchmark_all_real.py  # Main benchmark script
└── run_eval.slurm                      # SLURM job submission script
```

---

## Datasets

The benchmark currently includes 6 geochemistry datasets covering 2 tasks:

### Classification (2 datasets)
| Dataset | Description |
|---|---|
| `AutomaticRock_classification.csv` | Rock type classification from geochemical measurements |
| `TE_Data_classification.csv` | Trace element data classification |

### Regression (4 datasets)
| Dataset | Target |
|---|---|
| `Jorgenson_Liq_Regression.csv` | Liquid-phase geochemical property prediction |
| `Jorgenson_Pressure_Cpx_Regression.csv` | Clinopyroxene pressure estimation |
| `Jorgenson_Temp_Cpx_Regression.csv` | Clinopyroxene temperature estimation |
| `Jorgenson_Temp_Liq_Regression.csv` | Liquid-phase temperature estimation |

---

## Dataset Format

Place CSV files in the `DATA_DIR` directory. The task type is inferred from the filename:

- Files containing `classification` in the name -> evaluated as classification
- Files containing `regression` in the name -> evaluated as regression
- All other files are skipped

The **last column** of each CSV is treated as the target variable. All other columns are features.

Recommended naming convention:
```
{domain}_classification_{dataset_name}.csv
{domain}_regression_{dataset_name}.csv
```

---

## Configuration

Edit the constants near the top of `benchmark.py`:

```python
DATA_DIR = Path("/path/to/your/data/")
OUT_DIR  = Path("/path/to/your/results/")
```

You can also adjust these in `main()`:

| Parameter | Default | Description |
|---|---|---|
| `n_splits` | `5` | Number of CV folds |
| `seed` | `42` | Random seed for reproducibility |
| `max_rows` | `50000` | Max rows per dataset (sampled if exceeded) |
| `max_columns` | `2000` | Max features; dataset skipped if exceeded |

---

## Usage

```bash
python benchmark.py
```

or 

Use Slurm
```
sbatch run_eval.slurm
```

---

## Outputs

All outputs are written to `OUT_DIR/benchmark_outputs_combined/`:

| File / Folder | Description |
|---|---|
| `raw_combined_results.csv` | Per-fold raw scores for every dataset × model |
| `normalized_combined_results.csv` | Same, with min-max normalized scores added |
| `aggregate_barplot_ranked.pdf` | Bar chart of mean normalized score across all datasets (error bars = SEM) |
| `barplot_{dataset}.pdf` | Per-dataset normalized score bar chart |
| `big_table_all_datasets.csv` | Wide table: one row per dataset, columns for raw score / normalized score / rank per model |
| `raw_metric_plots/barplot_{dataset}.pdf` | Per-dataset raw metric plots (ROC-AUC or RMSE) |

---

## Models

### Classification
| Name | Notes |
|---|---|
| `LogReg` | Logistic Regression |
| `KNN` | k-Nearest Neighbours |
| `RandomForest` | Random Forest |
| `SVM` | Support Vector Machine (with probability calibration) |
| `MLP` | Multi-layer Perceptron |
| `XGBoost` | Requires `xgboost` |
| `LightGBM` | Requires `lightgbm` |
| `CatBoost` | Requires `catboost` |
| `TabPFN2.5` | Requires `tabpfn`; uses `ManyClassClassifier` wrapper for >10 classes |

### Regression
Same set as above, with `LinReg` (Linear Regression) and `SVR` replacing `LogReg` and `SVM`.

---

## Normalization & Scoring

- Classification metric: **ROC-AUC** (higher is better)
- Regression metric: **RMSE** (lower is better); stored as negative RMSE (`neg_rmse`) so higher is always better across both tasks
- Scores are normalized **per dataset per split** using min-max scaling, making cross-task and cross-dataset comparisons valid
- `LogReg` and `LinReg` are combined into a single `Linear` label in aggregate plots

---

## Preprocessing

Two preprocessing pipelines are used:

- **TabPFN2.5**: Ordinal-encodes categoricals, passes numerics through as-is (no imputation, as TabPFN handles missingness internally)
- **All other models**: Imputes missing categoricals with a `__MISSING__` token + ordinal encodes; imputes missing numerics with the column mean (configurable)

---

## Reproducibility

- All CV splits use a fixed `random_state=42`
- The same `KFold` / `StratifiedKFold` object is reused across models for a given dataset, guaranteeing identical train/test splits
- A `split_checksum` column (sum of training indices) is stored in results for verification
