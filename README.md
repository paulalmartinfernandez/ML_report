# Diabetes prediction from health indicators (BRFSS 2015)

Machine Learning project for **PRDL** and **MLLB** (MUTSC, Universidad Politécnica de Madrid).
A single, annotated Jupyter notebook covers the whole analysis: exploratory data analysis, regression of BMI,
calibrated three-class classification of diabetes status and unsupervised health profiles. 

## 1. Data
**Diabetes Health Indicators Dataset** (derived from the CDC BRFSS 2015 survey): 253,680 rows, 21 predictors and the
target `Diabetes_012` (0 = no diabetes, 1 = prediabetes, 2 = diabetes).

1. Download `diabetes_012_health_indicators_BRFSS2015.csv` from Kaggle ("Diabetes Health Indicators Dataset").
2. Place it in `Data/`. The file is not versioned (see `.gitignore`).

## 2. Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook Diabetes_ML.ipynb
```


## 3. How to run it
The first cell of Part 0 holds the configuration (`FAST`, `SEED`, `FORCE_RECOMPUTE`, `SAVE_FIGS`).

| Step | What to do | Time |
|---|---|---|
| 1. Smoke test | `FAST = True`, *Run All* | minutes |
| 2. Full run | **restart the kernel**, `FAST = False`, *Run All* | hours the first time |
| 3. Later runs | *Run All* again: everything already computed is read from the cache | minutes |

- `FAST = True` shrinks the expensive parts (folds, repetitions, search iterations and subsample sizes) to check that
  every cell runs. Its numbers are **not** the ones reported.
- **Restart the kernel before switching `FAST`.** Variables of the previous run stay in memory otherwise.
- `FORCE_RECOMPUTE = True` ignores the cache. The cache is keyed by the configuration, the data and the code of each
  step, so changing a model recomputes only what depends on it.
- The first full run is long mainly because of the tuning and the repeated cross-validation (for reference, the
  L1-penalised logistic model alone takes about 6 minutes, XGBoost about 1 minute per repeated CV).

## 4. What the notebook contains
| Part | Content |
|---|---|
| 0. Setup | imports, configuration, cache, figure saving |
| 1. EDA | types, outliers, categorical variables, Cramér's V / Kruskal–Wallis, Spearman, duplicates and weights, PCA, ordinal variables, label quality, interactions, train/test split |
| 2. Regression of BMI | association, simple vs multiple regression, inference and diagnostics, improvements, variable selection, ridge/PCR/PLS, information criteria, non-linear models, SVR, test evaluation |
| 3. Classification | preprocessing and metrics, model families, SVM/KNN/QDA/NB, tuning (nested CV, four search strategies), ablations, statistical tests, calibration, final model, decision rule and cost-sensitive threshold, subgroups, learning curves, imbalance (weights vs SMOTE) |
| 4. Unsupervised | PCA + k-means, silhouette and stability, profiles, Gaussian mixtures, DBSCAN, hierarchical linkages, clusters as features |
| 5. Scorecard | verdict on H1–H11, limitations, exported outputs |

## 5. Repository layout
| Path | Content |
|---|---|
| `Diabetes_ML.ipynb` | the full analysis (Parts 0–5) |
| `Data/` | input CSV; the notebook also writes `train.csv` and `test.csv` here |
| `results/` | CSV tables: scorecard, CV and test metrics of every model, thresholds, subgroup performance, environment |
| `results/cache/` | cached intermediate results (safe to delete) |
| `models/` | fitted models (`.joblib`, not versioned), including `FINAL_classifier.joblib` (model and decision rule) |
| `figures/` | every figure as PNG, named `<part>_<nn>.png` |
| `docs/` | reports (PRDL and MLLB) |

The `figures/` folder of this repository already contains the figures of the full run, so they can be inspected
without running anything.

## 6. Method in short
- 23,899 duplicated rows are collapsed into 229,781 unique profiles with a frequency weight `w`.
- One stratified 80/20 split on the unique profiles (`SEED = 42`); the test set is used once.
- Scaling, encoding, feature selection, calibration and tuning live inside a `Pipeline` (no data leakage).
- Every model is trained with `sample_weight = w` and every metric is weighted.
- Main metric of the classification: log-loss, with AUC (one-vs-rest), macro-F1, Brier score, calibration error
  (ECE) and bootstrap confidence intervals.
- Models: logistic regression (L2, lasso, splines), LDA, QDA, Naive Bayes, KNN, SVM, trees, random forest,
  HistGradientBoosting and XGBoost for classification; OLS, ridge, PCR, PLS, splines, polynomials, model trees,
  random forest, boosting and SVR for regression; k-means, Ward, Gaussian mixtures and DBSCAN for clustering.

## 7. Main results (test set)
- Classification (XGBoost + decision rule): AUC diabetes vs rest 0.837, prediabetes vs rest 0.712, log-loss 0.387,
  ECE of the joint risk 0.0035.
- Regression of BMI: test R² 0.157 (linear model with improvements) and 0.181 (boosting).
- Clustering: k = 3 profiles; they do not re-discover the target (ARI 0.140).
- The verdict on every hypothesis is in `results/hypothesis_scorecard.csv` and in Part 5 of the notebook.

## 8. Reproducibility
- Seed everywhere; library versions in `requirements.txt`.
- All randomness goes through `SEED`; the cache makes repeated runs return identical numbers.
- To start from scratch, delete `results/cache/`, `models/` and `figures/` (or set `FORCE_RECOMPUTE = True`).

## 9. Troubleshooting
| Problem | Cause and solution |
|---|---|
| `FileNotFoundError` for the CSV | the file is not in `Data/` or has a different name |
| Different numbers after switching `FAST` | the kernel was not restarted; restart and run all |
| Very slow lasso/saga cell | expected: the L1 fit is capped (`max_iter=200`); do not raise it |
| `xgboost not installed: skipped` | optional package; `pip install xgboost` to include it |
| Memory errors | close other kernels; the SVM, SVR and KNN already use fixed subsamples |
| Many repeated PNG files | the setup cell empties `figures/` when it runs and saves each figure once; re-run Part 0 |

## 10. Limitations
Self-reported cross-sectional data; sample skewed towards high education and income; irreducible noise from identical
profiles with different labels; fixed subsamples for SVM, SVR and KNN; LDA/QDA/KNN do not accept sample weights;
k-means on mixed binary/ordinal variables is an approximation. Results are associations, not causal effects.

## 11. Author
Paula Martín Fernández, MUTSC, Universidad Politécnica de Madrid. Course 2026/27.
