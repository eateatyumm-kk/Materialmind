# MaterialMind: Tabular Baseline Evaluation Report

This module evaluates baseline machine learning models on engineered material properties to predict Bulk Modulus ($\text{VRH}$). The goal is to establish strong tabular benchmarks before comparing performance against Graph Neural Network (GNN) architectures.

---

## Dataset & Split Strategy

The data ingestion pipeline splits the material records into strict, non-overlapping holdout sets based on `material_id`:
* **Feature Sets:**
  1. **Composition-Only (`comp`):** Magpie elemental features derived strictly from chemical formulas.
  2. **Composition + Structure (`comp_den`):** Magpie elemental features combined with physical density and structural properties.
* **Target Variable:** `log_bulk_modulus_vrh` (log-transformed to stabilize right-skewed physical values and equalize relative loss errors).

---

## Model Comparison Results

Models were evaluated using 5-fold cross-validation on the training set, followed by evaluation on the validation split.

### 1. Composition-Only Baseline (`comp`)

| Model                     | CV RMSE (Log) | CV MAE (Log) | CV $R^2$　 | Validation $R^2$ |
| **Dummy Baseline (Mean)** | 0.350         | 0.275        | -0.0482    | —                |
| **Scaled Linear (Ridge)** | 0.158         | 0.105        | 0.7815     | —                |
| **Random Forest**         | 0.123         | 0.072        | 0.8661     | —                |
| **HistGradientBoosting**  | **0.117**     | **0.068**    | **0.8807** | **0.8925**       |

### 2. Composition + Structure Baseline (`comp_den`)

| Model                     | CV RMSE (Log) | CV MAE (Log) | CV $R^2$ 　| Validation $R^2$ |
| **Dummy Baseline (Mean)** | 0.350         | 0.275        | -0.0482    | — 　　　　　　　　|
| **Scaled Linear (Ridge)** | 0.120         | 0.080        | 0.8733     | — 　　　　　　　　|
| **Random Forest**         | 0.104         | 0.063        | 0.9033     | — 　　　　　　　　|
| **HistGradientBoosting**  | **0.095**     | **0.056**    | **0.9196** | **0.9277** 　　　|

---

## Key Insights

* **Density Impact:** Incorporating physical density boosted model accuracy significantly, raising the top $R^2$ score from **0.8925** to **0.9277** and dropping log Validation MAE from **0.061** to **0.050**.
* **Model Champion:** `HistGradientBoostingRegressor` outperforms linear models and Random Forests across all metrics.
* **Residual Analysis:** Error distributions show tight, zero-centered residual peaks. Main prediction failures occur on extreme low-modulus outliers, where tree models tend to over-predict due to sparse training samples at boundaries.

---

## How to Run

Execute the baseline comparison script from the project root:

```bash
python Testing/tubular_data_model_test/Baseline_comparison.py


===================hyperparameter tuning result=====================

wandb:  l2_regularization: 0.0027333622120007876
wandb:  learning_rate: 0.26225955885339747
wandb:  max_depth: 3
wandb:  max_iter: 500
wandb:  max_leaf_nodes: 63
wandb:  min_samples_leaf: 50

wandb: Run summary:
wandb: n_estimators_used 271
wandb:           val_mae 0.05425
wandb:            val_r2 0.92983
wandb:          val_rmse 0.09366