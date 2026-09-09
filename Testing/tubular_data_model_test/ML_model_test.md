# MaterialMind: Tabular Baseline Evaluation Report

This module evaluates baseline machine learning models on engineered material properties to predict Bulk Modulus ($\text{VRH}$). The goal is to establish strong tabular benchmarks before comparing performance against Graph Neural Network (GNN) architectures.

Baseline_comparison.py : aim to find better dataset combination and model (linear regression, XGboost, Gradient Boosting (Hist), Random Forest)

hyperparameter_tuning.py : find good hyperparameter combination using wandb

Hist_test.py : testing the hist model with test dataset.
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

=== Composition-Only Baseline Comparison ===
Model                          CV RMSE  CV MAE   CV R²
1. Dummy Baseline (Mean)       0.381   0.299 -0.0550
2. Scaled Linear (Ridge)       0.197   0.130  0.7146
3. Random Forest               0.159      0.090  0.8166
4. Gradient Boosting (Hist)    0.153   0.088  0.8299
5. XGBoost                     0.162   0.092  0.8105

=== VALIDATION SET PERFORMANCE (COMP) ===
Model: 4. Gradient Boosting (Hist)
Val RMSE: 0.133 (Log Scale)
Val MAE:  0.077 (Log Scale)
Val R²:   0.8894

--- TOP 10 LARGEST PREDICTION FAILURES ---
      Actual_Log_Bulk_Modulus  Predicted_Log_Bulk_Modulus  Absolute_Error  Residual
CSO                  0.303628                    1.340365        1.036738 -1.036738
IN                   1.954088                    0.919524        1.034564  1.034564
GaCl3               -0.036684                    0.927298        0.963983 -0.963983
SiCl4               -0.283162                    0.612180        0.895342 -0.895342
CuBr                 0.775538                    1.664439        0.888901 -0.888901
AlBr3               -0.007446                    0.871290        0.878737 -0.878737
CuN3                 1.198135                    1.949972        0.751837 -0.751837
P2Pd3S8              0.739414                    1.488404        0.748990 -0.748990
Cu2WS4               0.978226                    1.701623        0.723397 -0.723397
GdMg                 2.336394                    1.637125        0.699268  0.699268

=== Composition + Structure Baseline Comparison ===
Model                          CV RMSE  CV MAE   CV R²
1. Dummy Baseline (Mean)       0.381   0.299 -0.0550
2. Scaled Linear (Ridge)       0.152   0.101  0.8283
3. Random Forest               0.123   0.072  0.8888
4. Gradient Boosting (Hist)    0.115   0.068  0.9040
5. XGBoost                     0.121   0.072  0.8957

=== VALIDATION SET PERFORMANCE (COMP_DEN) ===
Model: 4. Gradient Boosting (Hist)
Val RMSE: 0.105 (Log Scale)
Val MAE:  0.061 (Log Scale)
Val R²:   0.9319

--- TOP 10 LARGEST PREDICTION FAILURES ---
      Actual_Log_Bulk_Modulus  Predicted_Log_Bulk_Modulus  Absolute_Error  Residual
SiCl4               -0.283162                    0.755948        1.039110 -1.039110
CuBr                 0.775538                    1.589518        0.813981 -0.813981
WN2                  2.084805                    1.311166        0.773639  0.7736392
GdMg                 2.336394                    1.650472        0.685922  0.685922
GaCl3               -0.036684                    0.625684        0.662368 -0.662368
ZnSO4                1.848189                    1.193080        0.655109  0.655109
Ne                   0.502017                    1.101375        0.599358 -0.599358
AlBr3               -0.007446                    0.583827        0.591274 -0.591274
NpBi                 1.174322                    1.755923        0.581601 -0.581601
P2Pd3S8              0.739414                    1.245785        0.506371 -0.506371


## Key Insights

* **Density Impact:** Incorporating physical density boosted model accuracy significantly, raising the top $R^2$ score from **0.8894** to **0.9319** and dropping log Validation MAE from **0.077** to **0.061**.
* **Model Champion:** `HistGradientBoostingRegressor` outperforms linear models and Random Forests across all metrics.
* **Residual Analysis:** Error distributions show tight, zero-centered residual peaks. Main prediction failures occur on extreme low-modulus outliers, where tree models tend to over-predict due to sparse training samples at boundaries.

## Material Prediction failures

---

## How to Run

Execute the baseline comparison script from the project root:

```bash
python Testing/tubular_data_model_test/Baseline_comparison.py


===================hyperparameter tuning result=====================

better model: 

    learning_rate=0.12074849061070954,
    l2_regularization=0.032350973806895354,
    max_depth=3,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,

===================test result=====================

train: 7021, test: 1488

=== FINAL TEST SET PERFORMANCE (log10 GPa) — HistGB comp+density ===
Test RMSE: 0.0984
Test MAE:  0.0571
Test R²:   0.9330


