# MaterialMind: Tabular Baseline Evaluation Report

This module evaluates baseline machine learning models on engineered material properties to predict Bulk Modulus ($\text{VRH}$). The goal is to establish strong tabular benchmarks before comparing performance against Graph Neural Network (GNN) architectures.

Baseline_comparison.py : aim to find better dataset combination and model (linear regression, XGboost, Gradient Boosting (Hist), Random Forest)

hyperparameter_tuning.py : find good hyperparameter combination using wandb

Hist_test.py : testing the hist model with test dataset.

---
# Baseline_compariosn.py
## Model Comparison Results

=== Composition-Only Baseline Comparison ===
Model                          CV RMSE  CV MAE   CV R²
1. Dummy Baseline (Mean)       0.381   0.299 -0.0552
2. Scaled Linear (Ridge)       0.197   0.130  0.7145
3. Random Forest               0.158   0.089  0.8176
4. Gradient Boosting (Hist)    0.154   0.089  0.8282
5. XGBoost                     0.161   0.091  0.8136

=== VALIDATION SET PERFORMANCE (COMP) ===
Model: 4. Gradient Boosting (Hist)
Val RMSE: 0.133 (Log Scale)
Val MAE:  0.077 (Log Scale)
Val R²:   0.8895

--- TOP 10 LARGEST PREDICTION FAILURES ---
      Actual_Log_Bulk_Modulus  Predicted_Log_Bulk_Modulus  Absolute_Error  Residual
9041                 0.303628                    1.365423        1.061796 -1.061796
3438                 1.954088                    0.951258        1.002830  1.002830
9057                -0.283162                    0.652670        0.935833 -0.935833
9453                -0.036684                    0.876578        0.913263 -0.913263
8646                 0.775538                    1.673748        0.898210 -0.898210
8773                -0.007446                    0.863065        0.870512 -0.870512
276                  1.745059                    0.987127        0.757933  0.757933
5098                 0.739414                    1.495588        0.756174 -0.756174
9013                 1.198135                    1.928422        0.730288 -0.730288
6533                 0.978226                    1.706871        0.728645 -0.728645


## Key Insights


* **Model Champion:** `HistGradientBoostingRegressor` outperforms linear models and Random Forests across all metrics.
* **Residual Analysis:** Error distributions show tight, zero-centered residual peaks. Main prediction failures occur on extreme low-modulus outliers, where tree models tend to over-predict due to sparse training samples at boundaries.

## How to Run

Execute the baseline comparison script from the project root:

```bash
python Testing/tubular_data_model_test/Baseline_comparison.py
```
# hyperparameter_tuning.py

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

# Hist_test.py 

===================test result=====================

train: 7021, test: 1488

=== FINAL TEST SET PERFORMANCE (log10 GPa) — HistGB comp+density ===
Test RMSE: 0.0984
Test MAE:  0.0571
Test R²:   0.9330


