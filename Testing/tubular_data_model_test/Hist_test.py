"""
Final training + test evaluation for the tabular HistGradientBoosting
model, using the winning sweep config (ethereal-sweep-20).

This mirrors gat_final_train_eval.py's discipline exactly, so the two
models' numbers are directly comparable:
  1. Train on TRAIN only.
  2. Evaluate ONCE on TEST (never touched during the sweep).
  3. Report overall metrics AND a low-K subset metric (K < 3 GPa), since
     the baseline error analysis showed low-modulus / weakly-bonded
     materials (e.g. H4C) are where the tabular model struggles most —
     this is the specific regime the GNN/hybrid is meant to help with,
     so it needs its own row in the results table, not just an aggregate
     number.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def find_project_root(marker="data"):
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / marker).is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find a '{marker}' directory above {__file__}")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Low-K threshold in log10(GPa) space, computed exactly from the GPa cutoff
# rather than hardcoding the rounded log value.
LOW_K_THRESHOLD_GPA = 3.0
LOW_K_THRESHOLD_LOG = np.log10(LOW_K_THRESHOLD_GPA)

# --- Winning config from the W&B sweep report: ethereal-sweep-20 ---
BEST_PARAMS = dict(
    learning_rate=0.26225955885339747,
    l2_regularization=0.0027333622120007876,
    max_depth=3,
    max_iter=500,
    max_leaf_nodes=63,
    min_samples_leaf=50,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)

def load_data():
    df = pd.read_csv(DATA_DIR / "tabular_features.csv")

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
    # Defensive: exclude anything target-like by pattern, not just exact
    # old column names — this is the fix for the leakage bug found earlier
    # (log_bulk_modulus_vrh had leaked into tabular_features.csv).
    target_like = [c for c in df.columns if "bulk_modulus" in c.lower()]
    density_cols = [
        c for c in df.columns
        if c not in magpie_cols
        and c not in (["material_id"] + target_like)
    ]
    feature_cols = magpie_cols + density_cols

    train_df = df[df["material_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["material_id"].isin(test_ids)].reset_index(drop=True)

    y_col = "log_bulk_modulus_vrh"
    if y_col not in df.columns:
        raise KeyError(
            f"{y_col} not found in tabular_features.csv — "
            f"found target-like columns: {target_like}. Check which one "
            f"holds the log-transformed target and update y_col."
        )

    X_train, y_train = train_df[feature_cols], train_df[y_col]
    X_test, y_test = test_df[feature_cols], test_df[y_col]
    test_ids_ordered = test_df["material_id"]

    print(f"Feature set: {len(feature_cols)} columns "
          f"({len(magpie_cols)} Magpie + {len(density_cols)} density)")
    print(f"train: {len(X_train)}, test: {len(X_test)}")

    return X_train, y_train, X_test, y_test, test_ids_ordered, feature_cols


def evaluate_subset(y_true, y_pred, mask, label):
    if mask.sum() == 0:
        print(f"{label}: no materials in this subset — skipping.")
        return None
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    n = int(mask.sum())
    print(f"{label} (n={n}): MAE={mae:.4f}, RMSE={rmse:.4f}")
    return {"subset": label, "n": n, "mae": mae, "rmse": rmse}


def main():
    X_train, y_train, X_test, y_test, test_ids_ordered, feature_cols = load_data()

    model = HistGradientBoostingRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_arr = y_test.to_numpy()

    # --- Overall test metrics ---
    test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred))
    test_mae = mean_absolute_error(y_test_arr, y_pred)
    test_r2 = r2_score(y_test_arr, y_pred)

    print("\n=== FINAL TEST SET PERFORMANCE (log10 GPa) — HistGB comp+density ===")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R²:   {test_r2:.4f}")

    # --- Low-K subset metric ---
    print(f"\n=== LOW-K SUBSET (K < {LOW_K_THRESHOLD_GPA} GPa, "
          f"log10(K) < {LOW_K_THRESHOLD_LOG:.4f}) ===")
    low_k_mask = y_test_arr < LOW_K_THRESHOLD_LOG
    low_k_result = evaluate_subset(y_test_arr, y_pred, low_k_mask, "Low-K subset")

    # For context, also report the complement (K >= 3 GPa) so the low-K
    # number can be read against the rest of the distribution, not in
    # isolation.
    high_k_result = evaluate_subset(y_test_arr, y_pred, ~low_k_mask, "K >= 3 GPa subset")

    # --- Save everything for the results table / later report ---
    results_df = pd.DataFrame({
        "material_id": test_ids_ordered.to_numpy(),
        "actual_log_K": y_test_arr,
        "predicted_log_K": y_pred,
        "residual": y_test_arr - y_pred,
        "is_low_K": low_k_mask,
    })
    results_df.to_csv(RESULTS_DIR / "tabular_test_predictions.csv", index=False)

    metrics_rows = [{
        "model": "HistGB", "feature_set": "comp_plus_density",
        "subset": "overall", "n": len(y_test_arr),
        "rmse": test_rmse, "mae": test_mae, "r2": test_r2,
    }]
    if low_k_result:
        metrics_rows.append({
            "model": "HistGB", "feature_set": "comp_plus_density",
            "subset": f"low_K_(<{LOW_K_THRESHOLD_GPA}GPa)",
            "n": low_k_result["n"], "rmse": low_k_result["rmse"],
            "mae": low_k_result["mae"], "r2": None,
        })
    if high_k_result:
        metrics_rows.append({
            "model": "HistGB", "feature_set": "comp_plus_density",
            "subset": f"high_K_(>={LOW_K_THRESHOLD_GPA}GPa)",
            "n": high_k_result["n"], "rmse": high_k_result["rmse"],
            "mae": high_k_result["mae"], "r2": None,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "tabular_test_metrics.csv", index=False)
    print(f"\nSaved predictions and metrics to {RESULTS_DIR}")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_arr, y_pred, alpha=0.5, color='teal', edgecolors='k', linewidth=0.5)
    plt.plot([y_test_arr.min(), y_test_arr.max()], [y_test_arr.min(), y_test_arr.max()], 'r--', lw=2, label='Ideal (1:1)')
    plt.xlabel('Actual Log Bulk Modulus', fontsize=11)
    plt.ylabel('Predicted Log Bulk Modulus', fontsize=11)
    plt.title('Predicted vs. Actual', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    sns.histplot(y_test_arr - y_pred, kde=True, color='crimson', bins=30)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel('Residual Error (Actual - Predicted)', fontsize=11)
    plt.ylabel('Material Count', fontsize=11)
    plt.title('Error Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    return model, metrics_df


if __name__ == "__main__":
    main()