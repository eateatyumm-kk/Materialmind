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

# --- Winning config from the W&B sweep report: ethereal-sweep-20 --- -> need to rerun sweeping for hybrid model with embeddings and update the best params here.
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
    df = df[["material_id", "log_bulk_modulus_vrh"]]
    emb = pd.read_csv(DATA_DIR / "gnn_embeddings.csv")
    df = df.merge(emb, on="material_id", how="left") 

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    emb_cols = [c for c in emb.columns if c.startswith("gnn_embed")]

    train_df = df[df["material_id"].isin(train_ids)].reset_index(drop=True)

    test_df = df[df["material_id"].isin(test_ids)].reset_index(drop=True)

    X_train, y_train = train_df[emb_cols], train_df["log_bulk_modulus_vrh"]
    X_test, y_test = test_df[emb_cols], test_df["log_bulk_modulus_vrh"]
    test_ids_ordered = test_df["material_id"]

    print(f"Feature set: {len(emb_cols)} columns "
        f"({len(emb_cols)} GNN embeddings)")

    return X_train, y_train, X_test, y_test, test_ids_ordered, emb_cols

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
    X_train, y_train, X_test, y_test, test_ids_ordered, emb_cols = load_data()

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

if __name__ == "__main__":
    main()