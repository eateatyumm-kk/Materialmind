from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

LOW_K_THRESHOLD_GPA = 3.0
LOW_K_THRESHOLD_LOG = np.log10(LOW_K_THRESHOLD_GPA)

BEST_PARAMS = dict(
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
)

def load_data():
    df_features = pd.read_csv(DATA_DIR / "tabular_features.csv")
    emb = pd.read_csv(DATA_DIR / "gnn_embeddings.csv")
    df_targets = pd.read_csv(DATA_DIR / "targets.csv")

    if "log_bulk_modulus_vrh" not in df_targets.columns:
        df_targets["log_bulk_modulus_vrh"] = np.log10(df_targets["bulk_modulus_vrh"])

    df = pd.merge(df_features, df_targets[["material_id", "log_bulk_modulus_vrh"]], on="material_id", how="inner")

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    # Separate train and test embeddings to prevent data leakage
    emb_train = emb[emb["material_id"].isin(train_ids)].reset_index(drop=True)
    emb_test = emb[emb["material_id"].isin(test_ids)].reset_index(drop=True)

    # Fit PCA ONLY on training data
    pca = PCA(n_components=12)
    train_pca_feats = pca.fit_transform(emb_train.drop(columns=["material_id"]))
    test_pca_feats = pca.transform(emb_test.drop(columns=["material_id"]))

    cols = [f"gnn_embed_{i+1}" for i in range(12)]
    
    emb_train_pca = pd.DataFrame(train_pca_feats, columns=cols)
    emb_train_pca.insert(0, "material_id", emb_train["material_id"])

    emb_test_pca = pd.DataFrame(test_pca_feats, columns=cols)
    emb_test_pca.insert(0, "material_id", emb_test["material_id"])

    emb_pca_all = pd.concat([emb_train_pca, emb_test_pca], axis=0, ignore_index=True)

    # Merge embeddings back to main df
    df = df.merge(emb_pca_all, on="material_id", how="left")

    magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
    emb_cols = [c for c in emb_pca_all.columns if c.startswith("gnn_embed")]

    target_like = [c for c in df.columns if "bulk_modulus" in c.lower()]
    density_cols = [
        c for c in df.columns
        if c not in magpie_cols
        and c not in emb_cols
        and c not in (["material_id"] + target_like)
    ]
    feature_cols = magpie_cols + emb_cols + density_cols

    train_df = df[df["material_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["material_id"].isin(test_ids)].reset_index(drop=True)

    y_col = "log_bulk_modulus_vrh"
    X_train, y_train = train_df[feature_cols], train_df[y_col]
    X_test, y_test = test_df[feature_cols], test_df[y_col]
    test_ids_ordered = test_df["material_id"]

    print(f"Feature set: {len(feature_cols)} columns "
          f"({len(magpie_cols)} Magpie + {len(emb_cols)} GNN embeddings + {len(density_cols)} density)")

    return X_train, y_train, X_test, y_test, test_ids_ordered, feature_cols

def main():
    X_train, y_train, X_test, y_test, test_ids_ordered, feature_cols = load_data()

    model = HistGradientBoostingRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_arr = y_test.to_numpy()

    test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred))
    test_mae = mean_absolute_error(y_test_arr, y_pred)
    test_r2 = r2_score(y_test_arr, y_pred)

    print("\n=== FINAL TEST SET PERFORMANCE (log10 GPa) ===")
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