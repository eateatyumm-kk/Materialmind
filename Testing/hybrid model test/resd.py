"""
Residual-learning hybrid model.

Design (chosen over the reverse direction because HistGB, R2=0.9358, is
the stronger single model — it should own the primary prediction, with
the GNN correcting whatever systematic error pattern it leaves behind):

    1. HistGB (comp+density, tuned config) makes the primary prediction.
    2. residual = actual_log_K - histgb_pred, computed on TRAIN only.
    3. A second model is trained to predict that residual, using the GNN
       embedding as input (NOT concatenated with tabular features — the
       point is to isolate whether the embedding alone explains HistGB's
       mistakes, since concatenating it with tabular features is exactly
       what the earlier hybrid attempt already tried and found redundant).
    4. Final prediction = histgb_pred + residual_pred.

This directly tests the hypothesis from the concatenation-hybrid result:
if the embedding is redundant with density features, it should ALSO fail
to explain HistGB's residual (since the residual is precisely the part
density-based features couldn't capture). If the embedding contains
genuinely complementary structural information, it should predict a
non-trivial fraction of that residual.

Evaluated once on TEST, same discipline as all prior scripts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# Winning tabular config (ethereal-sweep-20)
HISTGB_PARAMS = dict(
    learning_rate=0.18823,
    l2_regularization=0.0029355,
    max_depth=3,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=5,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)

# Lighter config for the residual model — the residual is a much smaller,
# noisier signal than the raw target, and residual models are prone to
# overfitting the noise; the winning config's high max_iter/low
# regularization was tuned for the ORIGINAL target's scale, not this one.
RESIDUAL_PARAMS = dict(
    learning_rate=0.05,
    max_iter=200,
    max_depth=3,
    max_leaf_nodes=15,
    min_samples_leaf=20,   # stronger leaf-size regularization than the base model
    l2_regularization=0.1,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)


def load_tabular_and_embeddings():
    tab = pd.read_csv(DATA_DIR / "tabular_features.csv")
    emb = pd.read_csv(DATA_DIR / "gnn_embeddings.csv")

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    val_ids = set(pd.read_csv(DATA_DIR / "splits" / "val_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    magpie_cols = [c for c in tab.columns if c.startswith("MagpieData")]
    target_like = [c for c in tab.columns if "bulk_modulus" in c.lower()]
    density_cols = [
        c for c in tab.columns
        if c not in magpie_cols and c not in (["material_id"] + target_like)
    ]
    tabular_feature_cols = magpie_cols + density_cols
    embed_cols = [c for c in emb.columns if c.startswith("gnn_embed_")]

    y_col = "log_bulk_modulus_vrh"
    if y_col not in tab.columns:
        raise KeyError(f"{y_col} not found — found target-like columns: {target_like}")

    # Join tabular + embeddings by material_id — the actual "hybrid" join point
    merged = tab.merge(emb, on="material_id", how="inner")
    if len(merged) != len(tab):
        print(f"WARNING: {len(tab) - len(merged)} materials dropped in tabular/embedding "
              f"merge — check that gnn_embeddings.csv covers the full dataset.")

    train_df = merged[merged["material_id"].isin(train_ids)].reset_index(drop=True)
    val_df = merged[merged["material_id"].isin(val_ids)].reset_index(drop=True)
    test_df = merged[merged["material_id"].isin(test_ids)].reset_index(drop=True)

    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    return train_df, val_df, test_df, tabular_feature_cols, embed_cols, y_col


def main():
    train_df, val_df, test_df, tabular_cols, embed_cols, y_col = load_tabular_and_embeddings()

    # --- Stage 1: primary HistGB model on tabular features only ---
    X_train_tab = train_df[tabular_cols]
    y_train = train_df[y_col]
    X_val_tab = val_df[tabular_cols]
    y_val = val_df[y_col]
    X_test_tab = test_df[tabular_cols]
    y_test = test_df[y_col]

    primary_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    primary_model.fit(X_train_tab, y_train)

    train_primary_pred = primary_model.predict(X_train_tab)
    val_primary_pred = primary_model.predict(X_val_tab)
    test_primary_pred = primary_model.predict(X_test_tab)

    # --- Residuals, computed on TRAIN (and reported on val for sanity) ---
    train_residual = y_train.to_numpy() - train_primary_pred
    val_residual_actual = y_val.to_numpy() - val_primary_pred

    print(f"\nTrain residual stats: mean={train_residual.mean():.5f}, "
          f"std={train_residual.std():.5f}")

    # --- Stage 2: residual model, trained on GNN EMBEDDING ONLY ---
    X_train_embed = train_df[embed_cols]
    X_val_embed = val_df[embed_cols]
    X_test_embed = test_df[embed_cols]

    residual_model = HistGradientBoostingRegressor(**RESIDUAL_PARAMS)
    residual_model.fit(X_train_embed, train_residual)

    val_residual_pred = residual_model.predict(X_val_embed)
    test_residual_pred = residual_model.predict(X_test_embed)

    # How much of the residual does the embedding actually explain?
    # (on val, since this is a diagnostic, not the final reported number)
    residual_r2_val = r2_score(val_residual_actual, val_residual_pred)
    print(f"R² of embedding predicting HistGB's residual (val): {residual_r2_val:.4f}")
    print("(near 0 or negative => embedding does NOT explain HistGB's errors; "
          "meaningfully positive => embedding IS capturing complementary signal)")

    # --- Final combined prediction ---
    test_final_pred = test_primary_pred + test_residual_pred

    # --- Metrics: primary alone vs. residual-corrected, same test set ---
    def report(y_true, y_pred, label):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{label}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    print("\n=== TEST SET: PRIMARY (HistGB alone) vs RESIDUAL-CORRECTED ===")
    primary_metrics = report(y_test.to_numpy(), test_primary_pred, "HistGB alone (baseline)")
    residual_metrics = report(y_test.to_numpy(), test_final_pred, "HistGB + GNN residual correction")

    improvement = primary_metrics["r2"] - residual_metrics["r2"]
    if residual_metrics["r2"] > primary_metrics["r2"]:
        print(f"\nResidual correction IMPROVED R² by {residual_metrics['r2'] - primary_metrics['r2']:.4f}")
    else:
        print(f"\nResidual correction did NOT improve R² (delta {-improvement:.4f}). "
              f"This would support the hypothesis that the GNN embedding is largely "
              f"redundant with density features and doesn't explain HistGB's remaining errors.")

    # --- Save everything ---
    results_df = pd.DataFrame({
        "material_id": test_df["material_id"],
        "actual_log_K": y_test.to_numpy(),
        "histgb_pred": test_primary_pred,
        "residual_correction": test_residual_pred,
        "final_pred": test_final_pred,
        "histgb_residual": y_test.to_numpy() - test_primary_pred,
        "final_residual": y_test.to_numpy() - test_final_pred,
    })
    results_df.to_csv(RESULTS_DIR / "residual_hybrid_test_predictions.csv", index=False)

    metrics_df = pd.DataFrame([
        {"model": "HistGB_alone", "subset": "test", **primary_metrics},
        {"model": "HistGB_plus_GNN_residual", "subset": "test", **residual_metrics},
        {"model": "residual_r2_on_val_diagnostic", "subset": "val", "r2": residual_r2_val,
         "rmse": None, "mae": None},
    ])
    metrics_df.to_csv(RESULTS_DIR / "residual_hybrid_test_metrics.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()