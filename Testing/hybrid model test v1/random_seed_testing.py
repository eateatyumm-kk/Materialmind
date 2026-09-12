"""
Multi-seed reproducibility check for the PCA-embedding hybrid model.

The single-run result (comp+density R2=0.9330 vs PCA-hybrid R2=0.9349) is
a real observation, but a single run's R2 is itself a random variable --
PCA's random_state, HistGB's internal early-stopping validation split, and
tree construction all have randomness in them. This script re-runs the
SAME hyperparameter config multiple times with different seeds, for BOTH
the baseline (comp+density) and the PCA-hybrid, so you can see whether the
improvement holds up consistently or is within normal run-to-run noise.

Reading the output:
  - If PCA-hybrid R2 is consistently above baseline R2 across all seeds
    (even if the exact number wobbles a bit), that's a real, reportable
    effect.
  - If the two distributions overlap heavily (PCA-hybrid sometimes above,
    sometimes below, sometimes equal to baseline), the improvement is not
    distinguishable from noise, and should be reported as such honestly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_PATH = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_PATH / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- Fixed hyperparameters from your winning run (tuning set 1) ---
# EDIT THESE to match the exact config you already used for the PCA-hybrid
# result reported (RMSE=0.0970, MAE=0.0552, R2=0.9349).
HISTGB_PARAMS_FIXED = dict(
    learning_rate=0.18823,
    l2_regularization=0.0029355,
    max_depth=3,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=5,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)
PCA_COUNT = 12  # EDIT to match the pca_count you used for the reported result

SEEDS = [0, 1, 2, 3, 4]  # each seed reruns PCA + HistGB's own random_state


def load_data():
    df_features = pd.read_csv(ROOT_PATH / "data" / "tabular_features.csv")
    df_targets = pd.read_csv(ROOT_PATH / "data" / "materials_master.csv")
    df_embedding = pd.read_csv(ROOT_PATH / "data" / "gnn_embeddings.csv")

    train_ids = set(pd.read_csv(ROOT_PATH / "data" / "splits" / "train_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(ROOT_PATH / "data" / "splits" / "test_ids.csv")["material_id"])

    magpie_cols = [c for c in df_features.columns if c.startswith("MagpieData")]
    target_like = [c for c in df_features.columns if "bulk_modulus" in c.lower()]
    density_cols = [
        c for c in df_features.columns
        if c not in magpie_cols and c not in (["material_id"] + target_like)
    ]
    feature_cols = magpie_cols + density_cols

    train_feat = df_features[df_features["material_id"].isin(train_ids)][
        ["material_id"] + feature_cols].reset_index(drop=True)
    test_feat = df_features[df_features["material_id"].isin(test_ids)][
        ["material_id"] + feature_cols].reset_index(drop=True)

    train_y = df_targets[df_targets["material_id"].isin(train_ids)][
        ["material_id", "log_bulk_modulus_vrh"]].reset_index(drop=True)
    test_y = df_targets[df_targets["material_id"].isin(test_ids)][
        ["material_id", "log_bulk_modulus_vrh"]].reset_index(drop=True)

    emb_cols = [c for c in df_embedding.columns if c.startswith("gnn_embed_")]
    train_emb = df_embedding[df_embedding["material_id"].isin(train_ids)].set_index("material_id")
    test_emb = df_embedding[df_embedding["material_id"].isin(test_ids)].set_index("material_id")

    train_df = train_feat.merge(train_y, on="material_id")
    test_df = test_feat.merge(test_y, on="material_id")

    train_emb_aligned = train_emb.loc[train_df["material_id"]][emb_cols]
    test_emb_aligned = test_emb.loc[test_df["material_id"]][emb_cols]

    return train_df, test_df, train_emb_aligned, test_emb_aligned, feature_cols


def run_one_trial(seed, train_df, test_df, train_emb, test_emb, feature_cols, use_pca):
    X_train_tab = train_df[feature_cols]
    y_train = train_df["log_bulk_modulus_vrh"]
    X_test_tab = test_df[feature_cols]
    y_test = test_df["log_bulk_modulus_vrh"]

    if use_pca:
        pca = PCA(n_components=PCA_COUNT, random_state=seed)
        train_pca = pca.fit_transform(train_emb)
        test_pca = pca.transform(test_emb)
        pca_cols = [f"pca_{i}" for i in range(PCA_COUNT)]

        X_train = pd.concat([
            X_train_tab.reset_index(drop=True),
            pd.DataFrame(train_pca, columns=pca_cols)
        ], axis=1)
        X_test = pd.concat([
            X_test_tab.reset_index(drop=True),
            pd.DataFrame(test_pca, columns=pca_cols)
        ], axis=1)
    else:
        X_train, X_test = X_train_tab, X_test_tab

    model = HistGradientBoostingRegressor(random_state=seed, **HISTGB_PARAMS_FIXED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "seed": seed,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def main():
    train_df, test_df, train_emb, test_emb, feature_cols = load_data()

    baseline_results = []
    hybrid_results = []

    for seed in SEEDS:
        b = run_one_trial(seed, train_df, test_df, train_emb, test_emb, feature_cols, use_pca=False)
        h = run_one_trial(seed, train_df, test_df, train_emb, test_emb, feature_cols, use_pca=True)
        baseline_results.append(b)
        hybrid_results.append(h)
        print(f"Seed {seed}: baseline R2={b['r2']:.4f}  |  PCA-hybrid R2={h['r2']:.4f}  "
              f"|  delta={h['r2'] - b['r2']:+.4f}")

    baseline_df = pd.DataFrame(baseline_results)
    hybrid_df = pd.DataFrame(hybrid_results)

    print(f"\n=== SUMMARY ACROSS {len(SEEDS)} SEEDS ===")
    print(f"Baseline (comp+density):    R2 mean={baseline_df['r2'].mean():.4f}, "
          f"std={baseline_df['r2'].std():.4f}, "
          f"range=[{baseline_df['r2'].min():.4f}, {baseline_df['r2'].max():.4f}]")
    print(f"PCA-hybrid (comp+density+GNN emb): R2 mean={hybrid_df['r2'].mean():.4f}, "
          f"std={hybrid_df['r2'].std():.4f}, "
          f"range=[{hybrid_df['r2'].min():.4f}, {hybrid_df['r2'].max():.4f}]")

    delta_mean = hybrid_df['r2'].mean() - baseline_df['r2'].mean()
    n_wins = (hybrid_df['r2'].to_numpy() > baseline_df['r2'].to_numpy()).sum()
    print(f"\nMean R2 improvement: {delta_mean:+.4f}")
    print(f"PCA-hybrid beat baseline in {n_wins}/{len(SEEDS)} seeds")

    if n_wins == len(SEEDS) and delta_mean > baseline_df['r2'].std():
        print("\n-> Consistent improvement across all seeds, larger than baseline's own "
              "run-to-run variance. This supports a REAL effect, not noise.")
    elif n_wins >= len(SEEDS) - 1:
        print("\n-> Improvement holds in most seeds. Likely a real but small effect -- "
              "report with the seed range shown, not just a single point estimate.")
    else:
        print("\n-> Improvement is inconsistent across seeds. This should be reported as "
              "NOT distinguishable from run-to-run noise, consistent with the earlier "
              "residual-learning finding that the embedding adds little independent signal.")

    combined = pd.concat([
        baseline_df.assign(model="baseline_comp_density"),
        hybrid_df.assign(model="pca_hybrid"),
    ])
    combined.to_csv(RESULTS_DIR / "pca_hybrid_multiseed_check.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'pca_hybrid_multiseed_check.csv'}")


if __name__ == "__main__":
    main()