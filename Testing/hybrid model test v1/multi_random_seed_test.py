
from pathlib import Path

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_PATH / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT_PATH / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = ERROR_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

MAGPIE_PARAMS = dict(
    learning_rate=0.25008566386849446,
    l2_regularization=0.028004521423582184,
    max_depth=7,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=5,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)

RAW_HYBRID_PARAMS = dict(
    learning_rate=0.1086949308007322,
    l2_regularization=0.0002321764804508835,
    max_depth=5,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)

# --- Subgroup thresholds ---
# Fixed physical threshold (soft/extreme materials), PLUS percentile-based
# thirds (bottom 20% / middle 60% / top 20% by actual log-K), computed
# from the test set's own distribution rather than guessed constants —
# this avoids the earlier project mistake of a hardcoded GPa cutoff that
# caught only 3 materials out of a 1500-material test set.
EXTREME_LOW_K_GPA = 3.0
EXTREME_LOW_K_LOG = np.log10(EXTREME_LOW_K_GPA)


# ============================================================
# DATA LOADING
# ============================================================
def magpie_load_data():
    df_features = pd.read_csv(DATA_DIR / "tabular_features.csv")
    df_targets = pd.read_csv(DATA_DIR / "materials_master.csv")

    if "log_bulk_modulus_vrh" not in df_targets.columns:
        df_targets["log_bulk_modulus_vrh"] = np.log10(df_targets["bulk_modulus_vrh"])

    df = pd.merge(df_features, df_targets[["material_id", "log_bulk_modulus_vrh"]],
                  on="material_id", how="inner")

    train_ids = set(pd.read_csv(SPLITS_DIR / "train_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(SPLITS_DIR / "test_ids.csv")["material_id"])

    magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]

    train_df = df[df["material_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["material_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, test_df, magpie_cols


def emb_load_data():
    emb = pd.read_csv(DATA_DIR / "gnn_embeddings.csv")
    train_ids = set(pd.read_csv(SPLITS_DIR / "train_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(SPLITS_DIR / "test_ids.csv")["material_id"])

    emb_train = emb[emb["material_id"].isin(train_ids)].reset_index(drop=True)
    emb_test = emb[emb["material_id"].isin(test_ids)].reset_index(drop=True)
    emb_cols = [c for c in emb.columns if c.startswith("gnn_embed_")]
    return emb_train, emb_test, emb_cols


def assert_aligned(df_a, df_b, key="material_id"):
    """Hard check that two DataFrames' material_id columns match exactly
    in both membership AND order, before treating positional arrays as
    if they correspond to the same materials."""
    a_ids = df_a[key].to_numpy()
    b_ids = df_b[key].to_numpy()
    if not np.array_equal(a_ids, b_ids):
        raise ValueError(
            f"Row alignment check FAILED for key='{key}'. "
            f"This would silently corrupt results if not caught here. "
            f"Fix the merge/join logic before proceeding."
        )


# ============================================================
# MAGPIE MODEL — all seeds
# ============================================================
def run_magpie():
    results = []
    predictions = []

    train_df, test_df, magpie_cols = magpie_load_data()
    X_train = train_df[magpie_cols]
    y_train = train_df["log_bulk_modulus_vrh"]
    X_test = test_df[magpie_cols]
    y_test = test_df["log_bulk_modulus_vrh"].to_numpy()
    test_ids_ordered = test_df["material_id"].to_numpy()

    for seed in SEEDS:
        model = HistGradientBoostingRegressor(**MAGPIE_PARAMS, random_state=seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "model": "Magpie", "seed": seed,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        })

        for mat_id, yt, yp in zip(test_ids_ordered, y_test, pred):
            predictions.append({
                "model": "Magpie", "seed": seed, "material_id": mat_id,
                "y_true": yt, "y_pred": yp, "residual": yt - yp,
            })

    return results, predictions


# ============================================================
# RAW HYBRID MODEL — all seeds
# ============================================================
def run_raw_hybrid():
    results = []
    predictions = []

    train_df, test_df, magpie_cols = magpie_load_data()
    emb_train, emb_test, emb_cols = emb_load_data()

    # Merge features + embeddings + target all in ONE step, keeping
    # material_id attached throughout so alignment can be verified
    # explicitly rather than assumed.
    train_full = train_df[["material_id", "log_bulk_modulus_vrh"] + magpie_cols].merge(
        emb_train[["material_id"] + emb_cols], on="material_id", how="inner")
    test_full = test_df[["material_id", "log_bulk_modulus_vrh"] + magpie_cols].merge(
        emb_test[["material_id"] + emb_cols], on="material_id", how="inner")

    feature_cols = magpie_cols + emb_cols
    X_train = train_full[feature_cols]
    y_train = train_full["log_bulk_modulus_vrh"]
    X_test = test_full[feature_cols]
    y_test = test_full["log_bulk_modulus_vrh"].to_numpy()
    test_ids_ordered = test_full["material_id"].to_numpy()

    assert len(test_full) == len(test_df), (
        f"Row count changed after merging embeddings ({len(test_df)} -> "
        f"{len(test_full)}) — check for missing embeddings for some test materials."
    )

    for seed in SEEDS:
        model = HistGradientBoostingRegressor(**RAW_HYBRID_PARAMS, random_state=seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "model": "raw Hybrid", "seed": seed,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        })

        for mat_id, yt, yp in zip(test_ids_ordered, y_test, pred):
            predictions.append({
                "model": "raw Hybrid", "seed": seed, "material_id": mat_id,
                "y_true": yt, "y_pred": yp, "residual": yt - yp,
            })

    return results, predictions


# ============================================================
# GNN MODEL — all seeds
# ============================================================
def run_gnn():
    gnn_results = []
    gnn_predictions = []

    def load_split_graphs():
        graphs = torch.load(DATA_DIR / "graphs.pt", weights_only=False)
        train_ids = set(pd.read_csv(SPLITS_DIR / "train_ids.csv")["material_id"])
        val_ids = set(pd.read_csv(SPLITS_DIR / "val_ids.csv")["material_id"])
        test_ids = set(pd.read_csv(SPLITS_DIR / "test_ids.csv")["material_id"])

        train_graphs = [g for g in graphs if g.material_id in train_ids]
        val_graphs = [g for g in graphs if g.material_id in val_ids]
        test_graphs = [g for g in graphs if g.material_id in test_ids]
        print(f"train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
        return graphs, train_graphs, val_graphs, test_graphs

    all_graphs, train_graphs, val_graphs, test_graphs = load_split_graphs()

    train_y = torch.cat([g.y.view(-1) for g in train_graphs]).float()
    y_mean = train_y.mean()
    y_std = train_y.std()
    for g in all_graphs:
        g.y_scaled = ((g.y.float() - y_mean) / y_std).view(-1)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    class Config:
        learning_rate = 0.0001428170203496466
        batch_size = 64
        hidden_dim = 64
        fc_hidden = 128
        num_heads = 8
        dropout = 0.1
        attn_dropout = 0
        weight_decay = 0.000029904296072567457
        epochs = 300
        patience = 30

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class GaussianRBF(nn.Module):
        def __init__(self, start=0.0, stop=8.0, num_gaussians=16):
            super().__init__()
            self.offset = torch.linspace(start, stop, num_gaussians)
            self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2

        def forward(self, dist):
            dist = dist.unsqueeze(1)
            return torch.exp(self.coeff * (dist - self.offset.to(dist.device)) ** 2)

    class GATBulkModulus(nn.Module):
        def __init__(self, config, in_node_dim=6, in_edge_dim=2):
            super().__init__()
            self.num_heads = config.num_heads
            hidden = config.hidden_dim

            self.rbf = GaussianRBF(start=0.0, stop=8.0, num_gaussians=16)
            rbf_edge_dim = 17 if in_edge_dim == 2 else in_edge_dim

            self.node_embed = nn.Linear(in_node_dim, hidden)
            head_dim = hidden // self.num_heads
            self.conv1 = TransformerConv(hidden, head_dim, heads=self.num_heads, edge_dim=rbf_edge_dim, concat=True)
            self.conv2 = TransformerConv(hidden, head_dim, heads=self.num_heads, edge_dim=rbf_edge_dim, concat=True)
            self.conv3 = TransformerConv(hidden, head_dim, heads=self.num_heads, edge_dim=rbf_edge_dim, concat=True)

            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.norm3 = nn.LayerNorm(hidden)
            self.dropout = nn.Dropout(p=config.dropout)

            self.fc1 = nn.Linear(hidden, config.fc_hidden)
            self.fc2 = nn.Linear(config.fc_hidden, config.fc_hidden // 2)
            self.fc_out = nn.Linear(config.fc_hidden // 2, 1)

        def _graph_embedding(self, data):
            x, edge_index, edge_attr, batch = (
                data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
            )
            distance = edge_attr[:, 0]
            other_feature = edge_attr[:, 1]
            rbf_distance = self.rbf(distance)
            edge_attr = torch.cat([rbf_distance, other_feature.unsqueeze(1)], dim=1)

            x = F.relu(self.node_embed(x))
            x = self.dropout(x)

            h = self.conv1(x, edge_index, edge_attr)
            x = self.norm1(x + F.relu(h))
            x = self.dropout(x)

            h = self.conv2(x, edge_index, edge_attr)
            x = self.norm2(x + F.relu(h))
            x = self.dropout(x)

            h = self.conv3(x, edge_index, edge_attr)
            x = self.norm3(x + F.relu(h))

            return global_mean_pool(x, batch)

        def forward(self, data):
            graph_embedding = self._graph_embedding(data)
            h = F.relu(self.fc1(graph_embedding))
            h = self.dropout(h)
            h = F.relu(self.fc2(h))
            return self.fc_out(h).view(-1)

    def run_epoch(model, loader, criterion, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss, total_mae, total_n = 0.0, 0.0, 0

        for batch in loader:
            batch = batch.to(device)
            y = batch.y_scaled.view(-1).float()
            if is_train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(is_train):
                pred = model(batch)
                loss = criterion(pred, y)
            if torch.isnan(loss):
                continue
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            unscaled_pred = pred.detach() * y_std.to(device) + y_mean.to(device)
            unscaled_y = y * y_std.to(device) + y_mean.to(device)
            total_mae += torch.abs(unscaled_pred - unscaled_y).sum().item()
            total_n += bsz

        if total_n == 0:
            return None, None
        return total_loss / total_n, total_mae / total_n

    def train_final_model(seed):
        set_seed(seed)
        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False, num_workers=0)

        in_node_dim = train_graphs[0].x.size(1)
        edge_dim = train_graphs[0].edge_attr.size(1)

        model = GATBulkModulus(config, in_node_dim=in_node_dim, in_edge_dim=edge_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        ckpt_path = RESULTS_DIR / f"seed_gat_best_model_{seed}.pth"

        for epoch in range(config.epochs):
            train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer=None)

            if train_loss is None or val_loss is None:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1} (all-NaN epoch)")
                    break
                continue

            if epoch % 10 == 0 or epoch == config.epochs - 1:
                print(f"  Epoch {epoch+1:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}")
                break

        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model

    def evaluate_on_test(model, seed):
        test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)
        model.eval()
        all_preds, all_true = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_scaled = model(batch)
                pred_unscaled = (pred_scaled * y_std.to(device) + y_mean.to(device)).cpu().numpy()
                true_unscaled = batch.y.view(-1).cpu().numpy()
                all_preds.extend(pred_unscaled)
                all_true.extend(true_unscaled)

        all_ids = [g.material_id for g in test_graphs]
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        result = {
            "model": "GNN", "seed": seed,
            "RMSE": float(np.sqrt(mean_squared_error(all_true, all_preds))),
            "MAE": float(mean_absolute_error(all_true, all_preds)),
            "R2": float(r2_score(all_true, all_preds)),
        }
        preds = [
            {"model": "GNN", "seed": seed, "material_id": mid, "y_true": yt,
             "y_pred": yp, "residual": yt - yp}
            for mid, yt, yp in zip(all_ids, all_true, all_preds)
        ]
        return result, preds

    for seed in SEEDS:
        print(f"\n--- GNN seed {seed} ---")
        model = train_final_model(seed)
        result, preds = evaluate_on_test(model, seed)
        gnn_results.append(result)
        gnn_predictions.extend(preds)

    return gnn_results, gnn_predictions


# ============================================================
# PAIRED ANALYSIS: GNN vs Hybrid
# ============================================================
def paired_analysis(pred_df):
    """Paired comparison across seeds: for each seed, compute GNN R2 and
    Hybrid R2 on the SAME test set, then run a paired t-test on the
    per-seed R2 differences. Paired (not independent-sample) is correct
    here because both models are evaluated on identical materials each
    seed -- pairing controls for seed-to-seed test-set-difficulty noise
    that an unpaired test would miss."""
    gnn_by_seed = (pred_df[pred_df["model"] == "GNN"]
                   .groupby("seed")
                   .apply(lambda d: r2_score(d["y_true"], d["y_pred"])))
    hybrid_by_seed = (pred_df[pred_df["model"] == "raw Hybrid"]
                      .groupby("seed")
                      .apply(lambda d: r2_score(d["y_true"], d["y_pred"])))

    common_seeds = sorted(set(gnn_by_seed.index) & set(hybrid_by_seed.index))
    gnn_vals = gnn_by_seed.loc[common_seeds].to_numpy()
    hybrid_vals = hybrid_by_seed.loc[common_seeds].to_numpy()

    diffs = hybrid_vals - gnn_vals
    t_stat, p_value = stats.ttest_rel(hybrid_vals, gnn_vals)

    summary = pd.DataFrame({
        "seed": common_seeds, "GNN_R2": gnn_vals, "Hybrid_R2": hybrid_vals,
        "diff_hybrid_minus_gnn": diffs,
    })
    summary.attrs["t_stat"] = t_stat
    summary.attrs["p_value"] = p_value
    summary.attrs["mean_diff"] = float(diffs.mean())
    summary.attrs["std_diff"] = float(diffs.std())

    print(f"\nPaired GNN vs Hybrid (n={len(common_seeds)} seeds):")
    print(summary.to_string(index=False))
    print(f"Mean R2 diff (Hybrid - GNN): {diffs.mean():+.4f} (std {diffs.std():.4f})")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("-> Statistically significant difference at alpha=0.05.")
    else:
        print("-> NOT statistically significant at alpha=0.05 with this many seeds. "
              "More seeds would be needed to draw a confident conclusion either way.")

    return summary


# ============================================================
# SUBGROUP ANALYSIS
# ============================================================
def subgroup_analysis(pred_df):
    """Bottom 20% / middle 60% / top 20% by ACTUAL log-K (computed from
    the test set's own distribution, not a hardcoded constant), plus the
    fixed physical extreme-low-K cutoff (<3 GPa) as a separate, smaller
    diagnostic slice."""
    rows = []

    for model_name in pred_df["model"].unique():
        model_df = pred_df[pred_df["model"] == model_name]

        y_true_by_material = model_df.groupby("material_id")["y_true"].mean()
        p20 = y_true_by_material.quantile(0.20)
        p80 = y_true_by_material.quantile(0.80)

        for seed in model_df["seed"].unique():
            seed_df = model_df[model_df["seed"] == seed].copy()

            seed_df["subgroup_pct"] = np.select(
                [seed_df["y_true"] <= p20,
                 seed_df["y_true"] >= p80],
                ["bottom_20pct_lowK", "top_20pct_highK"],
                default="middle_60pct",
            )
            seed_df["is_extreme_low_k"] = seed_df["y_true"] < EXTREME_LOW_K_LOG

            for subgroup, group in seed_df.groupby("subgroup_pct"):
                rows.append({
                    "model": model_name, "seed": seed, "subgroup": subgroup,
                    "n": len(group),
                    "mae": mean_absolute_error(group["y_true"], group["y_pred"]),
                    "rmse": np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])),
                })

            extreme = seed_df[seed_df["is_extreme_low_k"]]
            if len(extreme) > 0:
                rows.append({
                    "model": model_name, "seed": seed,
                    "subgroup": f"extreme_low_K_(<{EXTREME_LOW_K_GPA}GPa)",
                    "n": len(extreme),
                    "mae": mean_absolute_error(extreme["y_true"], extreme["y_pred"]),
                    "rmse": np.sqrt(mean_squared_error(extreme["y_true"], extreme["y_pred"])),
                })

    subgroup_df = pd.DataFrame(rows)
    print("\n=== SUBGROUP ANALYSIS (mean across seeds) ===")
    print(subgroup_df.groupby(["model", "subgroup"])[["mae", "rmse", "n"]].mean().to_string())
    return subgroup_df


# ============================================================
# WHERE DO GNN AND HYBRID DISAGREE MOST?
# ============================================================
def find_biggest_disagreements(pred_df, top_n=15):
    gnn = pred_df[pred_df["model"] == "GNN"].groupby("material_id").agg(
        y_true=("y_true", "mean"), gnn_pred=("y_pred", "mean"))
    hybrid = pred_df[pred_df["model"] == "raw Hybrid"].groupby("material_id").agg(
        hybrid_pred=("y_pred", "mean"))

    merged = gnn.join(hybrid, how="inner")
    merged["abs_diff_gnn_vs_hybrid"] = (merged["gnn_pred"] - merged["hybrid_pred"]).abs()
    merged["gnn_error"] = (merged["y_true"] - merged["gnn_pred"]).abs()
    merged["hybrid_error"] = (merged["y_true"] - merged["hybrid_pred"]).abs()
    merged["which_is_better"] = np.where(
        merged["hybrid_error"] < merged["gnn_error"], "hybrid_better", "gnn_better")

    top_disagreements = merged.sort_values("abs_diff_gnn_vs_hybrid", ascending=False).head(top_n)
    print(f"\n=== TOP {top_n} MATERIALS WHERE GNN AND HYBRID DISAGREE MOST ===")
    print(top_disagreements.to_string())

    return merged.reset_index(), top_disagreements.reset_index()


# ============================================================
# PLOTS
# ============================================================
def make_plots(results_df, pred_df, subgroup_df):
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=results_df, x="model", y="R2")
    sns.stripplot(data=results_df, x="model", y="R2", color="black", alpha=0.6, jitter=True)
    plt.title("Test R² across seeds, by model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "r2_by_model_boxplot.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, pred_df["model"].nunique(), figsize=(15, 5), sharex=True, sharey=True)
    for ax, model_name in zip(axes, sorted(pred_df["model"].unique())):
        avg = pred_df[pred_df["model"] == model_name].groupby("material_id").agg(
            y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        ax.scatter(avg["y_true"], avg["y_pred"], alpha=0.4, s=12, edgecolors="k", linewidth=0.3)
        lims = [avg["y_true"].min(), avg["y_true"].max()]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_title(model_name)
        ax.set_xlabel("Actual log10(K)")
    axes[0].set_ylabel("Predicted log10(K) (seed-averaged)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predicted_vs_actual_by_model.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for model_name in pred_df["model"].unique():
        sns.kdeplot(pred_df[pred_df["model"] == model_name]["residual"], label=model_name, fill=True, alpha=0.3)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Residual (y_true - y_pred)")
    plt.title("Residual distribution by model (all seeds pooled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_distributions.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    subgroup_means = subgroup_df.groupby(["model", "subgroup"])["mae"].mean().reset_index()
    sns.barplot(data=subgroup_means, x="subgroup", y="mae", hue="model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Mean MAE by subgroup and model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "subgroup_mae_comparison.png", dpi=150)
    plt.close()

    print(f"\nSaved 4 plots to {PLOTS_DIR}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_results = []
    all_predictions = []

    print("=== Running Magpie ===")
    magpie_results, magpie_preds = run_magpie()
    all_results.extend(magpie_results)
    all_predictions.extend(magpie_preds)

    print("\n=== Running raw Hybrid ===")
    hybrid_results, hybrid_preds = run_raw_hybrid()
    all_results.extend(hybrid_results)
    all_predictions.extend(hybrid_preds)

    print("\n=== Running GNN ===")
    gnn_results, gnn_preds = run_gnn()
    all_results.extend(gnn_results)
    all_predictions.extend(gnn_preds)

    results_df = pd.DataFrame(all_results)
    pred_df = pd.DataFrame(all_predictions)

    results_df.to_csv(ERROR_DIR / "seed_results.csv", index=False)
    pred_df.to_csv(ERROR_DIR / "all_predictions_with_material_id.csv", index=False)

    print("\n" + "=" * 60)
    print("SEED TEST RESULTS")
    print("=" * 60)
    print(results_df)

    summary = results_df.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    print("\n" + "=" * 60)
    print("MEAN ± STD ACROSS SEEDS")
    print("=" * 60)
    print(summary)
    summary.to_csv(ERROR_DIR / "seed_summary.csv")

    paired_df = paired_analysis(pred_df)
    paired_df.to_csv(ERROR_DIR / "paired_gnn_vs_hybrid.csv", index=False)

    subgroup_df = subgroup_analysis(pred_df)
    subgroup_df.to_csv(ERROR_DIR / "subgroup_analysis.csv", index=False)

    all_disagreements, top_disagreements = find_biggest_disagreements(pred_df)
    all_disagreements.to_csv(ERROR_DIR / "gnn_vs_hybrid_all_materials.csv", index=False)
    top_disagreements.to_csv(ERROR_DIR / "gnn_vs_hybrid_top_disagreements.csv", index=False)

    make_plots(results_df, pred_df, subgroup_df)

    print(f"\nAll outputs saved under {ERROR_DIR}")