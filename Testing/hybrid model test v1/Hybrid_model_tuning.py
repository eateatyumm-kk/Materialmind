
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_PATH = Path(__file__).resolve().parents[2]

USE_EMBEDDING = False

RUN_LABEL = "with_embedding" if USE_EMBEDDING else "control_no_embedding"

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_r2", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.3},
        "max_iter": {"values": [100, 200, 300, 500]},
        "max_depth": {"values": [3, 5, 7, 10, None]},
        "max_leaf_nodes": {"values": [15, 31, 63, 127]},
        "min_samples_leaf": {"values": [5, 10, 20, 50]},
        "l2_regularization": {"distribution": "log_uniform_values", "min": 1e-4, "max": 10.0},
        "feature_set": {"value": RUN_LABEL},
    },
}

if USE_EMBEDDING:
    sweep_config["parameters"]["pca_count"] = {"values": [6, 12, 24, 48]}

def load_raw_data():
    df_features = pd.read_csv(ROOT_PATH / "data" / "tabular_features.csv")
    df_targets = pd.read_csv(ROOT_PATH / "data" / "materials_master.csv")

    train_ids = set(pd.read_csv(ROOT_PATH / "data" / "splits" / "train_ids.csv")["material_id"])
    val_ids = set(pd.read_csv(ROOT_PATH / "data" / "splits" / "val_ids.csv")["material_id"])

    magpie_cols = [c for c in df_features.columns if c.startswith("MagpieData")]

    feature_cols = magpie_cols

    print(f"Tabular feature set: {len(magpie_cols)} Magpie"
          f"= {len(feature_cols)} columns")

    train_features = df_features[df_features["material_id"].isin(train_ids)][
        ["material_id"] + feature_cols].reset_index(drop=True)
    val_features = df_features[df_features["material_id"].isin(val_ids)][
        ["material_id"] + feature_cols].reset_index(drop=True)

    train_target = df_targets[df_targets["material_id"].isin(train_ids)][
        ["material_id", "log_bulk_modulus_vrh"]].reset_index(drop=True)
    val_target = df_targets[df_targets["material_id"].isin(val_ids)][
        ["material_id", "log_bulk_modulus_vrh"]].reset_index(drop=True)

    if USE_EMBEDDING:
        df_embedding = pd.read_csv(ROOT_PATH / "data" / "gnn_embeddings.csv")
        train_emb = df_embedding[df_embedding["material_id"].isin(train_ids)].reset_index(drop=True)
        val_emb = df_embedding[df_embedding["material_id"].isin(val_ids)].reset_index(drop=True)
    else:
        train_emb, val_emb = None, None

    return (train_features, val_features, train_target, val_target,
            train_emb, val_emb, feature_cols)

# initially store the train and val, for each turbular and embbedding separatly first

(train_features_raw, val_features_raw, train_target_raw, val_target_raw,
 train_emb_raw, val_emb_raw, feature_cols) = load_raw_data()

def build_train_val(pca_count: int):

    train_df = train_features_raw.merge(train_target_raw, on="material_id")
    val_df = val_features_raw.merge(val_target_raw, on="material_id")

    if USE_EMBEDDING:
        emb_cols = [c for c in train_emb_raw.columns if c.startswith("gnn_embed_")]

        pca = PCA(n_components=pca_count, random_state=42)
        train_emb_aligned = train_emb_raw.set_index("material_id").loc[train_df["material_id"]]
        val_emb_aligned = val_emb_raw.set_index("material_id").loc[val_df["material_id"]]

        train_pca = pca.fit_transform(train_emb_aligned[emb_cols])
        val_pca = pca.transform(val_emb_aligned[emb_cols])

        pca_cols = [f"pca_{i}" for i in range(pca_count)]
        train_df = pd.concat([
            train_df.reset_index(drop=True),
            pd.DataFrame(train_pca, columns=pca_cols)
        ], axis=1)
        val_df = pd.concat([
            val_df.reset_index(drop=True),
            pd.DataFrame(val_pca, columns=pca_cols)
        ], axis=1)

        X_cols = feature_cols + pca_cols
    else:
        X_cols = feature_cols

    X_train, y_train = train_df[X_cols], train_df["log_bulk_modulus_vrh"]
    X_val, y_val = val_df[X_cols], val_df["log_bulk_modulus_vrh"]
    return X_train, y_train, X_val, y_val

def train():
    with wandb.init() as run:
        config = wandb.config

        pca_count = config.pca_count if USE_EMBEDDING else 0
        X_train, y_train, X_val, y_val = build_train_val(pca_count)

        model = HistGradientBoostingRegressor(
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_depth=config.max_depth,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_mae = float(mean_absolute_error(y_val, val_pred))
        val_r2 = float(r2_score(y_val, val_pred))

        run.log({"val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2})
        run.summary["val_rmse"] = val_rmse
        run.summary["val_mae"] = val_mae
        run.summary["val_r2"] = val_r2
        run.summary["run_label"] = RUN_LABEL


if __name__ == "__main__":
    print(f"\n{'='*60}\nRunning sweep: {RUN_LABEL}\n{'='*60}\n")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="88-eateatyumm-imperial-college-london",
        project="No_PCA_Hybrid_sweep",
    )
    wandb.agent(sweep_id, function=train, count=20)