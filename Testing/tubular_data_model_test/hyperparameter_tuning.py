
from pathlib import Path
import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

df = pd.read_csv(PROJECT_ROOT / "data" / "tabular_features.csv")

train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
val_ids = set(pd.read_csv(DATA_DIR / "splits" / "val_ids.csv")["material_id"])

magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
density_cols = [
    c for c in df.columns
    if c not in magpie_cols and c not in ("material_id", "log_bulk_modulus_vrh")
]
feature_cols = magpie_cols + density_cols

train_df = df[df["material_id"].isin(train_ids)]
val_df = df[df["material_id"].isin(val_ids)]

X_train, y_train = train_df[feature_cols], train_df["log_bulk_modulus_vrh"]
X_val, y_val = val_df[feature_cols], val_df["log_bulk_modulus_vrh"]

# --- 2. Sweep config ---
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_r2",
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3,
        },
        "max_iter": {
            "values": [100, 200, 300, 500],
        },
        "max_depth": {
            "values": [3, 5, 7, 10, None],
        },
        "max_leaf_nodes": {
            "values": [15, 31, 63, 127],
        },
        "min_samples_leaf": {
            "values": [5, 10, 20, 50],
        },
        "l2_regularization": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 10.0,
        },
        "feature_set": {
            "value": "comp_plus_density",
        },
    },
}


# --- 3. Training function ---
def train():
    with wandb.init() as run:
        config = wandb.config

        model = HistGradientBoostingRegressor(
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_depth=config.max_depth,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,  # HistGB's internal early-stopping split, drawn from X_train
            n_iter_no_change=15,
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)

        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_mae = float(mean_absolute_error(y_val, val_pred))
        val_r2 = float(r2_score(y_val, val_pred))

        run.log({
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "n_estimators_used": model.n_iter_,  # actual iterations before early stop
        })

        run.summary["val_rmse"] = val_rmse
        run.summary["val_mae"] = val_mae
        run.summary["val_r2"] = val_r2


# --- 4. Initialize & run sweep agent ---
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="88-eateatyumm-imperial-college-london",
        project="MaterialMind_ML",
    )

    wandb.agent(sweep_id, function=train, count=35)