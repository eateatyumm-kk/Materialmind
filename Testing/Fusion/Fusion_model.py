import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data
import pickle
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

class FiLMLayer(nn.Module):
    def __init__(self, tabular_dim, graph_dim):
        super().__init__()
        self.fc_gamma = nn.Linear(tabular_dim, graph_dim)
        self.fc_beta = nn.Linear(tabular_dim, graph_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x_graph, h_tab):
        gamma = self.fc_gamma(h_tab)  # [N, graph_dim]
        beta = self.fc_beta(h_tab)    # [N, graph_dim]
        return gamma * x_graph + beta


class GatedFusionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, h_graph, h_tab):
        cat_features = torch.cat([h_graph, h_tab], dim=-1)  # [Batch, 2 * hidden_dim]
        gate = torch.sigmoid(self.gate_fc(cat_features))     # [Batch, hidden_dim]
        h_fused = gate * h_graph + (1.0 - gate) * h_tab
        return h_fused, gate

class GaussianRBF(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=16):
        super().__init__()
        self.offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2

    def forward(self, dist):
        dist = dist.unsqueeze(1)
        return torch.exp(self.coeff * (dist - self.offset.to(dist.device)) ** 2)

class HybridFiLMGatedGAT(nn.Module):
    def __init__(self, config, in_node_dim=6, in_edge_dim=2, in_tabular_dim=132):
        super().__init__()
        self.num_heads = config.num_heads
        hidden = config.hidden_dim
        head_dim = hidden // self.num_heads

        # RBF expansion for continuous edge distances
        self.rbf = GaussianRBF(start=0.0, stop=8.0, num_gaussians=16)
        rbf_edge_dim = 17 if in_edge_dim == 2 else in_edge_dim

        # 1. Tabular Representation Pipeline
        self.tabular_mlp = nn.Sequential(
            nn.Linear(in_tabular_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
        self.node_embed = nn.Linear(in_node_dim, hidden)
        self.film = FiLMLayer(tabular_dim=hidden, graph_dim=hidden)

        self.conv1 = TransformerConv(hidden, head_dim, heads=self.num_heads, edge_dim=rbf_edge_dim, concat=True)
        self.conv2 = TransformerConv(hidden, head_dim, heads=self.num_heads, edge_dim=rbf_edge_dim, concat=True)
        self.conv3 = TransformerConv(hidden, head_dim, heads=self.num_heads, edge_dim=rbf_edge_dim, concat=True)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)

        self.dropout = nn.Dropout(p=config.dropout)

        self.gated_fusion = GatedFusionLayer(hidden_dim=hidden)

        self.fc1 = nn.Linear(hidden, config.fc_hidden)
        self.fc2 = nn.Linear(config.fc_hidden, config.fc_hidden // 2)
        self.fc_out = nn.Linear(config.fc_hidden // 2, 1)

    def forward(self, data, return_gate_weights=False):
        x, edge_index, edge_attr, batch = (
            data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
        )
        x_tabular = data.tabular_features.to(dtype=torch.float32, device=x.device)

        h_tab = self.tabular_mlp(x_tabular)  # [Batch, hidden]

        distance = edge_attr[:, 0]
        other_feature = edge_attr[:, 1]
        rbf_distance = self.rbf(distance)
        edge_attr = torch.cat([rbf_distance, other_feature.unsqueeze(1)], dim=1)

        x = F.relu(self.node_embed(x))
        h_tab_per_node = h_tab[batch]  # Broadcast batch-level tabular vector to all graph nodes
        x = self.film(x, h_tab_per_node)
        x = self.dropout(x)

        # Message Passing (Residual GAT Blocks)
        h = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x + F.relu(h))
        x = self.dropout(x)

        h = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x + F.relu(h))
        x = self.dropout(x)

        h = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x + F.relu(h))

        # Global Pooling -> Graph-Level Structural Vector
        h_graph_pooled = global_mean_pool(x, batch)  # [Batch, hidden]

        # Intermediate Gated Fusion
        h_fused, gate_weights = self.gated_fusion(h_graph_pooled, h_tab)  # [Batch, hidden]

        # Final MLP Regressor
        h = F.relu(self.fc1(h_fused))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        out = self.fc_out(h).view(-1)

        if return_gate_weights:
            return out, gate_weights
        return out

def load_split_graphs():
    graphs = torch.load(DATA_DIR / "graphs.pt", weights_only=False)
    tabular_df = pd.read_csv(DATA_DIR / "tabular_features.csv")

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    val_ids = set(pd.read_csv(DATA_DIR / "splits" / "val_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    if not train_ids.isdisjoint(val_ids):
        raise ValueError("Train and validation sets overlap")

    if not train_ids.isdisjoint(test_ids):
        raise ValueError("Train and test sets overlap")

    if not val_ids.isdisjoint(test_ids):
        raise ValueError("Validation and test sets overlap")

    print("Split overlap check passed: train/val/test are disjoint")

    target_like = [
        c for c in tabular_df.columns
        if "bulk_modulus" in c.lower()
    ]

    feature_cols = [
        c for c in tabular_df.columns
        if c.startswith("MagpieData")
    ]

    leaked_features = [
        c for c in feature_cols
        if c in target_like
    ]

    if leaked_features:
        raise ValueError(
            f"Target-like columns found among Magpie features: {leaked_features}"
        )

    if not all(pd.api.types.is_numeric_dtype(tabular_df[c]) for c in feature_cols):
        raise ValueError("Non-numeric column found in Magpie features")
    if tabular_df[feature_cols].isna().any().any():
        raise ValueError("NaN found in Magpie features")

    train_mask = tabular_df["material_id"].isin(train_ids)
    scaler = StandardScaler()
    scaler.fit(tabular_df.loc[train_mask, feature_cols])
    tabular_df[feature_cols] = scaler.transform(tabular_df[feature_cols])

    scaler_path = RESULTS_DIR / "fusion_tabular_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    print(f"Saved fitted tabular scaler to {scaler_path} "
          f"(needed at inference time to scale new materials identically)")

    tabular_lookup = {
        row["material_id"]: torch.tensor(row[feature_cols].to_numpy(dtype=np.float32))
        for _, row in tabular_df.iterrows()
    }

    for g in graphs:
        material_id = g.material_id
        if material_id not in tabular_lookup:
            raise ValueError(f"No Magpie features found for {material_id}")
        g.tabular_features = tabular_lookup[material_id].view(1, -1)

    train_graphs = [g for g in graphs if g.material_id in train_ids]
    val_graphs = [g for g in graphs if g.material_id in val_ids]
    test_graphs = [g for g in graphs if g.material_id in test_ids]

    print(f"Loaded graphs — train: {len(train_graphs)}, val: {len(val_graphs)}, "
          f"test: {len(test_graphs)}")
    print(f"Magpie feature dimension: {len(feature_cols)}")
    print(f"Example x_tabular shape (single graph): {train_graphs[0].tabular_features.shape}")


    check_loader = DataLoader(train_graphs, batch_size=8, shuffle=False)
    check_batch = next(iter(check_loader))
    expected_shape = (min(8, len(train_graphs)), len(feature_cols))
    actual_shape = tuple(check_batch.tabular_features.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Batched tabular_features shape mismatch: expected {expected_shape}, "
            f"got {actual_shape}. PyG's default batching is not concatenating "
            f"tabular_features as intended — a custom __cat_dim__ override is "
            f"needed (subclass torch_geometric.data.Data)."
        )
    print(f"Batch shape check passed: {actual_shape}")

    return train_graphs, val_graphs, test_graphs

train_graphs, val_graphs, test_graphs = load_split_graphs()

train_y = torch.cat([g.y.view(-1) for g in train_graphs]).float()
y_mean = train_y.mean()
y_std = train_y.std()

for g in train_graphs + val_graphs + test_graphs:
    g.y_scaled = ((g.y.float() - y_mean) / y_std).view(-1)

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values", "min": 0.0001, "max": 0.003,
        },
        "hidden_dim": {"values": [32, 64, 128]},        # Must be divisible by num_heads
        "num_heads": {"values": [2, 4, 8]},
        "fc_hidden": {"values": [32, 64, 128]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        "epochs": {"value": 300},
        "patience": {"value": 30},
        "batch_size": {"values": [16, 32, 64]},
        "weight_decay": {
            "distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3,
        },
        "architecture": {"value": "Hybrid-FiLM-Gated-GAT"},
        "dataset": {"value": "MaterialsProject_BulkModulus"},
    },
}

def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_mae, total_n = 0.0, 0.0, 0
    skipped_nan = 0
    gate_values = []

    for batch in loader:
        batch = batch.to(device)
        y = batch.y_scaled.view(-1).float()

        if is_train:
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                pred = model(batch)
                loss = criterion(pred, y)

        else:
            with torch.no_grad():
                pred, gate = model(
                    batch,
                    return_gate_weights=True
                )
                loss = criterion(pred, y)
                gate_values.append(gate.mean().item())

        if torch.isnan(loss):
            skipped_nan += 1
            continue

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()

        bsz = y.size(0)
        total_loss += loss.item() * bsz

        unscaled_pred = (
            pred.detach() * y_std.to(device)
            + y_mean.to(device)
        )

        unscaled_y = (
            y * y_std.to(device)
            + y_mean.to(device)
        )

        total_mae += torch.abs(
            unscaled_pred - unscaled_y
        ).sum().item()

        total_n += bsz

    if total_n == 0:
        return None, None, skipped_nan, None

    avg_gate = (
        sum(gate_values) / len(gate_values)
        if gate_values
        else None
    )

    if not is_train:
        return total_loss / total_n, total_mae / total_n, skipped_nan, avg_gate

    return total_loss / total_n, total_mae / total_n, skipped_nan, None

def train():

    with wandb.init() as run:
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False)

        in_node_dim = train_graphs[0].x.size(1)
        edge_dim = train_graphs[0].edge_attr.size(1)
        in_tabular_dim = train_graphs[0].tabular_features.size(1)  # Tabular dimension from data

        model = HybridFiLMGatedGAT(
            config, 
            in_node_dim=in_node_dim, 
            in_edge_dim=edge_dim, 
            in_tabular_dim=in_tabular_dim
        ).to(device)
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        ckpt_path = f"best_hybrid_gat_{run.id}.pth"

        for epoch in range(config.epochs):
            train_loss, train_mae, train_nan, _ = run_epoch(
                model, train_loader, device, criterion, optimizer)
            val_loss, val_mae, val_nan, avg_gate = run_epoch(
                model, val_loader, device, criterion, optimizer=None)

            if train_loss is None or val_loss is None:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1} (all-NaN epoch)")
                    break
                continue

            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_gate_mean": avg_gate,
                "skipped_nan_train_batches": train_nan,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        run.summary["best_val_loss"] = best_val_loss

        if os.path.exists(ckpt_path):
            wandb.save(ckpt_path)
            os.remove(ckpt_path)

if __name__ == "__main__":
    sweep_id = wandb.sweep(
                sweep=sweep_config,
                entity="88-eateatyumm-imperial-college-london",
                project="Final_Fusion_model_Sweep",
            )
    
    wandb.agent(
        sweep_id, 
        function=train, 
        count=20
    )