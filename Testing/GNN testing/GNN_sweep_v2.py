"""
GAT-style GNN for bulk modulus prediction, using PyG's TransformerConv
(attention over neighbors, edge-feature-aware — a proper drop-in upgrade
from the CGConv/CGCNN baseline that still uses interatomic distance).

Key structural differences from the CGCNN baseline script:
  1. Loads graphs.pt + the SHARED splits (train/val/test by material_id) —
     no more independent random_split. This is required for a valid
     comparison against the tabular model and the CGCNN baseline.
  2. Adds a genuine held-out TEST set, untouched by the sweep.
  3. Multi-head attention (TransformerConv) instead of CGConv.
  4. Residual connections between conv layers — attention-based GNNs
     benefit more from residuals than plain GCNs, since attention can
     otherwise over-smooth node features across layers.
  5. extract_embedding() method — returns the pooled graph representation
     before the final regression head, for the later hybrid model.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# 1. Model
# ============================================================

class GaussianRBF(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=16):
        super().__init__()
        self.offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2

    def forward(self, dist):
        # Expands scalar distances into continuous Gaussian functions
        return torch.exp(self.coeff * (dist - self.offset.to(dist.device)) ** 2)

class GATBulkModulus(nn.Module):
    def __init__(self, config, in_node_dim=6, in_edge_dim=2):
        super().__init__()
        self.num_heads = config.num_heads
        hidden = config.hidden_dim
        
        # 1. Expand raw distance into 16 Gaussian channels
        self.rbf = GaussianRBF(start=0.0, stop=8.0, num_gaussians=16)
        rbf_edge_dim = 16 if in_edge_dim == 1 else in_edge_dim

        self.node_embed = nn.Linear(in_node_dim, hidden)

        # 2. Convolutions using explicit head sizing
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

        # Expand distances if scalar edge attribute
        if edge_attr.dim() == 1 or edge_attr.size(1) == 1:
            edge_attr = self.rbf(edge_attr.squeeze())

        x = F.relu(self.node_embed(x))
        x = self.dropout(x)

        # Residual Blocks
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
        out = self.fc_out(h).view(-1)
        return out

    @torch.no_grad()
    def extract_embedding(self, data):
        """Returns the pooled graph representation (post-attention,
        pre-regression-head) as a fixed-length vector per structure.
        This is the feature the hybrid model concatenates with tabular
        features before feeding XGBoost."""
        self.eval()
        return self._graph_embedding(data)


# ============================================================
# 2. Data loading — SHARED splits, with a real test set
# ============================================================
def load_split_graphs():
    graphs = torch.load(DATA_DIR / "graphs.pt", weights_only=False)

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    val_ids = set(pd.read_csv(DATA_DIR / "splits" / "val_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    train_graphs = [g for g in graphs if g.material_id in train_ids]
    val_graphs = [g for g in graphs if g.material_id in val_ids]
    test_graphs = [g for g in graphs if g.material_id in test_ids]

    print(f"Loaded graphs — train: {len(train_graphs)}, val: {len(val_graphs)}, "
          f"test: {len(test_graphs)} (test held out from sweep entirely)")

    return train_graphs, val_graphs, test_graphs


train_graphs, val_graphs, test_graphs = load_split_graphs()

# Normalization stats from TRAIN ONLY (y is already log10(K) from
# build_graphs_and_splits.py — this z-scores the log target for training
# stability, same approach as the CGCNN baseline).
train_y = torch.cat([g.y.view(-1) for g in train_graphs]).float()
y_mean = train_y.mean()
y_std = train_y.std()

for g in train_graphs + val_graphs + test_graphs:
    g.y_scaled = ((g.y.float() - y_mean) / y_std).view(-1)


# ============================================================
# 3. Sweep config
# ============================================================
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values", "min": 0.0001, "max": 0.003,
        },
        "hidden_dim": {"values": [32, 64, 128]},        # must be divisible by num_heads
        "num_heads": {"values": [2, 4, 8]},
        "fc_hidden": {"values": [32, 64, 128]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        "attn_dropout": {"values": [0.0, 0.1, 0.2]},
        "epochs": {"value": 300},
        "patience": {"value": 30},
        "batch_size": {"values": [16, 32, 64]},
        "weight_decay": {
            "distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3,
        },
        "architecture": {"value": "GAT-TransformerConv"},
        "dataset": {"value": "MaterialsProject_BulkModulus"},
    },
}


# ============================================================
# 4. Training function (sweeps against VAL only — test untouched)
# ============================================================
def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_mae, total_n = 0.0, 0.0, 0
    skipped_nan = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y_scaled.view(-1).float()

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(batch)
            loss = criterion(pred, y)

        if torch.isnan(loss):
            skipped_nan += 1
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
        return None, None, skipped_nan
    return total_loss / total_n, total_mae / total_n, skipped_nan


def train():
    with wandb.init() as run:
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False)

        in_node_dim = train_graphs[0].x.size(1)
        edge_dim = train_graphs[0].edge_attr.size(1)

        model = GATBulkModulus(config, in_node_dim=in_node_dim, in_edge_dim=edge_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        ckpt_path = f"best_gat_{run.id}.pth"

        for epoch in range(config.epochs):
            train_loss, train_mae, train_nan = run_epoch(
                model, train_loader, device, criterion, optimizer)
            val_loss, val_mae, val_nan = run_epoch(
                model, val_loader, device, criterion, optimizer=None)

            if train_loss is None or val_loss is None:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1} (all-NaN epoch)")
                    break
                continue

            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_mae": train_mae,
                "val_loss": val_loss, "val_mae": val_mae,
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
        project="MaterialMind_GNN_v2",
    )
    wandb.agent(sweep_id, function=train, count=25)