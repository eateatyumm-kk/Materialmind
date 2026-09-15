from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Project Paths & Config ---
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

class Config:
    learning_rate = 0.0001428170203496466
    batch_size = 64
    hidden_dim = 64
    fc_hidden = 128
    num_heads = 8
    dropout = 0.1
    attn_dropout = 0.1
    weight_decay = 0.000029904296072567457
    epochs = 300
    patience = 30 

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianRBF(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=16):
        super().__init__()
        self.offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2

    def forward(self, dist):
        dist = dist.unsqueeze(1)
        return torch.exp(self.coeff * (dist - self.offset.to(dist.device)) ** 2)

class StandardGATBulkModulus(nn.Module):
    def __init__(self, config, in_node_dim=6, in_edge_dim=2):
        super().__init__()
        self.num_heads = config.num_heads
        hidden = config.hidden_dim
        head_dim = hidden // self.num_heads

        self.rbf = GaussianRBF(start=0.0, stop=8.0, num_gaussians=16)
        rbf_edge_dim = 17 if in_edge_dim == 2 else in_edge_dim

        self.node_embed = nn.Linear(in_node_dim, hidden)

        # Replacing TransformerConv with GATConv
        self.conv1 = GATConv(
            in_channels=hidden,
            out_channels=head_dim,
            heads=self.num_heads,
            concat=True,
            edge_dim=rbf_edge_dim,
            dropout=config.attn_dropout
        )
        self.conv2 = GATConv(
            in_channels=hidden,
            out_channels=head_dim,
            heads=self.num_heads,
            concat=True,
            edge_dim=rbf_edge_dim,
            dropout=config.attn_dropout
        )
        self.conv3 = GATConv(
            in_channels=hidden,
            out_channels=head_dim,
            heads=self.num_heads,
            concat=True,
            edge_dim=rbf_edge_dim,
            dropout=config.attn_dropout
        )

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

        # Residual Blocks using GATConv
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x + F.relu(h))
        x = self.dropout(x)

        h = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x + F.relu(h))
        x = self.dropout(x)

        h = self.conv3(x, edge_index, edge_attr=edge_attr)
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
        self.eval()
        return self._graph_embedding(data)