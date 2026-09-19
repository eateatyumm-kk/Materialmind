
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


class Config:
    learning_rate = 0.0005146899976585474
    batch_size = 64
    hidden_dim = 128
    fc_hidden = 128
    num_heads = 4
    dropout = 0.1
    weight_decay = 0.00000102818997199268
    epochs = 300
    patience = 30

config = Config()

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
        gamma = self.fc_gamma(h_tab)
        beta = self.fc_beta(h_tab)
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
        cat_features = torch.cat([h_graph, h_tab], dim=-1)
        gate = torch.sigmoid(self.gate_fc(cat_features))
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

        self.rbf = GaussianRBF(start=0.0, stop=8.0, num_gaussians=16)
        rbf_edge_dim = 17 if in_edge_dim == 2 else in_edge_dim

        self.tabular_mlp = nn.Sequential(
            nn.Linear(in_tabular_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
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

        h_tab = self.tabular_mlp(x_tabular)

        distance = edge_attr[:, 0]
        other_feature = edge_attr[:, 1]
        rbf_distance = self.rbf(distance)
        edge_attr = torch.cat([rbf_distance, other_feature.unsqueeze(1)], dim=1)

        x = F.relu(self.node_embed(x))
        h_tab_per_node = h_tab[batch]
        x = self.film(x, h_tab_per_node)
        x = self.dropout(x)

        h = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x + F.relu(h))
        x = self.dropout(x)

        h = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x + F.relu(h))
        x = self.dropout(x)

        h = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x + F.relu(h))

        h_graph_pooled = global_mean_pool(x, batch)
        h_fused, gate_weights = self.gated_fusion(h_graph_pooled, h_tab)

        h = F.relu(self.fc1(h_fused))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        out = self.fc_out(h).view(-1)

        if return_gate_weights:
            return out, gate_weights
        return out
