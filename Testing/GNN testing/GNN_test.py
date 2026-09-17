
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
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

        return torch.exp(
            self.coeff * (dist - self.offset.to(dist.device)) ** 2
        )

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
        out = self.fc_out(h).view(-1) #to match the shape with y [[1], [2], [3]] -> [1, 2, 3]
        return out

    @torch.no_grad()
    def extract_embedding(self, data):
        self.eval()
        return self._graph_embedding(data)

def load_split_graphs():
    graphs = torch.load(DATA_DIR / "graphs.pt", weights_only=False)

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "train_ids.csv")["material_id"])
    val_ids = set(pd.read_csv(DATA_DIR / "splits" / "val_ids.csv")["material_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "splits" / "test_ids.csv")["material_id"])

    train_graphs = [g for g in graphs if g.material_id in train_ids]
    val_graphs = [g for g in graphs if g.material_id in val_ids]
    test_graphs = [g for g in graphs if g.material_id in test_ids]

    print(f"train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
    return graphs, train_graphs, val_graphs, test_graphs


all_graphs, train_graphs, val_graphs, test_graphs = load_split_graphs() 
#train, val, and test is connected to same all graph object so updating all graph updates the connected object too.

train_y = torch.cat([g.y.view(-1) for g in train_graphs]).float()
y_mean = train_y.mean()
y_std = train_y.std()

for g in all_graphs:
    g.y_scaled = ((g.y.float() - y_mean) / y_std).view(-1) #normalise y and store in g.y_scaled for training and evaluation

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

def train_final_model():
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False, num_workers=0)

    in_node_dim = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1)

    model = GATBulkModulus(config, in_node_dim=in_node_dim, in_edge_dim=edge_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = RESULTS_DIR / "gat_best_model.pth"

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
            print(f"Epoch {epoch+1:3d} | train_loss {train_loss:.4f} train_mae {train_mae:.4f} "
                  f"| val_loss {val_loss:.4f} val_mae {val_mae:.4f}")

        # Save ONLY on improvement — this is what fixes the
        # best-vs-final-epoch drift the sweep report flagged.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}")
            break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}. "
          f"Best checkpoint saved to {ckpt_path}")

    # Reload the BEST checkpoint, not whatever the model holds after the
    # last epoch (which may already be past its best point, per the sweep
    # report's finding that val drifted upward after the checkpointed best
    # in several runs).
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model

def evaluate_on_test(model):
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)

    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_scaled = model(batch)
            pred_unscaled = (pred_scaled * y_std.to(device) + y_mean.to(device)).cpu().numpy()
            true_unscaled = batch.y.view(-1).cpu().numpy()  # y is already log10(K), unscaled

            all_preds.extend(pred_unscaled)
            all_true.extend(true_unscaled)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    test_rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    test_mae = mean_absolute_error(all_true, all_preds)
    test_r2 = r2_score(all_true, all_preds)

    print("\n=== FINAL TEST SET PERFORMANCE (log10 GPa) ===")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R²:   {test_r2:.4f}")

    results_df = pd.DataFrame({
        "material_id": [g.material_id for g in test_graphs],
        "actual_log_K": all_true,
        "predicted_log_K": all_preds,
        "residual": all_true - all_preds,
    })
    results_df.to_csv(RESULTS_DIR / "gat_test_predictions.csv", index=False)

    metrics_df = pd.DataFrame([{
        "model": "GAT-TransformerConv", "feature_set": "graph",
        "split": "test", "rmse": test_rmse, "mae": test_mae, "r2": test_r2,
    }])
    metrics_df.to_csv(RESULTS_DIR / "gat_test_metrics.csv", index=False)

    return results_df

def extract_all_embeddings(model):
    loader = DataLoader(all_graphs, batch_size=64, shuffle=False)

    model.eval()
    all_ids, all_embeddings = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.extract_embedding(batch).cpu().numpy()
            all_embeddings.append(emb)
            all_ids.extend(batch.material_id if isinstance(batch.material_id, list)
                            else [batch.material_id])

    all_ids = [g.material_id for g in all_graphs]
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    emb_cols = [f"gnn_embed_{i}" for i in range(all_embeddings.shape[1])]
    emb_df = pd.DataFrame(all_embeddings, columns=emb_cols)
    emb_df.insert(0, "material_id", all_ids)

    emb_path = DATA_DIR / "gnn_embeddings.csv"
    emb_df.to_csv(emb_path, index=False)
    print(f"\nSaved {emb_df.shape[0]} embeddings ({emb_df.shape[1]-1} dims each) to {emb_path}")

    return emb_df

if __name__ == "__main__":
    model = train_final_model()
    evaluate_on_test(model)
    extract_all_embeddings(model)