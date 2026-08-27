import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
import wandb

# --- 1. Model Definition ---
class GNN(nn.Module):
    def __init__(self, config, in_node_dim=3, edge_dim=1):
        super().__init__()
        # Project initial node features to hidden_1 dimension
        self.node_em = nn.Linear(in_node_dim, config.hidden_1)

        # PyG's CGConv requires input/output channels to match config.hidden_1
        self.conv1 = CGConv(channels=config.hidden_1, dim=edge_dim)
        self.conv2 = CGConv(channels=config.hidden_1, dim=edge_dim)

        self.fc1 = nn.Linear(config.hidden_1, config.hidden_2)
        self.fc2 = nn.Linear(config.hidden_2, config.hidden_3)
        self.fc3 = nn.Linear(config.hidden_3, 1)

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch

        # Embedding layer
        x = self.node_em(x)
        x = F.relu(x)
        x = self.dropout(x)

        # First convolution layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        # Second convolution layer
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Global Pooling (Graph Level Representation)
        x = global_mean_pool(x, batch)

        # MLP layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Flatten explicitly to 1D
        out = self.fc3(x).view(-1)

        return out


# --- 2. Dataset Path & Split (BEFORE normalization, to avoid leakage) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_materials_dataset.pt"

# weights_only=False is required here because the file stores torch_geometric
# Data objects, not just tensors. Only do this for files you trust.
dataset = torch.load(DATA_PATH, weights_only=False)

total_len = len(dataset)
train_size = int(0.8 * total_len)
val_size = total_len - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], generator=generator
)

# Compute normalization statistics from the TRAINING split only.
# Using the full dataset (train+val) here would leak validation-set
# information into the normalization, biasing val_loss/val_mae optimistically.
train_y = torch.cat([dataset[i].y.view(-1) for i in train_dataset.indices]).float()
y_mean = train_y.mean()
y_std = train_y.std()

# Apply the SAME (train-derived) stats to every example, train and val alike.
for d in dataset:
    d.y = ((d.y.float() - y_mean) / y_std).view(-1)

# --- 3. W&B Sweep Configuration ---
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.003
        },
        "hidden_1": {
            "values": [16, 32, 64]
        },
        "hidden_2": {
            "values": [16, 32, 64]
        },
        "hidden_3": {
            "values": [16, 32, 64]
        },
        "dropout": {
            "values": [0.0, 0.1, 0.2]
        },
        "epochs": {
            "value": 300
        },
        "patience": {
            "value": 30
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "architecture": {
            "value": "CGCNN"
        },
        "dataset": {
            "value": "MaterialsProject_BulkModulus"
        }
    }
}


# --- 4. Training Function ---
def train():
    with wandb.init() as run:
        config = wandb.config

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_node_dim = dataset[0].x.size(1) if hasattr(dataset[0], 'x') and dataset[0].x is not None else 3
        edge_dim = dataset[0].edge_attr.size(1) if hasattr(dataset[0], 'edge_attr') and dataset[0].edge_attr is not None else 1

        model = GNN(config, in_node_dim=in_node_dim, edge_dim=edge_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        patience = config.patience
        best_val_loss = float('inf')
        patience_counter = 0

        # Local device copies of the normalization stats, used to unscale
        # predictions back to real-world units for MAE reporting.
        y_mean_dev = y_mean.to(device)
        y_std_dev = y_std.to(device)

        ckpt_path = f"best_model_{run.id}.pth"

        for epoch in range(config.epochs):
            model.train()
            total_train_loss = 0.0
            total_train_mae = 0.0
            total_train_samples = 0
            skipped_nan_batches = 0

            for batch in train_loader:
                batch = batch.to(device)
                y = batch.y.view(-1).float()

                optimizer.zero_grad()
                train_pred = model(batch)

                train_loss = criterion(train_pred, y)

                # Guard against NaN before stepping; surface it instead of
                # silently swallowing it so divergence is visible in logs.
                if torch.isnan(train_loss):
                    skipped_nan_batches += 1
                    continue

                train_loss.backward()

                # Clip gradients strictly
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_sz = y.size(0)
                total_train_loss += train_loss.item() * batch_sz

                # Unscale MAE to get real-world units for reporting
                unscaled_pred = (train_pred * y_std_dev) + y_mean_dev
                unscaled_y = (y * y_std_dev) + y_mean_dev
                total_train_mae += torch.abs(unscaled_pred - unscaled_y).sum().item()

                total_train_samples += batch_sz

            if total_train_samples == 0:
                # Every batch this epoch was NaN; nothing to log/compare.
                run.log({"epoch": epoch + 1, "skipped_nan_batches": skipped_nan_batches})
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1} (all-NaN epochs)")
                    break
                continue

            epoch_train_loss = total_train_loss / total_train_samples
            epoch_train_mae = total_train_mae / total_train_samples

            # Validate
            model.eval()
            total_val_loss = 0.0
            total_val_mae = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    y = batch.y.view(-1).float()

                    val_pred = model(batch)
                    val_loss = criterion(val_pred, y)

                    batch_sz = y.size(0)
                    total_val_loss += val_loss.item() * batch_sz

                    unscaled_val_pred = (val_pred * y_std_dev) + y_mean_dev
                    unscaled_val_y = (y * y_std_dev) + y_mean_dev
                    total_val_mae += torch.abs(unscaled_val_pred - unscaled_val_y).sum().item()

                    total_val_samples += batch_sz

            epoch_val_loss = total_val_loss / total_val_samples
            epoch_val_mae = total_val_mae / total_val_samples

            # Logging
            run.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_mae": epoch_train_mae,
                "val_loss": epoch_val_loss,
                "val_mae": epoch_val_mae,
                "skipped_nan_batches": skipped_nan_batches
            })

            # Early Stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        run.summary["best_val_loss"] = best_val_loss

        # Upload the checkpoint to W&B and remove the local copy so 30 sweep
        # runs don't silently fill up disk with best_model_<run_id>.pth files.
        if os.path.exists(ckpt_path):
            wandb.save(ckpt_path)
            os.remove(ckpt_path)


# --- 5. Initialize & Run Sweep Agent ---
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="88-eateatyumm-imperial-college-london",
        project="MaterialMind"
    )

    wandb.agent(sweep_id, function=train, count=30)