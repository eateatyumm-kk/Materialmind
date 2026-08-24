import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from mp_api.client import MPRester
from emmet.core.summary import HasProps  # Import the property filter
import wandb

# --- Data Downloading & Graph Building ---

def download_material_data(limit=100):
    """Downloads materials that are guaranteed to have elastic/bulk modulus data."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("API_KEY")

    if api_key is None:
        raise ValueError("API_KEY not found in .env")

    with MPRester(api_key) as mpr:
        # Filter for materials WITH elasticity calculated
        docs = mpr.materials.summary.search(
            has_props=[HasProps.elasticity],
            fields=["material_id", "structure", "bulk_modulus"]
        )
    return docs[:limit]

def structure_to_graph_fast(structure, target, cutoff=5.0):
    """Fast structure conversion handling Periodic Boundary Conditions (PBCs)."""
    x = torch.tensor([[site.specie.Z] for site in structure], dtype=torch.float)
    all_neighbors = structure.get_all_neighbors(r=cutoff)
    
    edges = []
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor.index
            edges.append([i, j])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    y = torch.tensor([[target]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

# Fetch Data
docs = download_material_data(limit=200)

dataset = []
for doc in docs:
    # Extract target VRH bulk modulus value from dictionary
    target = None
    if doc.bulk_modulus and "vrh" in doc.bulk_modulus:
        target = doc.bulk_modulus["vrh"]

    if target is not None:
        g = structure_to_graph_fast(doc.structure, target)
        dataset.append(g)

print(f"Successfully processed {len(dataset)} graphs.")

# Split & Loaders
split = int(0.8 * len(dataset))
train_data = dataset[:split]
val_data = dataset[split:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# --- WandB & Setup ---

run = wandb.init(
    entity="88-eateatyumm-imperial-college-london",
    project="MaterialMind",
    config={
        "learning_rate": 0.0003,
        "architecture": "GCN",
        "dataset": "MaterialsProject_BulkModulus",
        "epochs": 200,
        "batch_size": 32,
        "cutoff_radius": 5.0,
        "hidden_dim_1": 64, 
        "hidden_dim_2": 32,
        "dropout": 0.2,
        "patience": 30,
    }
)
config = wandb.config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Model Definition ---

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_em = nn.Linear(1, config.hidden_dim_1)
        self.conv1 = GCNConv(config.hidden_dim_1, config.hidden_dim_1)
        self.conv2 = GCNConv(config.hidden_dim_1, config.hidden_dim_1)
        self.fc1 = nn.Linear(config.hidden_dim_1, config.hidden_dim_2)
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc2 = nn.Linear(config.hidden_dim_2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_em(x)
        x = F.relu(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)

        return out

model = Net().to(device)
loss_fn = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

# --- Training Loop ---

patience = config.patience
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(config.epochs):
    # Train
    model.train()
    total_train_loss = 0.0
    total_train_mae = 0.0
    total_train_samples = 0

    for batch in train_loader:
        batch = batch.to(device)  # Send batch to device (MPS/CPU)
        
        optimizer.zero_grad()
        train_pred = model(batch)
        train_loss = loss_fn(train_pred, batch.y)
        train_loss.backward()
        optimizer.step()

        batch_sz = batch.y.size(0)
        total_train_loss += train_loss.item() * batch_sz
        total_train_mae += torch.abs(train_pred - batch.y).sum().item()
        total_train_samples += batch_sz

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
            
            val_pred = model(batch)
            val_loss = loss_fn(val_pred, batch.y)

            batch_sz = batch.y.size(0)
            total_val_loss += val_loss.item() * batch_sz
            total_val_mae += torch.abs(val_pred - batch.y).sum().item()
            total_val_samples += batch_sz

    epoch_val_loss = total_val_loss / total_val_samples
    epoch_val_mae = total_val_mae / total_val_samples

    # Logging
    run.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_mae": epoch_train_mae,
        "val_loss": epoch_val_loss,
        "val_mae": epoch_val_mae
    })

    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break