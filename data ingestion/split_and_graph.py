# third data pipeline this code generates the node,edge data for GNN and splits the data into 70/15/15

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(exist_ok=True)

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> test


def structure_to_graph(structure, target: float, material_id: str, cutoff: float = 5.0) -> Data:
    node_features = []
    for site in structure:
        elem = site.specie
        z = elem.Z
        x_electroneg = elem.X if elem.X is not None else 0.0
        r_atomic = elem.atomic_radius if elem.atomic_radius is not None else 0.0
        node_features.append([z, x_electroneg, r_atomic])
    x = torch.tensor(node_features, dtype=torch.float)

    all_neighbors = structure.get_all_neighbors(r=cutoff)
    edges, edge_features = [], []
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            edges.append([i, neighbor.index])
            edge_features.append([neighbor.nn_distance])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    y = torch.tensor([[target]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.material_id = material_id  # keep the ID attached to the graph itself
    return data


def main():
    master = pd.read_csv(DATA_DIR / "materials_master.csv")
    with open(DATA_DIR / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    # log10 transform, matching the Matbench matbench_log_kvrh convention.
    # Do this ONCE here, upstream of both models, so tabular and GNN targets
    # are guaranteed identical.
    master = master[master["bulk_modulus_vrh"] > 0].reset_index(drop=True)
    master["log_bulk_modulus_vrh"] = np.log10(master["bulk_modulus_vrh"])

    graphs = []
    kept_ids = []
    for _, row in master.iterrows():
        mat_id = row["material_id"]
        structure = structures.get(mat_id)
        if structure is None:
            continue
        graph = structure_to_graph(structure, row["log_bulk_modulus_vrh"], mat_id)
        graphs.append(graph)
        kept_ids.append(mat_id)

    torch.save(graphs, DATA_DIR / "graphs.pt")
    print(f"Saved {len(graphs)} graphs to {DATA_DIR / 'graphs.pt'}")

    # --- Shared split, by material_id ---
    rng = np.random.default_rng(SEED)
    ids = np.array(kept_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    pd.Series(train_ids, name="material_id").to_csv(SPLIT_DIR / "train_ids.csv", index=False)
    pd.Series(val_ids, name="material_id").to_csv(SPLIT_DIR / "val_ids.csv", index=False)
    pd.Series(test_ids, name="material_id").to_csv(SPLIT_DIR / "test_ids.csv", index=False)

    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print("Both the tabular and GNN pipelines must filter to these IDs — "
          "do not re-split independently in either model script.")

    # Also save the log-transformed master so nothing downstream needs to
    # redo the transform differently.
    master.to_csv(DATA_DIR / "materials_master.csv", index=False)


if __name__ == "__main__":
    main()