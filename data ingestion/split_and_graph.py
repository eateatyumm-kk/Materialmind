import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(exist_ok=True)

SEED = 42

# Explicit physical fallbacks for noble gases (Group 18)
NOBLE_GAS_FALLBACKS = {
    "He": {"r": 1.40, "ie": 24.5874, "group": 18},
    "Ne": {"r": 1.54, "ie": 21.5645, "group": 18},
    "Ar": {"r": 1.88, "ie": 15.7596, "group": 18},
    "Kr": {"r": 2.02, "ie": 13.9996, "group": 18},
    "Xe": {"r": 2.16, "ie": 12.1298, "group": 18},
    "Rn": {"r": 2.20, "ie": 10.7485, "group": 18},
}

def structure_to_graph_knn(structure, target: float, material_id: str, impute_stats: dict, k: int = 12) -> Data:
    node_features = []
    coordination_numbers = [len(neighbors) for neighbors in structure.get_all_neighbors(r=3.0)]

    for i, site in enumerate(structure):
        elem = site.specie
        symbol = elem.symbol
        z = elem.Z

        # Noble gas physical override vs dataset median fallback
        if symbol in NOBLE_GAS_FALLBACKS:
            fb = NOBLE_GAS_FALLBACKS[symbol]
            x_electroneg = impute_stats["median_X"]  # neutral value, not an extreme 0.0
            r_atomic = fb["r"]
            ie = fb["ie"]
            group = fb["group"]
        else:
            x_electroneg = elem.X if (elem.X is not None and not np.isnan(elem.X)) else impute_stats["median_X"]
            r_atomic = elem.atomic_radius if (elem.atomic_radius is not None and not np.isnan(elem.atomic_radius)) else impute_stats["median_r"]
            ie = elem.ionization_energy if (elem.ionization_energy is not None and not np.isnan(elem.ionization_energy)) else impute_stats["median_ie"]
            group = elem.group if (elem.group is not None and not np.isnan(elem.group)) else impute_stats["median_group"]

        coord_num = float(coordination_numbers[i])
        node_features.append([z, x_electroneg, r_atomic, ie, group, coord_num])

    x = torch.tensor(node_features, dtype=torch.float)

    all_neighbors = structure.get_all_neighbors(r=8.0)
    edges, edge_features = [], []

    for i, neighbors in enumerate(all_neighbors):
        sorted_neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[:k]
        r_i = node_features[i][2]

        for neighbor in sorted_neighbors:
            j = neighbor.index
            dist = neighbor.nn_distance
            r_j = node_features[j][2]
            
            radius_sum = r_i + r_j
            bond_ratio = dist / radius_sum if radius_sum > 0 else 1.0

            edges.append([i, j])
            edge_features.append([dist, bond_ratio])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    y = torch.tensor([[target]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.material_id = material_id
    return data


def main():
    master = pd.read_csv(DATA_DIR / "materials_master.csv")
    with open(DATA_DIR / "structures.pkl", "rb") as f:
        structures = pickle.load(f)
    with open(DATA_DIR / "impute_stats.pkl", "rb") as f:
        impute_stats = pickle.load(f)

    master = master[master["bulk_modulus_vrh"] > 0].reset_index(drop=True)
    master["log_bulk_modulus_vrh"] = np.log10(master["bulk_modulus_vrh"])

    graphs = []
    kept_rows = []

    for _, row in master.iterrows():
        mat_id = row["material_id"]
        structure = structures.get(mat_id)
        if structure is None:
            continue
        graph = structure_to_graph_knn(structure, row["log_bulk_modulus_vrh"], mat_id, impute_stats)
        graphs.append(graph)
        kept_rows.append(row)

    df_kept = pd.DataFrame(kept_rows)
    torch.save(graphs, DATA_DIR / "graphs.pt")

    df_kept["formula"] = df_kept["formula"].fillna(df_kept["material_id"]).astype(str)

    # Group Split by Chemical Formula (70/15/15)
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_idx, val_test_idx = next(gss_train.split(df_kept, groups=df_kept["formula"]))

    df_train = df_kept.iloc[train_idx]
    df_val_test = df_kept.iloc[val_test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss_val.split(df_val_test, groups=df_val_test["formula"]))

    df_val = df_val_test.iloc[val_idx]
    df_test = df_val_test.iloc[test_idx]

    pd.Series(df_train["material_id"].values, name="material_id").to_csv(SPLIT_DIR / "train_ids.csv", index=False)
    pd.Series(df_val["material_id"].values, name="material_id").to_csv(SPLIT_DIR / "val_ids.csv", index=False)
    pd.Series(df_test["material_id"].values, name="material_id").to_csv(SPLIT_DIR / "test_ids.csv", index=False)

    print(f"Group Split Complete (Unique Formulas):")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    df_kept.to_csv(DATA_DIR / "materials_master.csv", index=False)


if __name__ == "__main__":
    main()