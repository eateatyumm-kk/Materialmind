import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import torch
from torch_geometric.data import Data
from mp_api.client import MPRester
from emmet.core.summary import HasProps

# --- 1. Graph Conversion with Node & Edge Features ---

def structure_to_graph_advanced(structure, target, cutoff=5.0):
    """Converts a pymatgen structure into a PyG Data object with node and edge features."""
    
    # Node Features: [Atomic Number (Z), Electronegativity, Atomic Radius]
    node_features = []
    for site in structure:
        elem = site.specie
        z = elem.Z
        x_electroneg = elem.X if elem.X is not None else 0.0
        r_atomic = elem.atomic_radius if elem.atomic_radius is not None else 0.0
        node_features.append([z, x_electroneg, r_atomic])
    
    x = torch.tensor(node_features, dtype=torch.float)

    # Edges & Edge Features (Interatomic Distances)
    all_neighbors = structure.get_all_neighbors(r=cutoff)
    
    edges = []
    edge_features = []
    
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor.index
            dist = neighbor.nn_distance  # Interatomic distance in Angstroms
            
            edges.append([i, j])
            edge_features.append([dist])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    y = torch.tensor([[target]], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# --- 2. Load CSV, Query MP Data, & Update CSV ---

def process_and_link_dataset(csv_path, limit=None):
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("API_KEY")

    if api_key is None:
        raise ValueError("API_KEY not found in .env")

    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[:limit]

    dataset = []
    material_ids = []
    valid_rows = []

    with MPRester(api_key) as mpr:
        for idx, row in df.iterrows():
            formula = row["formula"]
            target = row["bulk_modulus_vrh"]

            docs = mpr.materials.summary.search(
                formula=formula,
                has_props=[HasProps.elasticity],
                fields=["material_id", "structure"]
            )

            if docs:
                mat_id = str(docs[0].material_id)
                structure = docs[0].structure
                
                graph = structure_to_graph_advanced(structure, target)
                
                dataset.append(graph)
                material_ids.append(mat_id)
                valid_rows.append(idx)
            else:
                print(f"Warning: No structure found for formula {formula} (Row {idx})")

    # Update CSV with matching material_id metadata
    df_filtered = df.loc[valid_rows].copy()
    df_filtered["material_id"] = material_ids
    df_filtered.to_csv(csv_path, index=False)
    print(f"Updated '{csv_path}' with 'material_id' metadata.")

    return dataset


# --- 3. Dynamic Paths & Execution ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Exact path based on your VS Code directory tree
csv_file_path = PROJECT_ROOT / "data" / "materials_with_elasticity.csv"
output_pt_path = csv_file_path.parent / "processed_materials_dataset.pt"

# Run dataset processing and save binary file
dataset = process_and_link_dataset(csv_path=csv_file_path)
torch.save(dataset, output_pt_path)

print(f"Successfully processed {len(dataset)} graphs and saved to '{output_pt_path}'.")