import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition, Structure
from torch_geometric.data import Data

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Same fallbacks used in `data ingestion/split_and_graph.py` — pymatgen's
# Element has no reliable electronegativity/radius/ionization data for
# noble gases, so these are hardcoded physical values.
NOBLE_GAS_FALLBACKS = {
    "He": {"r": 1.40, "ie": 24.5874, "group": 18},
    "Ne": {"r": 1.54, "ie": 21.5645, "group": 18},
    "Ar": {"r": 1.88, "ie": 15.7596, "group": 18},
    "Kr": {"r": 2.02, "ie": 13.9996, "group": 18},
    "Xe": {"r": 2.16, "ie": 12.1298, "group": 18},
    "Rn": {"r": 2.20, "ie": 10.7485, "group": 18},
}


class MaterialsFeaturizer:
    """Turns a chemical formula + CIF structure into model-ready inputs
    for the Fusion model.

    The graph construction in `cif_to_graph` mirrors
    `structure_to_graph_knn` in `data ingestion/split_and_graph.py` so
    that inference-time graphs match what the model was trained on:
    same 6D node features, same KNN edge features, same imputation
    fallbacks for missing elemental data.
    """

    def __init__(self, max_radius: float = 8.0, k: int = 12):
        self.max_radius = max_radius
        self.k = k
        self.magpie = ElementProperty.from_preset("magpie")

        with open(WEIGHTS_DIR / "impute_stats.pkl", "rb") as f:
            self.impute_stats = pickle.load(f)

        with open(WEIGHTS_DIR / "fusion_tabular_scaler.pkl", "rb") as f:
            tabular_scaler = pickle.load(f)
        self.tabular_scaler = tabular_scaler["scaler"]
        self.tabular_feature_cols = tabular_scaler["feature_cols"]

    def formula_to_tabular(self, formula_str: str) -> torch.Tensor:
        """Parses a chemical formula into a scaled 132D Magpie feature vector.

        The model was trained on Magpie features scaled with a
        StandardScaler fit on the training split (see
        `Testing/Fusion/Fusion_model.py::load_split_graphs`), so raw
        features must be scaled the same way here before being fed to
        the model.
        """
        try:
            comp = Composition(formula_str)
            magpie_feats = self.magpie.featurize(comp)
        except Exception as e:
            raise ValueError(f"Failed to featurize formula '{formula_str}': {str(e)}")

        if len(magpie_feats) != len(self.tabular_feature_cols):
            raise ValueError(
                f"Expected {len(self.tabular_feature_cols)} Magpie features, "
                f"got {len(magpie_feats)}"
            )

        magpie_df = pd.DataFrame([magpie_feats], columns=self.tabular_feature_cols)
        scaled_feats = self.tabular_scaler.transform(magpie_df)
        tensor_feats = torch.tensor(scaled_feats, dtype=torch.float32)  # [1, 132]

        if torch.isnan(tensor_feats).any() or torch.isinf(tensor_feats).any():
            raise ValueError("Tabular features contain NaN or Inf values.")

        return tensor_feats

    def cif_to_graph(self, cif_content: str) -> Data:
        """Builds a KNN crystal graph from CIF text."""
        try:
            structure = Structure.from_str(cif_content, fmt="cif")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted CIF file format: {str(e)}")

        if len(structure) == 0:
            raise ValueError("CIF structure contains no atomic sites.")

        # 1. Node features (6D): [Z, electronegativity, atomic radius,
        #    ionization energy, group, coordination number]
        node_features = []
        coordination_numbers = [len(neighbors) for neighbors in structure.get_all_neighbors(r=3.0)]

        for i, site in enumerate(structure):
            # Safe element parsing for disordered/partial-occupancy CIFs
            elem = site.specie if hasattr(site, "specie") and site.specie else site.species.elements[0]
            symbol = elem.symbol
            z = float(elem.Z)

            if symbol in NOBLE_GAS_FALLBACKS:
                fb = NOBLE_GAS_FALLBACKS[symbol]
                x_electroneg = float(self.impute_stats["median_X"])
                r_atomic = float(fb["r"])
                ie = float(fb["ie"])
                group = float(fb["group"])
            else:
                x_electroneg = float(elem.X) if (elem.X is not None and not np.isnan(elem.X)) else float(self.impute_stats["median_X"])
                r_atomic = float(elem.atomic_radius) if (elem.atomic_radius is not None and not np.isnan(elem.atomic_radius)) else float(self.impute_stats["median_r"])
                ie = float(elem.ionization_energy) if (elem.ionization_energy is not None and not np.isnan(elem.ionization_energy)) else float(self.impute_stats["median_ie"])
                group = float(elem.group) if (elem.group is not None and not np.isnan(elem.group)) else float(self.impute_stats["median_group"])

            coord_num = float(coordination_numbers[i])
            node_features.append([z, x_electroneg, r_atomic, ie, group, coord_num])

        x = torch.tensor(node_features, dtype=torch.float32)

        # 2. Edges: k-nearest-neighbor bond graph within max_radius,
        #    edge features [distance, bond_ratio]
        all_neighbors = structure.get_all_neighbors(r=self.max_radius)
        edges, edge_features = [], []

        for i, neighbors in enumerate(all_neighbors):
            sorted_neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[:self.k]
            r_i = node_features[i][2]

            for neighbor in sorted_neighbors:
                j = int(neighbor.index)
                dist = float(neighbor.nn_distance)
                r_j = node_features[j][2]

                radius_sum = r_i + r_j
                bond_ratio = dist / radius_sum if radius_sum > 0 else 1.0

                edges.append([i, j])
                edge_features.append([dist, bond_ratio])

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        if torch.isnan(x).any() or torch.isnan(edge_attr).any():
            raise ValueError("Graph featurization generated NaN values.")

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def process_input(self, formula_str: str, cif_content: str) -> Data:
        """Combines tabular and graph processing into a single model-ready graph."""
        graph_data = self.cif_to_graph(cif_content)
        graph_data.tabular_features = self.formula_to_tabular(formula_str)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)

        return graph_data
