import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def diagnose_and_clean():
    master = pd.read_csv(DATA_DIR / "materials_master.csv")
    with open(DATA_DIR / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    all_X, all_r, all_ie, all_group = [], [], [], []

    for structure in structures.values():
        for site in structure:
            elem = site.specie
            if elem.X is not None and not np.isnan(elem.X):
                all_X.append(elem.X)
            if elem.atomic_radius is not None and not np.isnan(elem.atomic_radius):
                all_r.append(elem.atomic_radius)
            if elem.ionization_energy is not None and not np.isnan(elem.ionization_energy):
                all_ie.append(elem.ionization_energy)
            if elem.group is not None and not np.isnan(elem.group):
                all_group.append(elem.group)

    # Calculate dataset medians
    impute_stats = {
        "median_X": float(np.median(all_X)),
        "median_r": float(np.median(all_r)),
        "median_ie": float(np.median(all_ie)),
        "median_group": float(np.median(all_group)),
    }

    # Save imputation stats to disk for downstream graph generation
    with open(DATA_DIR / "impute_stats.pkl", "wb") as f:
        pickle.dump(impute_stats, f)

    valid_ids = {k for k, v in structures.items() if v is not None and len(v) > 0}

    master_clean = master[master["material_id"].isin(valid_ids)].reset_index(drop=True)
    structures_clean = {k: v for k, v in structures.items() if k in valid_ids}

    master_clean.to_csv(DATA_DIR / "materials_master.csv", index=False)
    with open(DATA_DIR / "structures.pkl", "wb") as f:
        pickle.dump(structures_clean, f)

    print(f"Dataset cleaned. Kept {len(structures_clean)} materials.")
    print("Saved median imputation stats to impute_stats.pkl")


if __name__ == "__main__":
    diagnose_and_clean()