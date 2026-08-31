
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}


def diagnose_and_clean():
    master = pd.read_csv(DATA_DIR / "materials_master.csv")
    with open(DATA_DIR / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    bad_electroneg_ids = []
    bad_radius_ids = []
    noble_gas_ids = []

    for mat_id, structure in structures.items():
        elements_here = {site.specie.symbol for site in structure}

        has_noble_gas = bool(elements_here & NOBLE_GASES)
        if has_noble_gas:
            noble_gas_ids.append(mat_id)

        for site in structure:
            elem = site.specie
            x = elem.X
            r = elem.atomic_radius
            if x is None or (isinstance(x, float) and np.isnan(x)):
                bad_electroneg_ids.append(mat_id)
                break
        for site in structure:
            elem = site.specie
            r = elem.atomic_radius
            if r is None or (isinstance(r, float) and np.isnan(r)):
                bad_radius_ids.append(mat_id)
                break

    bad_electroneg_ids = set(bad_electroneg_ids)
    bad_radius_ids = set(bad_radius_ids)
    noble_gas_ids = set(noble_gas_ids)

    to_drop = bad_electroneg_ids | bad_radius_ids

    print(f"Total materials: {len(structures)}")
    print(f"Contain a noble gas element: {len(noble_gas_ids)}")
    print(f"Have undefined electronegativity on some atom: {len(bad_electroneg_ids)}")
    print(f"Have undefined atomic_radius on some atom: {len(bad_radius_ids)}")
    print(f"Total materials to drop (union of both issues): {len(to_drop)}")

    if to_drop:
        print("\nExample dropped material_ids:", list(to_drop)[:10])

    # --- Clean and overwrite ---
    kept_ids = set(structures.keys()) - to_drop

    master_clean = master[master["material_id"].isin(kept_ids)].reset_index(drop=True)
    structures_clean = {k: v for k, v in structures.items() if k in kept_ids}

    master_clean.to_csv(DATA_DIR / "materials_master.csv", index=False)
    with open(DATA_DIR / "structures.pkl", "wb") as f:
        pickle.dump(structures_clean, f)

    print(f"\nKept {len(kept_ids)} / {len(structures)} materials.")
    print("Overwrote materials_master.csv and structures.pkl with cleaned versions.")
    print("Now re-run: featurize_tabular_fixed.py and build_graphs_and_splits.py")


if __name__ == "__main__":
    diagnose_and_clean()