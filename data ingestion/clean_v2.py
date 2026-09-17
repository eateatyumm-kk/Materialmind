import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

# Keep noble gas compounds so low-K / soft materials are preserved
DROP_NOBLE_GAS_MATERIALS = False


def diagnose_and_clean():
    master = pd.read_csv(DATA_DIR / "materials_master.csv")
    with open(DATA_DIR / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    all_X, all_r, all_ie, all_group = [], [], [], []
    bad_property_ids = []
    empty_structure_ids = []

    for mat_id, structure in structures.items():
        if structure is None or len(structure) == 0:
            empty_structure_ids.append(mat_id)
            continue

        has_bad_property = False
        elements_here = {site.specie.symbol for site in structure}
        has_noble_gas = bool(elements_here & NOBLE_GASES)

        for site in structure:
            elem = site.specie
            # Collect properties for median calculation (exclude noble gases from global medians)
            if elem.symbol not in NOBLE_GASES:
                if elem.X is not None and not np.isnan(elem.X):
                    all_X.append(elem.X)
                else:
                    has_bad_property = True

                if elem.atomic_radius is not None and not np.isnan(elem.atomic_radius):
                    all_r.append(elem.atomic_radius)
                else:
                    has_bad_property = True

                if elem.ionization_energy is not None and not np.isnan(elem.ionization_energy):
                    all_ie.append(elem.ionization_energy)

                if elem.group is not None and not np.isnan(elem.group):
                    all_group.append(elem.group)

        if has_bad_property or has_noble_gas:
            bad_property_ids.append(mat_id)

    # Median stats used strictly as a backup for non-noble gas missing values
    impute_stats = {
        "median_X": float(np.median(all_X)),
        "median_r": float(np.median(all_r)),
        "median_ie": float(np.median(all_ie)),
        "median_group": float(np.median(all_group)),
    }

    print("Computed baseline imputation stats (for non-noble gas fallback):")
    for k, v in impute_stats.items():
        print(f"  {k}: {v:.4f}")

    with open(DATA_DIR / "impute_stats.pkl", "wb") as f:
        pickle.dump(impute_stats, f)

    print(f"\nTotal materials: {len(structures)}")
    print(f"Empty/missing structures: {len(empty_structure_ids)}")
    print(f"Materials with a noble gas or undefined X/atomic_radius: {len(bad_property_ids)}")

    if DROP_NOBLE_GAS_MATERIALS:
        to_drop = set(bad_property_ids) | set(empty_structure_ids)
        print(f"DROP_NOBLE_GAS_MATERIALS=True -> dropping {len(to_drop)} materials")
    else:
        to_drop = set(empty_structure_ids)  # Always drop genuinely empty structures
        print(
            f"DROP_NOBLE_GAS_MATERIALS=False -> keeping noble-gas materials, "
            f"will use explicit physical fallbacks in build_graphs_and_splits.py ({len(bad_property_ids)} affected)"
        )

    valid_ids = set(structures.keys()) - to_drop

    master_clean = master[master["material_id"].isin(valid_ids)].reset_index(drop=True)
    structures_clean = {k: v for k, v in structures.items() if k in valid_ids}

    master_clean.to_csv(DATA_DIR / "materials_master.csv", index=False)
    with open(DATA_DIR / "structures.pkl", "wb") as f:
        pickle.dump(structures_clean, f)

    print(f"\nKept {len(structures_clean)} / {len(structures)} materials.")
    print("Saved imputation stats to impute_stats.pkl")
    print("\nNext step: Run featurize_tabular.py, then build_graphs_and_splits.py")


if __name__ == "__main__":
    diagnose_and_clean()