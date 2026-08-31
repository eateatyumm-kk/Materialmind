#Initial code to download the master material bulkmodulus data and structural data with material id

import os
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_material_data(num_chunks: int = 5, chunk_size: int = 1000):
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY not found in .env")

    with MPRester(api_key) as mpr:
        # Pull material_id and structure in the SAME query as the target,
        # so there is no second lookup that can pick a different polymorph.
        materials = mpr.materials.summary.search(
            is_stable=True,
            has_props=["elasticity"],
            fields=["material_id", "formula_pretty", "bulk_modulus", "structure"],
            chunk_size=chunk_size,
            num_chunks=num_chunks,
        )
        print(f"Downloaded {len(materials)} materials with elasticity data.")

    rows = []
    structures = {}
    for m in materials:
        if m.bulk_modulus is None:
            continue
        mat_id = str(m.material_id)
        rows.append({
            "material_id": mat_id,
            "formula": m.formula_pretty,
            "bulk_modulus_voigt": m.bulk_modulus["voigt"],
            "bulk_modulus_reuss": m.bulk_modulus["reuss"],
            "bulk_modulus_vrh": m.bulk_modulus["vrh"],
        })
        structures[mat_id] = m.structure

    df = pd.DataFrame(rows).drop_duplicates(subset="material_id")

    master_path = DATA_DIR / "materials_master.csv"
    struct_path = DATA_DIR / "structures.pkl"

    df.to_csv(master_path, index=False)
    with open(struct_path, "wb") as f:
        pickle.dump(structures, f)

    print(f"Saved {len(df)} rows to {master_path}")
    print(f"Saved {len(structures)} structures to {struct_path}")
    return df, structures


if __name__ == "__main__":
    df, structures = download_material_data()
    print(df.head())
    print(f"Unique formulas: {df['formula'].nunique()} / {len(df)} rows "
          f"(if these differ, you have polymorphs — that's fine now, "
          f"since we key by material_id, not formula)")