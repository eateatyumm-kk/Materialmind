import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_material_data(num_chunks: int = 10, chunk_size: int = 1000):
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY not found in .env")

    print(f"Connecting to Materials Project API (Max Chunks: {num_chunks}, Chunk Size: {chunk_size})...")
    with MPRester(api_key) as mpr:
        materials = mpr.materials.summary.search(
            has_props=["elasticity"],
            fields=[
                "material_id",
                "formula_pretty",
                "bulk_modulus",
                "structure",
                "energy_above_hull",
            ],
            chunk_size=chunk_size,
            num_chunks=num_chunks,
        )
        print(f"Downloaded {len(materials)} raw materials with elasticity metadata.")

    rows = []
    structures = {}
    
    # Audit counters for silent filtering/omissions
    n_missing_structure = 0
    n_missing_k = 0
    n_non_physical_k = 0

    for m in materials:
        # Bug Fix: Ensure structure exists before querying length
        if m.structure is None:
            n_missing_structure += 1
            continue

        if not m.bulk_modulus or m.bulk_modulus.get("vrh") is None:
            n_missing_k += 1
            continue

        vrh_k = m.bulk_modulus["vrh"]

        # Judgment Call 2: Count non-physical or negative bulk modulus records
        if vrh_k <= 0:
            n_non_physical_k += 1
            continue

        mat_id = str(m.material_id)

        rows.append({
            "material_id": mat_id,
            "formula": m.formula_pretty,
            "bulk_modulus_voigt": m.bulk_modulus.get("voigt"),
            "bulk_modulus_reuss": m.bulk_modulus.get("reuss"),
            "bulk_modulus_vrh": vrh_k,
            "energy_above_hull": m.energy_above_hull,
        })
        structures[mat_id] = m.structure

    df = pd.DataFrame(rows).drop_duplicates(subset="material_id")

    # Audit log printout
    print("\n--- INGESTION AUDIT REPORT ---")
    print(f"Total entries processed:          {len(materials)}")
    print(f"Skipped (Missing Structure):     {n_missing_structure}")
    print(f"Skipped (Missing K_VRH):         {n_missing_k}")
    print(f"Skipped (Non-physical K_VRH <= 0): {n_non_physical_k}")
    print(f"Final Valid Dataset Size:        {len(df)}")

    # Judgment Call 3: Quantile-based breakdown instead of hardcoded guesses
    print("\n--- DYNAMIC QUANTILE BREAKDOWN (K_VRH in GPa) ---")
    q25, q50, q75 = np.percentile(df["bulk_modulus_vrh"], [25, 50, 75])
    
    bin_q1 = (df["bulk_modulus_vrh"] <= q25).sum()
    bin_q2 = ((df["bulk_modulus_vrh"] > q25) & (df["bulk_modulus_vrh"] <= q50)).sum()
    bin_q3 = ((df["bulk_modulus_vrh"] > q50) & (df["bulk_modulus_vrh"] <= q75)).sum()
    bin_q4 = (df["bulk_modulus_vrh"] > q75).sum()

    print(f" Q1 Softest (K <= {q25:.1f} GPa):          {bin_q1} ({bin_q1/len(df):.1%})")
    print(f" Q2 Mid-Low  ({q25:.1f} < K <= {q50:.1f} GPa): {bin_q2} ({bin_q2/len(df):.1%})")
    print(f" Q3 Mid-High ({q50:.1f} < K <= {q75:.1f} GPa): {bin_q3} ({bin_q3/len(df):.1%})")
    print(f" Q4 Stiffest (K > {q75:.1f} GPa):          {bin_q4} ({bin_q4/len(df):.1%})")

    # File saving
    master_path = DATA_DIR / "materials_master.csv"
    struct_path = DATA_DIR / "structures.pkl"

    df.to_csv(master_path, index=False)
    with open(struct_path, "wb") as f:
        pickle.dump(structures, f)

    print(f"\nSaved master CSV to: {master_path}")
    print(f"Saved structures to:  {struct_path}")
    return df, structures

if __name__ == "__main__":
    download_material_data(num_chunks=10, chunk_size=1000)