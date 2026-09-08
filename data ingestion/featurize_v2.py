"""
Fixed version of your featurize script. One change: the target-exclusion
list now matches by PATTERN ("bulk_modulus" in the column name), not by
exact hardcoded names. This is the fix for the recurring leakage bug —
your original list didn't know about `log_bulk_modulus_vrh` (added later
by the graph-building script to materials_master.csv), so if this script
ran after that one, the log target would leak straight into
tabular_features.csv as if it were a regular feature.

This version is immune to that regardless of what script order you run
things in, or what target-like columns get added to materials_master.csv
in the future.
"""
import pickle
from pathlib import Path

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import DensityFeatures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    master = pd.read_csv(DATA_DIR / "materials_master.csv")
    with open(DATA_DIR / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    master["structure"] = master["material_id"].map(structures)
    master = master[master["structure"].notna()].reset_index(drop=True)

    master["composition"] = master["structure"].apply(lambda s: s.composition)

    ep = ElementProperty.from_preset(preset_name="magpie")
    df_feat = ep.featurize_dataframe(master, col_id="composition", ignore_errors=True)

    df_struct = DensityFeatures()
    df_feat = df_struct.featurize_dataframe(df_feat, col_id="structure", ignore_errors=True)

    # Pattern-based exclusion — catches bulk_modulus_vrh, bulk_modulus_voigt,
    # bulk_modulus_reuss, log_bulk_modulus_vrh, or any future variant,
    # regardless of when this script runs relative to others.
    target_like = [c for c in df_feat.columns if "bulk_modulus" in c.lower()]
    non_feature_cols = ["material_id", "formula", "composition", "structure"] + target_like

    feature_cols = [c for c in df_feat.columns if c not in non_feature_cols]

    print(f"Excluded as non-feature/target columns: {non_feature_cols}")

    X = df_feat[["material_id"] + feature_cols].copy()
    y = df_feat[["material_id", "bulk_modulus_vrh"]].copy()

    X.to_csv(DATA_DIR / "tabular_features.csv", index=False)
    y.to_csv(DATA_DIR / "targets.csv", index=False)

    print(f"Tabular features generated: {X.shape[0]} rows x {X.shape[1]-1} columns.")


if __name__ == "__main__":
    main()