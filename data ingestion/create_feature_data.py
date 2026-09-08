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

    feature_cols = [c for c in df_feat.columns if c not in (
        "material_id", "formula", "bulk_modulus_voigt", "bulk_modulus_reuss",
        "bulk_modulus_vrh", "composition", "structure"
    )]

    X = df_feat[["material_id"] + feature_cols].copy()
    y = df_feat[["material_id", "bulk_modulus_vrh"]].copy()

    X.to_csv(DATA_DIR / "tabular_features.csv", index=False)
    y.to_csv(DATA_DIR / "targets.csv", index=False)

    print(f"Tabular features generated: {X.shape[0]} rows x {X.shape[1]-1} columns.")


if __name__ == "__main__":
    main()