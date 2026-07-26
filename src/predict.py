import pandas as pd
from pathlib import Path
import pymatgen.core as pmg
from matminer.featurizers.composition import ElementProperty

SRC_DIR = Path(__file__).resolve().parent
DATA_PATH = SRC_DIR.parent / "data" / "materials_with_elasticity.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    df["composition"] = df["formula"].apply(pmg.Composition)
    
    ep = ElementProperty.from_preset(preset_name="magpie")
    df_features = ep.featurize_dataframe(df, col_id="composition")
    
    X = df_features.drop(columns=["formula", "bulk_modulus_voigt", "bulk_modulus_reuss", "bulk_modulus_vrh", "composition"])
    y = df_features["bulk_modulus_vrh"]
    
    OUTPUT_PATH = SRC_DIR.parent / "data" / "materials_data.csv"
    X.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Done! Created a matrix with {X.shape[0]} rows and {X.shape[1]} numeric features.")

"""
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""

if __name__ == "__main__":
    main()




    

        
        




