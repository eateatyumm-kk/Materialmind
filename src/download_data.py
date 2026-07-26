# download 5000 material data from materialsproject.org
from mp_api.client import MPRester
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os

def download_material_data():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("API_KEY")

    if api_key is None:
        raise ValueError("API_KEY not found in .env")

    with MPRester(api_key) as mpr:
        # search for materials with elasticity data (has_props=["elasticity"]) and is stable (is_stable=True)
        materials = mpr.materials.summary.search(
            is_stable=True, 
            has_props=["elasticity"], 
            fields=["formula_pretty", "bulk_modulus"], 
            chunk_size=1000,
            num_chunks=5
        )


        #Checking the format
        print(f"Downloaded {len(materials)} materials with elasticity data.")
        print(materials[0])
        print(materials[0].bulk_modulus)

    data_list = []

    for material in materials:
        if material.bulk_modulus is not None:
            data_list.append({
                "formula": material.formula_pretty,
                "bulk_modulus_voigt": material.bulk_modulus["voigt"],
                "bulk_modulus_reuss": material.bulk_modulus["reuss"],
                "bulk_modulus_vrh": material.bulk_modulus["vrh"]
            })

    df = pd.DataFrame(data_list)
    df.to_csv("materials_with_elasticity.csv", index=False)
    
    return df

if __name__ == "__main__":
    df = download_material_data()
    #checking the first few rows of the dataframe
    print(df.head())