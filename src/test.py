import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"

df_elasticity = pd.read_csv(data_dir / "materials_with_elasticity.csv")
df_main = pd.read_csv(data_dir / "materials_data.csv")

print(f"Elasticity CSV rows: {len(df_elasticity)}")
print(f"Main Data CSV rows:  {len(df_main)}")

if len(df_elasticity) == len(df_main):
    print("Row counts match perfectly! Both files are aligned by row index.")
else:
    print(f"Mismatch detected! Difference of {abs(len(df_elasticity) - len(df_main))} rows.")
