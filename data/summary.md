## download_v2:

Downloaded 10000 raw materials with elasticity metadata.

--- INGESTION AUDIT REPORT ---
Total entries processed:          10000
Skipped (Missing Structure):     0
Skipped (Missing K_VRH):         0
Skipped (Non-physical K_VRH <= 0): 0
Final Valid Dataset Size:        10000

--- DYNAMIC QUANTILE BREAKDOWN (K_VRH in GPa) ---
 Q1 Softest (K <= 46.6 GPa):          2500 (25.0%)
 Q2 Mid-Low  (46.6 < K <= 85.5 GPa): 2500 (25.0%)
 Q3 Mid-High (85.5 < K <= 147.3 GPa): 2500 (25.0%)
 Q4 Stiffest (K > 147.3 GPa):          2500 (25.0%)

Saved master CSV to: C:\projects\Materialmind\data\materials_master.csv
Saved structures to:  C:\projects\Materialmind\data\structures.pkl

=== MATERIAL STABILITY BREAKDOWN ===
Total Materials: 10000
Stable (E_above_hull == 0):     5588 (55.9%)
Metastable (E_above_hull > 0): 4409 (44.1%)
Missing Hull Energy Data:      3 (0.0%)

# data cleaning

Computed baseline imputation stats (for non-noble gas fallback):
  median_X: 1.9600
  median_r: 1.3500
  median_ie: 8.1517
  median_group: 13.0000

Total materials: 10000
Empty/missing structures: 0
Materials with a noble gas or undefined X/atomic_radius: 3
DROP_NOBLE_GAS_MATERIALS=False -> keeping noble-gas materials, will use explicit physical fallbacks in build_graphs_and_splits.py (3 affected)

Kept 10000 / 10000 materials.
Saved imputation stats to impute_stats.pkl

Next step: Run featurize_tabular.py, then build_graphs_and_splits.py

# feature


