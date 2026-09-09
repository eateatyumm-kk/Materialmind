## download_v2 (download_v2.py)

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

# data cleaning (clean_v2.py)

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

# splits (test.ipynb)

--- 1. DATASET ALIGNMENT & INTEGRITY ---
Master materials count:  10000
Tabular features count: 10000
Split counts -> Train: 7021 | Val: 1491 | Test: 1488
✓ All 10,000 samples accounted for across splits.
✓ Zero material_id overlap between splits.
✓ Formula Leakage Check: 0 overlapping chemical formulas.

--- 2. TARGET DISTRIBUTION (log_bulk_modulus_vrh) ---
Split  Count     Mean      Std       Min   Median      Max
Train   7021 1.885025 0.380239 -1.008774 1.929950 2.691606
  Val   1491 1.879221 0.400685 -0.283162 1.928437 2.584909
 Test   1488 1.889082 0.380321 -0.254145 1.946668 2.613860

--- 3. FEATURE COMPLETENESS ---
Total NaNs in tabular features: 9

# filling 

Noble gas data were filled with: 

NOBLE_GAS_FALLBACKS = {
    "He": {"X": 0.0, "r": 1.40, "ie": 24.5874, "group": 18},
    "Ne": {"X": 0.0, "r": 1.54, "ie": 21.5645, "group": 18},
    "Ar": {"X": 0.0, "r": 1.88, "ie": 15.7596, "group": 18},
    "Kr": {"X": 0.0, "r": 2.02, "ie": 13.9996, "group": 18},
    "Xe": {"X": 0.0, "r": 2.16, "ie": 12.1298, "group": 18},
    "Rn": {"X": 0.0, "r": 2.20, "ie": 10.7485, "group": 18},
}

the missing turbular features were filled with median or 0.0

# Which data to use

in the modeling turbular features.csv, splits/, and graph.pt will be mainly used for training, validating, and testing


