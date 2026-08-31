# Data Ingestion Pipeline

Three scripts turn raw Materials Project data into everything the tabular
(XGBoost) model and the GNN model need to train on **the same materials,
correctly matched, with the same train/val/test split.**

Run them in order. Each step reads the previous step's output.

```
1_download_material_data_fixed.py 
        │
        ▼
2_featurize_tabular_fixed.py          3_build_graphs_and_splits.py
        │                                       │
        ▼                                       ▼
  tabular features                    graphs + shared splits
```

---

## Step 1 — `download_material_data_fixed.py`

**Pulls raw data from the Materials Project API, once, per material.**

For each stable material with elasticity data, fetches in a single query:
- `material_id` (the unique, stable identifier — this is the key everything
  else joins on)
- `formula_pretty`
- `bulk_modulus` (voigt / reuss / vrh)
- `structure` — a raw `pymatgen.Structure`: a unit cell (lattice) + a list
  of atoms with element type and 3D position inside it. **Not a graph yet**
  — no concept of "neighbors" or "bonds" exists at this stage, just atom
  positions in space.

**Why fetch structure and bulk modulus together, in one call?** The old
version fetched them in two separate steps and joined by chemical
`formula`. Formulas aren't unique — multiple polymorphs (different crystal
structures, different `material_id`s) can share a formula — so that join
could silently attach the wrong structure to a bulk modulus value. Fetching
both fields in the same API call per `material_id` makes that mismatch
impossible.

**Output:**
- `data/materials_master.csv` — one row per material: `material_id`,
  `formula`, `bulk_modulus_vrh`, etc.
- `data/structures.pkl` — dict of `{material_id: pymatgen.Structure}`

---
## step 1.5 - 'cleaning data'

Diagnose and clean noble-gas / NaN-feature contamination before graph building.

Run this BEFORE build_graphs_and_splits.py (or re-run build_graphs_and_splits.py
after this, since it will regenerate graphs.pt and the splits from the cleaned
materials_master.csv / structures.pkl).

What it does:
  1. Reports how many materials contain elements with undefined Pauling
     electronegativity (mainly noble gases: He, Ne, Ar, Kr, Xe, Rn).
  2. Reports how many materials contain elements with missing atomic_radius.
  3. Filters those materials out of materials_master.csv and structures.pkl,
     so build_graphs_and_splits.py never sees a NaN node feature.

This is a filter, not a patch-with-0.0 — solid noble gases are cryogenic
edge cases that don't belong in a "stable at ambient conditions" bulk
modulus dataset anyway, so removing them is the more defensible choice
over silently imputing a physically meaningless placeholder value.

## Step 2 — `featurize_tabular_fixed.py`

**Converts each material into a fixed-length row of numbers for XGBoost.**

XGBoost can't take a `Structure` object as input — it needs a table.
This script is the "structure → numbers" translator for the tabular side.

1. Loads `materials_master.csv` + `structures.pkl`, attaches the correct
   structure to each row **by `material_id`** (not formula — this is the
   actual fix from the old pipeline).
2. **Magpie composition features** (~132 columns): statistics (mean, min,
   max, range, etc.) over the elements present — atomic weight,
   electronegativity, melting point, etc. These only depend on *which
   elements are present*, not how they're arranged in 3D. Two different
   polymorphs of the same formula get identical Magpie features.
3. **Density features** (a few extra columns): density, volume per atom,
   packing fraction — computed from the actual 3D structure. Coarse
   summary statistics of the geometry, not the detailed bonding topology.

**Output:**
- `data/tabular_features.csv` — feature matrix, keyed by `material_id`
- `data/targets.csv` — raw `bulk_modulus_vrh`, keyed by `material_id`
  (superseded by the log-transformed target in `materials_master.csv`
  after Step 3 runs — use that one for training)

At training time, select columns to get two reportable baselines from one
featurization run:
- **Composition-only**: Magpie columns
- **Composition + coarse geometry**: Magpie + density columns

---

## Step 3 — `build_graphs_and_splits.py`

**Builds the GNN's graph representation, and generates the one shared
data split both models must use.**

### Graph construction
For each structure, converts the raw atom positions into a PyTorch
Geometric `Data` object:
- **Nodes** — one per atom: `[atomic number, electronegativity, atomic
  radius]`
- **Edges** — `structure.get_all_neighbors(r=5.0)` searches a 5 Å radius
  sphere around every atom (accounting for the crystal's periodic
  boundary) and connects every atom pair found inside it
- **Edge features** — the interatomic distance (Å) for each connection

This is the actual "structure → graph" step; the `Structure` object from
Step 1 has no nodes/edges until this runs.

### Target transform
Applies `log10(bulk_modulus_vrh)` **once, here, upstream of both models**
— matches the standard convention (e.g. Matbench's `matbench_log_kvrh`)
and keeps the tabular and GNN targets identically transformed by
construction, not by two separate scripts trying to agree.

### Shared split
Generates **one** train/val/test split (70/15/15) by `material_id`, saved
to `data/splits/`. This is what makes tabular-vs-GNN-vs-hybrid comparisons
valid — neither model script should call its own
`train_test_split`/`random_split`; both must filter to these IDs.

**Output:**
- `data/graphs.pt` — list of PyG `Data` graphs
- `data/splits/train_ids.csv`, `val_ids.csv`, `test_ids.csv`
- `data/materials_master.csv` — updated in place with
  `log_bulk_modulus_vrh`

---

## Key invariant to preserve

Every downstream script (tabular training, GNN training, hybrid model)
must:
1. Join / filter by `material_id` — never by `formula` or row position
2. Load train/val/test membership from `data/splits/`, not generate its
   own split
3. Use `log_bulk_modulus_vrh` from `materials_master.csv` as the target

Breaking any of these three reintroduces the original bug or makes the
model comparison invalid.