# MaterialMind

[![GitHub Repository](https://img.shields.io/badge/GitHub-eateatyumm--kk%2FMaterialmind-blue?logo=github)](https://github.com/eateatyumm-kk/Materialmind)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-PyG-EE4C2C?logo=pytorch)](https://pytorch-geometric.readthedocs.io/)

**MaterialMind** is a materials informatics project investigating whether explicit crystal-structure representations learned via **Graph Neural Networks (GNNs)** provide complementary predictive power beyond traditional **composition-based tabular descriptors** when predicting a material's **bulk modulus (K)**. I chose to do this project to learn more about GNNs and their application to materials science.

---

## Overview

MaterialMind is a materials informatics framework investigating whether
explicit 3D crystal-structure representations learned by Graph Neural
Networks (GNNs) provide complementary predictive information beyond
standard composition-based descriptors (Magpie) for predicting material
bulk modulus.

The long-term goal is to develop a high-throughput machine-learning
surrogate model for DFT-calculated bulk modulus, reducing reliance on
computationally expensive calculations.

---

## Research Question

> Can learned 3D crystal-structure representations improve bulk modulus
> prediction beyond chemical composition alone, and can combining
> structural and compositional representations provide additional
> predictive performance?

---

## Dataset

The dataset contains approximately 10,000 materials from the Materials
Project, including both stable and unstable materials with available
elasticity data.

### Target

- Property: Bulk Modulus (K)
- Target representation: `log10(K)`

### Composition Features

132 Magpie elemental composition descriptors.

### Crystal Structure Graph

Each crystal structure is represented as a graph:

- **Nodes (atoms)**
  - Atomic number
  - Electronegativity
  - Atomic radius
  - Ionization energy
  - Periodic-table group
  - Coordination number

- **Edges**
  - Interatomic distance
  - Bond ratio

## Model

### Tabular Baseline: 
HistGradientBoosting Regressor (HistGB) trained exclusively on Magpie features (selected after benchmarking against Random Forest and XGBoost). 

### GNN Model: 
Graph Attention Network (GAT) capturing directional and structural atomic interactions via multi-head attention. 

<p align="center">
  <img width="465" height="712" alt="スクリーンショット 2026-09-15 104946" src="https://github.com/user-attachments/assets/9793dcd2-4401-45ae-b950-e2583a110a0d" />
</p>

### Hybrid Model: 
A late-fusion approach concatenating frozen GNN graph embeddings with Magpie tabular features, fed directly into a HistGB regressor.

<p align="center">
  <img width="643" height="527" alt="スクリーンショット 2026-09-15 123823" src="https://github.com/user-attachments/assets/15040c1e-e080-43b1-993a-b8ac979aa3cf" />
</p>

### Fusion Model (FiLM + Gated GAT):
An end-to-end hybrid: a tabular MLP branch conditions the graph branch through a **FiLM layer** (feature-wise scale and shift of node embeddings), three residual `TransformerConv` blocks learn structural embeddings, and a **learned sigmoid gate** blends the pooled graph embedding with the tabular embedding before the regression head. Unlike the Hybrid model, both branches are trained jointly rather than concatenating frozen features.

## Result Plot

<table>
  <tr>
    <td align="center">
      <b>Magpi Tubular Data (HistGB)</b><br>
      <img width="400" alt="Final_materialmind_HistGB_test" src="https://github.com/user-attachments/assets/f9e922a2-1dd5-442b-98ce-1d1894e58612" />
    </td>
    <td align="center">
      <b>GNN (GAT)</b><br>
      <img width="400" alt="Final_materialmind_GNN_test" src="https://github.com/user-attachments/assets/1db6bdbe-1f04-4861-9c88-c1535ea6ebc9" />
    </td>
    <td align="center">
      <b>Hybrid (Frozen GNN embedding + tubular data through HistGB)</b><br>
      <img width="400" alt="Final_materialmind_Hybrid_test" src="https://github.com/user-attachments/assets/7c45d386-5447-4c81-a7cd-3a0d35d78c20" />
    </td>
  </tr>
  <tr>
    <td align="center" colspan="3">
      <b>Fusion (FiLM + Gated GAT, trained end-to-end)</b><br>
      <img width="400" alt="Final_materialmind_Fusion_test" src="docs/images/Final_materialmind_Fusion_test.png" /><br>
      <sub>Seed 123: R² = 0.933, RMSE = 0.098, MAE = 0.053 (the seed closest to the 5-seed mean; regenerate with <code>Testing/Fusion/plot.py</code>)</sub>
    </td>
  </tr>
</table>

## Result Table

### 1. Overall Performance Across Seeds (Mean ± Std)

| Model | RMSE (Mean) | RMSE (Std) | MAE (Mean) | MAE (Std) | R2 (Mean) | R2 (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNN** | 0.101292 | 0.003779 | 0.060661 | 0.002698 | 0.928988 | 0.005259 |
| **Magpie** | 0.130382 | 0.001122 | 0.077400 | 0.001561 | 0.882467 | 0.002030 |
| **raw Hybrid** | 0.101953 | 0.002787 | 0.057978 | 0.001691 | 0.928094 | 0.003941 |
| **Fusion** | 0.098397 | 0.002876 | 0.054478 | 0.002426 | 0.933017 | 0.003903 |

---

### 2. Paired GNN vs Hybrid R2 Comparison (n = 5 seeds)

| Seed | GNN R2 | Hybrid R2 | Diff (Hybrid - GNN) |
| :---: | :---: | :---: | :---: |
| **42** | 0.926648 | 0.932056 | 0.005408 |
| **123** | 0.935578 | 0.927418 | -0.008160 |
| **456** | 0.924251 | 0.922591 | -0.001660 |
| **789** | 0.933656 | 0.926631 | -0.007025 |
| **2024** | 0.924807 | 0.931775 | 0.006968 |

> **Statistical Test Summary**
> * **Mean R2 Diff (Hybrid - GNN):** -0.0009 (std: 0.0062)
> * **Paired t-test:** t = -0.288, p = 0.7876
> * **Conclusion:** **NOT statistically significant** at alpha = 0.05. More seeds are needed to draw a confident conclusion.

Fusion vs. the other structure-aware models (same 5 seeds, paired t-test on R2):
Fusion - GNN = +0.0040 (p = 0.30, Fusion higher in 3 of 5 seeds); Fusion - raw Hybrid = +0.0049
(p = 0.20, Fusion higher in 4 of 5 seeds). Neither difference is statistically significant.

---

### 3. Subgroup Analysis (Mean Across Seeds)

| Model | Subgroup | MAE | RMSE | Count |
| :--- | :--- | :---: | :---: | :---: |
| **GNN** | `bottom_20pct_lowK` | 0.109324 | 0.172406 | 298.0 |
| | `extreme_low_K_(<3.0GPa)` | 0.452363 | 0.560878 | 5.0 |
| | `middle_60pct` | 0.053309 | 0.079867 | 892.0 |
| | `top_20pct_highK` | 0.034005 | 0.047744 | 298.0 |
| **Magpie** | `bottom_20pct_lowK` | 0.144663 | 0.225607 | 298.0 |
| | `extreme_low_K_(<3.0GPa)` | 0.593106 | 0.664448 | 5.0 |
| | `middle_60pct` | 0.060317 | 0.088725 | 892.0 |
| | `top_20pct_highK` | 0.061274 | 0.101879 | 298.0 |
| **raw Hybrid** | `bottom_20pct_lowK` | 0.107778 | 0.179017 | 298.0 |
| | `extreme_low_K_(<3.0GPa)` | 0.568495 | 0.653917 | 5.0 |
| | `middle_60pct` | 0.049149 | 0.076478 | 892.0 |
| | `top_20pct_highK` | 0.034606 | 0.048189 | 298.0 |
| **Fusion** | `bottom_20pct_lowK` | 0.106668 | 0.172356 | 298.0 |
| | `extreme_low_K_(<3.0GPa)` | 0.417256 | 0.472083 | 5.0 |
| | `middle_60pct` | 0.044544 | 0.074571 | 892.0 |
| | `top_20pct_highK` | 0.032024 | 0.043605 | 298.0 |

### Key findings

- **Structure matters.** The composition-only Magpie baseline (R² 0.88) is clearly
  outperformed by every model that sees the crystal graph (R² 0.93).
- **Fusion has the best mean scores** (R² 0.933, MAE 0.054) and the lowest mean error in
  every subgroup above, but its margin over GNN and raw Hybrid (about +0.004 to +0.005 R²)
  is comparable to the seed-to-seed spread and is not statistically significant with 5 seeds.
- **The Fusion gate is not strongly material-specific.** Its mean value is about 0.43–0.45
  (roughly equal graph/tabular weighting) with a small std across materials (about 0.03–0.045),
  so it behaves more like a near-constant blend than an adaptive per-material switch.
- **All models struggle most on the extreme low-K tail** (`K < 3 GPa`, n = 5), with MAE roughly
  an order of magnitude above the bulk of the distribution. This is likely a data-scarcity
  effect. Fusion is the least affected (MAE 0.417 vs. 0.452 for GNN), but with only 5 test
  materials this is weak evidence. See `results/error_analysis/` for the full breakdowns.

---

## Repository Structure

```
data ingestion/       # Raw MP data -> cleaned -> featurized -> graphs + splits
Testing/
  Fusion/              # End-to-end Fusion (FiLM + Gated GAT) model + training
  GNN testing/         # GAT baseline, hyperparameter sweeps
  hybrid model test v1/  # Raw hybrid (GNN embeddings + Magpie -> HistGB)
  tubular_data_model_test/  # Magpie-only tabular baseline
Fusion_service/        # FastAPI inference microservice for the Fusion model (+ tests)
data/                  # Processed features, graphs, and train/val/test splits (gitignored)
results/               # Metrics, predictions, trained weights, error analysis (gitignored)
```

## Data Pipeline

Four scripts turn raw Materials Project data into everything the tabular
and GNN models train on, using **the same materials, correctly matched,
with the same train/val/test split**. Run in order (each step reads the
previous step's output):

1. `download_v2.py` — pull raw structures/properties from the Materials Project API
2. `clean_v2.py` — filter and clean records
3. `featurize_v2.py` — compute Magpie composition descriptors
4. `split_and_graph.py` — build the train/val/test split and crystal graphs

See `data ingestion/README.data_ingestion.md` for details.

## Inference Service

`Fusion_service/` packages the trained Fusion model behind a FastAPI
`/predict` endpoint (formula + CIF file in, predicted bulk modulus out),
containerized via the included `dockerfile`. This is under active
development.

Run it with `cd Fusion_service && python -m uvicorn app.main:app`, then open:

- **http://localhost:8000/** — web app: type a formula, drag in a CIF file (or click a built-in example),
  and see the prediction with an ensemble-uncertainty plot.
- **http://localhost:8000/docs** — interactive API documentation.
- `GET /health` — service status and the model's reference test error.

Predictions come from a **5-model deep ensemble** — the same 5
independently-seeded checkpoints reported in the Results table above
(`Fusion_best_model_seed{42,123,456,789,2024}.pth`), all run on the same
input graph. The response reports the ensemble mean/std of `log10(K)` and
a derived (asymmetric) GPa interval, alongside each member's individual
prediction. This uncertainty is **epistemic only** — disagreement among 5
networks of the same architecture on the same input — not measurement
noise in the training labels, and not an out-of-distribution signal: all
5 members share one architecture, training set, and featurizer, so they
can agree closely while being systematically wrong together on an
unfamiliar material. A low reported std should not be read as high
accuracy.

### Tests and CI

The service has its own pytest suite and dependency files, independent of the repo-wide
`requirements.txt` used for the research code. GitHub Actions
(`.github/workflows/fusion-service-ci.yml`) runs lint and tests on changes under `Fusion_service/` only.

```bash
cd Fusion_service
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (optional locally)
pip install -r requirements-dev.txt
ruff check . && pytest --cov=app
```

---

## Tech Stack

Python, PyTorch, PyTorch Geometric, scikit-learn, Matminer/Magpie,
XGBoost/HistGradientBoosting, Weights & Biases (experiment tracking and
hyperparameter sweeps), FastAPI, Docker, pytest, GitHub Actions.
