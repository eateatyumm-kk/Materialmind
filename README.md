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

---

## Models Compared

Four modeling approaches were trained and evaluated on the same
train/val/test split to isolate the effect of adding structural
information to a composition-only baseline:

| Model | Description |
|---|---|
| **Magpie (tabular)** | HistGradientBoosting on 132 Magpie composition descriptors only — no structural information. |
| **GNN (GAT)** | A `TransformerConv`-based graph attention network trained end-to-end on the crystal graph, with Gaussian RBF-expanded edge distances. |
| **Raw Hybrid** | Magpie descriptors concatenated with pooled GNN node embeddings, fed into a HistGradientBoosting model (late fusion, two-stage training). |
| **Fusion (FiLM + Gated GAT)** | An end-to-end hybrid architecture: a tabular MLP branch conditions the graph branch via a **FiLM layer**, three residual `TransformerConv` blocks learn structural embeddings, and a **learned gate** combines the pooled graph embedding with the tabular embedding before the final regression head. |

---

## Results

5-seed test-set performance predicting `log10(K)` (GPa):

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Magpie (tabular) | 0.1304 ± 0.0011 | 0.0774 ± 0.0016 | 0.8825 ± 0.0020 |
| GNN (GAT) | 0.1006 ± 0.0044 | 0.0604 ± 0.0012 | 0.9299 ± 0.0061 |
| Raw Hybrid | 0.1020 ± 0.0028 | 0.0580 ± 0.0017 | 0.9281 ± 0.0039 |
| **Fusion (FiLM + Gated GAT)** | **0.0980 ± 0.0035** | **0.0537 ± 0.0021** | **0.9336 ± 0.0048** |

### Key findings

- Structure matters: the composition-only Magpie baseline (R² 0.88) is
  clearly outperformed by every model that has access to crystal-graph
  information (R² 0.93+).
- End-to-end fusion beats late fusion: jointly learning the graph and
  tabular branches with FiLM conditioning and a learned gate (Fusion)
  outperforms both the GNN alone and the two-stage "raw hybrid" (GNN
  embeddings + Magpie fed into a separate tree model), suggesting the
  gate learns to weight structural vs. compositional signal per-material
  rather than treating the embeddings as static features.
- Error analysis by subgroup shows all models struggle most on the
  extreme low-bulk-modulus tail (`K < 3 GPa`, n=5), with MAE roughly an
  order of magnitude higher than on the bulk of the distribution — this
  is likely a data-scarcity effect rather than a model-architecture
  effect. See `results/error_analysis/` for full breakdowns and plots.

---

## Repository Structure

```
data ingestion/       # Raw MP data -> cleaned -> featurized -> graphs + splits
Testing/
  Fusion/              # End-to-end Fusion (FiLM + Gated GAT) model + training
  GNN testing/         # GAT baseline, hyperparameter sweeps
  hybrid model test v1/  # Raw hybrid (GNN embeddings + Magpie -> HistGB)
  tubular_data_model_test/  # Magpie-only tabular baseline
Fusion_service/        # FastAPI inference microservice for the Fusion model
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

---

## Tech Stack

Python, PyTorch, PyTorch Geometric, scikit-learn, Matminer/Magpie,
XGBoost/HistGradientBoosting, Weights & Biases (experiment tracking and
hyperparameter sweeps), FastAPI, Docker.
