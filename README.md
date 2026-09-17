# MaterialMind

[![GitHub Repository](https://img.shields.io/badge/GitHub-eateatyumm--kk%2FMaterialmind-blue?logo=github)](https://github.com/eateatyumm-kk/Materialmind)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-PyG-EE4C2C?logo=pytorch)](https://pytorch-geometric.readthedocs.io/)

**MaterialMind** is a materials informatics project investigating whether explicit crystal-structure representations learned via **Graph Neural Networks (GNNs)** provide complementary predictive power beyond traditional **composition-based tabular descriptors** when predicting a material's **bulk modulus (K)**. I chose to do This project to learn more about GNN and it's application. 

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
</table>

## Result Table

### 1. Overall Performance Across Seeds (Mean ± Std)

| Model | RMSE (Mean) | RMSE (Std) | MAE (Mean) | MAE (Std) | R2 (Mean) | R2 (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNN** | 0.101292 | 0.003779 | 0.060661 | 0.002698 | 0.928988 | 0.005259 |
| **Magpie** | 0.130382 | 0.001122 | 0.077400 | 0.001561 | 0.882467 | 0.002030 |
| **raw Hybrid** | 0.101953 | 0.002787 | 0.057978 | 0.001691 | 0.928094 | 0.003941 |

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


