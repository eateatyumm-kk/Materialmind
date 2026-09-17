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









