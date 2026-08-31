# Predicting Bulk Modulus with a Hybrid Tabular + Graph Neural Network Model
### Research & Engineering Plan (Open-Source Project)

**Author:** [Your Name]
**Status:** Planning
**Target duration:** 8–10 weeks, part-time (~10–15 hrs/week)
**License:** MIT or Apache-2.0 (recommended for resume visibility + adoption)

---

## 1. Motivation & Framing

Bulk modulus (K) measures a material's resistance to uniform compression and is a key
screening property for structural materials, battery electrodes, superhard materials,
and thermal-barrier coatings. It is expensive to compute via DFT (hours to days per
structure) and even more expensive to measure experimentally, which is why it is a
standard machine-learning target in materials informatics.

The field has largely split into two camps:

- **Tabular / descriptor-based models** (Random Forest, XGBoost, LightGBM on
  Magpie/Matminer composition and structure statistics) — fast, interpretable,
  strong with small data, but blind to explicit atomic connectivity.
- **Graph Neural Networks** (CGCNN, MEGNet, ALIGNN, coGN) operating directly on the
  crystal graph — capture bonding/geometry but are more data-hungry, slower to
  train, and less interpretable.

Published work (ORNL HydraGNN, GATGNN, IGNN, CGCNN-based studies on Matbench) shows
bulk modulus is a **harder target than formation energy or band gap** because it
depends weakly and nonlinearly on composition and is sensitive to fine structural
detail (bond stiffness, packing) that summary statistics can wash out. This is
exactly the gap a **hybrid model** is positioned to close: let the GNN learn local
bonding/geometric embeddings, and let a gradient-boosted tree combine those learned
embeddings with global compositional/statistical descriptors that trees are
already excellent at exploiting.

**This is a legitimate, publishable-quality research question, not just a portfolio
exercise** — which is exactly what makes it a strong resume project: it has a clear
baseline to beat, a known benchmark to report against, and a real engineering
surface area (data pipeline, model architecture, evaluation, packaging, docs).

---

## 2. Project Goal & Success Criteria

**Primary goal:** Build and open-source a hybrid model (`XGBoost + GNN embeddings`)
that predicts `log10(K_VRH)` on the standard **Matbench `matbench_log_kvrh`**
benchmark (10,987 inorganic crystals from Materials Project, DFT Voigt-Reuss-Hill
averaged bulk modulus), and demonstrate it is competitive with or better than either
component model alone.

**Success criteria (in priority order):**

1. **Correctness / reproducibility bar:** full pipeline runs end-to-end from raw
   Materials Project data → trained model → benchmark report, with a fixed seed and
   documented environment (this alone is what most portfolio projects skip).
2. **Scientific bar:** hybrid model MAE on `matbench_log_kvrh` test folds is
   statistically indistinguishable from or better than your own best single-model
   baseline (GNN-only and XGBoost-only), evaluated with the benchmark's prescribed
   5-fold nested CV. You are *not* required to beat published SOTA (coGN/ALIGNN
   ~0.055–0.07 MAE range) to call this a success — beating your own honest
   baselines and explaining *why* the hybrid helps (or doesn't) is the actual
   scientific contribution.
3. **Engineering bar:** clean installable Python package, tests, CI, model card,
   inference API/CLI, and a results notebook that anyone can re-run.
4. **Communication bar:** README + short write-up (blog-style) explaining the
   architecture, ablations, and what you learned — this is what recruiters and
   interviewers will actually read.

---

## 3. Scope Decisions (make these explicit early)

| Decision | Recommendation | Why |
|---|---|---|
| Target property | `log10(K_VRH)` (log-bulk-modulus, GPa) | Matches Matbench convention; log-scale stabilizes variance across orders of magnitude |
| Benchmark dataset | `matbench_log_kvrh` (10,987 structures) | Standardized, citable, has known baselines to compare against |
| Extension dataset (optional, stretch) | JARVIS-DFT elastic tensor data | Larger, lets you test generalization beyond Matbench's curated set |
| Structure representation | Crystal graph (pymatgen `Structure` → graph) | Needed for GNN; matches CGCNN/MEGNet convention |
| GNN architecture | Start with CGCNN-style message passing; consider ALIGNN-lite (line graph) as stretch goal | CGCNN is well-documented, tractable to implement from scratch, and is the standard "readable" baseline |
| Tabular features | Magpie composition features + Matminer structural descriptors (density, packing fraction, coordination stats) via `matminer` | Avoids reinventing feature engineering; matches literature (e.g., 132-feature Matminer sets used in recent bulk modulus papers) |
| Hybrid fusion strategy | GNN produces a learned structure embedding → concatenated with tabular descriptors → fed into XGBoost | Simplest defensible hybrid; also enables ablations (tabular-only, GNN-only, hybrid) |
| Compute budget | Single GPU (Colab/Kaggle free tier or a rented spot instance) for GNN training; CPU fine for XGBoost | Keep the project reproducible without needing a cluster |

Decide these once, write them into the README on day 1, and don't relitigate scope
mid-project — scope creep is the #1 killer of portfolio projects.

---

## 4. System Architecture

```
                         ┌─────────────────────────┐
                         │   Materials Project /    │
                         │   Matbench raw data       │
                         └────────────┬──────────────┘
                                      │
                     ┌────────────────┴─────────────────┐
                     ▼                                   ▼
        ┌───────────────────────┐          ┌───────────────────────────┐
        │  Tabular feature        │          │  Crystal graph               │
        │  pipeline (matminer,    │          │  construction (pymatgen,     │
        │  Magpie composition,    │          │  radius/knn cutoff graph,    │
        │  structural stats)      │          │  one-hot + Gaussian expand   │
        └───────────┬─────────────┘          └────────────┬───────────────┘
                     │                                     │
                     │                                     ▼
                     │                     ┌───────────────────────────┐
                     │                     │  GNN encoder (CGCNN-style   │
                     │                     │  message passing, 3-4       │
                     │                     │  conv layers + pooling)     │
                     │                     └────────────┬───────────────┘
                     │                                     │
                     │                       ┌─────────────┴──────────────┐
                     │                       ▼                             ▼
                     │           GNN direct prediction         Learned graph embedding
                     │           (baseline, own loss)           (fixed-length vector)
                     │                                                     │
                     └─────────────────────┬───────────────────────────────┘
                                           ▼
                            ┌───────────────────────────────┐
                            │  Feature fusion: [tabular       │
                            │  descriptors ‖ GNN embedding]   │
                            └────────────────┬─────────────────┘
                                             ▼
                            ┌───────────────────────────────┐
                            │  XGBoost regressor (final       │
                            │  bulk modulus prediction)       │
                            └────────────────┬─────────────────┘
                                             ▼
                                 log10(K_VRH) prediction
```

Three models get trained and reported, not just the hybrid — this is what makes the
ablation credible:

1. **Tabular-only baseline** — XGBoost on Matminer/Magpie features.
2. **GNN-only baseline** — CGCNN-style model trained end-to-end with its own
   regression head.
3. **Hybrid model** — GNN trained (or fine-tuned) to produce embeddings, concatenated
   with tabular features, fed to XGBoost.

---

## 5. Detailed Phase Plan

### Phase 0 — Setup & Literature Grounding (Week 1)
- Read and take structured notes on 4–6 core papers:
  - Xie & Grossman, *CGCNN*, PRL 2018
  - Chen et al., *MEGNet*, Chem. Mater. 2019
  - Choudhary & DeCost, *ALIGNN*, npj Comp. Mater. 2021
  - Dunn et al., *Matbench*, npj Comp. Mater. 2020 (this defines your benchmark protocol)
  - Lupo Pasini et al. (ORNL), *HydraGNN* — explicitly notes bulk modulus is harder to
    predict than formation energy; good citation for your risk section
  - One recent (2024–2025) tabular-ML bulk modulus paper (Matminer feature approach)
    for a concrete tabular-only baseline number to compare against
- Set up repo skeleton, `pyproject.toml`, environment (conda or `uv`), pre-commit
  hooks (black, ruff, isort), GitHub Actions CI stub (lint + a trivial test).
- Write the README scope section (Section 3 of this plan, adapted) — commit it.
- Deliverable: repo scaffold + `NOTES.md` literature summary + environment that
  installs cleanly on a fresh machine.

### Phase 1 — Data Pipeline (Week 2)
- Pull `matbench_log_kvrh` via the `matbench` Python package (handles the official
  5-fold CV split so your numbers are directly comparable to published results).
- Build a `Structure → graph` converter using `pymatgen`: nodes = atoms (one-hot
  element + Magpie atomic features), edges = neighbors within a cutoff radius (or
  k-NN), edge features = Gaussian-expanded interatomic distances.
- Build the tabular featurizer using `matminer.featurizers.composition` (Magpie) and
  `matminer.featurizers.structure` (density, packing fraction, coordination number
  stats, etc.). Cache featurized outputs to disk (Parquet) — DFT-structure
  featurization is slow, don't recompute it every run.
- Write data validation checks: no NaNs leaking into training, no target leakage
  (e.g., don't accidentally include a feature derived from K itself), consistent
  ordering across folds.
- Deliverable: `data/` module with `load_matbench_kvrh()`, `structure_to_graph()`,
  `featurize_tabular()`, all covered by unit tests on a handful of fixture structures.

### Phase 2 — Baseline 1: Tabular XGBoost (Week 3)
- Train XGBoost with a documented hyperparameter search (Optuna, small budget —
  50–100 trials) over max_depth, learning_rate, n_estimators, subsample,
  colsample_bytree, min_child_weight.
- Use the Matbench-prescribed nested CV protocol so your MAE is directly comparable
  to literature numbers.
- Log everything with a lightweight experiment tracker (Weights & Biases free tier,
  or MLflow if you want to self-host — both look good on a resume, pick one).
- Deliverable: reproducible baseline MAE/RMSE/R² on `log10(K_VRH)`, feature
  importance plot (SHAP), committed model artifact + config.

### Phase 3 — Baseline 2: GNN from Scratch (Weeks 4–5)
- Implement a CGCNN-style graph convolution in PyTorch + PyTorch Geometric:
  message passing layer that updates atom embeddings from neighbor embeddings and
  edge (bond) features, followed by global pooling (mean or attention-weighted) and
  a small MLP regression head.
- Train with early stopping, learning-rate scheduling, and the same CV protocol as
  Phase 2.
- This is the highest-risk phase — budget extra time. Known failure modes to expect
  and pre-empt: overfitting on the pooled embedding, unstable training without edge
  feature normalization, slow convergence without proper weight initialization.
- Deliverable: trained GNN checkpoint, training curves, MAE/RMSE/R² comparable to
  Phase 2's baseline, and a saved embedding-extraction hook (needed for Phase 4).

### Phase 4 — Hybrid Model (Week 6)
- Extract fixed-length graph embeddings from the trained (or a lightly fine-tuned)
  GNN's pre-head pooled layer for every structure.
- Concatenate with the Phase 1 tabular features; re-run the Phase 2 XGBoost
  hyperparameter search on this expanded feature set (embeddings + descriptors may
  need different regularization than tabular-only).
- Run the **actual ablation study** — this is the intellectual core of the project:
  - Tabular-only vs. GNN-only vs. Hybrid, same CV folds, same metrics
  - Feature importance: how much weight does XGBoost put on GNN-derived features
    vs. hand-engineered ones?
  - Error analysis: are the hybrid model's improvements concentrated in specific
    chemistries/structure types (e.g., does it help most on complex, low-symmetry
    structures where hand-crafted descriptors lose information)?
- Deliverable: ablation table + plots, a written explanation of *why* the hybrid
  does or doesn't help (a honest null result here is still a good result if
  explained well).

### Phase 5 — Packaging, Docs, and Polish (Week 7)
- Package as an installable library (`pip install .`) with a clean public API:
  ```python
  from bulkmod_hybrid import HybridPredictor
  model = HybridPredictor.from_pretrained("checkpoints/hybrid_v1")
  model.predict(structure)  # pymatgen Structure -> float
  ```
- Add a CLI entry point (`bulkmod-predict --cif my_structure.cif`).
- Write a **model card** (intended use, training data provenance, known
  limitations, out-of-distribution behavior warning — e.g., don't trust
  predictions on chemistries far outside Materials Project's coverage).
- Add CI: lint, unit tests, and a lightweight "does inference run" smoke test on
  every PR.
- Deliverable: `pip`-installable package, model card, passing CI badge on README.

### Phase 6 — Write-up & Release (Week 8)
- Write a results-focused README with the ablation table front and center, an
  architecture diagram, and clear "how to reproduce" instructions.
- Optional but high-value: a short blog post / Medium-style write-up walking through
  the hybrid architecture decision and what the ablation revealed — this is what
  you'll actually link from your resume/LinkedIn, not the raw repo.
- Tag a `v1.0` release, archive on Zenodo for a DOI (free, takes 10 minutes, makes
  the project citable — a nice detail for a portfolio piece).
- Deliverable: public GitHub repo, README, optional blog post, tagged release.

### Stretch Goals (only after core plan is solid)
- Swap CGCNN-style GNN for an ALIGNN-lite (line-graph, angle-aware) encoder and
  re-run the ablation — angle information is known to matter for elastic properties.
- Add uncertainty quantification (quantile XGBoost or MC-dropout on the GNN) so
  predictions come with confidence intervals — valuable for a "screening tool"
  framing.
- Extend to a second target (shear modulus, `log_gvrh`) to show the pipeline
  generalizes, not just overfit to one Matbench task.
- Build a small Streamlit/Gradio demo where someone uploads a CIF and gets a
  prediction — great for a resume link people can actually click and use.

---

## 6. Tech Stack

| Layer | Tool |
|---|---|
| Structure handling | `pymatgen` |
| Tabular featurization | `matminer` |
| GNN framework | PyTorch + PyTorch Geometric (or DGL) |
| Gradient boosting | `xgboost` |
| Hyperparameter search | `optuna` |
| Experiment tracking | Weights & Biases (or MLflow) |
| Benchmark harness | `matbench` |
| Interpretability | `shap` |
| Packaging | `pyproject.toml` (hatch or setuptools), `pip`-installable |
| CI/CD | GitHub Actions |
| Environment | `conda` or `uv` + lockfile for reproducibility |

---

## 7. Risks & Honest Mitigations

| Risk | Mitigation |
|---|---|
| Bulk modulus is known in the literature to be a harder GNN target than formation energy (weak/nonlinear composition dependence) | Set expectations in the README up front; frame the ablation as the contribution, not a SOTA claim |
| GNN training instability / slow convergence eating your timeline | Time-box Phase 3 to 2 weeks hard; if it's not converging well by day 10, fall back to a smaller, well-tested architecture (plain CGCNN) rather than debugging a novel one |
| Hybrid model shows no improvement over baselines | This is a valid, publishable outcome — document it honestly with error analysis rather than hunting for a config that "makes the number go up" |
| Compute limits (no GPU cluster) | Matbench's dataset (~11k structures) is small enough to train on a single free-tier GPU (Colab/Kaggle) in a few hours per fold |
| Scope creep (adding more properties, architectures, datasets mid-project) | Lock scope in Phase 0, keep stretch goals clearly separated and optional |

---

## 8. What This Gives You for Your Resume

- A concrete, defensible line: *"Built and open-sourced a hybrid GNN + gradient-
  boosted-tree model for materials property prediction, benchmarked on Matbench;
  designed and ran an ablation study comparing tabular, graph, and hybrid
  representations."*
- Evidence of: ML engineering (data pipelines, packaging, CI), research skills
  (literature grounding, honest ablation design, benchmark-aware evaluation), and
  communication (README, write-up, model card).
- A live, clickable artifact (GitHub repo, optionally a demo) that interviewers can
  actually look at — far stronger than a bullet point alone.

---

## 9. Immediate Next Steps

1. Confirm scope decisions in Section 3 (especially: single-property Matbench focus
   vs. also pulling in JARVIS data from day one).
2. Set up the repo skeleton and environment (Phase 0).
3. Read the 4–6 core papers and write `NOTES.md` before writing any model code —
   this will save you from re-deriving decisions the literature already made.