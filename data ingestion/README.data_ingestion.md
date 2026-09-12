# Data Ingestion Pipeline

Three scripts turn raw Materials Project data into everything the tabular
(XGBoost) model and the GNN model need to train on **the same materials,
correctly matched, with the same train/val/test split.**

# running order
Run them in order. Each step reads the previous step's output.

1_download_v2
        │
        ▼
2_clean_v2
        │
        ▼
3_featurize_v2
        │
        ▼
4_split_and_graph
