## before Tuning the model

# Hybrid testing result

Feature set: 264 columns (132 Magpie + 4 density + 128 GNN embeddings)

=== FINAL TEST SET PERFORMANCE (log10 GPa) — HistGB comp+density ===
Test RMSE: 0.1006
Test MAE:  0.0585
Test R²:   0.9300

# GNN emb + magpi

=== FINAL TEST SET PERFORMANCE (log10 GPa) — HistGB comp+density ===
Test RMSE: 0.1043
Test MAE:  0.0616
Test R²:   0.9248

# GNN emb to hist

Feature set: 128 columns (128 GNN embeddings)

=== FINAL TEST SET PERFORMANCE (log10 GPa) — HistGB emb+comp+density ===
Test RMSE: 0.1111
Test MAE:  0.0697
Test R²:   0.9147

# with PCA emb + dens + comp

=== FINAL TEST SET PERFORMANCE (log10 GPa) ===
Test RMSE: 0.0970
Test MAE:  0.0552
Test R²:   0.9349   improved