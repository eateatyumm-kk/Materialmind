
# GNN performance

# ======== hyperparameter　v2 ========

learning_rate = 0.0002299
batch_size = 64
hidden_dim = 128
fc_hidden = 32
num_heads = 8
dropout = 0.0
attn_dropout = 0.1
weight_decay = 0.000010194
epochs = 300
patience = 30

# GAT test

Early stopping at epoch 124. Best val_loss: 0.0705

Training complete. Best val_loss: 0.0705. Best checkpoint saved to C:\projects\Materialmind\results\gat_best_model.pth

=== FINAL TEST SET PERFORMANCE (log10 GPa) ===
Test RMSE: 0.0998
Test MAE:  0.0579
Test R²:   0.9312
