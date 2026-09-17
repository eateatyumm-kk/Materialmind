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





# Tuning model

## With PCA

    learning_rate=0.05589461485225799,
    l2_regularization=0.002266953931102471,
    max_depth=10,
    max_iter=500,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    pca_count=48,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,

## With out PCA

    learning_rate=0.1086949308007322,
    l2_regularization=0.0002321764804508835,
    max_depth=5,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,

## test result 

Feature set: 180 columns (132 Magpie + 48 GNN embeddings)

=== FINAL TEST SET PERFORMANCE (log10 GPa) ===
Test RMSE: 0.1014
Test MAE:  0.0566
Test R²:   0.9289

===== biggest error =====
  material_id    y_true    y_pred  residual   formula
0     mp-3142  0.573800  1.834888  1.261088  Ca(NO3)2
1    mp-23280 -0.254145  0.424198  0.678343     AsCl3
2     mp-8224  1.749914  1.097383  0.652531    CaSnF6
3     mp-2793  0.954966  1.597962  0.642996      AuSe
4   mp-542570  0.292699  0.924189  0.631490      AsSe
5    mp-31038  0.749891  1.328144  0.578254   CuSe2Cl
6     mp-1999  1.108227  1.677765  0.569539     Sb2O3
7   mp-850275  0.996862  1.552865  0.556003   Li3SbS4
8   mp-560553  0.749659  1.280211  0.530552      IrF6
9   mp-541785  0.911743  1.414852  0.503109    GePdS3
Raw GNN embedding dimensions: 64
Feature set: 196 columns (132 Magpie + 64 GNN embeddings)

=== FINAL TEST SET PERFORMANCE (log10 GPa) ===
Test RMSE: 0.0991
Test MAE:  0.0554
Test R²:   0.9321

===== biggest error =====
  material_id    y_true    y_pred  residual   formula
0     mp-3142  0.573800  1.947033  1.373233  Ca(NO3)2
1   mp-542738  0.452093  1.522969  1.070876        SN
2   mp-542570  0.292699  0.883321  0.590622      AsSe
3     mp-1999  1.108227  1.677336  0.569109     Sb2O3
4     mp-8224  1.749914  1.213959  0.535955    CaSnF6
5    mp-30139  0.427486  0.949339  0.521853     BeBr2
6   mp-850275  0.996862  1.507245  0.510383   Li3SbS4
7   mp-545974  2.083305  1.587358  0.495947     AlPO4
8    mp-31038  0.749891  1.234923  0.485032   CuSe2Cl
9    mp-20438  1.287062  1.761100  0.474037      PuTe

## SEED test 

============================================================
SEED TEST RESULTS
============================================================
         model  seed      RMSE       MAE        R2
0       Magpie    42  0.129640  0.076776  0.883807
1       Magpie   123  0.130229  0.077705  0.882750
2       Magpie   456  0.130500  0.079230  0.882260
3       Magpie   789  0.132209  0.078190  0.879156
4       Magpie  2024  0.129330  0.075100  0.884363
5   raw Hybrid    42  0.099135  0.055401  0.932056
6   raw Hybrid   123  0.102462  0.057592  0.927418
7   raw Hybrid   456  0.105814  0.059548  0.922591
8   raw Hybrid   789  0.103016  0.059455  0.926631
9   raw Hybrid  2024  0.099339  0.057893  0.931775
10         GNN    42  0.099434  0.060746  0.931646
11         GNN   123  0.094807  0.058937  0.937859
12         GNN   456  0.105276  0.060900  0.923377
13         GNN   789  0.098730  0.059480  0.932610
14         GNN  2024  0.104745  0.062101  0.924148

============================================================
MEAN ± STD
============================================================
                RMSE                 MAE                  R2          
                mean       std      mean       std      mean       std
model                                                                 
GNN         0.100598  0.004401  0.060433  0.001249  0.929928  0.006111
Magpie      0.130382  0.001122  0.077400  0.001561  0.882467  0.002030
raw Hybrid  0.101953  0.002787  0.057978  0.001691  0.928094  0.003941