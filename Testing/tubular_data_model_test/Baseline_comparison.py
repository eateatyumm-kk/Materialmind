import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from pathlib import Path
from xgboost import XGBRegressor

# --------- Splitting the data into train/test/val and prepare two datasets ------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load files reliably
df = pd.read_csv(PROJECT_ROOT / "data" / "tabular_features.csv")

test_ids = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test_ids.csv")["material_id"]
train_ids = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train_ids.csv")["material_id"]
val_ids = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "val_ids.csv")["material_id"]

# 2. Define column groups
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
density_cols = [
    c for c in df.columns 
    if c not in magpie_cols and c not in ("material_id", "log_bulk_modulus_vrh")
]
comp_den_cols = magpie_cols + density_cols

# 3. Split the main dataframe first using sets for faster lookup
train_df = df[df["material_id"].isin(set(train_ids))]
val_df   = df[df["material_id"].isin(set(val_ids))]
test_df  = df[df["material_id"].isin(set(test_ids))]

# ---------- Only composition datasets -----------
X_train_subset_comp = train_df[magpie_cols]
y_train_subset_comp = train_df["log_bulk_modulus_vrh"]

X_val_subset_comp = val_df[magpie_cols]
y_val_subset_comp = val_df["log_bulk_modulus_vrh"]

X_test_subset_comp = test_df[magpie_cols]
y_test_subset_comp = test_df["log_bulk_modulus_vrh"]

# --------- Data with composition and density ----
X_train_subset_comp_den = train_df[comp_den_cols]
y_train_subset_comp_den = train_df["log_bulk_modulus_vrh"]

X_val_subset_comp_den = val_df[comp_den_cols]
y_val_subset_comp_den = val_df["log_bulk_modulus_vrh"]

X_test_subset_comp_den = test_df[comp_den_cols]
y_test_subset_comp_den = test_df["log_bulk_modulus_vrh"]

# -------- Pipelines for ML models ----------
models = {
    "1. Dummy Baseline (Mean)": Pipeline([
        ('model', DummyRegressor(strategy='mean'))
    ]),
    "2. Scaled Linear (Ridge)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    "3. Random Forest": Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "4. Gradient Boosting (Hist)": Pipeline([
        ('model', HistGradientBoostingRegressor(learning_rate=0.1, random_state=42))
    ]),
    "5. XGBoost": Pipeline([
        ('model', XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42))
    ])
}

# -------- Baseline cross-validation evaluation ------------
def baseline(dataset_type="comp"):
    results = []
    metrics = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    if dataset_type == "comp":
        X_train, y_train = X_train_subset_comp, y_train_subset_comp
    elif dataset_type == "comp_den":
        X_train, y_train = X_train_subset_comp_den, y_train_subset_comp_den

    for name, pipeline in models.items():
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=metrics, n_jobs=-1)
        
        rmse = -cv_scores['test_neg_root_mean_squared_error'].mean()
        mae = -cv_scores['test_neg_mean_absolute_error'].mean()
        r2 = cv_scores['test_r2'].mean()
        
        results.append({
            'Model': name,
            'CV RMSE': round(rmse, 3),
            'CV MAE': round(mae, 3),
            'CV R²': round(r2, 4)
        })

    results_df = pd.DataFrame(results)
    return results_df

# -------- Model evaluation and plotting ------
def result_analysis():
    comp_result = baseline(dataset_type="comp")
    comp_den_result = baseline(dataset_type="comp_den")
    
    # FIXED: Capitalized 'Model', fixed idmax -> idxmax, corrected dataframe references
    comp_best_model = str(comp_result.loc[comp_result['CV R²'].idxmax(), "Model"])
    comp_den_model = str(comp_den_result.loc[comp_den_result['CV R²'].idxmax(), "Model"])

    def model_eval(model_name, dataset_type="comp"):
        best_pipeline = models[model_name]

        if dataset_type == "comp":
            X_train, y_train = X_train_subset_comp, y_train_subset_comp
            X_val, y_val = X_val_subset_comp, y_val_subset_comp
        elif dataset_type == "comp_den":
            X_train, y_train = X_train_subset_comp_den, y_train_subset_comp_den
            X_val, y_val = X_val_subset_comp_den, y_val_subset_comp_den

        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_val)

        print(f"\n=== VALIDATION SET PERFORMANCE ({dataset_type.upper()}) ===")
        print(f"Model: {model_name}")
        print(f"Val RMSE: {root_mean_squared_error(y_val, y_pred):.3f} (Log Scale)")
        print(f"Val MAE:  {mean_absolute_error(y_val, y_pred):.3f} (Log Scale)")
        print(f"Val R²:   {r2_score(y_val, y_pred):.4f}")

        residuals = np.abs(y_val - y_pred)
        error_df = pd.DataFrame({
            'Actual_Log_Bulk_Modulus': y_val,
            'Predicted_Log_Bulk_Modulus': y_pred,
            'Absolute_Error': residuals,
            'Residual': y_val - y_pred 
        })
        return error_df

    def graph(error, title_suffix=""):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(error['Actual_Log_Bulk_Modulus'], error['Predicted_Log_Bulk_Modulus'], alpha=0.5, color='teal', edgecolors='k', linewidth=0.5)
        plt.plot([error['Actual_Log_Bulk_Modulus'].min(), error['Actual_Log_Bulk_Modulus'].max()], 
                 [error['Actual_Log_Bulk_Modulus'].min(), error['Actual_Log_Bulk_Modulus'].max()], 'r--', lw=2, label='Ideal (1:1)')
        plt.xlabel('Actual Log Bulk Modulus', fontsize=11)
        plt.ylabel('Predicted Log Bulk Modulus', fontsize=11)
        plt.title(f'Predicted vs. Actual ({title_suffix})', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        sns.histplot(error['Residual'], kde=True, color='crimson', bins=30)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        plt.xlabel('Residual Error (Actual - Predicted)', fontsize=11)
        plt.ylabel('Material Count', fontsize=11)
        plt.title(f'Error Distribution ({title_suffix})', fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

        print("\n--- TOP 10 LARGEST PREDICTION FAILURES ---")
        worst_errors = error.sort_values(by='Absolute_Error', ascending=False).head(10)
        print(worst_errors.to_string())

    # ------- Printing Composition Results ---------
    print("=== Composition-Only Baseline Comparison ===")
    print(comp_result.to_string(index=False))

    error = model_eval(comp_best_model, dataset_type="comp")
    graph(error, title_suffix="Composition Only")

    # ------- Printing Composition + Density Results --------
    print("\n=== Composition + Structure Baseline Comparison ===")
    print(comp_den_result.to_string(index=False))

    error_den = model_eval(comp_den_model, dataset_type="comp_den")
    graph(error_den, title_suffix="Comp + Density")

if __name__ == "__main__":
    result_analysis()