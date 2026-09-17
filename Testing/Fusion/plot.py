import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd

def find_project_root(marker="data"):
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / marker).is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find a '{marker}' directory above {__file__}")

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "Fusion_test_predictions_seed42.csv")

y_test_arr = df["actual_log_K"].values
y_pred = df["predicted_log_K"].values

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_arr, y_pred, alpha=0.5, color='teal', edgecolors='k', linewidth=0.5)
plt.plot([y_test_arr.min(), y_test_arr.max()], [y_test_arr.min(), y_test_arr.max()], 'r--', lw=2, label='Ideal (1:1)')
plt.xlabel('Actual Log Bulk Modulus', fontsize=11)
plt.ylabel('Predicted Log Bulk Modulus', fontsize=11)
plt.title('Predicted vs. Actual', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
sns.histplot(y_test_arr - y_pred, kde=True, color='crimson', bins=30)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Residual Error (Actual - Predicted)', fontsize=11)
plt.ylabel('Material Count', fontsize=11)
plt.title('Error Distribution', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()