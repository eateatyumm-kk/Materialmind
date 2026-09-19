"""Regenerate tests/golden/nacl.json from the checkpoints currently in weights/.

Run this ONLY after intentionally retraining (or changing featurization/scaling), from Fusion_service/:

    python tests/golden/regenerate.py

Then review the diff, and commit the new JSON together with the new weights. Never hand-edit the JSON.
"""

import json
import sys
from pathlib import Path

import torch

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))

from app.ensemble import FusionEnsemble  # noqa: E402
from app.featurizer import MaterialsFeaturizer  # noqa: E402
from tests import structures  # noqa: E402

ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]
OUTPUT_PATH = Path(__file__).resolve().parent / "nacl.json"


def main():
    cif = structures.to_cif(structures.nacl_2atom_cscl_type())
    data = MaterialsFeaturizer().process_input("NaCl", cif)
    ensemble = FusionEnsemble.from_weight_dir(APP_ROOT / "weights", ENSEMBLE_SEEDS, device=torch.device("cpu"))
    result = ensemble.predict(data)

    golden = {
        "structure": "nacl_2atom_cscl_type",
        "formula": "NaCl",
        "per_seed_log10": {str(seed): value for seed, value in result.per_seed_log10.items()},
        "mean_log10": result.mean_log10,
        "std_log10": result.std_log10,
    }
    OUTPUT_PATH.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(golden, indent=2))


if __name__ == "__main__":
    main()
