import random
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.ensemble import UNCERTAINTY_SCOPE, FusionEnsemble
from app.featurizer import MaterialsFeaturizer
from app.schemas import PredictResponse

# The 5 seeds trained together in `Testing/Fusion/Fusion_test_2.py`, all
# sharing one Config() architecture. `results/Fusion_best_model.pth` (no
# seed suffix, from an earlier wandb sweep) uses a different fc_hidden and
# must never be added here — it fails a strict state_dict load.
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]


def set_seed(seed: int = 456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(456)

WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights"

app = FastAPI(title="Materials Fusion ML Microservice", version="1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load featurizer & the 5-model ensemble once on startup
featurizer = MaterialsFeaturizer()
ensemble = FusionEnsemble.from_weight_dir(WEIGHT_DIR, seeds=ENSEMBLE_SEEDS, device=device)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    formula: str = Form(..., description="Chemical formula, e.g. Fe2O3"),
    cif_file: UploadFile = File(..., description="Crystal structure as a CIF file"),
):
    try:
        cif_content = (await cif_file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CIF file must be UTF-8 encoded text.")

    try:
        graph_data = featurizer.process_input(formula, cif_content).to(device)

        result = ensemble.predict(graph_data)

        gpa_mean = float(10 ** result.mean_log10)
        gpa_lower = float(10 ** (result.mean_log10 - result.std_log10))
        gpa_upper = float(10 ** (result.mean_log10 + result.std_log10))

        return PredictResponse(
            formula=formula,
            log10_bulk_modulus_mean=result.mean_log10,
            log10_bulk_modulus_std=result.std_log10,
            bulk_modulus_gpa_mean=gpa_mean,
            bulk_modulus_gpa_lower=gpa_lower,
            bulk_modulus_gpa_upper=gpa_upper,
            per_seed_log10_bulk_modulus=result.per_seed_log10,
            ensemble_size=len(result.per_seed_log10),
            uncertainty_scope=UNCERTAINTY_SCOPE,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
