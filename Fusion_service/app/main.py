import random
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ensemble import REFERENCE_TEST_RMSE_LOG10, UNCERTAINTY_SCOPE, FusionEnsemble
from app.featurizer import MaterialsFeaturizer
from app.schemas import ErrorResponse, HealthResponse, PredictResponse

# The 5 seeds trained together in `Testing/Fusion/Fusion_test_2.py`, all
# sharing one Config() architecture. `results/Fusion_best_model.pth` (no
# seed suffix, from an earlier wandb sweep) uses a different fc_hidden and
# must never be added here — it fails a strict state_dict load.
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]

API_DESCRIPTION = """
Predicts a material's **bulk modulus K** (in GPa) from its chemical formula and crystal structure, using a
5-model ensemble of Fusion GNNs (crystal graph + Magpie composition features) trained on Materials Project data.

**Web app:** open [`/`](/) for a point-and-click interface with drag-and-drop CIF upload and example materials.

**From code:**

```bash
curl -X POST http://localhost:8000/predict -F "formula=NaCl" -F "cif_file=@structure.cif"
```

### About the uncertainty
`log10_bulk_modulus_std` is the spread across the 5 ensemble members. It captures *model* disagreement only:
it is **not** measurement noise, **not** an out-of-distribution warning, and **not** a calibrated confidence
interval. The model's typical test error (see `/health`) is usually larger than this spread.
"""

TAGS_METADATA = [
    {"name": "prediction", "description": "Predict bulk modulus (with ensemble uncertainty) for a material."},
    {"name": "system", "description": "Service status and model information."},
]

STATIC_DIR = Path(__file__).resolve().parent / "static"


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

app = FastAPI(
    title="MaterialMind — Bulk Modulus Predictor",
    version="1.1",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    swagger_ui_parameters={"displayRequestDuration": True, "defaultModelsExpandDepth": -1, "tryItOutEnabled": True},
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load featurizer & the 5-model ensemble once on startup
featurizer = MaterialsFeaturizer()
ensemble = FusionEnsemble.from_weight_dir(WEIGHT_DIR, seeds=ENSEMBLE_SEEDS, device=device)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="Service status and model info")
def health():
    return HealthResponse(
        status="ok",
        ensemble_size=len(ensemble.models),
        seeds=ENSEMBLE_SEEDS,
        device=str(device),
        reference_test_rmse_log10=REFERENCE_TEST_RMSE_LOG10,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict bulk modulus with uncertainty",
    description=(
        "Upload a crystal structure as a **CIF file** together with the material's **chemical formula**. "
        "The formula should describe the same material as the CIF. Returns the ensemble mean prediction, "
        "its spread across the 5 members, and each member's individual prediction."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "The formula or CIF could not be parsed (invalid formula, corrupted CIF, non-UTF-8 file).",
        },
        500: {"model": ErrorResponse, "description": "Unexpected failure while running the model."},
    },
)
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
