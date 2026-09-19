import os
from pathlib import Path

import pytest

from tests import structures

APP_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR = APP_ROOT / "weights"
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]


def _weights_available() -> bool:
    return all((WEIGHT_DIR / f"Fusion_best_model_seed{seed}.pth").exists() for seed in ENSEMBLE_SEEDS)


def pytest_collection_modifyitems(config, items):
    # Locally, tests that need the trained checkpoints skip when they're absent. In CI (CI env var set)
    # they are left alone so missing weights fail loudly instead of silently passing.
    if _weights_available() or os.environ.get("CI"):
        return
    skip = pytest.mark.skip(reason="trained checkpoints not found in weights/ (this fails in CI)")
    for item in items:
        if "weights" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def nacl_2atom_cscl_type_cif() -> str:
    return structures.to_cif(structures.nacl_2atom_cscl_type())


@pytest.fixture(scope="session")
def nacl_rocksalt_cif() -> str:
    return structures.to_cif(structures.nacl_rocksalt())


@pytest.fixture(scope="session")
def featurizer():
    from app.featurizer import MaterialsFeaturizer

    return MaterialsFeaturizer()


@pytest.fixture(scope="session")
def real_ensemble():
    import torch

    from app.ensemble import FusionEnsemble

    return FusionEnsemble.from_weight_dir(WEIGHT_DIR, ENSEMBLE_SEEDS, device=torch.device("cpu"))


@pytest.fixture(scope="session")
def client():
    # app.main builds the featurizer and loads all 5 checkpoints at import time, so import lazily.
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
