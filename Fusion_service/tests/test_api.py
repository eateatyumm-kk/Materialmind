import math

import pytest

from app.ensemble import UNCERTAINTY_SCOPE
from app.schemas import PredictResponse

# Every test here needs app.main, which loads the real checkpoints at import time.
pytestmark = pytest.mark.weights


def _post(client, formula, cif, filename="structure.cif"):
    data = {} if formula is None else {"formula": formula}
    files = {} if cif is None else {"cif_file": (filename, cif, "text/plain")}
    return client.post("/predict", data=data, files=files or None)


def test_predict_happy_path_returns_valid_response(client, nacl_2atom_cscl_type_cif):
    response = _post(client, "NaCl", nacl_2atom_cscl_type_cif.encode())

    assert response.status_code == 200
    body = response.json()
    parsed = PredictResponse.model_validate(body)
    assert parsed.status == "success"
    assert parsed.formula == "NaCl"
    assert parsed.ensemble_size == 5
    assert len(body["per_seed_log10_bulk_modulus"]) == 5
    assert parsed.uncertainty_scope == UNCERTAINTY_SCOPE
    assert parsed.bulk_modulus_gpa_lower < parsed.bulk_modulus_gpa_mean < parsed.bulk_modulus_gpa_upper
    assert parsed.bulk_modulus_gpa_mean == pytest.approx(10**parsed.log10_bulk_modulus_mean)
    assert parsed.log10_bulk_modulus_std > 0
    # 10**x is convex, so the GPa interval is asymmetric: the upper gap exceeds the lower gap.
    assert parsed.bulk_modulus_gpa_upper - parsed.bulk_modulus_gpa_mean > (
        parsed.bulk_modulus_gpa_mean - parsed.bulk_modulus_gpa_lower
    )


def test_predict_accepts_a_true_rocksalt_structure(client, nacl_rocksalt_cif):
    response = _post(client, "NaCl", nacl_rocksalt_cif.encode())

    assert response.status_code == 200
    parsed = PredictResponse.model_validate(response.json())
    assert math.isfinite(parsed.bulk_modulus_gpa_mean) and parsed.bulk_modulus_gpa_mean > 0
    assert parsed.bulk_modulus_gpa_lower < parsed.bulk_modulus_gpa_mean < parsed.bulk_modulus_gpa_upper


def test_missing_formula_is_422(client, nacl_2atom_cscl_type_cif):
    assert _post(client, None, nacl_2atom_cscl_type_cif.encode()).status_code == 422


def test_missing_cif_file_is_422(client):
    assert _post(client, "NaCl", None).status_code == 422


def test_non_utf8_upload_is_400(client):
    response = _post(client, "NaCl", b"\xff\xfe\x00")
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_garbage_cif_is_400(client):
    assert _post(client, "NaCl", b"this is not a cif").status_code == 400


def test_bad_formula_is_400(client, nacl_2atom_cscl_type_cif):
    assert _post(client, "Xx9", nacl_2atom_cscl_type_cif.encode()).status_code == 400


def test_unexpected_failure_is_500_and_value_error_is_400(client, nacl_2atom_cscl_type_cif, monkeypatch):
    import app.main as main_module

    def _boom(_data):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(main_module.ensemble, "predict", _boom)
    response = _post(client, "NaCl", nacl_2atom_cscl_type_cif.encode())
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Prediction failed")

    def _bad_input(_data):
        raise ValueError("bad input")

    monkeypatch.setattr(main_module.ensemble, "predict", _bad_input)
    assert _post(client, "NaCl", nacl_2atom_cscl_type_cif.encode()).status_code == 400


def test_openapi_schema_exposes_predict(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]
