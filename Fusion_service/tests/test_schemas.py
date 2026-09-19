import pytest
from pydantic import ValidationError

from app.schemas import PredictResponse


def _payload() -> dict:
    return {
        "formula": "NaCl",
        "log10_bulk_modulus_mean": 0.59,
        "log10_bulk_modulus_std": 0.08,
        "bulk_modulus_gpa_mean": 3.88,
        "bulk_modulus_gpa_lower": 3.26,
        "bulk_modulus_gpa_upper": 4.63,
        "per_seed_log10_bulk_modulus": {42: 0.58, 123: 0.5},
        "ensemble_size": 2,
        "uncertainty_scope": "epistemic only",
    }


def test_valid_payload_builds_and_status_defaults_to_success():
    response = PredictResponse(**_payload())
    assert response.status == "success"
    assert response.formula == "NaCl"


@pytest.mark.parametrize("missing", list(_payload()))
def test_omitting_any_required_field_raises(missing):
    payload = _payload()
    del payload[missing]
    with pytest.raises(ValidationError):
        PredictResponse(**payload)


def test_json_string_seed_keys_coerce_to_int():
    payload = _payload()
    payload["per_seed_log10_bulk_modulus"] = {"42": 0.58, "123": 0.5}
    response = PredictResponse(**payload)
    assert set(response.per_seed_log10_bulk_modulus) == {42, 123}


def test_dump_and_revalidate_round_trip_is_lossless():
    original = PredictResponse(**_payload())
    assert PredictResponse.model_validate(original.model_dump()) == original
    assert PredictResponse.model_validate_json(original.model_dump_json()) == original


def test_required_fields_are_declared_in_json_schema():
    required = set(PredictResponse.model_json_schema()["required"])
    expected = set(_payload())
    assert expected <= required
    assert "status" not in required
