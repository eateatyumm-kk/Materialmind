import json
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "nacl.json"

# ~0.23% in GPa: far above float32 cross-platform noise (~1e-5) yet far below the ~0.07 between-seed spread,
# so any real change to weights, scaler, featurizer or un-scaling constants trips it.
TOLERANCE = 1e-3


@pytest.mark.weights
def test_nacl_prediction_matches_golden_values(real_ensemble, featurizer, nacl_2atom_cscl_type_cif):
    golden = json.loads(GOLDEN_PATH.read_text())

    result = real_ensemble.predict(featurizer.process_input(golden["formula"], nacl_2atom_cscl_type_cif))

    assert {str(seed) for seed in result.per_seed_log10} == set(golden["per_seed_log10"])
    for seed, value in result.per_seed_log10.items():
        assert value == pytest.approx(golden["per_seed_log10"][str(seed)], abs=TOLERANCE), f"seed {seed}"
    assert result.mean_log10 == pytest.approx(golden["mean_log10"], abs=TOLERANCE)
    assert result.std_log10 == pytest.approx(golden["std_log10"], abs=TOLERANCE)
