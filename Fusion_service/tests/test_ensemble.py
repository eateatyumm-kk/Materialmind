import hashlib
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from app.ensemble import TARGET_MEAN, TARGET_STD, FusionEnsemble
from app.model import Config, HybridFiLMGatedGAT

CPU = torch.device("cpu")


class _ConstantModel(nn.Module):
    """Stub ensemble member that ignores its input and emits a fixed z-scored prediction."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, data):
        return torch.tensor([self.value])


def _tiny_graph() -> Data:
    n = 4
    src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j], strict=True)
    return Data(
        x=torch.randn(n, 6),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.rand(len(src), 2) * 4.0,
        tabular_features=torch.randn(1, 132),
        batch=torch.zeros(n, dtype=torch.long),
    )


def _write_checkpoints(directory, seeds, config_cls=Config):
    for seed in seeds:
        torch.manual_seed(seed)
        torch.save(HybridFiLMGatedGAT(config_cls()).state_dict(), directory / f"Fusion_best_model_seed{seed}.pth")


# ---------------- hermetic: aggregation maths ----------------
def test_predict_unscales_each_member_and_aggregates_with_sample_std():
    scaled = [-1.0, -0.5, 0.0, 0.5, 1.5]
    seeds = [1, 2, 3, 4, 5]
    ensemble = FusionEnsemble(models=[_ConstantModel(v) for v in scaled], seeds=seeds, device=CPU)

    result = ensemble.predict(Data())

    expected = [v * TARGET_STD + TARGET_MEAN for v in scaled]
    assert result.per_seed_log10 == pytest.approx(dict(zip(seeds, expected, strict=True)))
    assert result.mean_log10 == pytest.approx(np.mean(expected))
    assert result.std_log10 == pytest.approx(TARGET_STD * np.std(scaled, ddof=1))
    assert result.std_log10 != pytest.approx(TARGET_STD * np.std(scaled, ddof=0))


# ---------------- hermetic: loading ----------------
def test_from_weight_dir_loads_distinct_eval_mode_members(tmp_path):
    seeds = [1, 2, 3, 4, 5]
    _write_checkpoints(tmp_path, seeds)

    ensemble = FusionEnsemble.from_weight_dir(tmp_path, seeds, CPU)

    assert len(ensemble.models) == 5
    assert all(not model.training for model in ensemble.models)
    result = ensemble.predict(_tiny_graph())
    assert all(math.isfinite(v) for v in result.per_seed_log10.values())
    assert len(set(result.per_seed_log10.values())) == 5


def test_missing_checkpoints_raise_file_not_found_naming_the_seeds(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"\[1, 2\]"):
        FusionEnsemble.from_weight_dir(tmp_path, [1, 2], CPU)


def test_architecture_mismatch_fails_strict_load(tmp_path):
    class NarrowerConfig(Config):
        fc_hidden = 64

    _write_checkpoints(tmp_path, [1], config_cls=NarrowerConfig)
    with pytest.raises(RuntimeError):
        FusionEnsemble.from_weight_dir(tmp_path, [1], CPU)


# ---------------- real trained checkpoints ----------------
@pytest.mark.weights
def test_real_ensemble_members_are_five_distinct_networks(real_ensemble):
    # Regression guard: "the same checkpoint accidentally loaded into every slot" would give zero spread.
    assert len(real_ensemble.models) == 5
    digests = {
        hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        for model in real_ensemble.models
    }
    assert len(digests) == 5
    assert all(not model.training for model in real_ensemble.models)


@pytest.mark.weights
def test_real_ensemble_predictions_are_finite_and_pairwise_distinct(real_ensemble, featurizer, nacl_2atom_cscl_type_cif):
    result = real_ensemble.predict(featurizer.process_input("NaCl", nacl_2atom_cscl_type_cif))
    values = list(result.per_seed_log10.values())
    assert len(values) == 5
    assert all(math.isfinite(v) for v in values)
    assert len(set(values)) == 5
    assert 0 < result.std_log10 < 0.5
