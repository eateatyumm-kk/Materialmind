from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from app.model import Config, HybridFiLMGatedGAT

# The target (log10 bulk modulus) was z-score normalized during training
# using statistics from the training split — see
# `Testing/Fusion/Fusion_model.py::load_split_graphs`. All 5 seed
# checkpoints share these same constants: y_mean/y_std were computed
# once from data/graphs.pt + splits/train_ids.csv (n_train=7021) before
# the per-seed training loop, not per seed. The model predicts in that
# scaled space, so every member's raw output must be un-scaled with
# these same constants to recover log10(K).
TARGET_MEAN = 1.8850252628326416
TARGET_STD = 0.3802662193775177

# Held-out test RMSE of the Fusion model in log10(GPa): mean over the 5 seeds, from
# results/Fusion_5seed_summary.csv (RMSE_mean = 0.09840). Shown in the UI and /health so users can compare
# the ensemble spread with the model's real error. Update this after retraining.
REFERENCE_TEST_RMSE_LOG10 = 0.0984

# What this uncertainty is and is not. Returned verbatim on every response
# (see `uncertainty_scope` in schemas.py) so the semantics travel with the
# number, not just in code comments.
UNCERTAINTY_SCOPE = (
    "log10_bulk_modulus_std reflects EPISTEMIC/MODEL uncertainty only: the "
    "disagreement among 5 independently-trained networks of the same "
    "architecture on the same input. It is NOT aleatoric/measurement noise "
    "in the DFT training labels, NOT an out-of-distribution signal (all 5 "
    "members share one architecture, one training set, and one featurizer, "
    "so they can agree closely while being systematically wrong together on "
    "an unfamiliar material — a low std does not mean high accuracy), and "
    "NOT a statistically calibrated confidence interval (n=5 is a "
    "descriptive spread, not a rigorous 68%/95% coverage guarantee)."
)


@dataclass
class EnsemblePrediction:
    mean_log10: float
    std_log10: float
    per_seed_log10: dict[int, float]


class FusionEnsemble:
    """Runs a crystal graph through all 5 independently-seeded Fusion
    checkpoints and aggregates their predictions.

    This is a deep ensemble, not MC-Dropout: `HybridFiLMGatedGAT.tabular_mlp`
    contains BatchNorm1d, which would use degenerate live batch-of-1
    statistics if any member were switched to `.train()` for dropout
    sampling. Keeping every member in `.eval()` (frozen running BatchNorm
    stats) avoids that entirely, so every member here stays in eval mode.
    """

    def __init__(self, models: list[HybridFiLMGatedGAT], seeds: list[int], device: torch.device):
        self.models = models
        self.seeds = seeds
        self.device = device

    @classmethod
    def from_weight_dir(cls, weight_dir: Path, seeds: list[int], device: torch.device) -> "FusionEnsemble":
        missing = [
            seed for seed in seeds
            if not (weight_dir / f"Fusion_best_model_seed{seed}.pth").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing Fusion checkpoints for seeds {missing} in {weight_dir}. "
                f"Expected Fusion_best_model_seed<seed>.pth for each of {seeds}."
            )

        models = []
        for seed in seeds:
            model = HybridFiLMGatedGAT(Config())
            state_dict = torch.load(weight_dir / f"Fusion_best_model_seed{seed}.pth", map_location=device)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            models.append(model)

        return cls(models=models, seeds=seeds, device=device)

    def predict(self, data: Data) -> EnsemblePrediction:
        """Runs every ensemble member on the SAME featurized graph.

        `HybridFiLMGatedGAT.forward` only ever creates local tensors from
        `data` (e.g. `data.x.float()`) and never writes back onto it, so
        reusing one `Data` object across all 5 models is safe and avoids
        re-featurizing per member.
        """
        per_seed_log10 = {}

        with torch.no_grad():
            for seed, model in zip(self.seeds, self.models, strict=True):
                pred_scaled = model(data)
                log10_k = (pred_scaled.item() * TARGET_STD) + TARGET_MEAN
                per_seed_log10[seed] = log10_k

        values = np.array(list(per_seed_log10.values()))

        return EnsemblePrediction(
            mean_log10=float(values.mean()),
            std_log10=float(values.std(ddof=1)),
            per_seed_log10=per_seed_log10,
        )
