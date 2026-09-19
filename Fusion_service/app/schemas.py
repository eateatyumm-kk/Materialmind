from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    formula: str
    log10_bulk_modulus_mean: float = Field(
        ..., description="Mean of log10(K) (K in GPa) across the 5-seed ensemble."
    )
    log10_bulk_modulus_std: float = Field(
        ...,
        description=(
            "Sample std (ddof=1) of log10(K) across the 5-seed ensemble — epistemic "
            "spread across independently-trained models, not a calibrated confidence "
            "interval. See `uncertainty_scope`."
        ),
    )
    bulk_modulus_gpa_mean: float = Field(
        ...,
        description=(
            "10 ** log10_bulk_modulus_mean — the geometric mean of the 5 members' "
            "individual GPa predictions."
        ),
    )
    bulk_modulus_gpa_lower: float = Field(
        ...,
        description=(
            "10 ** (log10_bulk_modulus_mean - log10_bulk_modulus_std). Note this interval "
            "is NOT symmetric around bulk_modulus_gpa_mean in GPa units — 10**x is "
            "monotonic but convex, so the upper gap is always larger than the lower gap "
            "for the same log-space std, growing with the predicted magnitude."
        ),
    )
    bulk_modulus_gpa_upper: float = Field(
        ..., description="10 ** (log10_bulk_modulus_mean + log10_bulk_modulus_std)."
    )
    per_seed_log10_bulk_modulus: dict[int, float] = Field(
        ..., description="Individual log10(K) prediction from each ensemble member, keyed by seed."
    )
    ensemble_size: int = Field(..., description="Number of models actually used for this prediction.")
    uncertainty_scope: str = Field(
        ...,
        description="What the reported uncertainty does and does not capture — see field value.",
    )
    status: str = "success"
