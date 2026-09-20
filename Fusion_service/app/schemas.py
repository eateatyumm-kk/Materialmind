from pydantic import BaseModel, ConfigDict, Field


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "formula": "NaCl",
                "log10_bulk_modulus_mean": 1.3553,
                "log10_bulk_modulus_std": 0.0155,
                "bulk_modulus_gpa_mean": 22.66,
                "bulk_modulus_gpa_lower": 21.87,
                "bulk_modulus_gpa_upper": 23.48,
                "per_seed_log10_bulk_modulus": {"42": 1.3547, "123": 1.3324, "456": 1.3707, "789": 1.3502, "2024": 1.3685},
                "ensemble_size": 5,
                "uncertainty_scope": "log10_bulk_modulus_std reflects EPISTEMIC/MODEL uncertainty only: ...",
                "status": "success",
            }
        }
    )

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


class ErrorResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"detail": "Invalid or corrupted CIF file format: ..."}})

    detail: str = Field(..., description="Human-readable explanation of what was wrong with the request.")


class HealthResponse(BaseModel):
    status: str = Field(..., description='Always "ok" when the service is up and the ensemble is loaded.')
    ensemble_size: int = Field(..., description="Number of trained models loaded in the ensemble.")
    seeds: list[int] = Field(..., description="Training seeds of the loaded ensemble members.")
    device: str = Field(..., description='Device inference runs on, e.g. "cpu" or "cuda".')
    reference_test_rmse_log10: float = Field(
        ...,
        description=(
            "Typical prediction error (RMSE, in log10 GPa) of the Fusion model on held-out test "
            "materials, for context when interpreting the ensemble spread."
        ),
    )
