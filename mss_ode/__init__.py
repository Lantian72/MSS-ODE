"""Core physics-constrained ZOH state-space model and recursive EM estimation."""
from .model import (
    zoh_jac,
    objective,
    coefficients,
    step_filter,
    optimize_components,
    forecast_held,
    forecast_conditional,
    reference_arrays,
    fixed_pass,
)
from .detection import (
    pelt,
    scan_chunk,
    ar_pvalue,
    trend_pvalue,
    confirm_references,
)

__all__ = [
    "zoh_jac",
    "objective",
    "coefficients",
    "step_filter",
    "optimize_components",
    "forecast_held",
    "forecast_conditional",
    "reference_arrays",
    "fixed_pass",
    "pelt",
    "scan_chunk",
    "ar_pvalue",
    "trend_pvalue",
    "confirm_references",
]
