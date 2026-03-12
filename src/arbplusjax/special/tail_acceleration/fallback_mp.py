from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate

from .diagnostics import TailEvaluationDiagnostics
from .quadrature import finite_panel_tail_quadrature


def mp_tail_integral_fallback(
    integrand: Callable[[float], float],
    lower_limit: float,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
) -> tuple[float, TailEvaluationDiagnostics]:
    note = "Host-side fallback only; true arbitrary-precision fallback is still pending."
    try:
        value, err = integrate.quad(lambda t: float(integrand(t)), float(lower_limit), np.inf, limit=200)
        diagnostics = TailEvaluationDiagnostics(
            method="mpfallback",
            chunk_count=1,
            panel_count=0,
            recurrence_steps=0,
            estimated_tail_remainder=float(err),
            instability_flags=(),
            fallback_used=True,
            precision_warning=True,
            note=note,
        )
        return float(value), diagnostics
    except Exception:
        value, diagnostics = finite_panel_tail_quadrature(
            integrand,
            lower_limit,
            panel_width=panel_width,
            max_panels=max_panels * 2,
            samples_per_panel=max(samples_per_panel * 2, 64),
            method="mpfallback",
            fallback_used=True,
            precision_warning=True,
            note=note,
        )
        return float(value), diagnostics
