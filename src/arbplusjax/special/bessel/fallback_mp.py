from __future__ import annotations

import jax.numpy as jnp

from ..tail_acceleration import TailEvaluationDiagnostics
from ..tail_acceleration.fallback_mp import mp_tail_integral_fallback
from .incomplete_bessel_base import incomplete_bessel_k_integrand


def incomplete_bessel_k_mpfallback(
    nu,
    z,
    lower_limit,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
) -> tuple[jnp.ndarray, TailEvaluationDiagnostics]:
    integrand = incomplete_bessel_k_integrand(nu, z)
    value, diagnostics = mp_tail_integral_fallback(
        integrand,
        lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jnp.asarray(value, dtype=jnp.float64), diagnostics
