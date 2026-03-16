from __future__ import annotations

import jax.numpy as jnp

from ..tail_acceleration import TailEvaluationDiagnostics
from .incomplete_bessel_base import incomplete_bessel_k_integrand


def incomplete_bessel_k_asymptotic(
    nu,
    z,
    lower_limit,
) -> tuple[jnp.ndarray, TailEvaluationDiagnostics]:
    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)
    lower_v = jnp.asarray(lower_limit, dtype=jnp.float64)
    integrand = incomplete_bessel_k_integrand(nu_v, z_v)
    f0 = integrand(lower_v)
    denom = z_v * jnp.sinh(lower_v) - nu_v * jnp.tanh(nu_v * lower_v)
    clipped = jnp.abs(denom) <= 1e-12
    safe_denom = jnp.where(clipped, jnp.asarray(1e-12, dtype=jnp.float64), denom)
    value = f0 / safe_denom
    instability_flags: list[str] = []
    if bool(clipped):
        instability_flags.append("small_endpoint_denominator")
    if bool(lower_v < 0.2):
        instability_flags.append("small_lower_limit")
    diagnostics = TailEvaluationDiagnostics(
        method="asymptotic",
        chunk_count=0,
        panel_count=0,
        recurrence_steps=0,
        estimated_tail_remainder=jnp.abs(value) * jnp.asarray(5e-2, dtype=jnp.float64),
        instability_flags=tuple(instability_flags),
        fallback_used=False,
        precision_warning=bool(instability_flags),
        note="Endpoint Laplace asymptotic for large z*cosh(a) and positive lower limit.",
    )
    return value, diagnostics
