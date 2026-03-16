from __future__ import annotations

import jax.numpy as jnp

from ..tail_acceleration import TailRegimeMetadata


def incomplete_gamma_upper_regime_metadata(s, z) -> TailRegimeMetadata:
    s_val = jnp.asarray(s, dtype=jnp.float64)
    z_val = jnp.asarray(z, dtype=jnp.float64)
    small_lower_limit = z_val < 0.2
    near_transition = jnp.abs(z_val - jnp.maximum(s_val - 1.0, 0.0)) < 0.5 * (jnp.sqrt(jnp.maximum(s_val, 1.0)) + 1.0)
    cancellation_risk = jnp.logical_or((s_val > 8.0) & (z_val < s_val), small_lower_limit & (s_val < 1.5))
    return TailRegimeMetadata(
        decay_rate=jnp.maximum(z_val, 0.0),
        oscillation_level=0.0,
        near_singularity=small_lower_limit,
        cancellation_risk=jnp.logical_or(cancellation_risk, near_transition),
        note="Real incomplete-gamma upper tail with fragile small-lower-limit and transition-zone regimes.",
    )


def choose_incomplete_gamma_upper_method(
    s,
    z,
    *,
    requested_method: str,
) -> str:
    if requested_method == "mpfallback":
        requested_method = "high_precision_refine"
    if requested_method != "auto":
        return requested_method

    s_v = jnp.asarray(s, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)
    if any(jnp.ndim(val) > 0 for val in (s_v, z_v)):
        return "quadrature"

    metadata = incomplete_gamma_upper_regime_metadata(s, z)
    if bool(metadata.near_singularity) and float(s_v) < 1.5:
        return "high_precision_refine"
    if bool(metadata.cancellation_risk):
        return "high_precision_refine"
    if float(z_v) >= 1.0:
        return "quadrature"
    return "aitken"
