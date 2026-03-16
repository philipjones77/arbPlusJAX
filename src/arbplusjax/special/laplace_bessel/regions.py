from __future__ import annotations

import jax.numpy as jnp

from ..tail_acceleration import TailRegimeMetadata


def laplace_bessel_k_tail_regime_metadata(nu, z, lam, lower_limit) -> TailRegimeMetadata:
    nu_abs = jnp.abs(jnp.asarray(nu, dtype=jnp.float64))
    z_val = jnp.asarray(z, dtype=jnp.float64)
    lam_val = jnp.asarray(lam, dtype=jnp.float64)
    lower_val = jnp.asarray(lower_limit, dtype=jnp.float64)
    total_decay = jnp.maximum(z_val + lam_val, 0.0)
    near_singularity = lower_val < 0.15
    cancellation_risk = jnp.logical_or((nu_abs > 10.0) & (z_val < 1.0), total_decay < 0.5)
    return TailRegimeMetadata(
        decay_rate=total_decay,
        oscillation_level=0.0,
        near_singularity=near_singularity,
        cancellation_risk=cancellation_risk,
        note="Real Laplace-Bessel-K tail with fragile small-lower and slow combined-decay regimes.",
    )


def choose_laplace_bessel_k_tail_method(
    nu,
    z,
    lam,
    lower_limit,
    *,
    requested_method: str,
) -> str:
    if requested_method == "mpfallback":
        requested_method = "high_precision_refine"
    if requested_method != "auto":
        return requested_method

    values = tuple(jnp.asarray(arg, dtype=jnp.float64) for arg in (nu, z, lam, lower_limit))
    if any(jnp.ndim(val) > 0 for val in values):
        return "quadrature"

    metadata = laplace_bessel_k_tail_regime_metadata(nu, z, lam, lower_limit)
    if bool(metadata.near_singularity) and float(jnp.asarray(z, dtype=jnp.float64)) < 0.5:
        return "high_precision_refine"
    if bool(metadata.cancellation_risk):
        return "high_precision_refine"
    if float(jnp.asarray(lam, dtype=jnp.float64) + jnp.asarray(z, dtype=jnp.float64)) < 0.75:
        return "aitken"
    return "quadrature"
