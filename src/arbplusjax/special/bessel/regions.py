from __future__ import annotations

import jax.numpy as jnp

from ..tail_acceleration import TailRegimeMetadata


def incomplete_bessel_k_regime_metadata(nu, z, lower_limit) -> TailRegimeMetadata:
    nu_abs = jnp.abs(jnp.asarray(nu, dtype=jnp.float64))
    z_val = jnp.asarray(z, dtype=jnp.float64)
    lower_val = jnp.asarray(lower_limit, dtype=jnp.float64)
    sinh_lower = jnp.sinh(lower_val)
    decay_rate = jnp.maximum(z_val, z_val * jnp.maximum(sinh_lower, 0.0))
    near_singularity = lower_val < 0.1
    cancellation_risk = jnp.logical_or((nu_abs > 10.0) & (z_val < 1.0), (decay_rate < 0.5) & (lower_val < 0.2))
    return TailRegimeMetadata(
        decay_rate=decay_rate,
        oscillation_level=0.0,
        near_singularity=near_singularity,
        cancellation_risk=cancellation_risk,
        note="Real incomplete-K uses the standard exp(-z cosh t) cosh(nu t) tail representation with fragile small-lower and large-order/low-z regimes.",
    )


def incomplete_bessel_i_regime_metadata(nu, z, upper_limit) -> TailRegimeMetadata:
    nu_abs = jnp.abs(jnp.asarray(nu, dtype=jnp.float64))
    z_val = jnp.asarray(z, dtype=jnp.float64)
    upper_val = jnp.asarray(upper_limit, dtype=jnp.float64)
    near_zero = upper_val < 0.15
    near_full_interval = jnp.abs(upper_val - jnp.pi) < 0.15
    cancellation_risk = jnp.logical_or((nu_abs > 10.0) & (jnp.abs(z_val) > 5.0), near_full_interval & (jnp.abs(z_val) > 8.0))
    return TailRegimeMetadata(
        decay_rate=jnp.abs(z_val),
        oscillation_level=nu_abs,
        near_singularity=near_zero,
        cancellation_risk=cancellation_risk,
        note="Angular incomplete-I truncation object over a finite interval with fragile small-upper and near-pi/high-z regimes.",
    )


def choose_incomplete_bessel_k_method(
    nu,
    z,
    lower_limit,
    *,
    requested_method: str,
) -> str:
    if requested_method == "mpfallback":
        requested_method = "high_precision_refine"
    if requested_method != "auto":
        return requested_method

    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)
    lower_v = jnp.asarray(lower_limit, dtype=jnp.float64)
    if any(jnp.ndim(val) > 0 for val in (nu_v, z_v, lower_v)):
        return "quadrature"

    zf = float(z_v)
    lowerf = float(lower_v)
    nuf = abs(float(nu_v))
    scaled_decay = zf * float(jnp.cosh(lower_v))
    metadata = incomplete_bessel_k_regime_metadata(nu, z, lower_limit)
    if metadata.near_singularity and zf < 0.25:
        return "high_precision_refine"
    if metadata.cancellation_risk:
        return "high_precision_refine"
    if (zf < 0.1 and lowerf < 0.1) or (nuf > 12.0 and zf < 0.75):
        return "high_precision_refine"
    if lowerf > 1.0 and scaled_decay >= 20.0 and nuf < 6.0:
        return "recurrence"
    if lowerf > 0.25 and scaled_decay >= 18.0:
        return "asymptotic"
    return "quadrature"


def choose_incomplete_bessel_i_method(
    nu,
    z,
    upper_limit,
    *,
    requested_method: str,
) -> str:
    if requested_method == "mpfallback":
        requested_method = "high_precision_refine"
    if requested_method != "auto":
        return requested_method

    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)
    upper_v = jnp.asarray(upper_limit, dtype=jnp.float64)
    if any(jnp.ndim(val) > 0 for val in (nu_v, z_v, upper_v)):
        return "quadrature"

    metadata = incomplete_bessel_i_regime_metadata(nu, z, upper_limit)
    if bool(metadata.near_singularity):
        return "high_precision_refine"
    if bool(metadata.cancellation_risk):
        return "high_precision_refine"
    if float(jnp.abs(z_v)) > 10.0 and abs(float(upper_v) - float(jnp.pi)) < 0.25:
        return "high_precision_refine"
    return "quadrature"
