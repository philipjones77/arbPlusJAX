from __future__ import annotations

import jax.numpy as jnp

from ..tail_acceleration import TailRegimeMetadata


def incomplete_bessel_k_regime_metadata(nu, z, lower_limit) -> TailRegimeMetadata:
    nu_abs = jnp.abs(jnp.asarray(nu, dtype=jnp.float64))
    z_val = jnp.asarray(z, dtype=jnp.float64)
    lower_val = jnp.asarray(lower_limit, dtype=jnp.float64)
    sinh_lower = jnp.sinh(lower_val)
    decay_rate = jnp.maximum(z_val, z_val * jnp.maximum(sinh_lower, 0.0))
    return TailRegimeMetadata(
        decay_rate=decay_rate,
        oscillation_level=0.0,
        near_singularity=False,
        cancellation_risk=False,
        note="Real incomplete-K uses the standard exp(-z cosh t) cosh(nu t) tail representation.",
    )
