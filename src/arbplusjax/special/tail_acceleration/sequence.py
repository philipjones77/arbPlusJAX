from __future__ import annotations

import jax.numpy as jnp


def aitken_delta_squared(sequence: jnp.ndarray) -> jnp.ndarray:
    if sequence.shape[0] < 3:
        raise ValueError("Aitken acceleration requires at least 3 sequence values.")
    s0 = sequence[:-2]
    s1 = sequence[1:-1]
    s2 = sequence[2:]
    delta1 = s1 - s0
    delta2 = s2 - 2.0 * s1 + s0
    eps = jnp.asarray(1e-30, dtype=sequence.dtype)
    return s0 - (delta1 * delta1) / jnp.where(jnp.abs(delta2) > eps, delta2, eps)


def wynn_epsilon(sequence: jnp.ndarray) -> jnp.ndarray:
    if sequence.shape[0] < 3:
        raise ValueError("Wynn epsilon requires at least 3 sequence values.")
    diffs = sequence[1:] - sequence[:-1]
    eps = jnp.asarray(1e-30, dtype=sequence.dtype)
    inv = 1.0 / jnp.where(jnp.abs(diffs) > eps, diffs, eps)
    return sequence[1:-1] + 1.0 / (inv[1:] - inv[:-1])
