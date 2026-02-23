from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def dlog_log1p(x: jax.Array) -> jax.Array:
    return jnp.log1p(jnp.asarray(x, dtype=jnp.float64))


def dlog_log1p_batch(x: jax.Array) -> jax.Array:
    return dlog_log1p(x)


dlog_log1p_batch_jit = jax.jit(dlog_log1p_batch)


__all__ = [
    "dlog_log1p",
    "dlog_log1p_batch",
    "dlog_log1p_batch_jit",
]
