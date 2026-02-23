from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fmpr_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.float64) + jnp.asarray(b, dtype=jnp.float64)


def fmpr_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.float64) * jnp.asarray(b, dtype=jnp.float64)


def fmpr_add_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return fmpr_add(a, b)


def fmpr_mul_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return fmpr_mul(a, b)


fmpr_add_batch_jit = jax.jit(fmpr_add_batch)
fmpr_mul_batch_jit = jax.jit(fmpr_mul_batch)


__all__ = [
    "fmpr_add",
    "fmpr_mul",
    "fmpr_add_batch",
    "fmpr_mul_batch",
    "fmpr_add_batch_jit",
    "fmpr_mul_batch_jit",
]
