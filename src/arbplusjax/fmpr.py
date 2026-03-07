from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fmpr_add(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64
    return jnp.asarray(aa, dtype=dtype) + jnp.asarray(bb, dtype=dtype)


def fmpr_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64
    return jnp.asarray(aa, dtype=dtype) * jnp.asarray(bb, dtype=dtype)


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
