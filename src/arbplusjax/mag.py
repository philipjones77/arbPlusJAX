from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def mag_add(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64
    return jnp.asarray(aa, dtype=dtype) + jnp.asarray(bb, dtype=dtype)


def mag_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64
    return jnp.asarray(aa, dtype=dtype) * jnp.asarray(bb, dtype=dtype)


def mag_add_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return mag_add(a, b)


def mag_mul_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return mag_mul(a, b)


mag_add_batch_jit = jax.jit(mag_add_batch)
mag_mul_batch_jit = jax.jit(mag_mul_batch)


__all__ = [
    "mag_add",
    "mag_mul",
    "mag_add_batch",
    "mag_mul_batch",
    "mag_add_batch_jit",
    "mag_mul_batch_jit",
]
