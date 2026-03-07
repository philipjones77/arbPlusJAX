from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def arf_add(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64
    return jnp.asarray(aa, dtype=dtype) + jnp.asarray(bb, dtype=dtype)


def arf_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64
    return jnp.asarray(aa, dtype=dtype) * jnp.asarray(bb, dtype=dtype)


def arf_add_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return arf_add(a, b)


def arf_mul_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return arf_mul(a, b)


arf_add_batch_jit = jax.jit(arf_add_batch)
arf_mul_batch_jit = jax.jit(arf_mul_batch)


__all__ = [
    "arf_add",
    "arf_mul",
    "arf_add_batch",
    "arf_mul_batch",
    "arf_add_batch_jit",
    "arf_mul_batch_jit",
]
