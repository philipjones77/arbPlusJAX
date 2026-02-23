from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def mag_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.float64) + jnp.asarray(b, dtype=jnp.float64)


def mag_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.float64) * jnp.asarray(b, dtype=jnp.float64)


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
