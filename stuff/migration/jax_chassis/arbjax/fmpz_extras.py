from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fmpz_extras_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.int64) + jnp.asarray(b, dtype=jnp.int64)


def fmpz_extras_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.int64) * jnp.asarray(b, dtype=jnp.int64)


def fmpz_extras_add_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return fmpz_extras_add(a, b)


def fmpz_extras_mul_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return fmpz_extras_mul(a, b)


fmpz_extras_add_batch_jit = jax.jit(fmpz_extras_add_batch)
fmpz_extras_mul_batch_jit = jax.jit(fmpz_extras_mul_batch)


__all__ = [
    "fmpz_extras_add",
    "fmpz_extras_mul",
    "fmpz_extras_add_batch",
    "fmpz_extras_mul_batch",
    "fmpz_extras_add_batch_jit",
    "fmpz_extras_mul_batch_jit",
]
