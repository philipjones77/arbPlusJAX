from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def acf_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.complex128) + jnp.asarray(b, dtype=jnp.complex128)


def acf_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.asarray(a, dtype=jnp.complex128) * jnp.asarray(b, dtype=jnp.complex128)


def acf_add_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return acf_add(a, b)


def acf_mul_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return acf_mul(a, b)


acf_add_batch_jit = jax.jit(acf_add_batch)
acf_mul_batch_jit = jax.jit(acf_mul_batch)


__all__ = [
    "acf_add",
    "acf_mul",
    "acf_add_batch",
    "acf_mul_batch",
    "acf_add_batch_jit",
    "acf_mul_batch_jit",
]
