from __future__ import annotations

import jax
import jax.numpy as jnp



def acf_add(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    elif not jnp.issubdtype(dtype, jnp.complexfloating):
        dtype = jnp.complex128
    return jnp.asarray(aa, dtype=dtype) + jnp.asarray(bb, dtype=dtype)


def acf_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = jnp.result_type(aa, bb)
    if jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    elif not jnp.issubdtype(dtype, jnp.complexfloating):
        dtype = jnp.complex128
    return jnp.asarray(aa, dtype=dtype) * jnp.asarray(bb, dtype=dtype)


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
