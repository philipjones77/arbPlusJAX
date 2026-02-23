from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def as_interval(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.int64)
    if arr.shape[-1] != 2:
        raise ValueError("expected integer intervals with last dimension 2")
    lo = arr[..., 0]
    hi = arr[..., 1]
    return jnp.stack([jnp.minimum(lo, hi), jnp.maximum(lo, hi)], axis=-1)


def interval(lo: jax.Array, hi: jax.Array) -> jax.Array:
    lo_arr = jnp.asarray(lo, dtype=jnp.int64)
    hi_arr = jnp.asarray(hi, dtype=jnp.int64)
    return jnp.stack([jnp.minimum(lo_arr, hi_arr), jnp.maximum(lo_arr, hi_arr)], axis=-1)


def fmpzi_add(a: jax.Array, b: jax.Array) -> jax.Array:
    a = as_interval(a)
    b = as_interval(b)
    return interval(a[..., 0] + b[..., 0], a[..., 1] + b[..., 1])


def fmpzi_sub(a: jax.Array, b: jax.Array) -> jax.Array:
    a = as_interval(a)
    b = as_interval(b)
    return interval(a[..., 0] - b[..., 1], a[..., 1] - b[..., 0])


def fmpzi_add_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return fmpzi_add(a, b)


def fmpzi_sub_batch(a: jax.Array, b: jax.Array) -> jax.Array:
    return fmpzi_sub(a, b)


fmpzi_add_batch_jit = jax.jit(fmpzi_add_batch)
fmpzi_sub_batch_jit = jax.jit(fmpzi_sub_batch)


__all__ = [
    "as_interval",
    "interval",
    "fmpzi_add",
    "fmpzi_sub",
    "fmpzi_add_batch",
    "fmpzi_sub_batch",
    "fmpzi_add_batch_jit",
    "fmpzi_sub_batch_jit",
]
