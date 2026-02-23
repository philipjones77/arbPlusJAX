from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def as_bool_mat2x2(a: jax.Array) -> jax.Array:
    arr = jnp.asarray(a, dtype=jnp.uint8)
    if arr.shape[-1] != 4:
        raise ValueError("expected flattened 2x2 matrices with last dimension 4")
    return arr & jnp.uint8(1)


def bool_mat_2x2_det(a: jax.Array) -> jax.Array:
    a = as_bool_mat2x2(a)
    a00 = a[..., 0]
    a01 = a[..., 1]
    a10 = a[..., 2]
    a11 = a[..., 3]
    return jnp.bitwise_xor(jnp.bitwise_and(a00, a11), jnp.bitwise_and(a01, a10))


def bool_mat_2x2_trace(a: jax.Array) -> jax.Array:
    a = as_bool_mat2x2(a)
    a00 = a[..., 0]
    a11 = a[..., 3]
    return jnp.bitwise_xor(a00, a11) & jnp.uint8(1)


def bool_mat_2x2_det_batch(a: jax.Array) -> jax.Array:
    return bool_mat_2x2_det(a)


def bool_mat_2x2_trace_batch(a: jax.Array) -> jax.Array:
    return bool_mat_2x2_trace(a)


bool_mat_2x2_det_batch_jit = jax.jit(bool_mat_2x2_det_batch)
bool_mat_2x2_trace_batch_jit = jax.jit(bool_mat_2x2_trace_batch)


__all__ = [
    "as_bool_mat2x2",
    "bool_mat_2x2_det",
    "bool_mat_2x2_trace",
    "bool_mat_2x2_det_batch",
    "bool_mat_2x2_trace_batch",
    "bool_mat_2x2_det_batch_jit",
    "bool_mat_2x2_trace_batch_jit",
]
