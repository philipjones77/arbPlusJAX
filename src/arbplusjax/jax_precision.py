from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def reduction_dtype(x: jax.Array) -> jnp.dtype:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return jnp.complex128
    return jnp.float64


def safe_sum(x: jax.Array, axis=None, keepdims: bool = False) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.sum(xx.astype(reduction_dtype(xx)), axis=axis, keepdims=keepdims)


def safe_mean(x: jax.Array, axis=None, keepdims: bool = False) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.mean(xx.astype(reduction_dtype(xx)), axis=axis, keepdims=keepdims)


def safe_dot(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = reduction_dtype(aa if aa.size else bb)
    return jnp.sum(aa.astype(dtype) * bb.astype(dtype))


def safe_vdot_real(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    dtype = reduction_dtype(aa if aa.size else bb)
    return jnp.vdot(aa.astype(dtype), bb.astype(dtype)).real


def safe_norm(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.sqrt(jnp.maximum(safe_vdot_real(xx, xx), jnp.asarray(0.0, dtype=jnp.float64)))


def kahan_sum(x: jax.Array) -> jax.Array:
    xx = jnp.ravel(jnp.asarray(x))
    dtype = reduction_dtype(xx)
    zeros = jnp.asarray(0.0 + 0.0j if jnp.issubdtype(dtype, jnp.complexfloating) else 0.0, dtype=dtype)

    def body(carry, xi):
        s, c = carry
        y = xi.astype(dtype) - c
        t = s + y
        c = (t - s) - y
        s = t
        return (s, c), None

    (s, _), _ = jax.lax.scan(body, (zeros, zeros), xx)
    return s


def safe_logsumexp(v: jax.Array, axis: int = -1, keepdims: bool = False) -> jax.Array:
    vv = jnp.asarray(v)
    m = jnp.max(vv.astype(vv.dtype), axis=axis, keepdims=True)
    acc = safe_sum(jnp.exp(vv.astype(vv.dtype) - m), axis=axis, keepdims=True)
    out = m.astype(reduction_dtype(m)) + jnp.log(acc)
    if keepdims:
        return out
    return jnp.squeeze(out, axis=axis)


def all_finite(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return jnp.all(jnp.isfinite(jnp.real(xx)) & jnp.isfinite(jnp.imag(xx)))
    return jnp.all(jnp.isfinite(xx))


__all__ = [
    "reduction_dtype",
    "safe_sum",
    "safe_mean",
    "safe_dot",
    "safe_vdot_real",
    "safe_norm",
    "kahan_sum",
    "safe_logsumexp",
    "all_finite",
]
