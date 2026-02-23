from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import double_interval as di

jax.config.update("jax_enable_x64", True)


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _as_coeffs(coeffs: jax.Array) -> jax.Array:
    arr = di.as_interval(coeffs)
    if arr.shape[-2:] != (4, 2):
        raise ValueError(f"Expected coeffs shape (..., 4, 2), got {arr.shape}")
    return arr


def arb_poly_eval_cubic(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    x = di.as_interval(x)
    c = di.midpoint(coeffs)
    xm = di.midpoint(x)
    v = ((c[..., 3] * xm + c[..., 2]) * xm + c[..., 1]) * xm + c[..., 0]
    finite = jnp.isfinite(v)
    out = di.interval(di._below(v), di._above(v))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_poly_eval_cubic_prec(
    coeffs: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_poly_eval_cubic(coeffs, x), prec_bits)


def arb_poly_eval_cubic_batch(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    x = di.as_interval(x)
    return jax.vmap(arb_poly_eval_cubic)(coeffs, x)


def arb_poly_eval_cubic_batch_prec(
    coeffs: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_poly_eval_cubic_batch(coeffs, x), prec_bits)


arb_poly_eval_cubic_batch_jit = jax.jit(arb_poly_eval_cubic_batch)
arb_poly_eval_cubic_batch_prec_jit = jax.jit(arb_poly_eval_cubic_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "arb_poly_eval_cubic",
    "arb_poly_eval_cubic_prec",
    "arb_poly_eval_cubic_batch",
    "arb_poly_eval_cubic_batch_prec",
    "arb_poly_eval_cubic_batch_jit",
    "arb_poly_eval_cubic_batch_prec_jit",
]
