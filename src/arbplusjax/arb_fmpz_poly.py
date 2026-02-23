from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import checks

jax.config.update("jax_enable_x64", True)


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _as_coeffs(coeffs: jax.Array) -> jax.Array:
    arr = di.as_interval(coeffs)
    checks.check_tail_shape(arr, (4, 2), "arb_fmpz_poly._as_coeffs")
    return arr


def arb_fmpz_poly_eval_cubic(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    x = di.as_interval(x)
    c = di.midpoint(coeffs)
    xm = di.midpoint(x)
    v = ((c[..., 3] * xm + c[..., 2]) * xm + c[..., 1]) * xm + c[..., 0]
    finite = jnp.isfinite(v)
    out = di.interval(di._below(v), di._above(v))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


def arb_fmpz_poly_eval_cubic_rigorous(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    x = di.as_interval(x)
    c0 = coeffs[..., 0, :]
    c1 = coeffs[..., 1, :]
    c2 = coeffs[..., 2, :]
    c3 = coeffs[..., 3, :]
    v = di.fast_add(di.fast_mul(c3, x), c2)
    v = di.fast_add(di.fast_mul(v, x), c1)
    v = di.fast_add(di.fast_mul(v, x), c0)
    finite = jnp.isfinite(v[..., 0]) & jnp.isfinite(v[..., 1])
    return jnp.where(finite[..., None], v, _full_interval_like(x))


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_fmpz_poly_eval_cubic_prec(
    coeffs: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_fmpz_poly_eval_cubic(coeffs, x), prec_bits)


def arb_fmpz_poly_eval_cubic_batch(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    x = di.as_interval(x)
    return jax.vmap(arb_fmpz_poly_eval_cubic)(coeffs, x)


def arb_fmpz_poly_eval_cubic_batch_rigorous(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    x = di.as_interval(x)
    return jax.vmap(arb_fmpz_poly_eval_cubic_rigorous)(coeffs, x)


def arb_fmpz_poly_eval_cubic_batch_prec(
    coeffs: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_fmpz_poly_eval_cubic_batch(coeffs, x), prec_bits)


arb_fmpz_poly_eval_cubic_batch_jit = jax.jit(arb_fmpz_poly_eval_cubic_batch)
arb_fmpz_poly_eval_cubic_batch_prec_jit = jax.jit(
    arb_fmpz_poly_eval_cubic_batch_prec, static_argnames=("prec_bits",)
)


__all__ = [
    "arb_fmpz_poly_eval_cubic",
    "arb_fmpz_poly_eval_cubic_rigorous",
    "arb_fmpz_poly_eval_cubic_prec",
    "arb_fmpz_poly_eval_cubic_batch",
    "arb_fmpz_poly_eval_cubic_batch_rigorous",
    "arb_fmpz_poly_eval_cubic_batch_prec",
    "arb_fmpz_poly_eval_cubic_batch_jit",
    "arb_fmpz_poly_eval_cubic_batch_prec_jit",
]
