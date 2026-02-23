from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DI_ULP_FACTOR = jnp.float64(4.440892098500626e-16)
DI_HUGE = jnp.float64(1e300)
DI_TINY = jnp.float64(1e-300)
DEFAULT_PREC_BITS = 53


def as_interval(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got shape {arr.shape}")
    return arr


def interval(lo: jax.Array, hi: jax.Array) -> jax.Array:
    lo_arr = jnp.asarray(lo, dtype=jnp.float64)
    hi_arr = jnp.asarray(hi, dtype=jnp.float64)
    return jnp.stack([lo_arr, hi_arr], axis=-1)


def lower(x: jax.Array) -> jax.Array:
    return as_interval(x)[..., 0]


def upper(x: jax.Array) -> jax.Array:
    return as_interval(x)[..., 1]


def _below(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    t = jnp.abs(x) + DI_TINY
    finite_branch = x - t * DI_ULP_FACTOR
    overflow_branch = jnp.where(jnp.isnan(x), -jnp.inf, DI_HUGE)
    return jnp.where(x <= DI_HUGE, finite_branch, overflow_branch)


def _above(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    t = jnp.abs(x) + DI_TINY
    finite_branch = x + t * DI_ULP_FACTOR
    overflow_branch = jnp.where(jnp.isnan(x), jnp.inf, -DI_HUGE)
    return jnp.where(x >= -DI_HUGE, finite_branch, overflow_branch)


def _binary_step(x: jax.Array, prec_bits: int) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    p = max(int(prec_bits), 1)
    ax = jnp.abs(x)
    exp2 = jnp.floor(jnp.log2(ax))
    return jnp.exp2(exp2 - jnp.float64(p - 1))


def round_down_to_prec(x: jax.Array, prec_bits: int) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    step = _binary_step(x, prec_bits)
    safe = jnp.isfinite(x) & (step > 0.0)
    q = jnp.floor(x / step)
    return jnp.where(safe, q * step, x)


def round_up_to_prec(x: jax.Array, prec_bits: int) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    step = _binary_step(x, prec_bits)
    safe = jnp.isfinite(x) & (step > 0.0)
    q = jnp.ceil(x / step)
    return jnp.where(safe, q * step, x)


def round_interval_outward(x: jax.Array, prec_bits: int) -> jax.Array:
    x = as_interval(x)
    lo = _below(round_down_to_prec(lower(x), prec_bits))
    hi = _above(round_up_to_prec(upper(x), prec_bits))
    return interval(lo, hi)


def neg(x: jax.Array) -> jax.Array:
    x = as_interval(x)
    return interval(-upper(x), -lower(x))


def fast_add(x: jax.Array, y: jax.Array) -> jax.Array:
    x = as_interval(x)
    y = as_interval(y)
    return interval(_below(lower(x) + lower(y)), _above(upper(x) + upper(y)))


def fast_sub(x: jax.Array, y: jax.Array) -> jax.Array:
    x = as_interval(x)
    y = as_interval(y)
    return interval(_below(lower(x) - upper(y)), _above(upper(x) - lower(y)))


def fast_mul(x: jax.Array, y: jax.Array) -> jax.Array:
    x = as_interval(x)
    y = as_interval(y)

    xa, xb = lower(x), upper(x)
    ya, yb = lower(y), upper(y)

    c1 = (xa > 0.0) & (ya > 0.0)
    c2 = (xa > 0.0) & (yb < 0.0)
    c3 = (xb < 0.0) & (ya > 0.0)
    c4 = (xb < 0.0) & (yb < 0.0)

    a1, b1 = xa * ya, xb * yb
    a2, b2 = xb * ya, xa * yb
    a3, b3 = xa * yb, xb * ya
    a4, b4 = xb * yb, xa * ya

    prods = jnp.stack([xa * ya, xa * yb, xb * ya, xb * yb], axis=-1)
    any_nan = jnp.any(jnp.isnan(prods), axis=-1)
    ag = jnp.where(any_nan, -jnp.inf, jnp.min(prods, axis=-1))
    bg = jnp.where(any_nan, jnp.inf, jnp.max(prods, axis=-1))

    a_raw = jnp.where(c1, a1, jnp.where(c2, a2, jnp.where(c3, a3, jnp.where(c4, a4, ag))))
    b_raw = jnp.where(c1, b1, jnp.where(c2, b2, jnp.where(c3, b3, jnp.where(c4, b4, bg))))
    return interval(_below(a_raw), _above(b_raw))


def fast_div(x: jax.Array, y: jax.Array) -> jax.Array:
    x = as_interval(x)
    y = as_interval(y)

    xa, xb = lower(x), upper(x)
    ya, yb = lower(y), upper(y)

    y_pos = ya > 0.0
    y_neg = yb < 0.0
    x_nonneg = xa >= 0.0
    x_nonpos = xb <= 0.0

    a_pos = jnp.where(x_nonneg, xa / yb, jnp.where(x_nonpos, xa / ya, xa / ya))
    b_pos = jnp.where(x_nonneg, xb / ya, jnp.where(x_nonpos, xb / yb, xb / ya))

    a_neg = jnp.where(x_nonneg, xb / yb, jnp.where(x_nonpos, xb / ya, xb / yb))
    b_neg = jnp.where(x_nonneg, xa / ya, jnp.where(x_nonpos, xa / yb, xa / yb))

    a_raw = jnp.where(y_pos, a_pos, jnp.where(y_neg, a_neg, -jnp.inf))
    b_raw = jnp.where(y_pos, b_pos, jnp.where(y_neg, b_neg, jnp.inf))
    return interval(_below(a_raw), _above(b_raw))


def fast_sqr(x: jax.Array) -> jax.Array:
    x = as_interval(x)
    xa, xb = lower(x), upper(x)

    is_nonneg = xa >= 0.0
    is_nonpos = xb <= 0.0

    a_raw = jnp.where(is_nonneg, xa * xa, jnp.where(is_nonpos, xb * xb, 0.0))
    b_cross = jnp.maximum(xa * xa, xb * xb)
    b_raw = jnp.where(is_nonneg, xb * xb, jnp.where(is_nonpos, xa * xa, b_cross))

    a_adj = jnp.where(a_raw != 0.0, _below(a_raw), a_raw)
    b_adj = _above(b_raw)
    return interval(a_adj, b_adj)


def fast_log_nonnegative(x: jax.Array) -> jax.Array:
    x = as_interval(x)
    xa, xb = lower(x), upper(x)
    a = jnp.where(xa <= 0.0, -jnp.inf, _below(jnp.log(xa)))
    b = _above(jnp.log(xb))
    return interval(a, b)


def midpoint(x: jax.Array) -> jax.Array:
    x = as_interval(x)
    return 0.5 * (lower(x) + upper(x))


def ubound_radius(x: jax.Array) -> jax.Array:
    x = as_interval(x)
    return _above((upper(x) - lower(x)) * 0.5)


def contains(outer: jax.Array, inner: jax.Array) -> jax.Array:
    outer = as_interval(outer)
    inner = as_interval(inner)
    return (lower(outer) <= lower(inner)) & (upper(outer) >= upper(inner))


@partial(jax.jit, static_argnames=("prec_bits",))
def fast_add_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return round_interval_outward(fast_add(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def fast_sub_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return round_interval_outward(fast_sub(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def fast_mul_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return round_interval_outward(fast_mul(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def fast_div_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return round_interval_outward(fast_div(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def fast_sqr_prec(x: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return round_interval_outward(fast_sqr(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def fast_log_nonnegative_prec(x: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return round_interval_outward(fast_log_nonnegative(x), prec_bits)


fast_add_jit = jax.jit(fast_add)
fast_sub_jit = jax.jit(fast_sub)
fast_mul_jit = jax.jit(fast_mul)
fast_div_jit = jax.jit(fast_div)
fast_sqr_jit = jax.jit(fast_sqr)
fast_log_nonnegative_jit = jax.jit(fast_log_nonnegative)

batch_fast_add = jax.jit(jax.vmap(fast_add, in_axes=(0, 0)))
batch_fast_sub = jax.jit(jax.vmap(fast_sub, in_axes=(0, 0)))
batch_fast_mul = jax.jit(jax.vmap(fast_mul, in_axes=(0, 0)))
batch_fast_div = jax.jit(jax.vmap(fast_div, in_axes=(0, 0)))
batch_fast_sqr = jax.jit(jax.vmap(fast_sqr, in_axes=0))
batch_fast_log_nonnegative = jax.jit(jax.vmap(fast_log_nonnegative, in_axes=0))


def batch_fast_add_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return jax.vmap(lambda a, b: fast_add_prec(a, b, prec_bits), in_axes=(0, 0))(x, y)


def batch_fast_sub_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return jax.vmap(lambda a, b: fast_sub_prec(a, b, prec_bits), in_axes=(0, 0))(x, y)


def batch_fast_mul_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return jax.vmap(lambda a, b: fast_mul_prec(a, b, prec_bits), in_axes=(0, 0))(x, y)


def batch_fast_div_prec(x: jax.Array, y: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return jax.vmap(lambda a, b: fast_div_prec(a, b, prec_bits), in_axes=(0, 0))(x, y)


def batch_fast_sqr_prec(x: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return jax.vmap(lambda a: fast_sqr_prec(a, prec_bits), in_axes=0)(x)


def batch_fast_log_nonnegative_prec(x: jax.Array, prec_bits: int = DEFAULT_PREC_BITS) -> jax.Array:
    return jax.vmap(lambda a: fast_log_nonnegative_prec(a, prec_bits), in_axes=0)(x)


batch_fast_add_prec_jit = jax.jit(batch_fast_add_prec, static_argnames=("prec_bits",))
batch_fast_sub_prec_jit = jax.jit(batch_fast_sub_prec, static_argnames=("prec_bits",))
batch_fast_mul_prec_jit = jax.jit(batch_fast_mul_prec, static_argnames=("prec_bits",))
batch_fast_div_prec_jit = jax.jit(batch_fast_div_prec, static_argnames=("prec_bits",))
batch_fast_sqr_prec_jit = jax.jit(batch_fast_sqr_prec, static_argnames=("prec_bits",))
batch_fast_log_nonnegative_prec_jit = jax.jit(batch_fast_log_nonnegative_prec, static_argnames=("prec_bits",))
