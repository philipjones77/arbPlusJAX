from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import arb_core
from . import double_interval as di
from . import checks

jax.config.update("jax_enable_x64", True)

_INTEGRANDS = ("exp", "sin", "cos")


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _eval_integrand(x: jax.Array, integrand: str) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "arb_calc._eval_integrand")
    if integrand == "exp":
        return jnp.exp(x)
    if integrand == "sin":
        return jnp.sin(x)
    if integrand == "cos":
        return jnp.cos(x)
    return jnp.exp(x)


def _eval_integrand_interval(x: jax.Array, integrand: str) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "arb_calc._eval_integrand_interval")
    if integrand == "exp":
        return arb_core.arb_exp(x)
    if integrand == "sin":
        return arb_core.arb_sin(x)
    if integrand == "cos":
        return arb_core.arb_cos(x)
    return arb_core.arb_exp(x)


def _integrate_line_midpoint(a: jax.Array, b: jax.Array, integrand: str, n_steps: int) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    if n_steps <= 0:
        n_steps = 1
    am = di.midpoint(a)
    bm = di.midpoint(b)
    delta = bm - am
    ts = (jnp.arange(n_steps, dtype=jnp.float64) + 0.5) / jnp.float64(n_steps)
    xs = am + delta * ts
    fx = _eval_integrand(xs, integrand)
    out = jnp.sum(fx) * delta / jnp.float64(n_steps)
    finite = jnp.isfinite(out)
    out_interval = di.interval(di._below(out), di._above(out))
    return jnp.where(finite[..., None], out_interval, _full_interval_like(a))


def _integrate_line_interval(a: jax.Array, b: jax.Array, integrand: str, n_steps: int) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    if n_steps <= 0:
        n_steps = 1
    delta = di.fast_sub(b, a)
    ts = (jnp.arange(n_steps, dtype=jnp.float64) + 0.5) / jnp.float64(n_steps)

    def sample(t):
        t_iv = di.interval(t, t)
        xt = di.fast_add(a, di.fast_mul(delta, t_iv))
        return _eval_integrand_interval(xt, integrand)

    vals = jax.vmap(sample)(ts)
    lo = jnp.sum(vals[..., 0], axis=0)
    hi = jnp.sum(vals[..., 1], axis=0)
    sum_iv = di.interval(di._below(lo), di._above(hi))
    scale = di.interval(jnp.float64(1.0 / n_steps), jnp.float64(1.0 / n_steps))
    out = di.fast_mul(sum_iv, di.fast_mul(delta, scale))
    finite = jnp.isfinite(out[..., 0]) & jnp.isfinite(out[..., 1])
    return jnp.where(finite[..., None], out, _full_interval_like(a))


def _intersect_or_hull(x: jax.Array, y: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    y = di.as_interval(y)
    lo = jnp.maximum(x[..., 0], y[..., 0])
    hi = jnp.minimum(x[..., 1], y[..., 1])
    overlap = lo <= hi
    lo = jnp.where(overlap, lo, jnp.minimum(x[..., 0], y[..., 0]))
    hi = jnp.where(overlap, hi, jnp.maximum(x[..., 1], y[..., 1]))
    return di.interval(lo, hi)


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def arb_calc_integrate_line(a: jax.Array, b: jax.Array, integrand: str = "exp", n_steps: int = 64) -> jax.Array:
    return _integrate_line_midpoint(a, b, integrand, n_steps)


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def arb_calc_integrate_line_rigorous(a: jax.Array, b: jax.Array, integrand: str = "exp", n_steps: int = 64) -> jax.Array:
    coarse = _integrate_line_interval(a, b, integrand, n_steps)
    fine = _integrate_line_interval(a, b, integrand, max(1, n_steps * 2))
    return _intersect_or_hull(coarse, fine)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def arb_calc_integrate_line_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_calc_integrate_line(a, b, integrand, n_steps), prec_bits)


def arb_calc_integrate_line_batch(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    return jax.vmap(lambda ai, bi: arb_calc_integrate_line(ai, bi, integrand, n_steps))(a, b)


def arb_calc_integrate_line_batch_rigorous(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    return jax.vmap(lambda ai, bi: arb_calc_integrate_line_rigorous(ai, bi, integrand, n_steps))(a, b)


def arb_calc_integrate_line_batch_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        arb_calc_integrate_line_batch(a, b, integrand, n_steps), prec_bits
    )


arb_calc_integrate_line_batch_jit = jax.jit(arb_calc_integrate_line_batch, static_argnames=("integrand", "n_steps"))
arb_calc_integrate_line_batch_prec_jit = jax.jit(
    arb_calc_integrate_line_batch_prec, static_argnames=("integrand", "n_steps", "prec_bits")
)


__all__ = [
    "arb_calc_integrate_line",
    "arb_calc_integrate_line_rigorous",
    "arb_calc_integrate_line_prec",
    "arb_calc_integrate_line_batch",
    "arb_calc_integrate_line_batch_rigorous",
    "arb_calc_integrate_line_batch_prec",
    "arb_calc_integrate_line_batch_jit",
    "arb_calc_integrate_line_batch_prec_jit",
]
