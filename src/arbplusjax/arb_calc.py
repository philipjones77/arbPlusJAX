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


def _coerce_n(n: int) -> int:
    return max(1, int(n))


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


@partial(jax.jit, static_argnames=("parts",))
def arb_calc_partition(a: jax.Array, b: jax.Array, parts: int = 8) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    n = _coerce_n(parts)
    am = di.midpoint(a)
    bm = di.midpoint(b)
    ts = jnp.linspace(0.0, 1.0, n + 1, dtype=jnp.float64)
    pts = am + (bm - am) * ts
    return di.interval(di._below(pts), di._above(pts))


@partial(jax.jit, static_argnames=("parts", "prec_bits"))
def arb_calc_partition_prec(
    a: jax.Array,
    b: jax.Array,
    parts: int = 8,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_calc_partition(a, b, parts), prec_bits)


def arb_calc_partition_batch(a: jax.Array, b: jax.Array, parts: int = 8) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    return jax.vmap(lambda ai, bi: arb_calc_partition(ai, bi, parts))(a, b)


def arb_calc_partition_batch_prec(
    a: jax.Array,
    b: jax.Array,
    parts: int = 8,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_calc_partition_batch(a, b, parts), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_newton_conv_factor(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    x = di.as_interval(x)
    m = di.midpoint(x)
    fac = 1.0 / (1.0 + jnp.abs(m))
    out = di.interval(di._below(fac), di._above(fac))
    return di.round_interval_outward(out, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_newton_conv_factor_prec(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_calc_newton_conv_factor(x, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_newton_step(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    x = di.as_interval(x)
    m = di.midpoint(x)
    step = m - jnp.sin(m) / (jnp.cos(m) + 1e-12)
    r = di.ubound_radius(x)
    out = di.interval(di._below(step - r), di._above(step + r))
    return di.round_interval_outward(out, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_newton_step_prec(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_calc_newton_step(x, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_refine_root_bisect(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    x = di.as_interval(x)
    m = di.midpoint(x)
    r = 0.5 * di.ubound_radius(x)
    out = di.interval(di._below(m - r), di._above(m + r))
    return di.round_interval_outward(out, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_refine_root_bisect_prec(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_calc_refine_root_bisect(x, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_refine_root_newton(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    step = arb_calc_newton_step(x, prec_bits=prec_bits)
    r = 0.5 * di.ubound_radius(di.as_interval(x))
    m = di.midpoint(step)
    out = di.interval(di._below(m - r), di._above(m + r))
    return di.round_interval_outward(out, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_calc_refine_root_newton_prec(
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_calc_refine_root_newton(x, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("max_roots", "prec_bits"))
def arb_calc_isolate_roots(
    a: jax.Array,
    b: jax.Array,
    max_roots: int = 8,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    parts = arb_calc_partition(a, b, parts=max_roots)
    left = parts[:-1, :]
    right = parts[1:, :]
    seg_mid = 0.5 * (di.midpoint(left) + di.midpoint(right))
    val = jnp.sin(seg_mid)
    s = jnp.sign(val)
    s_prev = jnp.concatenate([s[:1], s[:-1]], axis=0)
    has_cross = s == 0.0
    has_cross = has_cross | (s * s_prev <= 0.0)
    cand = di.interval(left[:, 0], right[:, 1])
    width = 0.5 * (cand[:, 1] - cand[:, 0])
    tiny = di.interval(seg_mid - 0.125 * width, seg_mid + 0.125 * width)
    out = jnp.where(has_cross[:, None], cand, tiny)
    return di.round_interval_outward(out, prec_bits)


@partial(jax.jit, static_argnames=("max_roots", "prec_bits"))
def arb_calc_isolate_roots_prec(
    a: jax.Array,
    b: jax.Array,
    max_roots: int = 8,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_calc_isolate_roots(a, b, max_roots=max_roots, prec_bits=prec_bits)


arb_calc_integrate_line_batch_jit = jax.jit(arb_calc_integrate_line_batch, static_argnames=("integrand", "n_steps"))
arb_calc_integrate_line_batch_prec_jit = jax.jit(
    arb_calc_integrate_line_batch_prec, static_argnames=("integrand", "n_steps", "prec_bits")
)
arb_calc_partition_batch_jit = jax.jit(arb_calc_partition_batch, static_argnames=("parts",))
arb_calc_partition_batch_prec_jit = jax.jit(
    arb_calc_partition_batch_prec, static_argnames=("parts", "prec_bits")
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
    "arb_calc_partition",
    "arb_calc_partition_prec",
    "arb_calc_partition_batch",
    "arb_calc_partition_batch_prec",
    "arb_calc_partition_batch_jit",
    "arb_calc_partition_batch_prec_jit",
    "arb_calc_newton_conv_factor",
    "arb_calc_newton_conv_factor_prec",
    "arb_calc_newton_step",
    "arb_calc_newton_step_prec",
    "arb_calc_refine_root_bisect",
    "arb_calc_refine_root_bisect_prec",
    "arb_calc_refine_root_newton",
    "arb_calc_refine_root_newton_prec",
    "arb_calc_isolate_roots",
    "arb_calc_isolate_roots_prec",
]
