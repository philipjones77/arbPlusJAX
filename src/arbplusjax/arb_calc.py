from __future__ import annotations

"""Arb-like real calc/integration surface.

This module is part of the canonical Arb/FLINT-style public surface for this
repo. The calc method names are Arb-like, while point/basic/adaptive/rigorous
mode dispatch is handled one layer up by the wrapper/API surface.

Provenance:
- classification: arb_like
- base_names: calc_integrate_line, calc_integrate, calc_integrate_gl_auto_deg, calc_integrate_taylor
- module lineage: Arb/FLINT-style calc/integration surface
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import arb_core
from . import baseline_wrappers
from . import double_interval as di
from . import checks

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "arb_like",
    "base_names": (
        "calc_integrate_line",
        "calc_integrate",
        "calc_integrate_gl_auto_deg",
        "calc_integrate_taylor",
    ),
    "module_lineage": "Arb/FLINT-style calc/integration surface",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}

_POINT_EVALS = {
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "log1p": jnp.log1p,
    "expm1": jnp.expm1,
    "sin_pi": lambda x: jnp.sin(jnp.pi * x),
    "cos_pi": lambda x: jnp.cos(jnp.pi * x),
    "tan_pi": lambda x: jnp.tan(jnp.pi * x),
    "sinc": lambda x: jnp.where(x == 0.0, 1.0, jnp.sin(x) / x),
    "sinc_pi": lambda x: jnp.where(x == 0.0, 1.0, jnp.sin(jnp.pi * x) / (jnp.pi * x)),
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "asinh": jnp.arcsinh,
    "acosh": jnp.arccosh,
    "atanh": jnp.arctanh,
    "cbrt": lambda x: jnp.sign(x) * jnp.power(jnp.abs(x), 1.0 / 3.0),
}

_INTERVAL_EVALS = {
    "exp": arb_core.arb_exp,
    "log": arb_core.arb_log,
    "sqrt": arb_core.arb_sqrt,
    "sin": arb_core.arb_sin,
    "cos": arb_core.arb_cos,
    "tan": arb_core.arb_tan,
    "sinh": arb_core.arb_sinh,
    "cosh": arb_core.arb_cosh,
    "tanh": arb_core.arb_tanh,
    "log1p": arb_core.arb_log1p,
    "expm1": arb_core.arb_expm1,
    "sin_pi": arb_core.arb_sin_pi,
    "cos_pi": arb_core.arb_cos_pi,
    "tan_pi": arb_core.arb_tan_pi,
    "sinc": arb_core.arb_sinc,
    "sinc_pi": arb_core.arb_sinc_pi,
    "asin": arb_core.arb_asin,
    "acos": arb_core.arb_acos,
    "atan": arb_core.arb_atan,
    "asinh": arb_core.arb_asinh,
    "acosh": arb_core.arb_acosh,
    "atanh": arb_core.arb_atanh,
    "cbrt": arb_core.arb_cbrt,
    "gamma": lambda x: baseline_wrappers.arb_gamma_mp(x, mode="rigorous"),
    "erf": lambda x: baseline_wrappers.arb_erf_mp(x, mode="rigorous"),
    "erfc": lambda x: baseline_wrappers.arb_erfc_mp(x, mode="rigorous"),
    "erfi": lambda x: baseline_wrappers.arb_erfi_mp(x, mode="rigorous"),
    "barnesg": lambda x: baseline_wrappers.arb_barnesg_mp(x, mode="rigorous"),
}

_INTEGRANDS = tuple(_INTERVAL_EVALS)


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _eval_integrand(x: jax.Array, integrand: str) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "arb_calc._eval_integrand")
    fn = _POINT_EVALS.get(integrand)
    if fn is not None:
        return fn(x)
    flat_x = jnp.ravel(x)
    flat_out = jax.vmap(lambda t: di.midpoint(_INTERVAL_EVALS[integrand](di.interval(t, t))))(flat_x)
    return flat_out.reshape(jnp.shape(x))


def _eval_integrand_interval(x: jax.Array, integrand: str) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "arb_calc._eval_integrand_interval")
    return _INTERVAL_EVALS[integrand](x)


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
