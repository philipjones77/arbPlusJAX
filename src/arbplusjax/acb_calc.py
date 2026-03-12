from __future__ import annotations

"""Arb-like complex calc/integration surface.

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

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from . import acb_core
from . import ball_wrappers
from . import checks
from . import double_interval as di
from . import elementary as el
from . import kernel_helpers as kh
from . import series_utils

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
    "sin_pi": el.sin_pi,
    "cos_pi": el.cos_pi,
    "tan_pi": el.tan_pi,
    "sinc": lambda z: jnp.where(z == 0.0, 1.0 + 0.0j, jnp.sin(z) / z),
    "sinc_pi": el.sinc_pi,
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "asinh": jnp.arcsinh,
    "acosh": jnp.arccosh,
    "atanh": jnp.arctanh,
}

_BOX_EVALS = {
    "exp": lambda z, prec_bits: acb_core.acb_exp(z),
    "log": lambda z, prec_bits: acb_core.acb_log(z),
    "sqrt": lambda z, prec_bits: acb_core.acb_sqrt(z),
    "sin": lambda z, prec_bits: acb_core.acb_sin(z),
    "cos": lambda z, prec_bits: acb_core.acb_cos(z),
    "tan": lambda z, prec_bits: acb_core.acb_tan(z),
    "sinh": lambda z, prec_bits: acb_core.acb_sinh(z),
    "cosh": lambda z, prec_bits: acb_core.acb_cosh(z),
    "tanh": lambda z, prec_bits: acb_core.acb_tanh(z),
    "log1p": lambda z, prec_bits: acb_core.acb_log1p(z),
    "expm1": lambda z, prec_bits: acb_core.acb_expm1(z),
    "sin_pi": lambda z, prec_bits: acb_core.acb_sin_pi(z),
    "cos_pi": lambda z, prec_bits: acb_core.acb_cos_pi(z),
    "tan_pi": lambda z, prec_bits: acb_core.acb_tan_pi(z),
    "sinc": lambda z, prec_bits: acb_core.acb_sinc(z),
    "sinc_pi": lambda z, prec_bits: acb_core.acb_sinc_pi(z),
    "asin": lambda z, prec_bits: acb_core.acb_asin(z),
    "acos": lambda z, prec_bits: acb_core.acb_acos(z),
    "atan": lambda z, prec_bits: acb_core.acb_atan(z),
    "asinh": lambda z, prec_bits: acb_core.acb_asinh(z),
    "acosh": lambda z, prec_bits: acb_core.acb_acosh(z),
    "atanh": lambda z, prec_bits: acb_core.acb_atanh(z),
}

_RIGOROUS_BOX_EVALS = {
    **_BOX_EVALS,
    "gamma": lambda z, prec_bits: ball_wrappers.acb_ball_gamma(z, prec_bits=prec_bits),
    "erf": lambda z, prec_bits: ball_wrappers.acb_ball_erf(z, prec_bits=prec_bits),
    "erfc": lambda z, prec_bits: ball_wrappers.acb_ball_erfc(z, prec_bits=prec_bits),
    "erfi": lambda z, prec_bits: ball_wrappers.acb_ball_erfi(z, prec_bits=prec_bits),
    "barnesg": lambda z, prec_bits: ball_wrappers.acb_ball_barnesg(z, prec_bits=prec_bits),
}

_INTEGRANDS = tuple(_RIGOROUS_BOX_EVALS)


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def _intersect_or_hull_interval(x: jax.Array, y: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    y = di.as_interval(y)
    lo = jnp.maximum(x[..., 0], y[..., 0])
    hi = jnp.minimum(x[..., 1], y[..., 1])
    overlap = lo <= hi
    lo = jnp.where(overlap, lo, jnp.minimum(x[..., 0], y[..., 0]))
    hi = jnp.where(overlap, hi, jnp.maximum(x[..., 1], y[..., 1]))
    return di.interval(lo, hi)


def _intersect_or_hull_box(x: jax.Array, y: jax.Array) -> jax.Array:
    xr = acb_core.acb_real(x)
    xi = acb_core.acb_imag(x)
    yr = acb_core.acb_real(y)
    yi = acb_core.acb_imag(y)
    re = _intersect_or_hull_interval(xr, yr)
    im = _intersect_or_hull_interval(xi, yi)
    return acb_core.acb_box(re, im)


def _acb_from_complex(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def _eval_integrand(z: jax.Array, integrand: str) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "acb_calc._eval_integrand")
    fn = _POINT_EVALS.get(integrand)
    if fn is not None:
        return fn(z)
    flat_z = jnp.ravel(z)
    flat_out = jax.vmap(
        lambda t: acb_core.acb_midpoint(_RIGOROUS_BOX_EVALS[integrand](_acb_from_complex(t), di.DEFAULT_PREC_BITS))
    )(flat_z)
    return flat_out.reshape(jnp.shape(z))


def _eval_integrand_box(z: jax.Array, integrand: str, prec_bits: int) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "acb_calc._eval_integrand_box")
    return _RIGOROUS_BOX_EVALS[integrand](z, prec_bits)


def _integrate_line_midpoint(a: jax.Array, b: jax.Array, integrand: str, n_steps: int) -> jax.Array:
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    if n_steps <= 0:
        n_steps = 1
    z0 = acb_core.acb_midpoint(a)
    z1 = acb_core.acb_midpoint(b)
    delta = z1 - z0
    ts = (jnp.arange(n_steps, dtype=jnp.float64) + 0.5) / jnp.float64(n_steps)
    zs = z0 + delta * ts
    fz = _eval_integrand(zs, integrand)
    dz = delta / jnp.float64(n_steps)
    out = jnp.sum(fz * dz)
    finite = jnp.isfinite(jnp.real(out)) & jnp.isfinite(jnp.imag(out))
    return jnp.where(finite[..., None], _acb_from_complex(out), _full_box_like(a))


def _integrate_line_point_value(a: jax.Array, b: jax.Array, integrand: str, n_steps: int) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if n_steps <= 0:
        n_steps = 1
    cdtype = jnp.dtype(jnp.result_type(a, b, jnp.complex64))
    rd = el.real_dtype_from_complex_dtype(cdtype)
    delta = b - a
    ts = (jnp.arange(n_steps, dtype=rd) + jnp.asarray(0.5, dtype=rd)) / jnp.asarray(n_steps, dtype=rd)
    zs = a + delta * ts
    fz = _eval_integrand(zs, integrand)
    dz = delta / jnp.asarray(n_steps, dtype=rd)
    return jnp.sum(fz * dz)


def _integrate_line_interval(a: jax.Array, b: jax.Array, integrand: str, n_steps: int, prec_bits: int) -> jax.Array:
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    if n_steps <= 0:
        n_steps = 1
    delta = acb_core.acb_sub(b, a)
    ts = (jnp.arange(n_steps, dtype=jnp.float64) + 0.5) / jnp.float64(n_steps)

    def sample(t):
        t_box = acb_core.acb_box(di.interval(t, t), di.interval(0.0, 0.0))
        zt = acb_core.acb_add(a, acb_core.acb_mul(delta, t_box))
        return _eval_integrand_box(zt, integrand, prec_bits)

    vals = jax.vmap(sample)(ts)
    re = acb_core.acb_real(vals)
    im = acb_core.acb_imag(vals)
    lo_re = jnp.sum(re[..., 0], axis=0)
    hi_re = jnp.sum(re[..., 1], axis=0)
    lo_im = jnp.sum(im[..., 0], axis=0)
    hi_im = jnp.sum(im[..., 1], axis=0)
    sum_box = acb_core.acb_box(di.interval(di._below(lo_re), di._above(hi_re)),
                               di.interval(di._below(lo_im), di._above(hi_im)))
    scale = acb_core.acb_box(di.interval(jnp.float64(1.0 / n_steps), jnp.float64(1.0 / n_steps)),
                             di.interval(0.0, 0.0))
    out = acb_core.acb_mul(sum_box, acb_core.acb_mul(delta, scale))
    finite = jnp.isfinite(acb_core.acb_real(out)[..., 0]) & jnp.isfinite(acb_core.acb_real(out)[..., 1])
    finite = finite & jnp.isfinite(acb_core.acb_imag(out)[..., 0]) & jnp.isfinite(acb_core.acb_imag(out)[..., 1])
    return jnp.where(finite[..., None], out, _full_box_like(a))


@lru_cache(maxsize=32)
def _gauss_legendre_nodes_weights(degree: int) -> tuple[jax.Array, jax.Array]:
    nodes, weights = np.polynomial.legendre.leggauss(int(degree))
    return jnp.asarray(nodes, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)


def _gauss_legendre_degree(n_steps: int, prec_bits: int) -> int:
    base = max(8, int(n_steps))
    extra = max(2, int(prec_bits) // 24)
    return base + extra


def _integrate_line_gauss_legendre(
    a: jax.Array,
    b: jax.Array,
    integrand: str,
    degree: int,
    prec_bits: int,
) -> jax.Array:
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    delta = acb_core.acb_sub(b, a)
    nodes, weights = _gauss_legendre_nodes_weights(degree)
    ts = 0.5 * (nodes + 1.0)
    scale = acb_core.acb_box(di.interval(0.5, 0.5), di.interval(0.0, 0.0))

    def sample(node_t, node_w):
        t_box = acb_core.acb_box(di.interval(node_t, node_t), di.interval(0.0, 0.0))
        w_box = acb_core.acb_box(di.interval(node_w, node_w), di.interval(0.0, 0.0))
        zt = acb_core.acb_add(a, acb_core.acb_mul(delta, t_box))
        return acb_core.acb_mul(_eval_integrand_box(zt, integrand, prec_bits), w_box)

    vals = jax.vmap(sample)(ts, weights)
    re = acb_core.acb_real(vals)
    im = acb_core.acb_imag(vals)
    sum_box = acb_core.acb_box(
        di.interval(di._below(jnp.sum(re[..., 0], axis=0)), di._above(jnp.sum(re[..., 1], axis=0))),
        di.interval(di._below(jnp.sum(im[..., 0], axis=0)), di._above(jnp.sum(im[..., 1], axis=0))),
    )
    out = acb_core.acb_mul(sum_box, acb_core.acb_mul(delta, scale))
    finite = jnp.isfinite(acb_core.acb_real(out)[..., 0]) & jnp.isfinite(acb_core.acb_real(out)[..., 1])
    finite = finite & jnp.isfinite(acb_core.acb_imag(out)[..., 0]) & jnp.isfinite(acb_core.acb_imag(out)[..., 1])
    return jnp.where(finite[..., None], out, _full_box_like(a))


def _integrate_line_taylor_series(
    a: jax.Array,
    b: jax.Array,
    integrand: str,
    degree: int,
    prec_bits: int,
) -> jax.Array:
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    z0 = acb_core.acb_midpoint(a)
    z1 = acb_core.acb_midpoint(b)
    center = 0.5 * (z0 + z1)
    half_delta = 0.5 * (z1 - z0)
    coeffs = series_utils.taylor_series_unary_complex(
        lambda z: _eval_integrand(z, integrand),
        center,
        degree + 2,
    )
    orders = jnp.arange(coeffs.shape[0], dtype=jnp.float64)
    moments = jnp.where(
        (orders % 2) == 0,
        2.0 / (orders + 1.0),
        0.0,
    )
    powers = jnp.power(half_delta, orders + 1.0)
    terms = coeffs * powers * moments
    estimate = jnp.sum(terms[:-1])
    tail = jnp.abs(terms[-1]) + jnp.abs(terms[-2])
    eps = jnp.exp2(-jnp.float64(prec_bits)) * (1.0 + jnp.abs(estimate))
    rad = tail + eps
    re = jnp.real(estimate)
    im = jnp.imag(estimate)
    return acb_core.acb_box(
        di.interval(di._below(re - rad), di._above(re + rad)),
        di.interval(di._below(im - rad), di._above(im + rad)),
    )


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate_line(a: jax.Array, b: jax.Array, integrand: str = "exp", n_steps: int = 64) -> jax.Array:
    return _integrate_line_midpoint(a, b, integrand, n_steps)


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate_line_point(a: jax.Array, b: jax.Array, integrand: str = "exp", n_steps: int = 64) -> jax.Array:
    return _integrate_line_point_value(a, b, integrand, n_steps)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_line_rigorous(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    coarse = _integrate_line_interval(a, b, integrand, n_steps, prec_bits)
    fine = _integrate_line_interval(a, b, integrand, max(1, n_steps * 2), prec_bits)
    return _intersect_or_hull_box(coarse, fine)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_line_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_calc_integrate_line(a, b, integrand, n_steps), prec_bits)


def acb_calc_integrate_line_batch(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    return jax.vmap(lambda ai, bi: acb_calc_integrate_line(ai, bi, integrand, n_steps))(a, b)


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate_line_batch_fixed_point(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return jax.vmap(lambda ai, bi: acb_calc_integrate_line_point(ai, bi, integrand, n_steps))(a, b)


def acb_calc_integrate_line_batch_padded_point(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last((jnp.asarray(a), jnp.asarray(b)), pad_to=pad_to)
    out = acb_calc_integrate_line_batch_fixed_point(*call_args, integrand=integrand, n_steps=n_steps)
    return kh.trim_batch_out(out, trim_n)


def acb_calc_integrate_line_batch_rigorous(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    return jax.vmap(lambda ai, bi: acb_calc_integrate_line_rigorous(ai, bi, integrand, n_steps, prec_bits))(a, b)


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate_line_batch_fixed(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    return jax.vmap(lambda ai, bi: acb_calc_integrate_line(ai, bi, integrand, n_steps))(acb_core.as_acb_box(a), acb_core.as_acb_box(b))


def acb_calc_integrate_line_batch_padded(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last((acb_core.as_acb_box(a), acb_core.as_acb_box(b)), pad_to=pad_to)
    out = acb_calc_integrate_line_batch_fixed(*call_args, integrand=integrand, n_steps=n_steps)
    return kh.trim_batch_out(out, trim_n)


def acb_calc_integrate_line_batch_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_calc_integrate_line_batch(a, b, integrand, n_steps), prec_bits
    )


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_line_batch_fixed_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_calc_integrate_line_batch_fixed(a, b, integrand=integrand, n_steps=n_steps), prec_bits
    )


def acb_calc_integrate_line_batch_padded_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last((acb_core.as_acb_box(a), acb_core.as_acb_box(b)), pad_to=pad_to)
    out = acb_calc_integrate_line_batch_fixed_prec(*call_args, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits)
    return kh.trim_batch_out(out, trim_n)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_line_batch_fixed_rigorous(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return jax.vmap(lambda ai, bi: acb_calc_integrate_line_rigorous(ai, bi, integrand, n_steps, prec_bits))(acb_core.as_acb_box(a), acb_core.as_acb_box(b))


def acb_calc_integrate_line_batch_padded_rigorous(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last((acb_core.as_acb_box(a), acb_core.as_acb_box(b)), pad_to=pad_to)
    out = acb_calc_integrate_line_batch_fixed_rigorous(*call_args, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits)
    return kh.trim_batch_out(out, trim_n)


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
) -> jax.Array:
    return acb_calc_integrate_line(a, b, integrand=integrand, n_steps=n_steps)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_calc_integrate(a, b, integrand=integrand, n_steps=n_steps), prec_bits
    )


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_gl_auto_deg(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    degree = _gauss_legendre_degree(n_steps, prec_bits)
    coarse = _integrate_line_gauss_legendre(a, b, integrand, degree, prec_bits)
    fine = _integrate_line_gauss_legendre(a, b, integrand, degree + max(4, degree // 2), prec_bits)
    return _intersect_or_hull_box(coarse, fine)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_gl_auto_deg_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_calc_integrate_gl_auto_deg(
        a, b, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits
    )


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_taylor(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    degree = max(8, int(n_steps)) + max(2, int(prec_bits) // 32)
    return _integrate_line_taylor_series(a, b, integrand, degree, prec_bits)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_taylor_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_calc_integrate_taylor(
        a, b, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits
    )


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_opt_init(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    # Return a stable initial enclosure used by higher-level integration paths.
    return acb_calc_integrate_prec(a, b, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_integrate_opt_init_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_calc_integrate_opt_init(
        a, b, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits
    )


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_cauchy_bound(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    box = acb_calc_integrate_prec(a, b, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits)
    re = acb_core.acb_real(box)
    im = acb_core.acb_imag(box)
    re_abs = jnp.maximum(jnp.abs(re[..., 0]), jnp.abs(re[..., 1]))
    im_abs = jnp.maximum(jnp.abs(im[..., 0]), jnp.abs(im[..., 1]))
    ub = jnp.sqrt(re_abs * re_abs + im_abs * im_abs)
    return di.interval(di._below(jnp.zeros_like(ub)), di._above(ub))


@partial(jax.jit, static_argnames=("integrand", "n_steps", "prec_bits"))
def acb_calc_cauchy_bound_prec(
    a: jax.Array,
    b: jax.Array,
    integrand: str = "exp",
    n_steps: int = 64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        acb_calc_cauchy_bound(a, b, integrand=integrand, n_steps=n_steps, prec_bits=prec_bits),
        prec_bits,
    )


acb_calc_integrate_line_batch_jit = jax.jit(acb_calc_integrate_line_batch, static_argnames=("integrand", "n_steps"))
acb_calc_integrate_line_batch_prec_jit = jax.jit(
    acb_calc_integrate_line_batch_prec, static_argnames=("integrand", "n_steps", "prec_bits")
)
acb_calc_integrate_line_batch_fixed_point_jit = acb_calc_integrate_line_batch_fixed_point
acb_calc_integrate_line_batch_fixed_jit = acb_calc_integrate_line_batch_fixed
acb_calc_integrate_line_batch_fixed_prec_jit = acb_calc_integrate_line_batch_fixed_prec
acb_calc_integrate_line_batch_fixed_rigorous_jit = acb_calc_integrate_line_batch_fixed_rigorous


__all__ = [
    "acb_calc_integrate_line",
    "acb_calc_integrate_line_point",
    "acb_calc_integrate_line_rigorous",
    "acb_calc_integrate_line_prec",
    "acb_calc_integrate_line_batch",
    "acb_calc_integrate_line_batch_fixed_point",
    "acb_calc_integrate_line_batch_padded_point",
    "acb_calc_integrate_line_batch_fixed",
    "acb_calc_integrate_line_batch_padded",
    "acb_calc_integrate_line_batch_rigorous",
    "acb_calc_integrate_line_batch_fixed_rigorous",
    "acb_calc_integrate_line_batch_padded_rigorous",
    "acb_calc_integrate_line_batch_prec",
    "acb_calc_integrate_line_batch_fixed_prec",
    "acb_calc_integrate_line_batch_padded_prec",
    "acb_calc_integrate_line_batch_jit",
    "acb_calc_integrate_line_batch_prec_jit",
    "acb_calc_integrate_line_batch_fixed_point_jit",
    "acb_calc_integrate_line_batch_fixed_jit",
    "acb_calc_integrate_line_batch_fixed_prec_jit",
    "acb_calc_integrate_line_batch_fixed_rigorous_jit",
    "acb_calc_integrate",
    "acb_calc_integrate_prec",
    "acb_calc_integrate_gl_auto_deg",
    "acb_calc_integrate_gl_auto_deg_prec",
    "acb_calc_integrate_taylor",
    "acb_calc_integrate_taylor_prec",
    "acb_calc_integrate_opt_init",
    "acb_calc_integrate_opt_init_prec",
    "acb_calc_cauchy_bound",
    "acb_calc_cauchy_bound_prec",
]
