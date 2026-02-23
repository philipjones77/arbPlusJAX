from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di
from . import core_wrappers
from . import checks

jax.config.update("jax_enable_x64", True)

_INTEGRANDS = ("exp", "sin", "cos")


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
    if integrand == "exp":
        return jnp.exp(z)
    if integrand == "sin":
        return jnp.sin(z)
    if integrand == "cos":
        return jnp.cos(z)
    return jnp.exp(z)


def _eval_integrand_box(z: jax.Array, integrand: str, prec_bits: int) -> jax.Array:
    checks.check_in_set(integrand, _INTEGRANDS, "acb_calc._eval_integrand_box")
    if integrand == "exp":
        return core_wrappers.acb_exp_mode(z, impl="rigorous", prec_bits=prec_bits)
    if integrand == "sin":
        return core_wrappers.acb_sin_mode(z, impl="rigorous", prec_bits=prec_bits)
    if integrand == "cos":
        return core_wrappers.acb_cos_mode(z, impl="rigorous", prec_bits=prec_bits)
    return core_wrappers.acb_exp_mode(z, impl="rigorous", prec_bits=prec_bits)


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


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate_line(a: jax.Array, b: jax.Array, integrand: str = "exp", n_steps: int = 64) -> jax.Array:
    return _integrate_line_midpoint(a, b, integrand, n_steps)


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


acb_calc_integrate_line_batch_jit = jax.jit(acb_calc_integrate_line_batch, static_argnames=("integrand", "n_steps"))
acb_calc_integrate_line_batch_prec_jit = jax.jit(
    acb_calc_integrate_line_batch_prec, static_argnames=("integrand", "n_steps", "prec_bits")
)


__all__ = [
    "acb_calc_integrate_line",
    "acb_calc_integrate_line_rigorous",
    "acb_calc_integrate_line_prec",
    "acb_calc_integrate_line_batch",
    "acb_calc_integrate_line_batch_rigorous",
    "acb_calc_integrate_line_batch_prec",
    "acb_calc_integrate_line_batch_jit",
    "acb_calc_integrate_line_batch_prec_jit",
]
