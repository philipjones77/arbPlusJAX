from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di

jax.config.update("jax_enable_x64", True)

_INTEGRANDS = ("exp", "sin", "cos")


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def _acb_from_complex(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def _eval_integrand(z: jax.Array, integrand: str) -> jax.Array:
    if integrand == "exp":
        return jnp.exp(z)
    if integrand == "sin":
        return jnp.sin(z)
    if integrand == "cos":
        return jnp.cos(z)
    raise ValueError(f"Unsupported integrand '{integrand}'. Use one of: {', '.join(_INTEGRANDS)}")


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


@partial(jax.jit, static_argnames=("integrand", "n_steps"))
def acb_calc_integrate_line(a: jax.Array, b: jax.Array, integrand: str = "exp", n_steps: int = 64) -> jax.Array:
    return _integrate_line_midpoint(a, b, integrand, n_steps)


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
    "acb_calc_integrate_line_prec",
    "acb_calc_integrate_line_batch",
    "acb_calc_integrate_line_batch_prec",
    "acb_calc_integrate_line_batch_jit",
    "acb_calc_integrate_line_batch_prec_jit",
]
