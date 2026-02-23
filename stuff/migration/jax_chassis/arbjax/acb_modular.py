from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di

jax.config.update("jax_enable_x64", True)


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


def acb_modular_j(tau: jax.Array) -> jax.Array:
    tau = acb_core.as_acb_box(tau)
    t = acb_core.acb_midpoint(tau)
    q = jnp.exp(2j * jnp.pi * t)
    qnorm = jnp.real(q) ** 2 + jnp.imag(q) ** 2
    v = 1.0 / q + 744.0 + 196884.0 * q + 21493760.0 * q * q
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v)) & (qnorm != 0.0)
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(tau))


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_modular_j_prec(tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_modular_j(tau), prec_bits)


def acb_modular_j_batch(tau: jax.Array) -> jax.Array:
    tau = acb_core.as_acb_box(tau)
    return jax.vmap(acb_modular_j)(tau)


def acb_modular_j_batch_prec(tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_modular_j_batch(tau), prec_bits)


acb_modular_j_batch_jit = jax.jit(acb_modular_j_batch)
acb_modular_j_batch_prec_jit = jax.jit(acb_modular_j_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "acb_modular_j",
    "acb_modular_j_prec",
    "acb_modular_j_batch",
    "acb_modular_j_batch_prec",
    "acb_modular_j_batch_jit",
    "acb_modular_j_batch_prec_jit",
]
