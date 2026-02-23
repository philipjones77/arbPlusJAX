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


def _agm(a: jax.Array, b: jax.Array, iters: int = 8) -> jax.Array:
    def body(val, _):
        aa, bb = val
        a_next = 0.5 * (aa + bb)
        b_next = jnp.sqrt(aa * bb)
        return (a_next, b_next), None

    (a_out, _), _ = jax.lax.scan(body, (a, b), None, length=iters)
    return a_out


def acb_elliptic_k(m: jax.Array) -> jax.Array:
    m = acb_core.as_acb_box(m)
    mm = acb_core.acb_midpoint(m)
    k = jnp.sqrt(1.0 - mm)
    agm = _agm(1.0 + 0.0j, k, iters=8)
    v = 0.5 * jnp.pi / agm
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(m))


def acb_elliptic_e(m: jax.Array) -> jax.Array:
    m = acb_core.as_acb_box(m)
    mm = acb_core.acb_midpoint(m)
    k = jnp.sqrt(1.0 - mm)
    agm = _agm(1.0 + 0.0j, k, iters=8)
    v = 0.5 * jnp.pi * agm
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(m))


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_elliptic_k_prec(m: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_elliptic_k(m), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_elliptic_e_prec(m: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_elliptic_e(m), prec_bits)


def acb_elliptic_k_batch(m: jax.Array) -> jax.Array:
    m = acb_core.as_acb_box(m)
    return jax.vmap(acb_elliptic_k)(m)


def acb_elliptic_e_batch(m: jax.Array) -> jax.Array:
    m = acb_core.as_acb_box(m)
    return jax.vmap(acb_elliptic_e)(m)


def acb_elliptic_k_batch_prec(m: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_elliptic_k_batch(m), prec_bits)


def acb_elliptic_e_batch_prec(m: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_elliptic_e_batch(m), prec_bits)


acb_elliptic_k_batch_jit = jax.jit(acb_elliptic_k_batch)
acb_elliptic_e_batch_jit = jax.jit(acb_elliptic_e_batch)
acb_elliptic_k_batch_prec_jit = jax.jit(acb_elliptic_k_batch_prec, static_argnames=("prec_bits",))
acb_elliptic_e_batch_prec_jit = jax.jit(acb_elliptic_e_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "acb_elliptic_k",
    "acb_elliptic_e",
    "acb_elliptic_k_prec",
    "acb_elliptic_e_prec",
    "acb_elliptic_k_batch",
    "acb_elliptic_e_batch",
    "acb_elliptic_k_batch_prec",
    "acb_elliptic_e_batch_prec",
    "acb_elliptic_k_batch_jit",
    "acb_elliptic_e_batch_jit",
    "acb_elliptic_k_batch_prec_jit",
    "acb_elliptic_e_batch_prec_jit",
]
