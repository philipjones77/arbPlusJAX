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


def _zeta_series(s: jax.Array, n_terms: int) -> jax.Array:
    if n_terms <= 0:
        n_terms = 1
    n = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    return jnp.sum(jnp.exp(-s * jnp.log(n)))


def acb_dirichlet_zeta(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = acb_core.as_acb_box(s)
    sm = acb_core.acb_midpoint(s)
    v = _zeta_series(sm, n_terms)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(s))


def acb_dirichlet_eta(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = acb_core.as_acb_box(s)
    sm = acb_core.acb_midpoint(s)
    zeta = _zeta_series(sm, n_terms)
    factor = 1.0 - jnp.exp((1.0 - sm) * jnp.log(2.0))
    v = factor * zeta
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(s))


@partial(jax.jit, static_argnames=("n_terms", "prec_bits"))
def acb_dirichlet_zeta_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dirichlet_zeta(s, n_terms), prec_bits)


@partial(jax.jit, static_argnames=("n_terms", "prec_bits"))
def acb_dirichlet_eta_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dirichlet_eta(s, n_terms), prec_bits)


def acb_dirichlet_zeta_batch(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = acb_core.as_acb_box(s)
    return jax.vmap(lambda si: acb_dirichlet_zeta(si, n_terms))(s)


def acb_dirichlet_eta_batch(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = acb_core.as_acb_box(s)
    return jax.vmap(lambda si: acb_dirichlet_eta(si, n_terms))(s)


def acb_dirichlet_zeta_batch_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dirichlet_zeta_batch(s, n_terms), prec_bits)


def acb_dirichlet_eta_batch_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dirichlet_eta_batch(s, n_terms), prec_bits)


acb_dirichlet_zeta_batch_jit = jax.jit(acb_dirichlet_zeta_batch, static_argnames=("n_terms",))
acb_dirichlet_eta_batch_jit = jax.jit(acb_dirichlet_eta_batch, static_argnames=("n_terms",))
acb_dirichlet_zeta_batch_prec_jit = jax.jit(
    acb_dirichlet_zeta_batch_prec, static_argnames=("n_terms", "prec_bits")
)
acb_dirichlet_eta_batch_prec_jit = jax.jit(
    acb_dirichlet_eta_batch_prec, static_argnames=("n_terms", "prec_bits")
)


__all__ = [
    "acb_dirichlet_zeta",
    "acb_dirichlet_eta",
    "acb_dirichlet_zeta_prec",
    "acb_dirichlet_eta_prec",
    "acb_dirichlet_zeta_batch",
    "acb_dirichlet_eta_batch",
    "acb_dirichlet_zeta_batch_prec",
    "acb_dirichlet_eta_batch_prec",
    "acb_dirichlet_zeta_batch_jit",
    "acb_dirichlet_eta_batch_jit",
    "acb_dirichlet_zeta_batch_prec_jit",
    "acb_dirichlet_eta_batch_prec_jit",
]
