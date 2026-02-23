from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import double_interval as di

jax.config.update("jax_enable_x64", True)


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _from_scalar(v: jax.Array) -> jax.Array:
    return di.interval(di._below(v), di._above(v))


def _zeta_series(s: jax.Array, n_terms: int) -> jax.Array:
    if n_terms <= 0:
        n_terms = 1
    n = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    return jnp.sum(jnp.exp(-s[..., None] * jnp.log(n)), axis=-1)


def dirichlet_zeta(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = di.as_interval(s)
    sm = di.midpoint(s)
    v = _zeta_series(sm, n_terms)
    finite = jnp.isfinite(v)
    out = _from_scalar(v)
    return jnp.where(finite[..., None], out, _full_interval_like(s))


def dirichlet_eta(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = di.as_interval(s)
    sm = di.midpoint(s)
    zeta = _zeta_series(sm, n_terms)
    factor = 1.0 - jnp.exp((1.0 - sm) * jnp.log(2.0))
    v = factor * zeta
    finite = jnp.isfinite(v)
    out = _from_scalar(v)
    return jnp.where(finite[..., None], out, _full_interval_like(s))


@partial(jax.jit, static_argnames=("n_terms", "prec_bits"))
def dirichlet_zeta_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(dirichlet_zeta(s, n_terms), prec_bits)


@partial(jax.jit, static_argnames=("n_terms", "prec_bits"))
def dirichlet_eta_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(dirichlet_eta(s, n_terms), prec_bits)


def dirichlet_zeta_batch(s: jax.Array, n_terms: int = 64) -> jax.Array:
    return dirichlet_zeta(s, n_terms)


def dirichlet_eta_batch(s: jax.Array, n_terms: int = 64) -> jax.Array:
    return dirichlet_eta(s, n_terms)


def dirichlet_zeta_batch_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(dirichlet_zeta_batch(s, n_terms), prec_bits)


def dirichlet_eta_batch_prec(
    s: jax.Array, n_terms: int = 64, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(dirichlet_eta_batch(s, n_terms), prec_bits)


dirichlet_zeta_batch_jit = jax.jit(dirichlet_zeta_batch, static_argnames=("n_terms",))
dirichlet_eta_batch_jit = jax.jit(dirichlet_eta_batch, static_argnames=("n_terms",))
dirichlet_zeta_batch_prec_jit = jax.jit(
    dirichlet_zeta_batch_prec, static_argnames=("n_terms", "prec_bits")
)
dirichlet_eta_batch_prec_jit = jax.jit(
    dirichlet_eta_batch_prec, static_argnames=("n_terms", "prec_bits")
)


__all__ = [
    "dirichlet_zeta",
    "dirichlet_eta",
    "dirichlet_zeta_prec",
    "dirichlet_eta_prec",
    "dirichlet_zeta_batch",
    "dirichlet_eta_batch",
    "dirichlet_zeta_batch_prec",
    "dirichlet_eta_batch_prec",
    "dirichlet_zeta_batch_jit",
    "dirichlet_eta_batch_jit",
    "dirichlet_zeta_batch_prec_jit",
    "dirichlet_eta_batch_prec_jit",
]
