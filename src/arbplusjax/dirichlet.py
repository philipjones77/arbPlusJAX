from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import arb_core
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


def _zeta_series_interval(s: jax.Array, n_terms: int) -> jax.Array:
    if n_terms <= 0:
        n_terms = 1
    s = di.as_interval(s)
    n = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    logn = jnp.log(n)
    logn_iv = di.interval(logn, logn)
    expo = di.neg(di.fast_mul(s[..., None, :], logn_iv))
    terms = arb_core.arb_exp(expo)
    lo = jnp.sum(terms[..., 0], axis=-1)
    hi = jnp.sum(terms[..., 1], axis=-1)
    return di.interval(di._below(lo), di._above(hi))


def _eta_series_interval(s: jax.Array, n_terms: int) -> jax.Array:
    if n_terms <= 0:
        n_terms = 1
    s = di.as_interval(s)
    n = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    logn = jnp.log(n)
    logn_iv = di.interval(logn, logn)
    expo = di.neg(di.fast_mul(s[..., None, :], logn_iv))
    terms = arb_core.arb_exp(expo)
    signs = jnp.where((n % 2) == 0, -1.0, 1.0)
    signs_iv = di.interval(signs, signs)
    terms = di.fast_mul(terms, signs_iv)
    lo = jnp.sum(terms[..., 0], axis=-1)
    hi = jnp.sum(terms[..., 1], axis=-1)
    return di.interval(di._below(lo), di._above(hi))


def _zeta_tail_bound(s: jax.Array, n_terms: int) -> tuple[jax.Array, jax.Array]:
    s = di.as_interval(s)
    sigma = s[0]
    n = jnp.float64(n_terms + 1)
    ok = sigma > 1.0
    tail = jnp.where(ok, jnp.exp((1.0 - sigma) * jnp.log(n)) / (sigma - 1.0), jnp.inf)
    return tail, ok


def _eta_tail_bound(s: jax.Array, n_terms: int) -> tuple[jax.Array, jax.Array]:
    s = di.as_interval(s)
    sigma = s[0]
    n = jnp.float64(n_terms + 1)
    ok = sigma > 0.0
    tail = jnp.where(ok, jnp.exp(-sigma * jnp.log(n)), jnp.inf)
    return tail, ok


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


def dirichlet_zeta_rigorous(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = di.as_interval(s)
    out = _zeta_series_interval(s, n_terms)
    tail, ok = _zeta_tail_bound(s, n_terms)
    lo = out[..., 0]
    hi = out[..., 1] + tail
    out = di.interval(di._below(lo), di._above(hi))
    finite = jnp.isfinite(out[..., 0]) & jnp.isfinite(out[..., 1])
    return jnp.where(ok & finite[..., None], out, _full_interval_like(s))


def dirichlet_eta_rigorous(s: jax.Array, n_terms: int = 64) -> jax.Array:
    s = di.as_interval(s)
    out = _eta_series_interval(s, n_terms)
    tail, ok = _eta_tail_bound(s, n_terms)
    lo = out[..., 0] - tail
    hi = out[..., 1] + tail
    out = di.interval(di._below(lo), di._above(hi))
    finite = jnp.isfinite(out[..., 0]) & jnp.isfinite(out[..., 1])
    return jnp.where(ok & finite[..., None], out, _full_interval_like(s))


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
    "dirichlet_zeta_rigorous",
    "dirichlet_eta_rigorous",
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
