from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from .spherical_asymptotics import (
    modified_spherical_bessel_i_asymptotic,
    modified_spherical_bessel_k_asymptotic,
    spherical_bessel_j_asymptotic,
    spherical_bessel_y_asymptotic,
)
from .spherical_recurrences import (
    modified_spherical_bessel_i_recurrence,
    modified_spherical_bessel_k_recurrence,
    spherical_bessel_j_recurrence,
    spherical_bessel_y_recurrence,
)

_SERIES_TERMS = 24


def _real_dtype(z):
    dt = jnp.asarray(z).dtype
    if jnp.issubdtype(dt, jnp.complexfloating):
        return jnp.float32 if dt == jnp.complex64 else jnp.float64
    return jnp.float32 if dt == jnp.float32 else jnp.float64


def _series_common(n, z, *, sign: float):
    n_int = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    rdt = _real_dtype(z_v)
    pref = jnp.asarray(0.5 * jnp.sqrt(jnp.asarray(jnp.pi, dtype=rdt)), dtype=rdt)
    half_z = jnp.asarray(0.5, dtype=z_v.dtype) * z_v
    nu = n_int.astype(rdt) + jnp.asarray(1.5, dtype=rdt)
    term0 = pref * (half_z ** n_int.astype(z_v.dtype)) / jnp.exp(lax.lgamma(nu))
    sum0 = term0
    z2 = z_v * z_v

    def body(k, state):
        term, total = state
        k1 = jnp.asarray(k + 1, dtype=rdt)
        den = k1 * (k1 + n_int.astype(rdt) + jnp.asarray(0.5, dtype=rdt))
        term = term * ((jnp.asarray(0.25 * sign, dtype=rdt) * z2) / den)
        return term, total + term

    _, out = lax.fori_loop(0, _SERIES_TERMS - 1, body, (term0, sum0))
    return out


@partial(jax.jit, static_argnames=("method",))
def spherical_bessel_j_point(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    if method == "asymptotic":
        return spherical_bessel_j_asymptotic(n, z)
    if method == "series":
        return _series_common(n, z, sign=-1.0)
    if method == "recurrence":
        return spherical_bessel_j_recurrence(n, z)
    threshold = jnp.maximum(jnp.asarray(16.0, dtype=_real_dtype(z_v)), n_v.astype(_real_dtype(z_v)) + jnp.asarray(8.0, dtype=_real_dtype(z_v)))
    return lax.cond(
        jnp.abs(z_v) < jnp.asarray(0.75, dtype=_real_dtype(z_v)),
        lambda _: _series_common(n_v, z_v, sign=-1.0),
        lambda _: lax.cond(jnp.abs(z_v) >= threshold, lambda __: spherical_bessel_j_asymptotic(n_v, z_v), lambda __: spherical_bessel_j_recurrence(n_v, z_v), operand=None),
        operand=None,
    )


@partial(jax.jit, static_argnames=("method",))
def spherical_bessel_y_point(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    if method == "asymptotic":
        return spherical_bessel_y_asymptotic(n, z)
    if method == "recurrence":
        return spherical_bessel_y_recurrence(n, z)
    threshold = jnp.maximum(jnp.asarray(16.0, dtype=_real_dtype(z_v)), n_v.astype(_real_dtype(z_v)) + jnp.asarray(8.0, dtype=_real_dtype(z_v)))
    return lax.cond(jnp.abs(z_v) >= threshold, lambda _: spherical_bessel_y_asymptotic(n_v, z_v), lambda _: spherical_bessel_y_recurrence(n_v, z_v), operand=None)


@partial(jax.jit, static_argnames=("method",))
def modified_spherical_bessel_i_point(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    if method == "asymptotic":
        return modified_spherical_bessel_i_asymptotic(n, z)
    if method == "series":
        return _series_common(n, z, sign=1.0)
    if method == "recurrence":
        return modified_spherical_bessel_i_recurrence(n, z)
    threshold = jnp.maximum(jnp.asarray(16.0, dtype=_real_dtype(z_v)), n_v.astype(_real_dtype(z_v)) + jnp.asarray(8.0, dtype=_real_dtype(z_v)))
    return lax.cond(
        jnp.abs(z_v) < jnp.asarray(0.75, dtype=_real_dtype(z_v)),
        lambda _: _series_common(n_v, z_v, sign=1.0),
        lambda _: lax.cond(jnp.abs(z_v) >= threshold, lambda __: modified_spherical_bessel_i_asymptotic(n_v, z_v), lambda __: modified_spherical_bessel_i_recurrence(n_v, z_v), operand=None),
        operand=None,
    )


@partial(jax.jit, static_argnames=("method",))
def modified_spherical_bessel_k_point(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    if method == "asymptotic":
        return modified_spherical_bessel_k_asymptotic(n, z)
    if method == "recurrence":
        return modified_spherical_bessel_k_recurrence(n, z)
    threshold = jnp.maximum(jnp.asarray(16.0, dtype=_real_dtype(z_v)), n_v.astype(_real_dtype(z_v)) + jnp.asarray(8.0, dtype=_real_dtype(z_v)))
    return lax.cond(jnp.abs(z_v) >= threshold, lambda _: modified_spherical_bessel_k_asymptotic(n_v, z_v), lambda _: modified_spherical_bessel_k_recurrence(n_v, z_v), operand=None)


def spherical_bessel_j(n, z, *, method: str = "auto"):
    return spherical_bessel_j_point(n, z, method=method)


def spherical_bessel_y(n, z, *, method: str = "auto"):
    return spherical_bessel_y_point(n, z, method=method)


def modified_spherical_bessel_i(n, z, *, method: str = "auto"):
    return modified_spherical_bessel_i_point(n, z, method=method)


def modified_spherical_bessel_k(n, z, *, method: str = "auto"):
    return modified_spherical_bessel_k_point(n, z, method=method)


__all__ = [
    "spherical_bessel_j",
    "spherical_bessel_j_point",
    "spherical_bessel_y",
    "spherical_bessel_y_point",
    "modified_spherical_bessel_i",
    "modified_spherical_bessel_i_point",
    "modified_spherical_bessel_k",
    "modified_spherical_bessel_k_point",
]
