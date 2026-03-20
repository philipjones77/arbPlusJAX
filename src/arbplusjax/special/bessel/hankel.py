from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from ... import bessel_kernels as bk
from .hankel_asymptotics import (
    hankel1_asymptotic,
    hankel2_asymptotic,
    scaled_hankel1_asymptotic,
    scaled_hankel2_asymptotic,
)


def _complex_dtype(*xs) -> jnp.dtype:
    return jnp.result_type(*[jnp.asarray(x).dtype for x in xs], jnp.complex64)


def _hankel_direct(kind: int, nu, z):
    cdt = _complex_dtype(nu, z)
    nu_v = jnp.asarray(nu, dtype=cdt)
    z_v = jnp.asarray(z, dtype=cdt)
    jv = bk.complex_bessel_series(nu_v, z_v, -1.0)
    yv = bk.complex_bessel_y(nu_v, z_v)
    imag = jnp.asarray(1j, dtype=cdt)
    if kind == 1:
        return jv + imag * yv
    return jv - imag * yv


@partial(jax.jit, static_argnames=("method",))
def hankel1_point(nu, z, *, method: str = "auto"):
    if method == "asymptotic":
        return hankel1_asymptotic(nu, z)
    if method == "direct":
        return _hankel_direct(1, nu, z)
    return lax.cond(jnp.abs(jnp.asarray(z)) >= 16.0, lambda _: hankel1_asymptotic(nu, z), lambda _: _hankel_direct(1, nu, z), operand=None)


@partial(jax.jit, static_argnames=("method",))
def hankel2_point(nu, z, *, method: str = "auto"):
    if method == "asymptotic":
        return hankel2_asymptotic(nu, z)
    if method == "direct":
        return _hankel_direct(2, nu, z)
    return lax.cond(jnp.abs(jnp.asarray(z)) >= 16.0, lambda _: hankel2_asymptotic(nu, z), lambda _: _hankel_direct(2, nu, z), operand=None)


@partial(jax.jit, static_argnames=("method",))
def scaled_hankel1_point(nu, z, *, method: str = "auto"):
    cdt = _complex_dtype(nu, z)
    if method == "asymptotic":
        return scaled_hankel1_asymptotic(nu, z)
    if method == "direct":
        return jnp.exp(-1j * jnp.asarray(z, dtype=cdt)) * hankel1_point(nu, z, method="direct")
    return lax.cond(
        jnp.abs(jnp.asarray(z)) >= 16.0,
        lambda _: scaled_hankel1_asymptotic(nu, z),
        lambda _: jnp.exp(-1j * jnp.asarray(z, dtype=cdt)) * hankel1_point(nu, z, method="direct"),
        operand=None,
    )


@partial(jax.jit, static_argnames=("method",))
def scaled_hankel2_point(nu, z, *, method: str = "auto"):
    cdt = _complex_dtype(nu, z)
    if method == "asymptotic":
        return scaled_hankel2_asymptotic(nu, z)
    if method == "direct":
        return jnp.exp(1j * jnp.asarray(z, dtype=cdt)) * hankel2_point(nu, z, method="direct")
    return lax.cond(
        jnp.abs(jnp.asarray(z)) >= 16.0,
        lambda _: scaled_hankel2_asymptotic(nu, z),
        lambda _: jnp.exp(1j * jnp.asarray(z, dtype=cdt)) * hankel2_point(nu, z, method="direct"),
        operand=None,
    )


def hankel1(nu, z, *, method: str = "auto"):
    return hankel1_point(nu, z, method=method)


def hankel2(nu, z, *, method: str = "auto"):
    return hankel2_point(nu, z, method=method)


def scaled_hankel1(nu, z, *, method: str = "auto"):
    return scaled_hankel1_point(nu, z, method=method)


def scaled_hankel2(nu, z, *, method: str = "auto"):
    return scaled_hankel2_point(nu, z, method=method)


__all__ = [
    "hankel1",
    "hankel1_point",
    "hankel2",
    "hankel2_point",
    "scaled_hankel1",
    "scaled_hankel1_point",
    "scaled_hankel2",
    "scaled_hankel2_point",
]
