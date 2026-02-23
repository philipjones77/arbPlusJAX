from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_LOG_A = jnp.float64(0.248754477)  # log Glaisher-Kinkelin constant
_LOG_2PI = jnp.log(jnp.float64(2.0 * jnp.pi))

_BARNESG_B = jnp.asarray(
    [
        -1.0 / 30.0,  # B4
        1.0 / 42.0,  # B6
        -1.0 / 30.0,  # B8
        5.0 / 66.0,  # B10
        -691.0 / 2730.0,  # B12
        7.0 / 6.0,  # B14
        -3617.0 / 510.0,  # B16
        43867.0 / 798.0,  # B18
        -174611.0 / 330.0,  # B20
    ],
    dtype=jnp.float64,
)

_LANCZOS = jnp.asarray(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ],
    dtype=jnp.float64,
)


def _complex_loggamma_lanczos(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    z1 = z - jnp.complex128(1.0 + 0.0j)
    x = jnp.complex128(_LANCZOS[0] + 0.0j)

    def body(i, acc):
        return acc + _LANCZOS[i] / (z1 + jnp.float64(i))

    x = lax.fori_loop(1, 9, body, x)
    t = z1 + jnp.float64(7.5)
    return jnp.float64(0.91893853320467274178) + (z1 + 0.5) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)

    def reflection(w):
        return jnp.log(jnp.pi) - jnp.log(jnp.sin(jnp.pi * w)) - _complex_loggamma_lanczos(1.0 - w)

    return lax.cond(jnp.real(z) < 0.5, reflection, _complex_loggamma_lanczos, z)


def _log_barnesg_asymptotic(z: jax.Array) -> jax.Array:
    # Asymptotic expansion for log G(z), using formula for log G(w+1) with w = z-1
    w = z - jnp.complex128(1.0 + 0.0j)
    logw = jnp.log(w)
    w2 = w * w
    term = (w2 / 2.0 - jnp.float64(1.0 / 12.0)) * logw - jnp.float64(0.75) * w2 + 0.5 * w * _LOG_2PI + _LOG_A

    def body(i, acc):
        k = jnp.float64(i + 1)
        coeff = _BARNESG_B[i]
        return acc + coeff / (4.0 * k * (k + 1.0) * (w ** (2.0 * k)))

    return lax.fori_loop(0, _BARNESG_B.shape[0], body, term)


def _is_nonpositive_integer_real(x: jax.Array) -> jax.Array:
    return (x <= 0.0) & (jnp.abs(x - jnp.round(x)) < 1e-12)


@partial(jax.jit, static_argnames=("target",))
def log_barnesg(z: jax.Array, target: float = 5.0) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    max_shift = 16
    n = jnp.maximum(0.0, jnp.ceil(target - jnp.real(z)))
    n = jnp.minimum(n, jnp.float64(max_shift))

    def body(i, state):
        zc, acc = state
        do = jnp.float64(i) < n
        acc = jnp.where(do, acc - _complex_loggamma(zc), acc)
        zc = jnp.where(do, zc + 1.0, zc)
        return zc, acc

    z_shift, acc = lax.fori_loop(0, max_shift, body, (z, jnp.complex128(0.0 + 0.0j)))
    return _log_barnesg_asymptotic(z_shift) + acc


@partial(jax.jit, static_argnames=("target",))
def barnesg(z: jax.Array, target: float = 5.0) -> jax.Array:
    return jnp.exp(log_barnesg(z, target=target))


@partial(jax.jit, static_argnames=("target",))
def barnesg_real(x: jax.Array, target: float = 5.0) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    pole = _is_nonpositive_integer_real(x)
    val = jnp.real(barnesg(x + 0.0j, target=target))
    return jnp.where(pole, jnp.nan, val)


@partial(jax.jit, static_argnames=("target",))
def barnesg_complex(z: jax.Array, target: float = 5.0) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    pole = (jnp.imag(z) == 0.0) & _is_nonpositive_integer_real(jnp.real(z))
    val = barnesg(z, target=target)
    return jnp.where(pole, jnp.nan + 1j * jnp.nan, val)


__all__ = [
    "log_barnesg",
    "barnesg",
    "barnesg_real",
    "barnesg_complex",
]
