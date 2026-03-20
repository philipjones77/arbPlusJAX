from __future__ import annotations

import jax.numpy as jnp

from ... import elementary as el


def _complex_dtype(*xs) -> jnp.dtype:
    return el.complex_dtype_from(*xs)


def _phase(nu, z):
    cdt = _complex_dtype(nu, z)
    quarter_pi = jnp.asarray(0.25 * el.PI, dtype=cdt)
    half_pi = jnp.asarray(el.HALF_PI, dtype=cdt)
    return jnp.asarray(z, dtype=cdt) - half_pi * jnp.asarray(nu, dtype=cdt) - quarter_pi


def hankel1_asymptotic(nu, z):
    cdt = _complex_dtype(nu, z)
    amplitude = jnp.sqrt(jnp.asarray(2.0 / el.PI, dtype=cdt) / jnp.asarray(z, dtype=cdt))
    return amplitude * jnp.exp(1j * _phase(nu, z))


def hankel2_asymptotic(nu, z):
    cdt = _complex_dtype(nu, z)
    amplitude = jnp.sqrt(jnp.asarray(2.0 / el.PI, dtype=cdt) / jnp.asarray(z, dtype=cdt))
    return amplitude * jnp.exp(-1j * _phase(nu, z))


def scaled_hankel1_asymptotic(nu, z):
    cdt = _complex_dtype(nu, z)
    return jnp.exp(-1j * jnp.asarray(z, dtype=cdt)) * hankel1_asymptotic(nu, z)


def scaled_hankel2_asymptotic(nu, z):
    cdt = _complex_dtype(nu, z)
    return jnp.exp(1j * jnp.asarray(z, dtype=cdt)) * hankel2_asymptotic(nu, z)


__all__ = [
    "hankel1_asymptotic",
    "hankel2_asymptotic",
    "scaled_hankel1_asymptotic",
    "scaled_hankel2_asymptotic",
]
