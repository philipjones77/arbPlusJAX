from __future__ import annotations

import jax.numpy as jnp

from ... import elementary as el


def _dtype(z):
    return jnp.asarray(z).dtype


def _phase(n, z):
    dt = _dtype(z)
    return jnp.asarray(z, dtype=dt) - jnp.asarray(0.5 * n * el.PI, dtype=dt)


def spherical_bessel_j_asymptotic(n, z):
    dt = _dtype(z)
    z_v = jnp.asarray(z, dtype=dt)
    return jnp.sin(_phase(n, z_v)) / z_v


def spherical_bessel_y_asymptotic(n, z):
    dt = _dtype(z)
    z_v = jnp.asarray(z, dtype=dt)
    return -jnp.cos(_phase(n, z_v)) / z_v


def modified_spherical_bessel_i_asymptotic(n, z):
    dt = _dtype(z)
    z_v = jnp.asarray(z, dtype=dt)
    return jnp.exp(z_v) / (2.0 * z_v)


def modified_spherical_bessel_k_asymptotic(n, z):
    dt = _dtype(z)
    z_v = jnp.asarray(z, dtype=dt)
    return jnp.asarray(0.5 * el.PI, dtype=dt) * jnp.exp(-z_v) / z_v


__all__ = [
    "spherical_bessel_j_asymptotic",
    "spherical_bessel_y_asymptotic",
    "modified_spherical_bessel_i_asymptotic",
    "modified_spherical_bessel_k_asymptotic",
]
