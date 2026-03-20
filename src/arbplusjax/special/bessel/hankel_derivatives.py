from __future__ import annotations

import jax.numpy as jnp

from .hankel import hankel1_point, hankel2_point, scaled_hankel1_point, scaled_hankel2_point


def hankel1_derivative(nu, z, *, method: str = "auto"):
    nu_v = jnp.asarray(nu)
    z_v = jnp.asarray(z)
    return 0.5 * (hankel1_point(nu_v - 1.0, z_v, method=method) - hankel1_point(nu_v + 1.0, z_v, method=method))


def hankel2_derivative(nu, z, *, method: str = "auto"):
    nu_v = jnp.asarray(nu)
    z_v = jnp.asarray(z)
    return 0.5 * (hankel2_point(nu_v - 1.0, z_v, method=method) - hankel2_point(nu_v + 1.0, z_v, method=method))


def scaled_hankel1_derivative(nu, z, *, method: str = "auto"):
    z_v = jnp.asarray(z)
    return -1j * scaled_hankel1_point(nu, z_v, method=method) + jnp.exp(-1j * z_v) * hankel1_derivative(nu, z_v, method=method)


def scaled_hankel2_derivative(nu, z, *, method: str = "auto"):
    z_v = jnp.asarray(z)
    return 1j * scaled_hankel2_point(nu, z_v, method=method) + jnp.exp(1j * z_v) * hankel2_derivative(nu, z_v, method=method)


__all__ = [
    "hankel1_derivative",
    "hankel2_derivative",
    "scaled_hankel1_derivative",
    "scaled_hankel2_derivative",
]
