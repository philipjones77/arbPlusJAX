from __future__ import annotations

import jax
import jax.numpy as jnp

from .spherical import (
    modified_spherical_bessel_i_point,
    modified_spherical_bessel_k_point,
    spherical_bessel_j_point,
    spherical_bessel_y_point,
)


def spherical_bessel_j_derivative(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    return jax.lax.cond(
        n_v == 0,
        lambda _: -spherical_bessel_j_point(1, z_v, method=method),
        lambda _: spherical_bessel_j_point(n_v - 1, z_v, method=method) - ((n_v.astype(z_v.dtype) + 1.0) / z_v) * spherical_bessel_j_point(n_v, z_v, method=method),
        operand=None,
    )


def spherical_bessel_y_derivative(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    return jax.lax.cond(
        n_v == 0,
        lambda _: -spherical_bessel_y_point(1, z_v, method=method),
        lambda _: spherical_bessel_y_point(n_v - 1, z_v, method=method) - ((n_v.astype(z_v.dtype) + 1.0) / z_v) * spherical_bessel_y_point(n_v, z_v, method=method),
        operand=None,
    )


def modified_spherical_bessel_i_derivative(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    return jax.lax.cond(
        n_v == 0,
        lambda _: modified_spherical_bessel_i_point(1, z_v, method=method),
        lambda _: modified_spherical_bessel_i_point(n_v - 1, z_v, method=method) - ((n_v.astype(z_v.dtype) + 1.0) / z_v) * modified_spherical_bessel_i_point(n_v, z_v, method=method),
        operand=None,
    )


def modified_spherical_bessel_k_derivative(n, z, *, method: str = "auto"):
    n_v = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z)
    return jax.lax.cond(
        n_v == 0,
        lambda _: -modified_spherical_bessel_k_point(1, z_v, method=method),
        lambda _: -modified_spherical_bessel_k_point(n_v - 1, z_v, method=method) - ((n_v.astype(z_v.dtype) + 1.0) / z_v) * modified_spherical_bessel_k_point(n_v, z_v, method=method),
        operand=None,
    )


__all__ = [
    "spherical_bessel_j_derivative",
    "spherical_bessel_y_derivative",
    "modified_spherical_bessel_i_derivative",
    "modified_spherical_bessel_k_derivative",
]
