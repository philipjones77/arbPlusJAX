from __future__ import annotations

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di

jax.config.update("jax_enable_x64", True)


def as_interval_mat_2x2(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check_tail_shape(arr, (2, 2, 2), label)
    return arr


def as_box_mat_2x2(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check_tail_shape(arr, (2, 2, 4), label)
    return arr


def full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def interval_from_point(x: jax.Array) -> jax.Array:
    return di.interval(di._below(x), di._above(x))


def box_from_point(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def interval_is_finite(x: jax.Array) -> jax.Array:
    return jnp.isfinite(x[..., 0]) & jnp.isfinite(x[..., 1])


def box_is_finite(x: jax.Array) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    return interval_is_finite(re) & interval_is_finite(im)


def complex_is_finite(z: jax.Array) -> jax.Array:
    return jnp.isfinite(jnp.real(z)) & jnp.isfinite(jnp.imag(z))


__all__ = [
    "as_interval_mat_2x2",
    "as_box_mat_2x2",
    "full_interval_like",
    "full_box_like",
    "interval_from_point",
    "box_from_point",
    "interval_is_finite",
    "box_is_finite",
    "complex_is_finite",
]
