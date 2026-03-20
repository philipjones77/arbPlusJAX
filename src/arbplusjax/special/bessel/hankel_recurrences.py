from __future__ import annotations

import jax.numpy as jnp

from .hankel import hankel1_point, hankel2_point


def hankel1_order_recurrence(nu, z, *, method: str = "auto"):
    nu_v = jnp.asarray(nu)
    z_v = jnp.asarray(z)
    return (2.0 * nu_v / z_v) * hankel1_point(nu_v, z_v, method=method) - hankel1_point(nu_v + 1.0, z_v, method=method)


def hankel2_order_recurrence(nu, z, *, method: str = "auto"):
    nu_v = jnp.asarray(nu)
    z_v = jnp.asarray(z)
    return (2.0 * nu_v / z_v) * hankel2_point(nu_v, z_v, method=method) - hankel2_point(nu_v + 1.0, z_v, method=method)


__all__ = ["hankel1_order_recurrence", "hankel2_order_recurrence"]
