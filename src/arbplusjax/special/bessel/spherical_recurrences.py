from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from ... import elementary as el


def _dtype(z):
    return jnp.asarray(z).dtype


def _recur_up(n, z, f0, f1, *, mode: str):
    n_int = jnp.asarray(n, dtype=jnp.int32)
    z_v = jnp.asarray(z, dtype=_dtype(z))

    def many(_):
        def cond_fn(state):
            k, _, _ = state
            return k < n_int

        def body_fn(state):
            k, prev, curr = state
            coeff = jnp.asarray(2, dtype=z_v.dtype) * k.astype(z_v.dtype) + jnp.asarray(1, dtype=z_v.dtype)
            scaled = (coeff / z_v) * curr
            if mode == "subtract_prev":
                nxt = scaled - prev
            elif mode == "prev_minus":
                nxt = prev - scaled
            else:
                nxt = prev + scaled
            return k + jnp.asarray(1, dtype=jnp.int32), curr, nxt

        _, _, out = lax.while_loop(cond_fn, body_fn, (jnp.asarray(1, dtype=jnp.int32), f0, f1))
        return out

    return lax.cond(
        n_int == 0,
        lambda _: f0,
        lambda _: lax.cond(n_int == 1, lambda __: f1, many, operand=None),
        operand=None,
    )


@partial(jax.jit, static_argnames=())
def spherical_bessel_j_recurrence(n, z):
    z_v = jnp.asarray(z, dtype=_dtype(z))
    f0 = jnp.sin(z_v) / z_v
    f1 = jnp.sin(z_v) / (z_v * z_v) - jnp.cos(z_v) / z_v
    return _recur_up(n, z_v, f0, f1, mode="subtract_prev")


@partial(jax.jit, static_argnames=())
def spherical_bessel_y_recurrence(n, z):
    z_v = jnp.asarray(z, dtype=_dtype(z))
    f0 = -jnp.cos(z_v) / z_v
    f1 = -jnp.cos(z_v) / (z_v * z_v) - jnp.sin(z_v) / z_v
    return _recur_up(n, z_v, f0, f1, mode="subtract_prev")


@partial(jax.jit, static_argnames=())
def modified_spherical_bessel_i_recurrence(n, z):
    z_v = jnp.asarray(z, dtype=_dtype(z))
    f0 = jnp.sinh(z_v) / z_v
    f1 = jnp.cosh(z_v) / z_v - jnp.sinh(z_v) / (z_v * z_v)
    return _recur_up(n, z_v, f0, f1, mode="prev_minus")


@partial(jax.jit, static_argnames=())
def modified_spherical_bessel_k_recurrence(n, z):
    z_v = jnp.asarray(z, dtype=_dtype(z))
    f0 = jnp.asarray(0.5 * el.PI, dtype=z_v.dtype) * jnp.exp(-z_v) / z_v
    f1 = f0 * (1.0 + 1.0 / z_v)
    return _recur_up(n, z_v, f0, f1, mode="prev_plus")


__all__ = [
    "spherical_bessel_j_recurrence",
    "spherical_bessel_y_recurrence",
    "modified_spherical_bessel_i_recurrence",
    "modified_spherical_bessel_k_recurrence",
]
