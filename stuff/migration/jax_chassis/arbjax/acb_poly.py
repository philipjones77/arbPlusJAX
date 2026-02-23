from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di

jax.config.update("jax_enable_x64", True)


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def _acb_from_complex(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def _as_coeffs(coeffs: jax.Array) -> jax.Array:
    arr = acb_core.as_acb_box(coeffs)
    if arr.shape[-2:] != (4, 4):
        raise ValueError(f"Expected coeffs shape (..., 4, 4), got {arr.shape}")
    return arr


def acb_poly_eval_cubic(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    z = acb_core.as_acb_box(z)
    if z.shape[-1] != 4:
        raise ValueError(f"Expected z shape (..., 4), got {z.shape}")
    c = acb_core.acb_midpoint(coeffs)
    zz = acb_core.acb_midpoint(z)
    v = c[..., 3]
    v = v * zz + c[..., 2]
    v = v * zz + c[..., 1]
    v = v * zz + c[..., 0]
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(z))


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_poly_eval_cubic_prec(
    coeffs: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_poly_eval_cubic(coeffs, z), prec_bits)


def acb_poly_eval_cubic_batch(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    z = acb_core.as_acb_box(z)
    return jax.vmap(acb_poly_eval_cubic)(coeffs, z)


def acb_poly_eval_cubic_batch_prec(
    coeffs: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_poly_eval_cubic_batch(coeffs, z), prec_bits)


acb_poly_eval_cubic_batch_jit = jax.jit(acb_poly_eval_cubic_batch)
acb_poly_eval_cubic_batch_prec_jit = jax.jit(
    acb_poly_eval_cubic_batch_prec, static_argnames=("prec_bits",)
)


__all__ = [
    "acb_poly_eval_cubic",
    "acb_poly_eval_cubic_prec",
    "acb_poly_eval_cubic_batch",
    "acb_poly_eval_cubic_batch_prec",
    "acb_poly_eval_cubic_batch_jit",
    "acb_poly_eval_cubic_batch_prec_jit",
]
