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


def _as_mat_2x2(x: jax.Array) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    if arr.shape[-3:] != (2, 2, 4):
        raise ValueError(f"Expected shape (..., 2, 2, 4), got {arr.shape}")
    return arr


def acb_mat_2x2_det(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = acb_core.acb_midpoint(a[..., 0, 0, :])
    a01 = acb_core.acb_midpoint(a[..., 0, 1, :])
    a10 = acb_core.acb_midpoint(a[..., 1, 0, :])
    a11 = acb_core.acb_midpoint(a[..., 1, 1, :])
    det = a00 * a11 - a01 * a10
    finite = jnp.isfinite(jnp.real(det)) & jnp.isfinite(jnp.imag(det))
    out = _acb_from_complex(det)
    return jnp.where(finite[..., None], out, _full_box_like(a[..., 0, 0, :]))


def acb_mat_2x2_trace(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = acb_core.acb_midpoint(a[..., 0, 0, :])
    a11 = acb_core.acb_midpoint(a[..., 1, 1, :])
    tr = a00 + a11
    finite = jnp.isfinite(jnp.real(tr)) & jnp.isfinite(jnp.imag(tr))
    out = _acb_from_complex(tr)
    return jnp.where(finite[..., None], out, _full_box_like(a[..., 0, 0, :]))


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_2x2_det_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_2x2_det(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_2x2_trace_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_2x2_trace(a), prec_bits)


def acb_mat_2x2_det_batch(a: jax.Array) -> jax.Array:
    return acb_mat_2x2_det(a)


def acb_mat_2x2_trace_batch(a: jax.Array) -> jax.Array:
    return acb_mat_2x2_trace(a)


def acb_mat_2x2_det_batch_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_2x2_det_batch(a), prec_bits)


def acb_mat_2x2_trace_batch_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_2x2_trace_batch(a), prec_bits)


acb_mat_2x2_det_batch_jit = jax.jit(acb_mat_2x2_det_batch)
acb_mat_2x2_trace_batch_jit = jax.jit(acb_mat_2x2_trace_batch)
acb_mat_2x2_det_batch_prec_jit = jax.jit(acb_mat_2x2_det_batch_prec, static_argnames=("prec_bits",))
acb_mat_2x2_trace_batch_prec_jit = jax.jit(acb_mat_2x2_trace_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "acb_mat_2x2_det",
    "acb_mat_2x2_trace",
    "acb_mat_2x2_det_prec",
    "acb_mat_2x2_trace_prec",
    "acb_mat_2x2_det_batch",
    "acb_mat_2x2_trace_batch",
    "acb_mat_2x2_det_batch_prec",
    "acb_mat_2x2_trace_batch_prec",
    "acb_mat_2x2_det_batch_jit",
    "acb_mat_2x2_trace_batch_jit",
    "acb_mat_2x2_det_batch_prec_jit",
    "acb_mat_2x2_trace_batch_prec_jit",
]
