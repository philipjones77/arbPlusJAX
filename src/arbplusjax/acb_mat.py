from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di
from . import mat_common

jax.config.update("jax_enable_x64", True)

def _as_mat_2x2(x: jax.Array) -> jax.Array:
    return mat_common.as_box_mat_2x2(x, "acb_mat._as_mat_2x2")


def acb_mat_2x2_det(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = acb_core.acb_midpoint(a[..., 0, 0, :])
    a01 = acb_core.acb_midpoint(a[..., 0, 1, :])
    a10 = acb_core.acb_midpoint(a[..., 1, 0, :])
    a11 = acb_core.acb_midpoint(a[..., 1, 1, :])
    det = a00 * a11 - a01 * a10
    finite = mat_common.complex_is_finite(det)
    out = mat_common.box_from_point(det)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(a[..., 0, 0, :]))


def acb_mat_2x2_trace(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = acb_core.acb_midpoint(a[..., 0, 0, :])
    a11 = acb_core.acb_midpoint(a[..., 1, 1, :])
    tr = a00 + a11
    finite = mat_common.complex_is_finite(tr)
    out = mat_common.box_from_point(tr)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(a[..., 0, 0, :]))


def acb_mat_2x2_det_rigorous(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    det = acb_core.acb_sub(acb_core.acb_mul(a00, a11), acb_core.acb_mul(a01, a10))
    finite = mat_common.box_is_finite(det)
    return jnp.where(finite[..., None], det, mat_common.full_box_like(a[..., 0, 0, :]))


def acb_mat_2x2_trace_rigorous(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    tr = acb_core.acb_add(a[..., 0, 0, :], a[..., 1, 1, :])
    finite = mat_common.box_is_finite(tr)
    return jnp.where(finite[..., None], tr, mat_common.full_box_like(a[..., 0, 0, :]))


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


def acb_mat_2x2_det_batch_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_2x2_det_rigorous(a)


def acb_mat_2x2_trace_batch_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_2x2_trace_rigorous(a)


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
    "acb_mat_2x2_det_rigorous",
    "acb_mat_2x2_trace_rigorous",
    "acb_mat_2x2_det_prec",
    "acb_mat_2x2_trace_prec",
    "acb_mat_2x2_det_batch",
    "acb_mat_2x2_trace_batch",
    "acb_mat_2x2_det_batch_rigorous",
    "acb_mat_2x2_trace_batch_rigorous",
    "acb_mat_2x2_det_batch_prec",
    "acb_mat_2x2_trace_batch_prec",
    "acb_mat_2x2_det_batch_jit",
    "acb_mat_2x2_trace_batch_jit",
    "acb_mat_2x2_det_batch_prec_jit",
    "acb_mat_2x2_trace_batch_prec_jit",
]
