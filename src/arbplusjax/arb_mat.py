from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import checks

jax.config.update("jax_enable_x64", True)


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _as_mat_2x2(x: jax.Array) -> jax.Array:
    arr = di.as_interval(x)
    checks.check_tail_shape(arr, (2, 2, 2), "arb_mat._as_mat_2x2")
    return arr


def arb_mat_2x2_det(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = di.midpoint(a[..., 0, 0, :])
    a01 = di.midpoint(a[..., 0, 1, :])
    a10 = di.midpoint(a[..., 1, 0, :])
    a11 = di.midpoint(a[..., 1, 1, :])
    det = a00 * a11 - a01 * a10
    finite = jnp.isfinite(det)
    out = di.interval(di._below(det), di._above(det))
    return jnp.where(finite[..., None], out, _full_interval_like(a[..., 0, 0, :]))


def arb_mat_2x2_trace(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = di.midpoint(a[..., 0, 0, :])
    a11 = di.midpoint(a[..., 1, 1, :])
    tr = a00 + a11
    finite = jnp.isfinite(tr)
    out = di.interval(di._below(tr), di._above(tr))
    return jnp.where(finite[..., None], out, _full_interval_like(a[..., 0, 0, :]))


def arb_mat_2x2_det_rigorous(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    det = di.fast_sub(di.fast_mul(a00, a11), di.fast_mul(a01, a10))
    finite = jnp.isfinite(det[..., 0]) & jnp.isfinite(det[..., 1])
    return jnp.where(finite[..., None], det, _full_interval_like(a[..., 0, 0, :]))


def arb_mat_2x2_trace_rigorous(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    tr = di.fast_add(a[..., 0, 0, :], a[..., 1, 1, :])
    finite = jnp.isfinite(tr[..., 0]) & jnp.isfinite(tr[..., 1])
    return jnp.where(finite[..., None], tr, _full_interval_like(a[..., 0, 0, :]))


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_2x2_det_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_2x2_det(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_2x2_trace_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_2x2_trace(a), prec_bits)


def arb_mat_2x2_det_batch(a: jax.Array) -> jax.Array:
    return arb_mat_2x2_det(a)


def arb_mat_2x2_trace_batch(a: jax.Array) -> jax.Array:
    return arb_mat_2x2_trace(a)


def arb_mat_2x2_det_batch_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_2x2_det_rigorous(a)


def arb_mat_2x2_trace_batch_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_2x2_trace_rigorous(a)


def arb_mat_2x2_det_batch_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_2x2_det_batch(a), prec_bits)


def arb_mat_2x2_trace_batch_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_2x2_trace_batch(a), prec_bits)


arb_mat_2x2_det_batch_jit = jax.jit(arb_mat_2x2_det_batch)
arb_mat_2x2_trace_batch_jit = jax.jit(arb_mat_2x2_trace_batch)
arb_mat_2x2_det_batch_prec_jit = jax.jit(arb_mat_2x2_det_batch_prec, static_argnames=("prec_bits",))
arb_mat_2x2_trace_batch_prec_jit = jax.jit(arb_mat_2x2_trace_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "arb_mat_2x2_det",
    "arb_mat_2x2_trace",
    "arb_mat_2x2_det_rigorous",
    "arb_mat_2x2_trace_rigorous",
    "arb_mat_2x2_det_prec",
    "arb_mat_2x2_trace_prec",
    "arb_mat_2x2_det_batch",
    "arb_mat_2x2_trace_batch",
    "arb_mat_2x2_det_batch_rigorous",
    "arb_mat_2x2_trace_batch_rigorous",
    "arb_mat_2x2_det_batch_prec",
    "arb_mat_2x2_trace_batch_prec",
    "arb_mat_2x2_det_batch_jit",
    "arb_mat_2x2_trace_batch_jit",
    "arb_mat_2x2_det_batch_prec_jit",
    "arb_mat_2x2_trace_batch_prec_jit",
]
