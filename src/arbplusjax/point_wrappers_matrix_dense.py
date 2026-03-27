from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di
from . import kernel_helpers as kh
from . import mat_common


def _pad_args_repeat_last(args, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)


def _arb_mat_point_matrix(a: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_matrix(a, "point_wrappers_matrix_dense.arb_mat_point_matrix"))


def _arb_mat_point_vector(x: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_vector(x, "point_wrappers_matrix_dense.arb_mat_point_vector"))


def _acb_mat_point_matrix(a: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_matrix(a, "point_wrappers_matrix_dense.acb_mat_point_matrix"))


def _acb_mat_point_vector(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_vector(x, "point_wrappers_matrix_dense.acb_mat_point_vector"))


def arb_mat_zero_point(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.zeros((n, n), dtype=dtype)


def arb_mat_identity_point(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.eye(n, dtype=dtype)


@partial(jax.jit, static_argnames=())
def arb_mat_matmul_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    b_mid = _arb_mat_point_matrix(b)
    checks.check_equal(a_mid.shape[-1], b_mid.shape[-2], "point_wrappers_matrix_dense.arb_mat_matmul_point.inner")
    return jnp.matmul(a_mid, b_mid)


@partial(jax.jit, static_argnames=())
def arb_mat_matvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    x_mid = _arb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-1], x_mid.shape[-1], "point_wrappers_matrix_dense.arb_mat_matvec_point.inner")
    return jnp.einsum("...ij,...j->...i", a_mid, x_mid)


@partial(jax.jit, static_argnames=())
def arb_mat_det_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.det(_arb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def arb_mat_trace_point(a: jax.Array) -> jax.Array:
    return jnp.trace(_arb_mat_point_matrix(a), axis1=-2, axis2=-1)


@partial(jax.jit, static_argnames=())
def arb_mat_sqr_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    return jnp.matmul(a_mid, a_mid)


@partial(jax.jit, static_argnames=())
def arb_mat_norm_fro_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_arb_mat_point_matrix(a), ord="fro", axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def arb_mat_norm_1_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_arb_mat_point_matrix(a), ord=1, axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def arb_mat_norm_inf_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_arb_mat_point_matrix(a), ord=jnp.inf, axis=(-2, -1))


def arb_mat_matmul_batch_fixed_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_matmul_point(a, b)


def arb_mat_matmul_batch_padded_point(a: jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b), pad_to)
    return arb_mat_matmul_point(*call_args)


def arb_mat_matvec_batch_fixed_point(a: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_point(a, x)


def arb_mat_matvec_batch_padded_point(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, x), pad_to)
    return arb_mat_matvec_point(*call_args)


def arb_mat_det_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_det_point(a)


def arb_mat_det_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_det_point(*call_args)


def arb_mat_trace_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_trace_point(a)


def arb_mat_trace_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_trace_point(*call_args)


def arb_mat_sqr_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_sqr_point(a)


def arb_mat_sqr_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_sqr_point(*call_args)


def arb_mat_norm_fro_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_norm_fro_point(a)


def arb_mat_norm_fro_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_norm_fro_point(*call_args)


def arb_mat_norm_1_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_norm_1_point(a)


def arb_mat_norm_1_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_norm_1_point(*call_args)


def arb_mat_norm_inf_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_norm_inf_point(a)


def arb_mat_norm_inf_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_norm_inf_point(*call_args)


def acb_mat_zero_point(n: int, *, dtype: jnp.dtype = jnp.complex128) -> jax.Array:
    return jnp.zeros((n, n), dtype=dtype)


def acb_mat_identity_point(n: int, *, dtype: jnp.dtype = jnp.complex128) -> jax.Array:
    return jnp.eye(n, dtype=dtype)


@partial(jax.jit, static_argnames=())
def acb_mat_matmul_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    b_mid = _acb_mat_point_matrix(b)
    checks.check_equal(a_mid.shape[-1], b_mid.shape[-2], "point_wrappers_matrix_dense.acb_mat_matmul_point.inner")
    return jnp.matmul(a_mid, b_mid)


@partial(jax.jit, static_argnames=())
def acb_mat_matvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    x_mid = _acb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-1], x_mid.shape[-1], "point_wrappers_matrix_dense.acb_mat_matvec_point.inner")
    return jnp.einsum("...ij,...j->...i", a_mid, x_mid)


@partial(jax.jit, static_argnames=())
def acb_mat_det_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.det(_acb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def acb_mat_trace_point(a: jax.Array) -> jax.Array:
    return jnp.trace(_acb_mat_point_matrix(a), axis1=-2, axis2=-1)


@partial(jax.jit, static_argnames=())
def acb_mat_sqr_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    return jnp.matmul(a_mid, a_mid)


@partial(jax.jit, static_argnames=())
def acb_mat_norm_fro_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_acb_mat_point_matrix(a), ord="fro", axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def acb_mat_norm_1_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_acb_mat_point_matrix(a), ord=1, axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def acb_mat_norm_inf_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_acb_mat_point_matrix(a), ord=jnp.inf, axis=(-2, -1))


def acb_mat_matmul_batch_fixed_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_matmul_point(a, b)


def acb_mat_matmul_batch_padded_point(a: jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b), pad_to)
    return acb_mat_matmul_point(*call_args)


def acb_mat_matvec_batch_fixed_point(a: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_point(a, x)


def acb_mat_matvec_batch_padded_point(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, x), pad_to)
    return acb_mat_matvec_point(*call_args)


def acb_mat_det_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_det_point(a)


def acb_mat_det_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_det_point(*call_args)


def acb_mat_trace_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_trace_point(a)


def acb_mat_trace_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_trace_point(*call_args)


def acb_mat_sqr_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_sqr_point(a)


def acb_mat_sqr_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_sqr_point(*call_args)


def acb_mat_norm_fro_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_norm_fro_point(a)


def acb_mat_norm_fro_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_norm_fro_point(*call_args)


def acb_mat_norm_1_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_norm_1_point(a)


def acb_mat_norm_1_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_norm_1_point(*call_args)


def acb_mat_norm_inf_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_norm_inf_point(a)


def acb_mat_norm_inf_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_norm_inf_point(*call_args)


__all__ = sorted(name for name in globals() if name.startswith(("arb_mat_", "acb_mat_")))
