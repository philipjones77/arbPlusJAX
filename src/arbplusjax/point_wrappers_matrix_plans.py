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
    return di.midpoint(mat_common.as_interval_matrix(a, "point_wrappers_matrix_plans.arb_mat_point_matrix"))


def _arb_mat_point_vector(x: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_vector(x, "point_wrappers_matrix_plans.arb_mat_point_vector"))


def _arb_mat_point_rhs(x: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_rhs(x, "point_wrappers_matrix_plans.arb_mat_point_rhs"))


def _acb_mat_point_matrix(a: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_matrix(a, "point_wrappers_matrix_plans.acb_mat_point_matrix"))


def _acb_mat_point_vector(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_vector(x, "point_wrappers_matrix_plans.acb_mat_point_vector"))


def _acb_mat_point_rhs(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_rhs(x, "point_wrappers_matrix_plans.acb_mat_point_rhs"))


@partial(jax.jit, static_argnames=())
def arb_mat_lu_point(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_mid = _arb_mat_point_matrix(a)
    lu, _, perm = jax.lax.linalg.lu(a_mid)
    n = a_mid.shape[-1]
    eye = jnp.eye(n, dtype=a_mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return p, l, u


@partial(jax.jit, static_argnames=())
def arb_mat_cho_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    sym = 0.5 * (a_mid + jnp.swapaxes(a_mid, -2, -1))
    return jnp.linalg.cholesky(sym)


def arb_mat_dense_lu_solve_plan_prepare_point(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    p, l, u = arb_mat_lu_point(a)
    return mat_common.DenseLUSolvePlan(p=p, l=l, u=u, rows=int(p.shape[-1]), algebra="arb")


def arb_mat_dense_spd_solve_plan_prepare_point(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    factor = arb_mat_cho_point(a)
    return mat_common.DenseCholeskySolvePlan(factor=factor, rows=int(factor.shape[-1]), algebra="arb", structure="symmetric")


def arb_mat_dense_matvec_plan_prepare_point(a: jax.Array) -> mat_common.DenseMatvecPlan:
    matrix = _arb_mat_point_matrix(a)
    return mat_common.DenseMatvecPlan(matrix=matrix, rows=int(matrix.shape[-2]), cols=int(matrix.shape[-1]), algebra="arb")


def arb_mat_dense_matvec_plan_apply_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    matrix = jnp.asarray(plan.matrix) if isinstance(plan, mat_common.DenseMatvecPlan) else _arb_mat_point_matrix(plan)
    return jnp.einsum("...ij,...j->...i", matrix, _arb_mat_point_vector(x))


def arb_mat_matvec_cached_prepare_point(a: jax.Array) -> jax.Array:
    return _arb_mat_point_matrix(a)


def arb_mat_matvec_cached_apply_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    x_mid = _arb_mat_point_vector(x)
    checks.check_equal(cache.shape[-1], x_mid.shape[-1], "point_wrappers_matrix_plans.arb_mat_matvec_cached_apply_point.inner")
    return jnp.einsum("...ij,...j->...i", jnp.asarray(cache), x_mid)


def arb_mat_dense_lu_solve_plan_apply_point(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="arb", label="point_wrappers_matrix_plans.arb_mat_dense_lu_solve_plan_apply_point")
    p_mid = jnp.asarray(plan.p)
    l_mid = jnp.asarray(plan.l)
    u_mid = jnp.asarray(plan.u)
    b_mid = _arb_mat_point_rhs(b)
    vector_rhs = b_mid.ndim == p_mid.ndim - 1
    pb = jnp.einsum("...ij,...j->...i", p_mid, b_mid) if vector_rhs else jnp.matmul(p_mid, b_mid)
    y = jax.lax.linalg.triangular_solve(l_mid, pb[..., None] if vector_rhs else pb, left_side=True, lower=True, unit_diagonal=True)
    out = jax.lax.linalg.triangular_solve(u_mid, y, left_side=True, lower=False, unit_diagonal=False)
    return out[..., 0] if vector_rhs else out


def arb_mat_dense_spd_solve_plan_apply_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else arb_mat_cho_point(plan)
    return mat_common.lower_cholesky_solve(factor, _arb_mat_point_rhs(b))


def arb_mat_spd_inv_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else arb_mat_cho_point(plan)
    eye = jnp.broadcast_to(jnp.eye(factor.shape[-1], dtype=factor.dtype), factor.shape)
    return mat_common.lower_cholesky_solve(factor, eye)


def arb_mat_matvec_cached_apply_batch_fixed_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply_point(cache, x)


def arb_mat_matvec_cached_apply_batch_padded_point(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((cache, x), pad_to)
    return arb_mat_matvec_cached_apply_point(*call_args)


def arb_mat_dense_matvec_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return arb_mat_dense_matvec_plan_prepare_point(a)


def arb_mat_dense_matvec_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseMatvecPlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_dense_matvec_plan_prepare_point(*call_args)


def arb_mat_dense_matvec_plan_apply_batch_fixed_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_dense_matvec_plan_apply_point(plan, x)


def arb_mat_dense_matvec_plan_apply_batch_padded_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    (x_pad,), _ = _pad_args_repeat_last((x,), pad_to)
    return arb_mat_dense_matvec_plan_apply_point(plan, x_pad)


def arb_mat_dense_lu_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    return arb_mat_dense_lu_solve_plan_prepare_point(a)


def arb_mat_dense_lu_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseLUSolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_dense_lu_solve_plan_prepare_point(*call_args)


def arb_mat_dense_spd_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    return arb_mat_dense_spd_solve_plan_prepare_point(a)


def arb_mat_dense_spd_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseCholeskySolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_dense_spd_solve_plan_prepare_point(*call_args)


def arb_mat_dense_lu_solve_plan_apply_batch_fixed_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return arb_mat_dense_lu_solve_plan_apply_point(plan, b)


def arb_mat_dense_lu_solve_plan_apply_batch_padded_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array, *, pad_to: int) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return arb_mat_dense_lu_solve_plan_apply_point(plan, b_pad)


def arb_mat_dense_spd_solve_plan_apply_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_dense_spd_solve_plan_apply_point(plan, b)


def arb_mat_dense_spd_solve_plan_apply_batch_padded_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return arb_mat_dense_spd_solve_plan_apply_point(plan, b_pad)


def arb_mat_spd_inv_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return arb_mat_spd_inv_point(plan)


def arb_mat_spd_inv_batch_padded_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, *, pad_to: int) -> jax.Array:
    del pad_to
    return arb_mat_spd_inv_point(plan)


@partial(jax.jit, static_argnames=())
def acb_mat_lu_point(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_mid = _acb_mat_point_matrix(a)
    lu, _, perm = jax.lax.linalg.lu(a_mid)
    n = a_mid.shape[-1]
    eye = jnp.eye(n, dtype=a_mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return p, l, u


@partial(jax.jit, static_argnames=())
def acb_mat_cho_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    herm = 0.5 * (a_mid + jnp.conj(jnp.swapaxes(a_mid, -2, -1)))
    return jnp.linalg.cholesky(herm)


def acb_mat_dense_lu_solve_plan_prepare_point(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    p, l, u = acb_mat_lu_point(a)
    return mat_common.DenseLUSolvePlan(p=p, l=l, u=u, rows=int(p.shape[-1]), algebra="acb")


def acb_mat_dense_hpd_solve_plan_prepare_point(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    factor = acb_mat_cho_point(a)
    return mat_common.DenseCholeskySolvePlan(factor=factor, rows=int(factor.shape[-1]), algebra="acb", structure="hermitian")


def acb_mat_dense_matvec_plan_prepare_point(a: jax.Array) -> mat_common.DenseMatvecPlan:
    matrix = _acb_mat_point_matrix(a)
    return mat_common.DenseMatvecPlan(matrix=matrix, rows=int(matrix.shape[-2]), cols=int(matrix.shape[-1]), algebra="acb")


def acb_mat_dense_matvec_plan_apply_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    matrix = jnp.asarray(plan.matrix) if isinstance(plan, mat_common.DenseMatvecPlan) else _acb_mat_point_matrix(plan)
    return jnp.einsum("...ij,...j->...i", matrix, _acb_mat_point_vector(x))


def acb_mat_matvec_cached_prepare_point(a: jax.Array) -> jax.Array:
    return _acb_mat_point_matrix(a)


def acb_mat_matvec_cached_apply_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    x_mid = _acb_mat_point_vector(x)
    checks.check_equal(cache.shape[-1], x_mid.shape[-1], "point_wrappers_matrix_plans.acb_mat_matvec_cached_apply_point.inner")
    return jnp.einsum("...ij,...j->...i", jnp.asarray(cache), x_mid)


def acb_mat_dense_lu_solve_plan_apply_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="acb", label="point_wrappers_matrix_plans.acb_mat_dense_lu_solve_plan_apply_point")
    p_mid = jnp.asarray(plan.p)
    l_mid = jnp.asarray(plan.l)
    u_mid = jnp.asarray(plan.u)
    b_mid = _acb_mat_point_rhs(b)
    vector_rhs = b_mid.ndim == p_mid.ndim - 1
    pb = jnp.einsum("...ij,...j->...i", p_mid, b_mid) if vector_rhs else jnp.matmul(p_mid, b_mid)
    y = jax.lax.linalg.triangular_solve(l_mid, pb[..., None] if vector_rhs else pb, left_side=True, lower=True, unit_diagonal=True)
    out = jax.lax.linalg.triangular_solve(u_mid, y, left_side=True, lower=False, unit_diagonal=False)
    return out[..., 0] if vector_rhs else out


def acb_mat_dense_hpd_solve_plan_apply_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else acb_mat_cho_point(plan)
    return mat_common.lower_cholesky_solve(factor, _acb_mat_point_rhs(b))


def acb_mat_hpd_inv_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else acb_mat_cho_point(plan)
    eye = jnp.broadcast_to(jnp.eye(factor.shape[-1], dtype=factor.dtype), factor.shape)
    return mat_common.lower_cholesky_solve(factor, eye)


def acb_mat_matvec_cached_apply_batch_fixed_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_apply_point(cache, x)


def acb_mat_matvec_cached_apply_batch_padded_point(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((cache, x), pad_to)
    return acb_mat_matvec_cached_apply_point(*call_args)


def acb_mat_dense_matvec_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return acb_mat_dense_matvec_plan_prepare_point(a)


def acb_mat_dense_matvec_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseMatvecPlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_dense_matvec_plan_prepare_point(*call_args)


def acb_mat_dense_matvec_plan_apply_batch_fixed_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_dense_matvec_plan_apply_point(plan, x)


def acb_mat_dense_matvec_plan_apply_batch_padded_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    (x_pad,), _ = _pad_args_repeat_last((x,), pad_to)
    return acb_mat_dense_matvec_plan_apply_point(plan, x_pad)


def acb_mat_dense_lu_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    return acb_mat_dense_lu_solve_plan_prepare_point(a)


def acb_mat_dense_lu_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseLUSolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_dense_lu_solve_plan_prepare_point(*call_args)


def acb_mat_dense_hpd_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    return acb_mat_dense_hpd_solve_plan_prepare_point(a)


def acb_mat_dense_hpd_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseCholeskySolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_dense_hpd_solve_plan_prepare_point(*call_args)


def acb_mat_dense_lu_solve_plan_apply_batch_fixed_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return acb_mat_dense_lu_solve_plan_apply_point(plan, b)


def acb_mat_dense_lu_solve_plan_apply_batch_padded_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array, *, pad_to: int) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return acb_mat_dense_lu_solve_plan_apply_point(plan, b_pad)


def acb_mat_dense_hpd_solve_plan_apply_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_dense_hpd_solve_plan_apply_point(plan, b)


def acb_mat_dense_hpd_solve_plan_apply_batch_padded_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return acb_mat_dense_hpd_solve_plan_apply_point(plan, b_pad)


def acb_mat_hpd_inv_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return acb_mat_hpd_inv_point(plan)


def acb_mat_hpd_inv_batch_padded_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, *, pad_to: int) -> jax.Array:
    del pad_to
    return acb_mat_hpd_inv_point(plan)


__all__ = sorted(name for name in globals() if name.startswith(("arb_mat_", "acb_mat_")))
