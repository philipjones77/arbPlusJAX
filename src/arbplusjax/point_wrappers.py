from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import acb_core
from . import acb_dirichlet
from . import acb_elliptic
from . import acb_modular
from . import barnesg
from . import checks
from . import double_interval as di
from . import hypgeom
from . import elementary as el
from . import kernel_helpers as kh
from . import mat_common
from .kernel_helpers import scalarize_binary_complex, scalarize_unary_complex, vmap_complex_scalar



# Point-only kernels (no interval or outward rounding)


def _broadcast_flatten(*args):
    arrs = [jnp.asarray(arg) for arg in args]
    bcast = jnp.broadcast_arrays(*arrs)
    shape = bcast[0].shape
    return tuple(jnp.ravel(a) for a in bcast), shape


def _vectorize_real_scalar(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out = jax.vmap(fn)(*flats)
    return out.reshape(shape)


def _vectorize_complex_scalar(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out = jax.vmap(fn)(*flats)
    return out.reshape(shape)


def _vectorize_real_scalar_tuple2(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out1, out2 = jax.vmap(fn)(*flats)
    return out1.reshape(shape), out2.reshape(shape)


def _vectorize_complex_scalar_tuple2(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out1, out2 = jax.vmap(fn)(*flats)
    return out1.reshape(shape), out2.reshape(shape)


def _pad_args_repeat_last(args, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)


def _fixed_unary_point(fn, x: jax.Array, **kwargs):
    return fn(x, **kwargs)


def _padded_unary_point(fn, x: jax.Array, *, pad_to: int, **kwargs):
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return fn(*call_args, **kwargs)


def _arb_mat_point_matrix(a: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_matrix(a, "point_wrappers.arb_mat_point_matrix"))


def _arb_mat_point_vector(x: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_vector(x, "point_wrappers.arb_mat_point_vector"))


def _arb_mat_point_rhs(x: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_rhs(x, "point_wrappers.arb_mat_point_rhs"))


def _arb_mat_point_2x2(a: jax.Array) -> jax.Array:
    return di.midpoint(mat_common.as_interval_mat_2x2(a, "point_wrappers.arb_mat_point_2x2"))


def _acb_mat_point_matrix(a: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_matrix(a, "point_wrappers.acb_mat_point_matrix"))


def _acb_mat_point_vector(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_vector(x, "point_wrappers.acb_mat_point_vector"))


def _acb_mat_point_rhs(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_rhs(x, "point_wrappers.acb_mat_point_rhs"))


def _acb_mat_point_2x2(a: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(mat_common.as_box_mat_2x2(a, "point_wrappers.acb_mat_point_2x2"))


def _band_mask(rows: int, cols: int, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    i = jnp.arange(rows)[:, None]
    j = jnp.arange(cols)[None, :]
    return (i - j <= lower_bandwidth) & (j - i <= upper_bandwidth)


def arb_mat_zero_point(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.zeros((n, n), dtype=dtype)


def arb_mat_identity_point(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.eye(n, dtype=dtype)


def arb_mat_permutation_matrix_point(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    return jnp.eye(perm.shape[-1], dtype=dtype)[perm]


@partial(jax.jit, static_argnames=())
def arb_mat_transpose_point(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(di.midpoint(mat_common.as_interval_rect_matrix(a, "point_wrappers.arb_mat_transpose_point")), -2, -1)


@partial(jax.jit, static_argnames=())
def arb_mat_add_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return _arb_mat_point_matrix(a) + _arb_mat_point_matrix(b)


@partial(jax.jit, static_argnames=())
def arb_mat_sub_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return _arb_mat_point_matrix(a) - _arb_mat_point_matrix(b)


@partial(jax.jit, static_argnames=())
def arb_mat_neg_point(a: jax.Array) -> jax.Array:
    return -_arb_mat_point_matrix(a)


@partial(jax.jit, static_argnames=())
def arb_mat_mul_entrywise_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return _arb_mat_point_matrix(a) * _arb_mat_point_matrix(b)


@partial(jax.jit, static_argnames=("row_start", "row_stop", "col_start", "col_stop"))
def arb_mat_submatrix_point(a: jax.Array, row_start: int, row_stop: int, col_start: int, col_stop: int) -> jax.Array:
    return _arb_mat_point_matrix(a)[..., row_start:row_stop, col_start:col_stop]


@partial(jax.jit, static_argnames=())
def arb_mat_diag_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    idx = jnp.arange(a_mid.shape[-1])
    return a_mid[..., idx, idx]


@partial(jax.jit, static_argnames=())
def arb_mat_diag_matrix_point(d: jax.Array) -> jax.Array:
    d_mid = _arb_mat_point_vector(d)
    eye = jnp.eye(d_mid.shape[-1], dtype=d_mid.dtype)
    return eye * d_mid[..., None, :]


def arb_mat_block_assemble_point(block_rows) -> jax.Array:
    checks.check(len(block_rows) > 0, "point_wrappers.arb_mat_block_assemble_point.rows")
    row_chunks = []
    for row in block_rows:
        row_chunks.append(jnp.concatenate([_arb_mat_point_matrix(block) for block in row], axis=-1))
    return jnp.concatenate(row_chunks, axis=-2)


def arb_mat_block_diag_point(blocks) -> jax.Array:
    checks.check(len(blocks) > 0, "point_wrappers.arb_mat_block_diag_point.blocks")
    matrices = [_arb_mat_point_matrix(block) for block in blocks]
    row_sizes = [int(block.shape[-2]) for block in matrices]
    col_sizes = [int(block.shape[-1]) for block in matrices]
    dtype = matrices[0].dtype
    rows = []
    for i, block in enumerate(matrices):
        row = []
        for j in range(len(matrices)):
            if i == j:
                row.append(block)
            else:
                row.append(jnp.zeros(block.shape[:-2] + (row_sizes[i], col_sizes[j]), dtype=dtype))
        rows.append(tuple(row))
    return arb_mat_block_assemble_point(tuple(rows))


def arb_mat_block_extract_point(a: jax.Array, row_block_sizes, col_block_sizes, row_block: int, col_block: int) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    row_offsets = [0]
    for size in row_block_sizes:
        row_offsets.append(row_offsets[-1] + int(size))
    col_offsets = [0]
    for size in col_block_sizes:
        col_offsets.append(col_offsets[-1] + int(size))
    return a_mid[..., row_offsets[row_block] : row_offsets[row_block + 1], col_offsets[col_block] : col_offsets[col_block + 1]]


def arb_mat_block_row_point(a: jax.Array, row_block_sizes, row_block: int) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    offsets = [0]
    for size in row_block_sizes:
        offsets.append(offsets[-1] + int(size))
    return a_mid[..., offsets[row_block] : offsets[row_block + 1], :]


def arb_mat_block_col_point(a: jax.Array, col_block_sizes, col_block: int) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    offsets = [0]
    for size in col_block_sizes:
        offsets.append(offsets[-1] + int(size))
    return a_mid[..., :, offsets[col_block] : offsets[col_block + 1]]


def arb_mat_block_matmul_point(a_blocks, b_blocks) -> jax.Array:
    out_rows = []
    for i, a_row in enumerate(a_blocks):
        out_row = []
        for j in range(len(b_blocks[0])):
            total = None
            for k in range(len(a_row)):
                prod = arb_mat_matmul_point(a_row[k], b_blocks[k][j])
                total = prod if total is None else total + prod
            out_row.append(total)
        out_rows.append(tuple(out_row))
    return arb_mat_block_assemble_point(tuple(out_rows))


@partial(jax.jit, static_argnames=())
def arb_mat_matmul_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    b_mid = _arb_mat_point_matrix(b)
    checks.check_equal(a_mid.shape[-1], b_mid.shape[-2], "point_wrappers.arb_mat_matmul_point.inner")
    return jnp.matmul(a_mid, b_mid)


@partial(jax.jit, static_argnames=())
def arb_mat_matvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    x_mid = _arb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-1], x_mid.shape[-1], "point_wrappers.arb_mat_matvec_point.inner")
    return jnp.einsum("...ij,...j->...i", a_mid, x_mid)


@partial(jax.jit, static_argnames=("lower_bandwidth", "upper_bandwidth"))
def arb_mat_banded_matvec_point(a: jax.Array, x: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    x_mid = _arb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-1], x_mid.shape[-1], "point_wrappers.arb_mat_banded_matvec_point.inner")
    mask = _band_mask(a_mid.shape[-2], a_mid.shape[-1], lower_bandwidth, upper_bandwidth)
    return jnp.einsum("...ij,...j->...i", jnp.where(mask, a_mid, jnp.zeros_like(a_mid)), x_mid)


@partial(jax.jit, static_argnames=())
def arb_mat_solve_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    b_mid = _arb_mat_point_rhs(b)
    rows = b_mid.shape[-1] if b_mid.ndim == a_mid.ndim - 1 else b_mid.shape[-2]
    checks.check_equal(a_mid.shape[-1], rows, "point_wrappers.arb_mat_solve_point.inner")
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    out = jnp.linalg.solve(a_mid, b_mid[..., None] if vector_rhs else b_mid)
    return out[..., 0] if vector_rhs else out


@partial(jax.jit, static_argnames=())
def arb_mat_inv_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.inv(_arb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def arb_mat_det_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.det(_arb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def arb_mat_trace_point(a: jax.Array) -> jax.Array:
    return jnp.trace(_arb_mat_point_matrix(a), axis1=-2, axis2=-1)


@partial(jax.jit, static_argnames=())
def arb_mat_norm_fro_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_arb_mat_point_matrix(a), ord="fro", axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def arb_mat_norm_1_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_arb_mat_point_matrix(a), ord=1, axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def arb_mat_norm_inf_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_arb_mat_point_matrix(a), ord=jnp.inf, axis=(-2, -1))


@partial(jax.jit, static_argnames=("lower", "unit_diagonal"))
def arb_mat_triangular_solve_point(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    b_mid = _arb_mat_point_rhs(b)
    rows = b_mid.shape[-1] if b_mid.ndim == a_mid.ndim - 1 else b_mid.shape[-2]
    checks.check_equal(a_mid.shape[-1], rows, "point_wrappers.arb_mat_triangular_solve_point.inner")
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    out = lax.linalg.triangular_solve(a_mid, b_mid[..., None] if vector_rhs else b_mid, left_side=True, lower=lower, unit_diagonal=unit_diagonal)
    return out[..., 0] if vector_rhs else out


@partial(jax.jit, static_argnames=())
def arb_mat_lu_point(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_mid = _arb_mat_point_matrix(a)
    lu, _, perm = lax.linalg.lu(a_mid)
    n = a_mid.shape[-1]
    eye = jnp.eye(n, dtype=a_mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return p, l, u


@partial(jax.jit, static_argnames=())
def arb_mat_qr_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.linalg.qr(_arb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def arb_mat_symmetric_part_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    return 0.5 * (a_mid + jnp.swapaxes(a_mid, -2, -1))


def arb_mat_is_symmetric_point(a: jax.Array) -> jax.Array:
    return mat_common.real_midpoint_is_symmetric(_arb_mat_point_matrix(a))


def arb_mat_is_spd_point(a: jax.Array) -> jax.Array:
    chol = jnp.linalg.cholesky(arb_mat_symmetric_part_point(a))
    return arb_mat_is_symmetric_point(a) & mat_common.lower_cholesky_finite(chol)


@partial(jax.jit, static_argnames=())
def arb_mat_cho_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.cholesky(arb_mat_symmetric_part_point(a))


def arb_mat_ldl_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol = arb_mat_cho_point(a)
    diag = jnp.diagonal(chol, axis1=-2, axis2=-1)
    return chol / diag[..., None, :], diag * diag


@partial(jax.jit, static_argnames=())
def arb_mat_eigvalsh_point(a: jax.Array) -> jax.Array:
    values, _ = jnp.linalg.eigh(arb_mat_symmetric_part_point(a))
    return values


@partial(jax.jit, static_argnames=())
def arb_mat_eigh_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.linalg.eigh(arb_mat_symmetric_part_point(a))


@partial(jax.jit, static_argnames=())
def arb_mat_charpoly_point(a: jax.Array) -> jax.Array:
    mid = _arb_mat_point_matrix(a)
    coeffs_general = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=False)
    coeffs_symmetric = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=True)
    return jnp.where(mat_common.real_midpoint_is_symmetric(mid)[..., None], jnp.real(coeffs_symmetric), jnp.real(coeffs_general))


@partial(jax.jit, static_argnames=("n",))
def arb_mat_pow_ui_point(a: jax.Array, n: int) -> jax.Array:
    return mat_common.matrix_power_ui(_arb_mat_point_matrix(a), n)


@partial(jax.jit, static_argnames=())
def arb_mat_exp_point(a: jax.Array) -> jax.Array:
    mid = _arb_mat_point_matrix(a)
    general = mat_common.matrix_exp(mid, hermitian=False)
    symmetric = mat_common.matrix_exp(mid, hermitian=True)
    return jnp.real(jnp.where(mat_common.real_midpoint_is_symmetric(mid)[..., None, None], symmetric, general))


def arb_mat_is_diag_point(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_diagonal(_arb_mat_point_matrix(a))


def arb_mat_is_tril_point(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_arb_mat_point_matrix(a), lower=True)


def arb_mat_is_triu_point(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_arb_mat_point_matrix(a), lower=False)


def arb_mat_is_zero_point(a: jax.Array) -> jax.Array:
    return jnp.all(_arb_mat_point_matrix(a) == 0, axis=(-2, -1))


def arb_mat_is_finite_point(a: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(_arb_mat_point_matrix(a)), axis=(-2, -1))


def arb_mat_is_exact_point(a: jax.Array) -> jax.Array:
    del a
    return jnp.asarray(True)


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


def arb_mat_rmatvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a_mid = di.midpoint(mat_common.as_interval_rect_matrix(a, "point_wrappers.arb_mat_rmatvec_point"))
    x_mid = _arb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-2], x_mid.shape[-1], "point_wrappers.arb_mat_rmatvec_point.inner")
    return jnp.einsum("...ji,...j->...i", a_mid, x_mid)


def arb_mat_lu_solve_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="arb", label="point_wrappers.arb_mat_lu_solve_point")
    p_mid = jnp.asarray(plan.p)
    l_mid = jnp.asarray(plan.l)
    u_mid = jnp.asarray(plan.u)
    b_mid = _arb_mat_point_rhs(b)
    vector_rhs = b_mid.ndim == p_mid.ndim - 1
    pb = jnp.einsum("...ij,...j->...i", p_mid, b_mid) if vector_rhs else jnp.matmul(p_mid, b_mid)
    y = lax.linalg.triangular_solve(
        l_mid,
        pb[..., None] if vector_rhs else pb,
        left_side=True,
        lower=True,
        unit_diagonal=True,
    )
    out = lax.linalg.triangular_solve(u_mid, y, left_side=True, lower=False, unit_diagonal=False)
    return out[..., 0] if vector_rhs else out


def arb_mat_dense_lu_solve_plan_apply_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return arb_mat_lu_solve_point(plan, b)


def arb_mat_solve_transpose_point(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        factor = jnp.asarray(a_or_plan.factor)
        return mat_common.lower_cholesky_solve_transpose(factor, _arb_mat_point_rhs(b))
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        plan = mat_common.as_dense_lu_solve_plan(a_or_plan, algebra="arb", label="point_wrappers.arb_mat_solve_transpose_point")
        p_mid = jnp.asarray(plan.p)
        l_mid = jnp.asarray(plan.l)
        u_mid = jnp.asarray(plan.u)
        b_mid = _arb_mat_point_rhs(b)
        vector_rhs = b_mid.ndim == p_mid.ndim - 1
        y = lax.linalg.triangular_solve(u_mid, b_mid[..., None] if vector_rhs else b_mid, left_side=True, lower=False, transpose_a=True, conjugate_a=False)
        z = lax.linalg.triangular_solve(l_mid, y, left_side=True, lower=True, unit_diagonal=True, transpose_a=True, conjugate_a=False)
        out = jnp.einsum("...ij,...j->...i", jnp.swapaxes(p_mid, -2, -1), z[..., 0]) if vector_rhs else jnp.matmul(jnp.swapaxes(p_mid, -2, -1), z)
        return out
    a_mid = _arb_mat_point_matrix(a_or_plan)
    b_mid = _arb_mat_point_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    out = jnp.linalg.solve(jnp.swapaxes(a_mid, -2, -1), b_mid[..., None] if vector_rhs else b_mid)
    return out[..., 0] if vector_rhs else out


def arb_mat_solve_lu_point(
    a_or_plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array] | jax.Array,
    b: jax.Array,
) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        return arb_mat_lu_solve_point(a_or_plan, b)
    return arb_mat_lu_solve_point(arb_mat_dense_lu_solve_plan_prepare_point(a_or_plan), b)


def arb_mat_solve_tril_point(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_triangular_solve_point(a, b, lower=True, unit_diagonal=unit_diagonal)


def arb_mat_solve_triu_point(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_triangular_solve_point(a, b, lower=False, unit_diagonal=unit_diagonal)


def arb_mat_solve_add_point(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        solved = arb_mat_spd_solve_point(a_or_plan, b)
    elif isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        solved = arb_mat_solve_lu_point(a_or_plan, b)
    else:
        solved = arb_mat_solve_point(a_or_plan, b)
    return _arb_mat_point_rhs(y) + solved


def arb_mat_solve_transpose_add_point(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return _arb_mat_point_rhs(y) + arb_mat_solve_transpose_point(a_or_plan, b)


def arb_mat_mat_solve_point(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return arb_mat_spd_solve_point(a_or_plan, b)
    if isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        return arb_mat_solve_lu_point(a_or_plan, b)
    return arb_mat_solve_point(a_or_plan, b)


def arb_mat_mat_solve_transpose_point(a_or_plan, b: jax.Array) -> jax.Array:
    return arb_mat_solve_transpose_point(a_or_plan, b)


def arb_mat_spd_solve_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else arb_mat_cho_point(plan)
    return mat_common.lower_cholesky_solve(factor, _arb_mat_point_rhs(b))


def arb_mat_dense_spd_solve_plan_apply_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_spd_solve_point(plan, b)


def arb_mat_spd_inv_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else arb_mat_cho_point(plan)
    eye = jnp.broadcast_to(jnp.eye(factor.shape[-1], dtype=factor.dtype), factor.shape)
    return mat_common.lower_cholesky_solve(factor, eye)


@partial(jax.jit, static_argnames=())
def arb_mat_2x2_det_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_2x2(a)
    return a_mid[..., 0, 0] * a_mid[..., 1, 1] - a_mid[..., 0, 1] * a_mid[..., 1, 0]


@partial(jax.jit, static_argnames=())
def arb_mat_2x2_trace_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_2x2(a)
    return a_mid[..., 0, 0] + a_mid[..., 1, 1]


def arb_mat_2x2_det_batch_point(a: jax.Array) -> jax.Array:
    return arb_mat_2x2_det_point(a)


def arb_mat_2x2_trace_batch_point(a: jax.Array) -> jax.Array:
    return arb_mat_2x2_trace_point(a)


@partial(jax.jit, static_argnames=())
def arb_mat_sqr_point(a: jax.Array) -> jax.Array:
    a_mid = _arb_mat_point_matrix(a)
    return jnp.matmul(a_mid, a_mid)


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


def arb_mat_banded_matvec_batch_fixed_point(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    return arb_mat_banded_matvec_point(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


def arb_mat_banded_matvec_batch_padded_point(
    a: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, x), pad_to)
    return arb_mat_banded_matvec_point(*call_args, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


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


def arb_mat_symmetric_part_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_symmetric_part_point(a)


def arb_mat_symmetric_part_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_symmetric_part_point(*call_args)


def arb_mat_is_symmetric_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_is_symmetric_point(a)


def arb_mat_is_symmetric_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_is_symmetric_point(*call_args)


def arb_mat_is_spd_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_is_spd_point(a)


def arb_mat_is_spd_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_is_spd_point(*call_args)


def arb_mat_cho_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_cho_point(a)


def arb_mat_cho_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_cho_point(*call_args)


def arb_mat_ldl_batch_fixed_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_mat_ldl_point(a)


def arb_mat_ldl_batch_padded_point(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array]:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_ldl_point(*call_args)


def arb_mat_matvec_cached_prepare_point(a: jax.Array) -> jax.Array:
    return _arb_mat_point_matrix(a)


def arb_mat_rmatvec_cached_prepare_point(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(di.midpoint(mat_common.as_interval_rect_matrix(a, "point_wrappers.arb_mat_rmatvec_cached_prepare_point")), -2, -1)


@partial(jax.jit, static_argnames=())
def arb_mat_matvec_cached_apply_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    x_mid = _arb_mat_point_vector(x)
    checks.check_equal(cache.shape[-1], x_mid.shape[-1], "point_wrappers.arb_mat_matvec_cached_apply_point.inner")
    return jnp.einsum("...ij,...j->...i", jnp.asarray(cache), x_mid)


@partial(jax.jit, static_argnames=())
def arb_mat_rmatvec_cached_apply_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply_point(cache, x)


def arb_mat_matvec_cached_apply_batch_fixed_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply_point(cache, x)


def arb_mat_rmatvec_cached_apply_batch_fixed_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_rmatvec_cached_apply_point(cache, x)


def arb_mat_matvec_cached_apply_batch_padded_point(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((cache, x), pad_to)
    return arb_mat_matvec_cached_apply_point(*call_args)


def arb_mat_rmatvec_cached_apply_batch_padded_point(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((cache, x), pad_to)
    return arb_mat_rmatvec_cached_apply_point(*call_args)


def arb_mat_dense_matvec_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return arb_mat_dense_matvec_plan_prepare_point(a)


def arb_mat_dense_matvec_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseMatvecPlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_dense_matvec_plan_prepare_point(*call_args)


def arb_mat_dense_matvec_plan_apply_batch_fixed_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_dense_matvec_plan_apply_point(plan, x)


def arb_mat_dense_matvec_plan_apply_batch_padded_point(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (x_pad,), _ = _pad_args_repeat_last((x,), pad_to)
    return arb_mat_dense_matvec_plan_apply_point(plan, x_pad)


def arb_mat_dense_lu_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    return arb_mat_dense_lu_solve_plan_prepare_point(a)


def arb_mat_dense_spd_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    return arb_mat_dense_spd_solve_plan_prepare_point(a)


def arb_mat_dense_spd_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseCholeskySolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_dense_spd_solve_plan_prepare_point(*call_args)


def arb_mat_dense_lu_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseLUSolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_dense_lu_solve_plan_prepare_point(*call_args)


def arb_mat_dense_lu_solve_plan_apply_batch_fixed_point(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return arb_mat_dense_lu_solve_plan_apply_point(plan, b)


def arb_mat_dense_lu_solve_plan_apply_batch_padded_point(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return arb_mat_dense_lu_solve_plan_apply_point(plan, b_pad)


def arb_mat_spd_solve_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_spd_solve_point(plan, b)


def arb_mat_spd_solve_batch_padded_point(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return arb_mat_spd_solve_point(plan, b_pad)


def arb_mat_dense_spd_solve_plan_apply_batch_fixed_point(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return arb_mat_dense_spd_solve_plan_apply_point(plan, b)


def arb_mat_dense_spd_solve_plan_apply_batch_padded_point(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return arb_mat_dense_spd_solve_plan_apply_point(plan, b_pad)


def arb_mat_spd_inv_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return arb_mat_spd_inv_point(plan)


def arb_mat_spd_inv_batch_padded_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, *, pad_to: int) -> jax.Array:
    del pad_to
    return arb_mat_spd_inv_point(plan)


def arb_mat_transpose_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_transpose_point(a)


def arb_mat_transpose_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_transpose_point(*call_args)


def arb_mat_diag_batch_fixed_point(a: jax.Array) -> jax.Array:
    return arb_mat_diag_point(a)


def arb_mat_diag_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return arb_mat_diag_point(*call_args)


def arb_mat_diag_matrix_batch_fixed_point(d: jax.Array) -> jax.Array:
    return arb_mat_diag_matrix_point(d)


def arb_mat_diag_matrix_batch_padded_point(d: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((d,), pad_to)
    return arb_mat_diag_matrix_point(*call_args)


@partial(jax.jit, static_argnames=())
def acb_mat_matmul_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    b_mid = _acb_mat_point_matrix(b)
    checks.check_equal(a_mid.shape[-1], b_mid.shape[-2], "point_wrappers.acb_mat_matmul_point.inner")
    return jnp.matmul(a_mid, b_mid)


@partial(jax.jit, static_argnames=())
def acb_mat_matvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    x_mid = _acb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-1], x_mid.shape[-1], "point_wrappers.acb_mat_matvec_point.inner")
    return jnp.einsum("...ij,...j->...i", a_mid, x_mid)


@partial(jax.jit, static_argnames=("lower_bandwidth", "upper_bandwidth"))
def acb_mat_banded_matvec_point(a: jax.Array, x: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    x_mid = _acb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-1], x_mid.shape[-1], "point_wrappers.acb_mat_banded_matvec_point.inner")
    mask = _band_mask(a_mid.shape[-2], a_mid.shape[-1], lower_bandwidth, upper_bandwidth)
    return jnp.einsum("...ij,...j->...i", jnp.where(mask, a_mid, jnp.zeros_like(a_mid)), x_mid)


@partial(jax.jit, static_argnames=())
def acb_mat_hermitian_part_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    return 0.5 * (a_mid + jnp.conj(jnp.swapaxes(a_mid, -2, -1)))


def acb_mat_is_hermitian_point(a: jax.Array) -> jax.Array:
    return mat_common.complex_midpoint_is_hermitian(_acb_mat_point_matrix(a))


def acb_mat_is_hpd_point(a: jax.Array) -> jax.Array:
    chol = jnp.linalg.cholesky(acb_mat_hermitian_part_point(a))
    return acb_mat_is_hermitian_point(a) & mat_common.lower_cholesky_finite(chol)


@partial(jax.jit, static_argnames=())
def acb_mat_cho_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.cholesky(acb_mat_hermitian_part_point(a))


def acb_mat_ldl_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol = acb_mat_cho_point(a)
    diag = jnp.diagonal(chol, axis1=-2, axis2=-1)
    return chol / diag[..., None, :], jnp.real(diag * jnp.conj(diag))


@partial(jax.jit, static_argnames=())
def acb_mat_eigvalsh_point(a: jax.Array) -> jax.Array:
    values, _ = jnp.linalg.eigh(acb_mat_hermitian_part_point(a))
    return values


@partial(jax.jit, static_argnames=())
def acb_mat_eigh_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.linalg.eigh(acb_mat_hermitian_part_point(a))


@partial(jax.jit, static_argnames=())
def acb_mat_charpoly_point(a: jax.Array) -> jax.Array:
    mid = _acb_mat_point_matrix(a)
    coeffs_general = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=False)
    coeffs_hermitian = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=True)
    return jnp.where(mat_common.complex_midpoint_is_hermitian(mid)[..., None], coeffs_hermitian, coeffs_general)


@partial(jax.jit, static_argnames=("n",))
def acb_mat_pow_ui_point(a: jax.Array, n: int) -> jax.Array:
    return mat_common.matrix_power_ui(_acb_mat_point_matrix(a), n)


@partial(jax.jit, static_argnames=())
def acb_mat_exp_point(a: jax.Array) -> jax.Array:
    mid = _acb_mat_point_matrix(a)
    general = mat_common.matrix_exp(mid, hermitian=False)
    hermitian = mat_common.matrix_exp(mid, hermitian=True)
    return jnp.where(mat_common.complex_midpoint_is_hermitian(mid)[..., None, None], hermitian, general)


def acb_mat_is_diag_point(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_diagonal(_acb_mat_point_matrix(a))


def acb_mat_is_tril_point(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_acb_mat_point_matrix(a), lower=True)


def acb_mat_is_triu_point(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_acb_mat_point_matrix(a), lower=False)


def acb_mat_is_zero_point(a: jax.Array) -> jax.Array:
    return jnp.all(_acb_mat_point_matrix(a) == 0, axis=(-2, -1))


def acb_mat_is_finite_point(a: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(_acb_mat_point_matrix(a)), axis=(-2, -1))


def acb_mat_is_exact_point(a: jax.Array) -> jax.Array:
    del a
    return jnp.asarray(True)


def acb_mat_is_real_point(a: jax.Array) -> jax.Array:
    return jnp.all(jnp.imag(_acb_mat_point_matrix(a)) == 0, axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def acb_mat_solve_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    b_mid = _acb_mat_point_rhs(b)
    rows = b_mid.shape[-1] if b_mid.ndim == a_mid.ndim - 1 else b_mid.shape[-2]
    checks.check_equal(a_mid.shape[-1], rows, "point_wrappers.acb_mat_solve_point.inner")
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    out = jnp.linalg.solve(a_mid, b_mid[..., None] if vector_rhs else b_mid)
    return out[..., 0] if vector_rhs else out


@partial(jax.jit, static_argnames=())
def acb_mat_inv_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.inv(_acb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def acb_mat_det_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.det(_acb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=())
def acb_mat_trace_point(a: jax.Array) -> jax.Array:
    return jnp.trace(_acb_mat_point_matrix(a), axis1=-2, axis2=-1)


@partial(jax.jit, static_argnames=("lower", "unit_diagonal"))
def acb_mat_triangular_solve_point(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    b_mid = _acb_mat_point_rhs(b)
    rows = b_mid.shape[-1] if b_mid.ndim == a_mid.ndim - 1 else b_mid.shape[-2]
    checks.check_equal(a_mid.shape[-1], rows, "point_wrappers.acb_mat_triangular_solve_point.inner")
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    out = lax.linalg.triangular_solve(a_mid, b_mid[..., None] if vector_rhs else b_mid, left_side=True, lower=lower, unit_diagonal=unit_diagonal)
    return out[..., 0] if vector_rhs else out


@partial(jax.jit, static_argnames=())
def acb_mat_lu_point(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_mid = _acb_mat_point_matrix(a)
    lu, _, perm = lax.linalg.lu(a_mid)
    n = a_mid.shape[-1]
    eye = jnp.eye(n, dtype=a_mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return p, l, u


@partial(jax.jit, static_argnames=())
def acb_mat_qr_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.linalg.qr(_acb_mat_point_matrix(a))


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


def acb_mat_rmatvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a_mid = acb_core.acb_midpoint(mat_common.as_box_rect_matrix(a, "point_wrappers.acb_mat_rmatvec_point"))
    x_mid = _acb_mat_point_vector(x)
    checks.check_equal(a_mid.shape[-2], x_mid.shape[-1], "point_wrappers.acb_mat_rmatvec_point.inner")
    return jnp.einsum("...ji,...j->...i", a_mid, x_mid)


def acb_mat_lu_solve_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="acb", label="point_wrappers.acb_mat_lu_solve_point")
    p_mid = jnp.asarray(plan.p)
    l_mid = jnp.asarray(plan.l)
    u_mid = jnp.asarray(plan.u)
    b_mid = _acb_mat_point_rhs(b)
    vector_rhs = b_mid.ndim == p_mid.ndim - 1
    pb = jnp.einsum("...ij,...j->...i", p_mid, b_mid) if vector_rhs else jnp.matmul(p_mid, b_mid)
    y = lax.linalg.triangular_solve(
        l_mid,
        pb[..., None] if vector_rhs else pb,
        left_side=True,
        lower=True,
        unit_diagonal=True,
    )
    out = lax.linalg.triangular_solve(u_mid, y, left_side=True, lower=False, unit_diagonal=False)
    return out[..., 0] if vector_rhs else out


def acb_mat_dense_lu_solve_plan_apply_point(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return acb_mat_lu_solve_point(plan, b)


def acb_mat_solve_transpose_point(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        factor = jnp.asarray(a_or_plan.factor)
        return mat_common.lower_cholesky_solve_transpose(factor, _acb_mat_point_rhs(b))
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        plan = mat_common.as_dense_lu_solve_plan(a_or_plan, algebra="acb", label="point_wrappers.acb_mat_solve_transpose_point")
        p_mid = jnp.asarray(plan.p)
        l_mid = jnp.asarray(plan.l)
        u_mid = jnp.asarray(plan.u)
        b_mid = _acb_mat_point_rhs(b)
        vector_rhs = b_mid.ndim == p_mid.ndim - 1
        y = lax.linalg.triangular_solve(u_mid, b_mid[..., None] if vector_rhs else b_mid, left_side=True, lower=False, transpose_a=True, conjugate_a=False)
        z = lax.linalg.triangular_solve(l_mid, y, left_side=True, lower=True, unit_diagonal=True, transpose_a=True, conjugate_a=False)
        out = jnp.einsum("...ij,...j->...i", jnp.swapaxes(p_mid, -2, -1), z[..., 0]) if vector_rhs else jnp.matmul(jnp.swapaxes(p_mid, -2, -1), z)
        return out
    a_mid = _acb_mat_point_matrix(a_or_plan)
    b_mid = _acb_mat_point_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    out = jnp.linalg.solve(jnp.swapaxes(a_mid, -2, -1), b_mid[..., None] if vector_rhs else b_mid)
    return out[..., 0] if vector_rhs else out


def acb_mat_solve_lu_point(
    a_or_plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array] | jax.Array,
    b: jax.Array,
) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        return acb_mat_lu_solve_point(a_or_plan, b)
    return acb_mat_lu_solve_point(acb_mat_dense_lu_solve_plan_prepare_point(a_or_plan), b)


def acb_mat_solve_tril_point(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return acb_mat_triangular_solve_point(a, b, lower=True, unit_diagonal=unit_diagonal)


def acb_mat_solve_triu_point(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return acb_mat_triangular_solve_point(a, b, lower=False, unit_diagonal=unit_diagonal)


def acb_mat_solve_add_point(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        solved = acb_mat_hpd_solve_point(a_or_plan, b)
    elif isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        solved = acb_mat_solve_lu_point(a_or_plan, b)
    else:
        solved = acb_mat_solve_point(a_or_plan, b)
    return _acb_mat_point_rhs(y) + solved


def acb_mat_solve_transpose_add_point(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return _acb_mat_point_rhs(y) + acb_mat_solve_transpose_point(a_or_plan, b)


def acb_mat_mat_solve_point(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return acb_mat_hpd_solve_point(a_or_plan, b)
    if isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        return acb_mat_solve_lu_point(a_or_plan, b)
    return acb_mat_solve_point(a_or_plan, b)


def acb_mat_mat_solve_transpose_point(a_or_plan, b: jax.Array) -> jax.Array:
    return acb_mat_solve_transpose_point(a_or_plan, b)


def acb_mat_hpd_solve_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else acb_mat_cho_point(plan)
    return mat_common.lower_cholesky_solve(factor, _acb_mat_point_rhs(b))


def acb_mat_dense_hpd_solve_plan_apply_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_hpd_solve_point(plan, b)


def acb_mat_hpd_inv_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    factor = jnp.asarray(plan.factor) if isinstance(plan, mat_common.DenseCholeskySolvePlan) else acb_mat_cho_point(plan)
    eye = jnp.broadcast_to(jnp.eye(factor.shape[-1], dtype=factor.dtype), factor.shape)
    return mat_common.lower_cholesky_solve(factor, eye)


@partial(jax.jit, static_argnames=())
def acb_mat_2x2_det_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_2x2(a)
    return a_mid[..., 0, 0] * a_mid[..., 1, 1] - a_mid[..., 0, 1] * a_mid[..., 1, 0]


@partial(jax.jit, static_argnames=())
def acb_mat_2x2_trace_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_2x2(a)
    return a_mid[..., 0, 0] + a_mid[..., 1, 1]


def acb_mat_2x2_det_batch_point(a: jax.Array) -> jax.Array:
    return acb_mat_2x2_det_point(a)


def acb_mat_2x2_trace_batch_point(a: jax.Array) -> jax.Array:
    return acb_mat_2x2_trace_point(a)


@partial(jax.jit, static_argnames=())
def acb_mat_sqr_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    return jnp.matmul(a_mid, a_mid)


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


def acb_mat_banded_matvec_batch_fixed_point(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    return acb_mat_banded_matvec_point(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


def acb_mat_banded_matvec_batch_padded_point(
    a: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, x), pad_to)
    return acb_mat_banded_matvec_point(*call_args, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


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


def acb_mat_hermitian_part_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_hermitian_part_point(a)


def acb_mat_hermitian_part_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_hermitian_part_point(*call_args)


def acb_mat_is_hermitian_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_is_hermitian_point(a)


def acb_mat_is_hermitian_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_is_hermitian_point(*call_args)


def acb_mat_is_hpd_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_is_hpd_point(a)


def acb_mat_is_hpd_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_is_hpd_point(*call_args)


def acb_mat_cho_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_cho_point(a)


def acb_mat_cho_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_cho_point(*call_args)


def acb_mat_ldl_batch_fixed_point(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_ldl_point(a)


def acb_mat_ldl_batch_padded_point(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array]:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_ldl_point(*call_args)


def acb_mat_matvec_cached_prepare_point(a: jax.Array) -> jax.Array:
    return _acb_mat_point_matrix(a)


def acb_mat_rmatvec_cached_prepare_point(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(acb_core.acb_midpoint(mat_common.as_box_rect_matrix(a, "point_wrappers.acb_mat_rmatvec_cached_prepare_point")), -2, -1)


@partial(jax.jit, static_argnames=())
def acb_mat_matvec_cached_apply_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    x_mid = _acb_mat_point_vector(x)
    checks.check_equal(cache.shape[-1], x_mid.shape[-1], "point_wrappers.acb_mat_matvec_cached_apply_point.inner")
    return jnp.einsum("...ij,...j->...i", jnp.asarray(cache), x_mid)


@partial(jax.jit, static_argnames=())
def acb_mat_rmatvec_cached_apply_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_apply_point(cache, x)


def acb_mat_matvec_cached_apply_batch_fixed_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_apply_point(cache, x)


def acb_mat_rmatvec_cached_apply_batch_fixed_point(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_rmatvec_cached_apply_point(cache, x)


def acb_mat_matvec_cached_apply_batch_padded_point(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((cache, x), pad_to)
    return acb_mat_matvec_cached_apply_point(*call_args)


def acb_mat_rmatvec_cached_apply_batch_padded_point(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((cache, x), pad_to)
    return acb_mat_rmatvec_cached_apply_point(*call_args)


def acb_mat_dense_matvec_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return acb_mat_dense_matvec_plan_prepare_point(a)


def acb_mat_dense_hpd_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    return acb_mat_dense_hpd_solve_plan_prepare_point(a)


def acb_mat_dense_hpd_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseCholeskySolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_dense_hpd_solve_plan_prepare_point(*call_args)


def acb_mat_dense_matvec_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseMatvecPlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_dense_matvec_plan_prepare_point(*call_args)


def acb_mat_dense_matvec_plan_apply_batch_fixed_point(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_dense_matvec_plan_apply_point(plan, x)


def acb_mat_dense_matvec_plan_apply_batch_padded_point(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (x_pad,), _ = _pad_args_repeat_last((x,), pad_to)
    return acb_mat_dense_matvec_plan_apply_point(plan, x_pad)


def acb_mat_dense_lu_solve_plan_prepare_batch_fixed_point(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    return acb_mat_dense_lu_solve_plan_prepare_point(a)


def acb_mat_dense_lu_solve_plan_prepare_batch_padded_point(a: jax.Array, *, pad_to: int) -> mat_common.DenseLUSolvePlan:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_dense_lu_solve_plan_prepare_point(*call_args)


def acb_mat_dense_lu_solve_plan_apply_batch_fixed_point(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return acb_mat_dense_lu_solve_plan_apply_point(plan, b)


def acb_mat_dense_lu_solve_plan_apply_batch_padded_point(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return acb_mat_dense_lu_solve_plan_apply_point(plan, b_pad)


def acb_mat_hpd_solve_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_hpd_solve_point(plan, b)


def acb_mat_hpd_solve_batch_padded_point(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return acb_mat_hpd_solve_point(plan, b_pad)


def acb_mat_dense_hpd_solve_plan_apply_batch_fixed_point(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return acb_mat_dense_hpd_solve_plan_apply_point(plan, b)


def acb_mat_dense_hpd_solve_plan_apply_batch_padded_point(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = _pad_args_repeat_last((b,), pad_to)
    return acb_mat_dense_hpd_solve_plan_apply_point(plan, b_pad)


def acb_mat_hpd_inv_batch_fixed_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return acb_mat_hpd_inv_point(plan)


def acb_mat_hpd_inv_batch_padded_point(plan: mat_common.DenseCholeskySolvePlan | jax.Array, *, pad_to: int) -> jax.Array:
    del pad_to
    return acb_mat_hpd_inv_point(plan)


def acb_mat_zero_point(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.zeros((n, n), dtype=jnp.result_type(dtype, jnp.complex64))


def acb_mat_identity_point(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.eye(n, dtype=jnp.result_type(dtype, jnp.complex64))


def acb_mat_permutation_matrix_point(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    return jnp.eye(perm.shape[-1], dtype=jnp.result_type(dtype, jnp.complex64))[perm]


@partial(jax.jit, static_argnames=())
def acb_mat_transpose_point(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(acb_core.acb_midpoint(mat_common.as_box_rect_matrix(a, "point_wrappers.acb_mat_transpose_point")), -2, -1)


@partial(jax.jit, static_argnames=())
def acb_mat_conjugate_transpose_point(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(jnp.conj(acb_core.acb_midpoint(mat_common.as_box_rect_matrix(a, "point_wrappers.acb_mat_conjugate_transpose_point"))), -2, -1)


@partial(jax.jit, static_argnames=())
def acb_mat_add_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return _acb_mat_point_matrix(a) + _acb_mat_point_matrix(b)


@partial(jax.jit, static_argnames=())
def acb_mat_sub_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return _acb_mat_point_matrix(a) - _acb_mat_point_matrix(b)


@partial(jax.jit, static_argnames=())
def acb_mat_neg_point(a: jax.Array) -> jax.Array:
    return -_acb_mat_point_matrix(a)


@partial(jax.jit, static_argnames=())
def acb_mat_mul_entrywise_point(a: jax.Array, b: jax.Array) -> jax.Array:
    return _acb_mat_point_matrix(a) * _acb_mat_point_matrix(b)


@partial(jax.jit, static_argnames=())
def acb_mat_conjugate_point(a: jax.Array) -> jax.Array:
    return jnp.conj(_acb_mat_point_matrix(a))


@partial(jax.jit, static_argnames=("row_start", "row_stop", "col_start", "col_stop"))
def acb_mat_submatrix_point(a: jax.Array, row_start: int, row_stop: int, col_start: int, col_stop: int) -> jax.Array:
    return _acb_mat_point_matrix(a)[..., row_start:row_stop, col_start:col_stop]


@partial(jax.jit, static_argnames=())
def acb_mat_diag_point(a: jax.Array) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    idx = jnp.arange(a_mid.shape[-1])
    return a_mid[..., idx, idx]


@partial(jax.jit, static_argnames=())
def acb_mat_diag_matrix_point(d: jax.Array) -> jax.Array:
    d_mid = _acb_mat_point_vector(d)
    eye = jnp.eye(d_mid.shape[-1], dtype=d_mid.dtype)
    return eye * d_mid[..., None, :]


def acb_mat_block_assemble_point(block_rows) -> jax.Array:
    checks.check(len(block_rows) > 0, "point_wrappers.acb_mat_block_assemble_point.rows")
    row_chunks = []
    for row in block_rows:
        row_chunks.append(jnp.concatenate([_acb_mat_point_matrix(block) for block in row], axis=-1))
    return jnp.concatenate(row_chunks, axis=-2)


def acb_mat_block_diag_point(blocks) -> jax.Array:
    checks.check(len(blocks) > 0, "point_wrappers.acb_mat_block_diag_point.blocks")
    matrices = [_acb_mat_point_matrix(block) for block in blocks]
    row_sizes = [int(block.shape[-2]) for block in matrices]
    col_sizes = [int(block.shape[-1]) for block in matrices]
    dtype = matrices[0].dtype
    rows = []
    for i, block in enumerate(matrices):
        row = []
        for j in range(len(matrices)):
            if i == j:
                row.append(block)
            else:
                row.append(jnp.zeros(block.shape[:-2] + (row_sizes[i], col_sizes[j]), dtype=dtype))
        rows.append(tuple(row))
    return acb_mat_block_assemble_point(tuple(rows))


def acb_mat_block_extract_point(a: jax.Array, row_block_sizes, col_block_sizes, row_block: int, col_block: int) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    row_offsets = [0]
    for size in row_block_sizes:
        row_offsets.append(row_offsets[-1] + int(size))
    col_offsets = [0]
    for size in col_block_sizes:
        col_offsets.append(col_offsets[-1] + int(size))
    return a_mid[..., row_offsets[row_block] : row_offsets[row_block + 1], col_offsets[col_block] : col_offsets[col_block + 1]]


def acb_mat_block_row_point(a: jax.Array, row_block_sizes, row_block: int) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    offsets = [0]
    for size in row_block_sizes:
        offsets.append(offsets[-1] + int(size))
    return a_mid[..., offsets[row_block] : offsets[row_block + 1], :]


def acb_mat_block_col_point(a: jax.Array, col_block_sizes, col_block: int) -> jax.Array:
    a_mid = _acb_mat_point_matrix(a)
    offsets = [0]
    for size in col_block_sizes:
        offsets.append(offsets[-1] + int(size))
    return a_mid[..., :, offsets[col_block] : offsets[col_block + 1]]


def acb_mat_block_matmul_point(a_blocks, b_blocks) -> jax.Array:
    out_rows = []
    for i, a_row in enumerate(a_blocks):
        out_row = []
        for j in range(len(b_blocks[0])):
            total = None
            for k in range(len(a_row)):
                prod = acb_mat_matmul_point(a_row[k], b_blocks[k][j])
                total = prod if total is None else total + prod
            out_row.append(total)
        out_rows.append(tuple(out_row))
    return acb_mat_block_assemble_point(tuple(out_rows))


@partial(jax.jit, static_argnames=())
def acb_mat_norm_fro_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_acb_mat_point_matrix(a), ord="fro", axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def acb_mat_norm_1_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_acb_mat_point_matrix(a), ord=1, axis=(-2, -1))


@partial(jax.jit, static_argnames=())
def acb_mat_norm_inf_point(a: jax.Array) -> jax.Array:
    return jnp.linalg.norm(_acb_mat_point_matrix(a), ord=jnp.inf, axis=(-2, -1))


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


def acb_mat_transpose_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_transpose_point(a)


def acb_mat_transpose_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_transpose_point(*call_args)


def acb_mat_conjugate_transpose_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_conjugate_transpose_point(a)


def acb_mat_conjugate_transpose_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_conjugate_transpose_point(*call_args)


def acb_mat_diag_batch_fixed_point(a: jax.Array) -> jax.Array:
    return acb_mat_diag_point(a)


def acb_mat_diag_batch_padded_point(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a,), pad_to)
    return acb_mat_diag_point(*call_args)


def acb_mat_diag_matrix_batch_fixed_point(d: jax.Array) -> jax.Array:
    return acb_mat_diag_matrix_point(d)


def acb_mat_diag_matrix_batch_padded_point(d: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((d,), pad_to)
    return acb_mat_diag_matrix_point(*call_args)


def _real_laguerre_l_scalar(n: int, m: jax.Array, x: jax.Array) -> jax.Array:
    def body(k, acc):
        kf = jnp.asarray(k, dtype=jnp.asarray(x).dtype)
        coeff = jnp.exp(
            hypgeom._gammaln_real(n + m + 1.0)
            - hypgeom._gammaln_real(n - k + 1.0)
            - hypgeom._gammaln_real(m + k + 1.0)
        )
        term = coeff * jnp.power(-x, kf) / jnp.exp(hypgeom._gammaln_real(kf + 1.0))
        return acc + term

    return lax.fori_loop(0, n + 1, body, jnp.asarray(0.0, dtype=jnp.asarray(x).dtype))


def _real_hermite_h_scalar(n: int, x: jax.Array) -> jax.Array:
    if n == 0:
        return jnp.asarray(1.0, dtype=jnp.asarray(x).dtype)
    if n == 1:
        return jnp.asarray(2.0, dtype=jnp.asarray(x).dtype) * x
    h0 = jnp.asarray(1.0, dtype=jnp.asarray(x).dtype)
    h1 = jnp.asarray(2.0, dtype=jnp.asarray(x).dtype) * x

    def body(k, state):
        h_prev, h_curr = state
        h_next = 2.0 * x * h_curr - 2.0 * jnp.asarray(k - 1, dtype=jnp.asarray(x).dtype) * h_prev
        return h_curr, h_next

    _, hn = lax.fori_loop(2, n + 1, body, (h0, h1))
    return hn


def _real_pfq_scalar(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool, n_terms: int) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    z = jnp.asarray(z, dtype=jnp.result_type(z, jnp.float64))

    def body(k, state):
        term, s = state
        k1 = jnp.asarray(k + 1, dtype=z.dtype)
        num = jnp.prod(a + k) if a.size else jnp.asarray(1.0, dtype=z.dtype)
        den = jnp.prod(b + k) if b.size else jnp.asarray(1.0, dtype=z.dtype)
        term = term * (num / den) * (z / k1)
        return term, s + term

    term0 = jnp.asarray(1.0, dtype=z.dtype)
    _, out = lax.fori_loop(0, n_terms - 1, body, (term0, term0))
    return jnp.where(reciprocal, 1.0 / out, out)


def _complex_pfq_scalar(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool, n_terms: int) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    z = jnp.asarray(z, dtype=jnp.result_type(z, jnp.complex128))

    def body(k, state):
        term, s = state
        k1 = jnp.asarray(k + 1, dtype=z.real.dtype)
        num = jnp.prod(a + k) if a.size else jnp.asarray(1.0 + 0.0j, dtype=z.dtype)
        den = jnp.prod(b + k) if b.size else jnp.asarray(1.0 + 0.0j, dtype=z.dtype)
        term = term * (num / den) * (z / k1)
        return term, s + term

    term0 = jnp.asarray(1.0 + 0.0j, dtype=z.dtype)
    _, out = lax.fori_loop(0, n_terms - 1, body, (term0, term0))
    return jnp.where(reciprocal, 1.0 / out, out)


def _complex_digamma_scalar(z: jax.Array) -> jax.Array:
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    h = jnp.asarray(1e-6 + 0.0j, dtype=zz.dtype)
    two = jnp.asarray(2.0, dtype=real_dtype)
    return (acb_core._complex_loggamma(zz + h) - acb_core._complex_loggamma(zz - h)) / (two * h)


def _complex_zeta_scalar(s: jax.Array, n_terms: int = 64) -> jax.Array:
    ss = el.as_complex(s)
    real_dtype = el.real_dtype_from_complex_dtype(ss.dtype)
    n = jnp.arange(1, n_terms + 1, dtype=real_dtype)
    return jnp.sum(jnp.exp(-ss * jnp.log(n)))


def _complex_hurwitz_zeta_scalar(
    s: jax.Array,
    a: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    ss = el.as_complex(s)
    aa = el.as_complex(a)
    real_dtype = el.real_dtype_from_complex_dtype(ss.dtype)
    re_s = jnp.real(ss)
    eps = jnp.asarray(1e-12, dtype=real_dtype)
    tail_target = eps * jnp.maximum(re_s - 1.0, jnp.asarray(1e-12, dtype=real_dtype))
    base = jnp.power(tail_target, 1.0 / jnp.maximum(1.0 - re_s, jnp.asarray(1e-12, dtype=real_dtype)))
    n_est = jnp.ceil(base + 1.0)
    n_eff = jnp.where(re_s > 1.1, n_est, jnp.asarray(terms, dtype=real_dtype))
    n_eff = jnp.clip(n_eff, jnp.asarray(min_terms, dtype=real_dtype), jnp.asarray(max_terms, dtype=real_dtype))
    ks = jnp.arange(max_terms, dtype=real_dtype)
    mask = ks < n_eff
    terms_arr = jnp.power(aa + ks, -ss)
    return jnp.sum(jnp.where(mask, terms_arr, jnp.zeros_like(terms_arr)))


def _complex_polygamma_scalar(
    m: int,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    if m == 0:
        return _complex_digamma_scalar(zz)
    re_z = jnp.real(zz)
    m_float = jnp.asarray(float(m), dtype=real_dtype)
    eps = jnp.asarray(1e-12, dtype=real_dtype)
    tail_target = eps * jnp.maximum(m_float, jnp.asarray(1.0, dtype=real_dtype))
    base = jnp.power(tail_target, -1.0 / jnp.maximum(m_float, eps))
    n_est = jnp.ceil(base - re_z)
    n_eff = jnp.where(m_float > 0, n_est, jnp.asarray(terms, dtype=real_dtype))
    n_eff = jnp.clip(n_eff, jnp.asarray(min_terms, dtype=real_dtype), jnp.asarray(max_terms, dtype=real_dtype))
    ks = jnp.arange(max_terms, dtype=real_dtype)
    mask = ks < n_eff
    factorial = jnp.exp(lax.lgamma(m_float + 1.0))
    series_terms = jnp.power(zz + ks, -(m_float + 1.0))
    series = jnp.sum(jnp.where(mask, series_terms, jnp.zeros_like(series_terms)))
    sign = -1.0 if (m + 1) % 2 else 1.0
    return jnp.asarray(sign, dtype=real_dtype) * factorial * series


def _complex_bernoulli_poly_ui_scalar(n: int, z: jax.Array) -> jax.Array:
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    if n == 0:
        return jnp.asarray(1.0 + 0.0j, dtype=zz.dtype)
    if n == 1:
        return zz - jnp.asarray(0.5, dtype=real_dtype)
    if n == 2:
        return zz * zz - zz + jnp.asarray(1.0 / 6.0, dtype=real_dtype)
    if n == 3:
        return zz * zz * zz - jnp.asarray(1.5, dtype=real_dtype) * zz * zz + jnp.asarray(0.5, dtype=real_dtype) * zz
    if n == 4:
        return zz**4 - jnp.asarray(2.0, dtype=real_dtype) * zz**3 + zz * zz - jnp.asarray(1.0 / 30.0, dtype=real_dtype)
    return jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=zz.dtype)


def _complex_polylog_scalar(
    s: jax.Array,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    ss = el.as_complex(s)
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    absz = jnp.abs(zz)
    eps = jnp.asarray(1e-12, dtype=real_dtype)
    base = jnp.ceil(jnp.asarray(8.0, dtype=real_dtype) / jnp.maximum(jnp.asarray(1.0, dtype=real_dtype) - absz, eps))
    n_eff = jnp.where(absz < 1.0, base, jnp.asarray(terms, dtype=real_dtype))
    n_eff = jnp.clip(n_eff, jnp.asarray(min_terms, dtype=real_dtype), jnp.asarray(max_terms, dtype=real_dtype))
    ks = jnp.arange(1, max_terms + 1, dtype=real_dtype)
    mask = ks <= n_eff
    series_terms = jnp.power(zz, ks) / jnp.power(ks, ss)
    series = jnp.sum(jnp.where(mask, series_terms, jnp.zeros_like(series_terms)))
    nanv = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=zz.dtype)
    return jnp.where(absz < 1.0, series, nanv)


def _complex_polylog_si_scalar(s: int, z: jax.Array, terms: int = 64, max_terms: int = 512, min_terms: int = 32) -> jax.Array:
    zz = el.as_complex(z)
    sval = jnp.asarray(float(s) + 0.0j, dtype=zz.dtype)
    return _complex_polylog_scalar(sval, zz, terms=terms, max_terms=max_terms, min_terms=min_terms)


def _complex_agm_scalar(a: jax.Array, b: jax.Array, iters: int = 10) -> jax.Array:
    aa = el.as_complex(a)
    bb = el.as_complex(b)

    def body(_, state):
        x, y = state
        return (0.5 * (x + y), jnp.sqrt(x * y))

    out, _ = lax.fori_loop(0, iters, body, (aa, bb))
    return out


def _pad_point_batch_last(args, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)

@partial(jax.jit, static_argnames=())
def arb_exp_point(x: jax.Array) -> jax.Array:
    return jnp.exp(x)


@partial(jax.jit, static_argnames=())
def arb_log_point(x: jax.Array) -> jax.Array:
    return jnp.log(x)


@partial(jax.jit, static_argnames=())
def arb_sqrt_point(x: jax.Array) -> jax.Array:
    return jnp.sqrt(x)


@partial(jax.jit, static_argnames=())
def arb_sin_point(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@partial(jax.jit, static_argnames=())
def arb_cos_point(x: jax.Array) -> jax.Array:
    return jnp.cos(x)


@partial(jax.jit, static_argnames=())
def arb_tan_point(x: jax.Array) -> jax.Array:
    return jnp.tan(x)


@partial(jax.jit, static_argnames=())
def arb_sinh_point(x: jax.Array) -> jax.Array:
    return jnp.sinh(x)


@partial(jax.jit, static_argnames=())
def arb_cosh_point(x: jax.Array) -> jax.Array:
    return jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def arb_tanh_point(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


@partial(jax.jit, static_argnames=())
def arb_abs_point(x: jax.Array) -> jax.Array:
    return jnp.abs(x)


@partial(jax.jit, static_argnames=())
def arb_add_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


@partial(jax.jit, static_argnames=())
def arb_sub_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x - y


@partial(jax.jit, static_argnames=())
def arb_mul_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@partial(jax.jit, static_argnames=())
def arb_div_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x / y


@partial(jax.jit, static_argnames=())
def arb_inv_point(x: jax.Array) -> jax.Array:
    return 1.0 / x


@partial(jax.jit, static_argnames=())
def arb_fma_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x * y + z


@partial(jax.jit, static_argnames=())
def arb_log1p_point(x: jax.Array) -> jax.Array:
    return jnp.log1p(x)


@partial(jax.jit, static_argnames=())
def arb_expm1_point(x: jax.Array) -> jax.Array:
    return jnp.expm1(x)


@partial(jax.jit, static_argnames=())
def arb_sin_cos_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sin(x), jnp.cos(x)


@partial(jax.jit, static_argnames=())
def arb_sinh_cosh_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sinh(x), jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def arb_sin_pi_point(x: jax.Array) -> jax.Array:
    return el.sin_pi(x)


@partial(jax.jit, static_argnames=())
def arb_cos_pi_point(x: jax.Array) -> jax.Array:
    return el.cos_pi(x)


@partial(jax.jit, static_argnames=())
def arb_tan_pi_point(x: jax.Array) -> jax.Array:
    return el.tan_pi(x)


@partial(jax.jit, static_argnames=())
def arb_sinc_point(x: jax.Array) -> jax.Array:
    return el.sinc(x)


@partial(jax.jit, static_argnames=())
def arb_sinc_pi_point(x: jax.Array) -> jax.Array:
    return el.sinc_pi(x)


@partial(jax.jit, static_argnames=())
def arb_asin_point(x: jax.Array) -> jax.Array:
    return jnp.arcsin(x)


@partial(jax.jit, static_argnames=())
def arb_acos_point(x: jax.Array) -> jax.Array:
    return jnp.arccos(x)


@partial(jax.jit, static_argnames=())
def arb_atan_point(x: jax.Array) -> jax.Array:
    return jnp.arctan(x)


@partial(jax.jit, static_argnames=())
def arb_asinh_point(x: jax.Array) -> jax.Array:
    return jnp.arcsinh(x)


@partial(jax.jit, static_argnames=())
def arb_acosh_point(x: jax.Array) -> jax.Array:
    return jnp.arccosh(x)


@partial(jax.jit, static_argnames=())
def arb_atanh_point(x: jax.Array) -> jax.Array:
    return jnp.arctanh(x)


@partial(jax.jit, static_argnames=())
def arb_sign_point(x: jax.Array) -> jax.Array:
    return jnp.sign(x)


@partial(jax.jit, static_argnames=())
def arb_pow_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(x, y)


@partial(jax.jit, static_argnames=("n",))
def arb_pow_ui_point(x: jax.Array, n: int) -> jax.Array:
    return jnp.power(x, n)


@partial(jax.jit, static_argnames=("k",))
def arb_root_ui_point(x: jax.Array, k: int) -> jax.Array:
    xx = jnp.asarray(x)
    kf = jnp.asarray(k, dtype=xx.dtype if jnp.issubdtype(xx.dtype, jnp.floating) else jnp.float64)
    root_abs = jnp.power(jnp.abs(xx), 1.0 / kf)
    if (k % 2) == 1:
        return jnp.sign(xx) * root_abs
    return jnp.where(xx < 0, jnp.nan, root_abs)


@partial(jax.jit, static_argnames=())
def arb_cbrt_point(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.power(jnp.abs(x), 1.0 / 3.0)


@partial(jax.jit, static_argnames=())
def arb_pow_fmpz_point(x: jax.Array, n: jax.Array | int) -> jax.Array:
    return jnp.power(x, jnp.asarray(n))


@partial(jax.jit, static_argnames=())
def arb_pow_fmpq_point(x: jax.Array, p: jax.Array, q: jax.Array) -> jax.Array:
    return jnp.power(x, jnp.asarray(p) / jnp.asarray(q))


@partial(jax.jit, static_argnames=("k",))
def arb_root_point(x: jax.Array, k: int) -> jax.Array:
    return arb_root_ui_point(x, k)


@partial(jax.jit, static_argnames=())
def arb_lgamma_point(x: jax.Array) -> jax.Array:
    return lax.lgamma(x)


@partial(jax.jit, static_argnames=())
def arb_rgamma_point(x: jax.Array) -> jax.Array:
    return jnp.exp(-lax.lgamma(x))


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_0f1_point(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda aa, zz: hypgeom._real_hyp0f1_scalar(aa, zz), a, z)
    if regularized:
        out = out * arb_rgamma_point(a)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_1f1_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda aa, bb, zz: hypgeom._real_hyp1f1_regime(aa, bb, zz), a, b, z)
    if regularized:
        out = out * arb_rgamma_point(b)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_m_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1_point(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_2f1_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda aa, bb, cc, zz: hypgeom._real_hyp2f1_regime(aa, bb, cc, zz), a, b, c, z)
    if regularized:
        out = out * arb_rgamma_point(c)
    return out


@partial(jax.jit, static_argnames=())
def arb_hypgeom_u_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda aa, bb, zz: hypgeom._real_hypu_regime(aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_gamma_lower_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda ss, zz: hypgeom._gammainc_real(ss, zz), s, z)
    if not regularized:
        out = out * hypgeom._gamma_real(s)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_gamma_upper_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda ss, zz: hypgeom._gammaincc_real(ss, zz), s, z)
    if not regularized:
        out = out * hypgeom._gamma_real(s)
    return out


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_legendre_p_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_real_scalar(lambda mm, zz: hypgeom._real_legendre_p_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_legendre_q_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_real_scalar(lambda mm, zz: hypgeom._real_legendre_q_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_jacobi_p_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda aa, bb, zz: hypgeom._real_jacobi_p_scalar(n, aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_gegenbauer_c_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda ll, zz: hypgeom._real_gegenbauer_c_scalar(n, ll, zz), lam, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_chebyshev_t_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    nf = jnp.asarray(n, dtype=x.dtype)
    return lax.cond(jnp.all(jnp.abs(x) <= 1.0), lambda t: jnp.cos(nf * jnp.arccos(t)), lambda t: jnp.cosh(nf * jnp.arccosh(t)), x)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_chebyshev_u_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    nf = jnp.asarray(n + 1, dtype=x.dtype)
    def in_range(t):
        ang = jnp.arccos(t)
        return jnp.sin(nf * ang) / jnp.sin(ang)
    def out_range(t):
        ach = jnp.arccosh(jnp.abs(t))
        return jnp.sinh(nf * ach) / jnp.sinh(ach)
    return lax.cond(jnp.all(jnp.abs(x) <= 1.0), in_range, out_range, x)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_laguerre_l_point(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda mm, zz: _real_laguerre_l_scalar(n, mm, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_hermite_h_point(n: int, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda zz: _real_hermite_h_scalar(n, zz), z)


@partial(jax.jit, static_argnames=("reciprocal", "n_terms"))
def arb_hypgeom_pfq_point(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    z_arr = jnp.asarray(z)
    if a_arr.ndim <= 1 and b_arr.ndim <= 1 and z_arr.ndim == 0:
        return _real_pfq_scalar(a_arr, b_arr, z_arr, reciprocal=reciprocal, n_terms=n_terms)
    if z_arr.ndim == 0:
        z_arr = jnp.broadcast_to(z_arr, (a_arr.shape[0],))
    return jax.vmap(lambda aa, bb, zz: _real_pfq_scalar(aa, bb, zz, reciprocal=reciprocal, n_terms=n_terms))(a_arr, b_arr, z_arr)


def arb_hypgeom_0f1_batch_fixed_point(a: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_0f1_point(a, z, regularized=regularized)


def arb_hypgeom_0f1_batch_padded_point(a: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, z), pad_to)
    return arb_hypgeom_0f1_point(*call_args, regularized=regularized)


def arb_hypgeom_1f1_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1_point(a, b, z, regularized=regularized)


def arb_hypgeom_1f1_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_1f1_point(*call_args, regularized=regularized)


def arb_hypgeom_m_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_m_point(a, b, z, regularized=regularized)


def arb_hypgeom_m_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_m_point(*call_args, regularized=regularized)


def arb_hypgeom_2f1_batch_fixed_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_2f1_point(a, b, c, z, regularized=regularized)


def arb_hypgeom_2f1_batch_padded_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, c, z), pad_to)
    return arb_hypgeom_2f1_point(*call_args, regularized=regularized)


def arb_hypgeom_u_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_u_point(a, b, z)


def arb_hypgeom_u_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_u_point(*call_args)


def arb_hypgeom_gamma_lower_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_gamma_lower_point(s, z, regularized=regularized)


def arb_hypgeom_gamma_lower_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return arb_hypgeom_gamma_lower_point(*call_args, regularized=regularized)


def arb_hypgeom_gamma_upper_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_gamma_upper_point(s, z, regularized=regularized)


def arb_hypgeom_gamma_upper_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return arb_hypgeom_gamma_upper_point(*call_args, regularized=regularized)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_gamma_point(x: jax.Array) -> jax.Array:
    return arb_gamma_point(x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erf_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_erf_series, x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfc_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: 1.0 - hypgeom._real_erf_series(t), x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfi_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_erfi, x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfinv_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_erfinv_scalar, x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfcinv_point(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    return arb_hypgeom_erfinv_point(1.0 - arr)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_ei_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_ei_scalar, z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_si_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: hypgeom._real_si_ci_scalar(t)[0], z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_ci_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: hypgeom._real_si_ci_scalar(t)[1], z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_shi_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: 0.5 * (hypgeom._real_ei_scalar(t) - hypgeom._real_ei_scalar(-t)), z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_chi_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: 0.5 * (hypgeom._real_ei_scalar(t) + hypgeom._real_ei_scalar(-t)), z)


@partial(jax.jit, static_argnames=("offset",))
def arb_hypgeom_li_point(z: jax.Array, offset: int = 0) -> jax.Array:
    offset_term = jnp.asarray(0.0, dtype=jnp.asarray(z).dtype)
    if offset > 0:
        offset_term = hypgeom._real_ei_scalar(jnp.log(jnp.asarray(offset, dtype=jnp.asarray(z).dtype)))
    return _vectorize_real_scalar(lambda t: hypgeom._real_ei_scalar(jnp.log(t)) - offset_term, z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_dilog_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_dilog_scalar, z)


@partial(jax.jit, static_argnames=("normalized",))
def arb_hypgeom_fresnel_point(z: jax.Array, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return _vectorize_real_scalar_tuple2(lambda t: hypgeom._real_fresnel_scalar(t, normalized), z)


def arb_hypgeom_pfq_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    return arb_hypgeom_pfq_point(a, b, z, reciprocal=reciprocal, n_terms=n_terms)


def arb_hypgeom_pfq_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_pfq_point(*call_args, reciprocal=reciprocal, n_terms=n_terms)


def arb_hypgeom_gamma_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_gamma_point(x)


def arb_hypgeom_gamma_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_gamma_point(*call_args)


def arb_hypgeom_erf_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erf_point(x)


def arb_hypgeom_erf_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erf_point(*call_args)


def arb_hypgeom_erfc_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfc_point(x)


def arb_hypgeom_erfc_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfc_point(*call_args)


def arb_hypgeom_erfi_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfi_point(x)


def arb_hypgeom_erfi_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfi_point(*call_args)


def arb_hypgeom_erfinv_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfinv_point(x)


def arb_hypgeom_erfinv_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfinv_point(*call_args)


def arb_hypgeom_erfcinv_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfcinv_point(x)


def arb_hypgeom_erfcinv_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfcinv_point(*call_args)


def arb_hypgeom_ei_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_ei_point(z)


def arb_hypgeom_ei_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_ei_point(*call_args)


def arb_hypgeom_si_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_si_point(z)


def arb_hypgeom_si_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_si_point(*call_args)


def arb_hypgeom_ci_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_ci_point(z)


def arb_hypgeom_ci_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_ci_point(*call_args)


def arb_hypgeom_shi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_shi_point(z)


def arb_hypgeom_shi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_shi_point(*call_args)


def arb_hypgeom_chi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_chi_point(z)


def arb_hypgeom_chi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_chi_point(*call_args)


def arb_hypgeom_li_batch_fixed_point(z: jax.Array, *, offset: int = 0) -> jax.Array:
    return arb_hypgeom_li_point(z, offset=offset)


def arb_hypgeom_li_batch_padded_point(z: jax.Array, *, pad_to: int, offset: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_li_point(*call_args, offset=offset)


def arb_hypgeom_dilog_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_dilog_point(z)


def arb_hypgeom_dilog_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_dilog_point(*call_args)


def arb_hypgeom_fresnel_batch_fixed_point(z: jax.Array, *, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_fresnel_point(z, normalized=normalized)


def arb_hypgeom_fresnel_batch_padded_point(z: jax.Array, *, pad_to: int, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_fresnel_point(*call_args, normalized=normalized)


def arb_hypgeom_legendre_p_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return arb_hypgeom_legendre_p_point(n, m, z, type=type)


def arb_hypgeom_legendre_p_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return arb_hypgeom_legendre_p_point(n, *call_args, type=type)


def arb_hypgeom_legendre_q_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return arb_hypgeom_legendre_q_point(n, m, z, type=type)


def arb_hypgeom_legendre_q_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return arb_hypgeom_legendre_q_point(n, *call_args, type=type)


def arb_hypgeom_jacobi_p_batch_fixed_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_jacobi_p_point(n, a, b, z)


def arb_hypgeom_jacobi_p_batch_padded_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_jacobi_p_point(n, *call_args)


def arb_hypgeom_gegenbauer_c_batch_fixed_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_gegenbauer_c_point(n, lam, z)


def arb_hypgeom_gegenbauer_c_batch_padded_point(n: int, lam: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((lam, z), pad_to)
    return arb_hypgeom_gegenbauer_c_point(n, *call_args)


def arb_hypgeom_chebyshev_t_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return arb_hypgeom_chebyshev_t_point(n, z)


def arb_hypgeom_chebyshev_t_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_chebyshev_t_point(n, *call_args)


def arb_hypgeom_chebyshev_u_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return arb_hypgeom_chebyshev_u_point(n, z)


def arb_hypgeom_chebyshev_u_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_chebyshev_u_point(n, *call_args)


def arb_hypgeom_laguerre_l_batch_fixed_point(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_laguerre_l_point(n, m, z)


def arb_hypgeom_laguerre_l_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return arb_hypgeom_laguerre_l_point(n, *call_args)


def arb_hypgeom_hermite_h_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return arb_hypgeom_hermite_h_point(n, z)


def arb_hypgeom_hermite_h_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_hermite_h_point(n, *call_args)


@partial(jax.jit, static_argnames=())
def acb_abs_point(z: jax.Array) -> jax.Array:
    return jnp.abs(z)


@partial(jax.jit, static_argnames=())
def acb_add_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


@partial(jax.jit, static_argnames=())
def acb_sub_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x - y


@partial(jax.jit, static_argnames=())
def acb_mul_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@partial(jax.jit, static_argnames=())
def acb_div_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x / y


@partial(jax.jit, static_argnames=())
def acb_inv_point(x: jax.Array) -> jax.Array:
    return 1.0 / x


@partial(jax.jit, static_argnames=())
def acb_fma_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x * y + z


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_0f1_point(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda aa, zz: hypgeom._complex_hyp0f1_scalar(aa, zz), a, z)
    if regularized:
        out = out * acb_rgamma_point(a)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_1f1_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda aa, bb, zz: hypgeom._complex_hyp1f1_regime(aa, bb, zz), a, b, z)
    if regularized:
        out = out * acb_rgamma_point(b)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_m_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1_point(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_2f1_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda aa, bb, cc, zz: hypgeom._complex_hyp2f1_regime(aa, bb, cc, zz), a, b, c, z)
    if regularized:
        out = out * acb_rgamma_point(c)
    return out


@partial(jax.jit, static_argnames=())
def acb_hypgeom_u_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda aa, bb, zz: hypgeom._complex_hypu_regime(aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_gamma_lower_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda ss, zz: hypgeom._complex_gamma_lower_scalar(ss, zz), s, z)
    if regularized:
        out = out / jnp.exp(acb_lgamma_point(s))
    return out


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_gamma_upper_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda ss, zz: hypgeom._complex_gamma_upper_scalar(ss, zz), s, z)
    if regularized:
        out = out / jnp.exp(acb_lgamma_point(s))
    return out


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_legendre_p_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_complex_scalar(lambda mm, zz: hypgeom._complex_legendre_p_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_legendre_q_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_complex_scalar(lambda mm, zz: hypgeom._complex_legendre_q_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_jacobi_p_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda aa, bb, zz: hypgeom._complex_jacobi_p_scalar(n, aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_gegenbauer_c_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda ll, zz: hypgeom._complex_gegenbauer_c_scalar(n, ll, zz), lam, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_chebyshev_t_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    nf = jnp.asarray(n, dtype=x.real.dtype)
    return jnp.cos(nf * jnp.arccos(x))


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_chebyshev_u_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    theta = jnp.arccos(x)
    nf = jnp.asarray(n + 1, dtype=x.real.dtype)
    return jnp.sin(nf * theta) / jnp.sin(theta)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_laguerre_l_point(n: int, a: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda aa, zz: jnp.exp(hypgeom._complex_loggamma(n + aa + 1.0) - hypgeom._complex_loggamma(n + 1.0) - hypgeom._complex_loggamma(aa + 1.0)) * hypgeom._complex_hyp1f1_scalar(-jnp.asarray(n, dtype=zz.real.dtype), aa + 1.0, zz), a, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_hermite_h_point(n: int, z: jax.Array) -> jax.Array:
    def scalar(w):
        if n == 0:
            return jnp.asarray(1.0 + 0.0j, dtype=jnp.asarray(w).dtype)
        if n == 1:
            return 2.0 * w
        h0 = jnp.asarray(1.0 + 0.0j, dtype=jnp.asarray(w).dtype)
        h1 = 2.0 * w

        def body(k, state):
            h_prev, h_curr = state
            h_next = 2.0 * w * h_curr - 2.0 * jnp.asarray(k - 1, dtype=jnp.asarray(w).real.dtype) * h_prev
            return h_curr, h_next

        _, hn = lax.fori_loop(2, n + 1, body, (h0, h1))
        return hn

    return _vectorize_complex_scalar(scalar, z)


@partial(jax.jit, static_argnames=("reciprocal", "n_terms"))
def acb_hypgeom_pfq_point(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    z_arr = jnp.asarray(z)
    if a_arr.ndim <= 1 and b_arr.ndim <= 1 and z_arr.ndim == 0:
        return _complex_pfq_scalar(a_arr, b_arr, z_arr, reciprocal=reciprocal, n_terms=n_terms)
    if z_arr.ndim == 0:
        z_arr = jnp.broadcast_to(z_arr, (a_arr.shape[0],))
    return jax.vmap(lambda aa, bb, zz: _complex_pfq_scalar(aa, bb, zz, reciprocal=reciprocal, n_terms=n_terms))(a_arr, b_arr, z_arr)


def acb_hypgeom_0f1_batch_fixed_point(a: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_0f1_point(a, z, regularized=regularized)


def acb_hypgeom_0f1_batch_padded_point(a: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, z), pad_to)
    return acb_hypgeom_0f1_point(*call_args, regularized=regularized)


def acb_hypgeom_1f1_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1_point(a, b, z, regularized=regularized)


def acb_hypgeom_1f1_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_1f1_point(*call_args, regularized=regularized)


def acb_hypgeom_m_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_m_point(a, b, z, regularized=regularized)


def acb_hypgeom_m_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_m_point(*call_args, regularized=regularized)


def acb_hypgeom_2f1_batch_fixed_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_2f1_point(a, b, c, z, regularized=regularized)


def acb_hypgeom_2f1_batch_padded_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, c, z), pad_to)
    return acb_hypgeom_2f1_point(*call_args, regularized=regularized)


def acb_hypgeom_u_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_u_point(a, b, z)


def acb_hypgeom_u_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_u_point(*call_args)


def acb_hypgeom_gamma_lower_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_gamma_lower_point(s, z, regularized=regularized)


def acb_hypgeom_gamma_lower_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return acb_hypgeom_gamma_lower_point(*call_args, regularized=regularized)


def acb_hypgeom_gamma_upper_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_gamma_upper_point(s, z, regularized=regularized)


def acb_hypgeom_gamma_upper_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return acb_hypgeom_gamma_upper_point(*call_args, regularized=regularized)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_gamma_point(x: jax.Array) -> jax.Array:
    return acb_gamma_point(x)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_erf_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_erf_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_erfc_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_erfc_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_erfi_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_erfi_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_ei_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_ei_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_si_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_si_ci_series(w)[0], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_ci_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_si_ci_series(w)[1], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_shi_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_shi_chi_series(w)[0], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_chi_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_shi_chi_series(w)[1], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_li_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_ei_series(jnp.log(w)), z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_dilog_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_dilog_series, z)


@partial(jax.jit, static_argnames=("normalized",))
def acb_hypgeom_fresnel_point(z: jax.Array, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return _vectorize_complex_scalar_tuple2(lambda w: hypgeom._complex_fresnel(w, normalized), z)


def acb_hypgeom_pfq_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    return acb_hypgeom_pfq_point(a, b, z, reciprocal=reciprocal, n_terms=n_terms)


def acb_hypgeom_pfq_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_pfq_point(*call_args, reciprocal=reciprocal, n_terms=n_terms)


def acb_hypgeom_gamma_batch_fixed_point(x: jax.Array) -> jax.Array:
    return acb_hypgeom_gamma_point(x)


def acb_hypgeom_gamma_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return acb_hypgeom_gamma_point(*call_args)


def acb_hypgeom_erf_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_erf_point(z)


def acb_hypgeom_erf_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_erf_point(*call_args)


def acb_hypgeom_erfc_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_erfc_point(z)


def acb_hypgeom_erfc_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_erfc_point(*call_args)


def acb_hypgeom_erfi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_erfi_point(z)


def acb_hypgeom_erfi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_erfi_point(*call_args)


def acb_hypgeom_ei_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_ei_point(z)


def acb_hypgeom_ei_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_ei_point(*call_args)


def acb_hypgeom_si_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_si_point(z)


def acb_hypgeom_si_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_si_point(*call_args)


def acb_hypgeom_ci_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_ci_point(z)


def acb_hypgeom_ci_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_ci_point(*call_args)


def acb_hypgeom_shi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_shi_point(z)


def acb_hypgeom_shi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_shi_point(*call_args)


def acb_hypgeom_chi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_chi_point(z)


def acb_hypgeom_chi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_chi_point(*call_args)


def acb_hypgeom_li_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_li_point(z)


def acb_hypgeom_li_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_li_point(*call_args)


def acb_hypgeom_dilog_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_dilog_point(z)


def acb_hypgeom_dilog_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_dilog_point(*call_args)


def acb_hypgeom_fresnel_batch_fixed_point(z: jax.Array, *, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return acb_hypgeom_fresnel_point(z, normalized=normalized)


def acb_hypgeom_fresnel_batch_padded_point(z: jax.Array, *, pad_to: int, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_fresnel_point(*call_args, normalized=normalized)


def acb_hypgeom_legendre_p_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return acb_hypgeom_legendre_p_point(n, m, z, type=type)


def acb_hypgeom_legendre_p_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return acb_hypgeom_legendre_p_point(n, *call_args, type=type)


def acb_hypgeom_legendre_q_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return acb_hypgeom_legendre_q_point(n, m, z, type=type)


def acb_hypgeom_legendre_q_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return acb_hypgeom_legendre_q_point(n, *call_args, type=type)


def acb_hypgeom_jacobi_p_batch_fixed_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_jacobi_p_point(n, a, b, z)


def acb_hypgeom_jacobi_p_batch_padded_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_jacobi_p_point(n, *call_args)


def acb_hypgeom_gegenbauer_c_batch_fixed_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_gegenbauer_c_point(n, lam, z)


def acb_hypgeom_gegenbauer_c_batch_padded_point(n: int, lam: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((lam, z), pad_to)
    return acb_hypgeom_gegenbauer_c_point(n, *call_args)


def acb_hypgeom_chebyshev_t_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return acb_hypgeom_chebyshev_t_point(n, z)


def acb_hypgeom_chebyshev_t_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_chebyshev_t_point(n, *call_args)


def acb_hypgeom_chebyshev_u_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return acb_hypgeom_chebyshev_u_point(n, z)


def acb_hypgeom_chebyshev_u_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_chebyshev_u_point(n, *call_args)


def acb_hypgeom_laguerre_l_batch_fixed_point(n: int, a: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_laguerre_l_point(n, a, z)


def acb_hypgeom_laguerre_l_batch_padded_point(n: int, a: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, z), pad_to)
    return acb_hypgeom_laguerre_l_point(n, *call_args)


def acb_hypgeom_hermite_h_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return acb_hypgeom_hermite_h_point(n, z)


def acb_hypgeom_hermite_h_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_hermite_h_point(n, *call_args)


@partial(jax.jit, static_argnames=())
def acb_log1p_point(x: jax.Array) -> jax.Array:
    return jnp.log1p(x)


@partial(jax.jit, static_argnames=())
def acb_expm1_point(x: jax.Array) -> jax.Array:
    return jnp.expm1(x)


@partial(jax.jit, static_argnames=())
def acb_sin_cos_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sin(x), jnp.cos(x)


@partial(jax.jit, static_argnames=())
def acb_asin_point(x: jax.Array) -> jax.Array:
    return jnp.arcsin(x)


@partial(jax.jit, static_argnames=())
def acb_acos_point(x: jax.Array) -> jax.Array:
    return jnp.arccos(x)


@partial(jax.jit, static_argnames=())
def acb_atan_point(x: jax.Array) -> jax.Array:
    return jnp.arctan(x)


@partial(jax.jit, static_argnames=())
def acb_asinh_point(x: jax.Array) -> jax.Array:
    return jnp.arcsinh(x)


@partial(jax.jit, static_argnames=())
def acb_acosh_point(x: jax.Array) -> jax.Array:
    return jnp.arccosh(x)


@partial(jax.jit, static_argnames=())
def acb_atanh_point(x: jax.Array) -> jax.Array:
    return jnp.arctanh(x)


@partial(jax.jit, static_argnames=())
def acb_exp_point(x: jax.Array) -> jax.Array:
    return jnp.exp(x)


@partial(jax.jit, static_argnames=())
def acb_log_point(x: jax.Array) -> jax.Array:
    return jnp.log(x)


@partial(jax.jit, static_argnames=())
def acb_sqrt_point(x: jax.Array) -> jax.Array:
    return jnp.sqrt(x)


@partial(jax.jit, static_argnames=())
def acb_rsqrt_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.sqrt(x)


@partial(jax.jit, static_argnames=())
def acb_sin_point(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@partial(jax.jit, static_argnames=())
def acb_cos_point(x: jax.Array) -> jax.Array:
    return jnp.cos(x)


@partial(jax.jit, static_argnames=())
def acb_tan_point(x: jax.Array) -> jax.Array:
    return jnp.tan(x)


@partial(jax.jit, static_argnames=())
def acb_cot_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.tan(x)


@partial(jax.jit, static_argnames=())
def acb_sinh_point(x: jax.Array) -> jax.Array:
    return jnp.sinh(x)


@partial(jax.jit, static_argnames=())
def acb_cosh_point(x: jax.Array) -> jax.Array:
    return jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def acb_tanh_point(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


@partial(jax.jit, static_argnames=())
def acb_sech_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def acb_csch_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.sinh(x)


@partial(jax.jit, static_argnames=())
def acb_sin_pi_point(x: jax.Array) -> jax.Array:
    return el.sin_pi(x)


@partial(jax.jit, static_argnames=())
def acb_cos_pi_point(x: jax.Array) -> jax.Array:
    return el.cos_pi(x)


@partial(jax.jit, static_argnames=())
def acb_sin_cos_pi_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return el.sin_pi(x), el.cos_pi(x)


@partial(jax.jit, static_argnames=())
def acb_tan_pi_point(x: jax.Array) -> jax.Array:
    return el.tan_pi(x)


@partial(jax.jit, static_argnames=())
def acb_cot_pi_point(x: jax.Array) -> jax.Array:
    return 1.0 / el.tan_pi(x)


@partial(jax.jit, static_argnames=())
def acb_csc_pi_point(x: jax.Array) -> jax.Array:
    return 1.0 / el.sin_pi(x)


@partial(jax.jit, static_argnames=())
def acb_sinc_point(x: jax.Array) -> jax.Array:
    return jnp.where(x == 0.0, 1.0 + 0.0j, jnp.sin(x) / x)


@partial(jax.jit, static_argnames=())
def acb_sinc_pi_point(x: jax.Array) -> jax.Array:
    return el.sinc_pi(x)


@partial(jax.jit, static_argnames=())
def acb_exp_pi_i_point(x: jax.Array) -> jax.Array:
    return el.exp_pi_i(x)


@partial(jax.jit, static_argnames=())
def acb_exp_invexp_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    ex = jnp.exp(x)
    return ex, 1.0 / ex


@partial(jax.jit, static_argnames=())
def acb_addmul_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x + y * z


@partial(jax.jit, static_argnames=())
def acb_submul_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x - y * z


@partial(jax.jit, static_argnames=())
def acb_pow_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(x, y)


@partial(jax.jit, static_argnames=())
def acb_pow_arb_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(x, y)


@partial(jax.jit, static_argnames=("n",))
def acb_pow_ui_point(x: jax.Array, n: int) -> jax.Array:
    return jnp.power(x, n)


@partial(jax.jit, static_argnames=("n",))
def acb_pow_si_point(x: jax.Array, n: int) -> jax.Array:
    return jnp.power(x, n)


@partial(jax.jit, static_argnames=())
def acb_pow_fmpz_point(x: jax.Array, n: int | jax.Array) -> jax.Array:
    return jnp.power(x, jnp.asarray(n))


@partial(jax.jit, static_argnames=())
def acb_sqr_point(x: jax.Array) -> jax.Array:
    return x * x


@partial(jax.jit, static_argnames=("k",))
def acb_root_ui_point(x: jax.Array, k: int) -> jax.Array:
    return jnp.power(x, 1.0 / jnp.asarray(k, dtype=jnp.asarray(x).real.dtype))


@partial(jax.jit, static_argnames=())
def acb_gamma_point(x: jax.Array) -> jax.Array:
    return vmap_complex_scalar(lambda t: jnp.exp(acb_core._complex_loggamma(t)))(x)


@partial(jax.jit, static_argnames=())
def acb_rgamma_point(x: jax.Array) -> jax.Array:
    return vmap_complex_scalar(lambda t: jnp.exp(-acb_core._complex_loggamma(t)))(x)


@partial(jax.jit, static_argnames=())
def acb_lgamma_point(x: jax.Array) -> jax.Array:
    return vmap_complex_scalar(acb_core._complex_loggamma)(x)


@partial(jax.jit, static_argnames=())
def acb_log_sin_pi_point(x: jax.Array) -> jax.Array:
    return el.log_sin_pi(x)


acb_digamma_point = scalarize_unary_complex(_complex_digamma_scalar)
acb_barnes_g_point = scalarize_unary_complex(barnesg.barnesg_complex)
acb_log_barnes_g_point = scalarize_unary_complex(barnesg.log_barnesg)
acb_zeta_point = scalarize_unary_complex(_complex_zeta_scalar)


@jax.jit
def acb_hurwitz_zeta_point(s: jax.Array, a: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(_complex_hurwitz_zeta_scalar, s, a)


@partial(jax.jit, static_argnames=("n",))
def acb_polygamma_point(n: int, x: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(partial(_complex_polygamma_scalar, n), x)


@partial(jax.jit, static_argnames=("n",))
def acb_bernoulli_poly_ui_point(n: int, x: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(partial(_complex_bernoulli_poly_ui_scalar, n), x)


acb_polylog_point = scalarize_binary_complex(_complex_polylog_scalar)


@partial(jax.jit, static_argnames=("s",))
def acb_polylog_si_point(s: int, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(partial(_complex_polylog_si_scalar, s), z)


acb_agm_point = scalarize_binary_complex(_complex_agm_scalar)
acb_agm1_point = scalarize_unary_complex(lambda x: _complex_agm_scalar(jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(x).dtype), x))
acb_agm1_cpx_point = scalarize_unary_complex(lambda x: _complex_agm_scalar(jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(x).dtype), x))


@partial(jax.jit, static_argnames=("n_terms",))
def acb_dirichlet_zeta_point(s: jax.Array, n_terms: int = 64) -> jax.Array:
    out = _vectorize_complex_scalar(lambda ss: _complex_zeta_scalar(ss, n_terms), s)
    return out.astype(el.complex_dtype_from(s))


@partial(jax.jit, static_argnames=("n_terms",))
def acb_dirichlet_eta_point(s: jax.Array, n_terms: int = 64) -> jax.Array:
    ss = el.as_complex(s)
    zeta = acb_dirichlet_zeta_point(ss, n_terms=n_terms)
    one = jnp.asarray(1.0, dtype=el.real_dtype_from_complex_dtype(ss.dtype))
    two = jnp.asarray(2.0, dtype=el.real_dtype_from_complex_dtype(ss.dtype))
    factor = one - jnp.exp((ss - one) * jnp.log(two))
    return (factor * zeta).astype(el.complex_dtype_from(s))


@partial(jax.jit, static_argnames=())
def acb_modular_j_point(tau: jax.Array) -> jax.Array:
    tt = el.as_complex(tau)
    real_dtype = el.real_dtype_from_complex_dtype(tt.dtype)
    q = jnp.exp(jnp.asarray(2j, dtype=tt.dtype) * jnp.asarray(el.PI, dtype=real_dtype) * tt)
    c744 = jnp.asarray(744.0, dtype=real_dtype)
    c1 = jnp.asarray(196884.0, dtype=real_dtype)
    c2 = jnp.asarray(21493760.0, dtype=real_dtype)
    return jnp.asarray(1.0, dtype=real_dtype) / q + c744 + c1 * q + c2 * q * q


@partial(jax.jit, static_argnames=())
def acb_elliptic_k_point(m: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(
        lambda mm: jnp.asarray(el.HALF_PI, dtype=el.real_dtype_from_complex_dtype(el.as_complex(mm).dtype))
        / acb_elliptic._agm(
            jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(mm).dtype),
            jnp.sqrt(1.0 - el.as_complex(mm)),
            iters=8,
        ),
        m,
    )


@partial(jax.jit, static_argnames=())
def acb_elliptic_e_point(m: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(
        lambda mm: jnp.asarray(el.HALF_PI, dtype=el.real_dtype_from_complex_dtype(el.as_complex(mm).dtype))
        * acb_elliptic._agm(
            jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(mm).dtype),
            jnp.sqrt(1.0 - el.as_complex(mm)),
            iters=8,
        ),
        m,
    )


def acb_dirichlet_zeta_batch_fixed_point(s: jax.Array, *, n_terms: int = 64) -> jax.Array:
    return acb_dirichlet_zeta_point(s, n_terms=n_terms)


def acb_dirichlet_zeta_batch_padded_point(s: jax.Array, *, pad_to: int, n_terms: int = 64) -> jax.Array:
    call_args, _ = _pad_point_batch_last((s,), pad_to)
    return acb_dirichlet_zeta_point(*call_args, n_terms=n_terms)


def acb_dirichlet_eta_batch_fixed_point(s: jax.Array, *, n_terms: int = 64) -> jax.Array:
    return acb_dirichlet_eta_point(s, n_terms=n_terms)


def acb_dirichlet_eta_batch_padded_point(s: jax.Array, *, pad_to: int, n_terms: int = 64) -> jax.Array:
    call_args, _ = _pad_point_batch_last((s,), pad_to)
    return acb_dirichlet_eta_point(*call_args, n_terms=n_terms)


def acb_modular_j_batch_fixed_point(tau: jax.Array) -> jax.Array:
    return acb_modular_j_point(tau)


def acb_modular_j_batch_padded_point(tau: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_point_batch_last((tau,), pad_to)
    return acb_modular_j_point(*call_args)


def acb_elliptic_k_batch_fixed_point(m: jax.Array) -> jax.Array:
    return acb_elliptic_k_point(m)


def acb_elliptic_k_batch_padded_point(m: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_point_batch_last((m,), pad_to)
    return acb_elliptic_k_point(*call_args)


def acb_elliptic_e_batch_fixed_point(m: jax.Array) -> jax.Array:
    return acb_elliptic_e_point(m)


def acb_elliptic_e_batch_padded_point(m: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_point_batch_last((m,), pad_to)
    return acb_elliptic_e_point(*call_args)


@partial(jax.jit, static_argnames=())
def arb_gamma_point(x: jax.Array) -> jax.Array:
    return jnp.exp(lax.lgamma(x))


@partial(jax.jit, static_argnames=())
def arb_erf_point(x: jax.Array) -> jax.Array:
    return hypgeom._real_erf_series(x)


@partial(jax.jit, static_argnames=())
def arb_erfc_point(x: jax.Array) -> jax.Array:
    return 1.0 - hypgeom._real_erf_series(x)


@partial(jax.jit, static_argnames=())
def arb_bessel_j_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_j(nu, z)


@partial(jax.jit, static_argnames=())
def arb_bessel_y_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_y(nu, z)


@partial(jax.jit, static_argnames=())
def arb_bessel_i_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_i(nu, z)


@partial(jax.jit, static_argnames=())
def arb_bessel_k_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_k(nu, z)


__all__ = [
    "arb_exp_point",
    "arb_log_point",
    "arb_sqrt_point",
    "arb_sin_point",
    "arb_cos_point",
    "arb_tan_point",
    "arb_sinh_point",
    "arb_cosh_point",
    "arb_tanh_point",
    "arb_abs_point",
    "arb_add_point",
    "arb_sub_point",
    "arb_mul_point",
    "arb_div_point",
    "arb_inv_point",
    "arb_fma_point",
    "arb_log1p_point",
    "arb_expm1_point",
    "arb_sin_cos_point",
    "acb_abs_point",
    "acb_add_point",
    "acb_sub_point",
    "acb_mul_point",
    "acb_div_point",
    "acb_inv_point",
    "acb_fma_point",
    "acb_log1p_point",
    "acb_expm1_point",
    "acb_sin_cos_point",
        "arb_gamma_point",
        "arb_erf_point",
        "arb_erfc_point",
        "arb_bessel_j_point",
        "arb_bessel_y_point",
        "arb_bessel_i_point",
        "arb_bessel_k_point",
        "arb_mat_matmul_point",
        "arb_mat_matvec_point",
        "arb_mat_banded_matvec_point",
        "arb_mat_solve_point",
        "arb_mat_inv_point",
        "arb_mat_det_point",
        "arb_mat_trace_point",
        "arb_mat_triangular_solve_point",
        "arb_mat_lu_point",
        "arb_mat_qr_point",
        "arb_mat_2x2_det_point",
        "arb_mat_2x2_trace_point",
        "arb_mat_2x2_det_batch_point",
        "arb_mat_2x2_trace_batch_point",
        "acb_mat_matmul_point",
        "acb_mat_matvec_point",
        "acb_mat_banded_matvec_point",
        "acb_mat_solve_point",
        "acb_mat_inv_point",
        "acb_mat_det_point",
        "acb_mat_trace_point",
        "acb_mat_triangular_solve_point",
        "acb_mat_lu_point",
        "acb_mat_qr_point",
        "acb_mat_2x2_det_point",
        "acb_mat_2x2_trace_point",
        "acb_mat_2x2_det_batch_point",
        "acb_mat_2x2_trace_batch_point",
]

__all__.extend(
    [
        "arb_mat_zero_point",
        "arb_mat_identity_point",
        "arb_mat_sqr_point",
        "arb_mat_matmul_batch_fixed_point",
        "arb_mat_matmul_batch_padded_point",
        "arb_mat_matvec_batch_fixed_point",
        "arb_mat_matvec_batch_padded_point",
        "arb_mat_banded_matvec_batch_fixed_point",
        "arb_mat_banded_matvec_batch_padded_point",
        "arb_mat_det_batch_fixed_point",
        "arb_mat_det_batch_padded_point",
        "arb_mat_trace_batch_fixed_point",
        "arb_mat_trace_batch_padded_point",
        "arb_mat_sqr_batch_fixed_point",
        "arb_mat_sqr_batch_padded_point",
        "arb_mat_norm_fro_point",
        "arb_mat_norm_1_point",
        "arb_mat_norm_inf_point",
        "arb_mat_norm_fro_batch_fixed_point",
        "arb_mat_norm_fro_batch_padded_point",
        "arb_mat_norm_1_batch_fixed_point",
        "arb_mat_norm_1_batch_padded_point",
        "arb_mat_norm_inf_batch_fixed_point",
        "arb_mat_norm_inf_batch_padded_point",
        "arb_mat_matvec_cached_prepare_point",
        "arb_mat_matvec_cached_apply_point",
        "arb_mat_matvec_cached_apply_batch_fixed_point",
        "arb_mat_matvec_cached_apply_batch_padded_point",
        "acb_mat_zero_point",
        "acb_mat_identity_point",
        "acb_mat_sqr_point",
        "acb_mat_matmul_batch_fixed_point",
        "acb_mat_matmul_batch_padded_point",
        "acb_mat_matvec_batch_fixed_point",
        "acb_mat_matvec_batch_padded_point",
        "acb_mat_banded_matvec_batch_fixed_point",
        "acb_mat_banded_matvec_batch_padded_point",
        "acb_mat_det_batch_fixed_point",
        "acb_mat_det_batch_padded_point",
        "acb_mat_trace_batch_fixed_point",
        "acb_mat_trace_batch_padded_point",
        "acb_mat_sqr_batch_fixed_point",
        "acb_mat_sqr_batch_padded_point",
        "acb_mat_norm_fro_point",
        "acb_mat_norm_1_point",
        "acb_mat_norm_inf_point",
        "acb_mat_norm_fro_batch_fixed_point",
        "acb_mat_norm_fro_batch_padded_point",
        "acb_mat_norm_1_batch_fixed_point",
        "acb_mat_norm_1_batch_padded_point",
        "acb_mat_norm_inf_batch_fixed_point",
        "acb_mat_norm_inf_batch_padded_point",
        "acb_mat_matvec_cached_prepare_point",
        "acb_mat_matvec_cached_apply_point",
        "acb_mat_matvec_cached_apply_batch_fixed_point",
        "acb_mat_matvec_cached_apply_batch_padded_point",
        "arb_sinh_cosh_point",
        "arb_sin_pi_point",
        "arb_cos_pi_point",
        "arb_tan_pi_point",
        "arb_sinc_point",
        "arb_sinc_pi_point",
        "arb_asin_point",
        "arb_acos_point",
        "arb_atan_point",
        "arb_asinh_point",
        "arb_acosh_point",
        "arb_atanh_point",
        "arb_sign_point",
        "arb_pow_point",
        "arb_pow_ui_point",
        "arb_pow_fmpz_point",
        "arb_pow_fmpq_point",
        "arb_root_ui_point",
        "arb_root_point",
        "arb_cbrt_point",
        "arb_lgamma_point",
        "arb_rgamma_point",
        "acb_exp_point",
        "acb_log_point",
        "acb_sqrt_point",
        "acb_rsqrt_point",
        "acb_sin_point",
        "acb_cos_point",
        "acb_tan_point",
        "acb_cot_point",
        "acb_sinh_point",
        "acb_cosh_point",
        "acb_tanh_point",
        "acb_asin_point",
        "acb_acos_point",
        "acb_atan_point",
        "acb_asinh_point",
        "acb_acosh_point",
        "acb_atanh_point",
        "acb_sech_point",
        "acb_csch_point",
        "acb_sin_pi_point",
        "acb_cos_pi_point",
        "acb_sin_cos_pi_point",
        "acb_tan_pi_point",
        "acb_cot_pi_point",
        "acb_csc_pi_point",
        "acb_sinc_point",
        "acb_sinc_pi_point",
        "acb_exp_pi_i_point",
        "acb_exp_invexp_point",
        "acb_addmul_point",
        "acb_submul_point",
        "acb_pow_point",
        "acb_pow_arb_point",
        "acb_pow_ui_point",
        "acb_pow_si_point",
        "acb_pow_fmpz_point",
        "acb_sqr_point",
        "acb_root_ui_point",
        "acb_gamma_point",
        "acb_rgamma_point",
        "acb_lgamma_point",
        "acb_log_sin_pi_point",
        "acb_digamma_point",
        "acb_barnes_g_point",
        "acb_log_barnes_g_point",
        "acb_zeta_point",
        "acb_hurwitz_zeta_point",
        "acb_polygamma_point",
        "acb_bernoulli_poly_ui_point",
        "acb_polylog_point",
        "acb_polylog_si_point",
        "acb_agm_point",
        "acb_agm1_point",
        "acb_agm1_cpx_point",
    ]
)
