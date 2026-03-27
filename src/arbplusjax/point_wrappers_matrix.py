from __future__ import annotations
from functools import partial
import importlib

import jax
import jax.numpy as jnp
from jax import lax

from . import acb_core
from . import checks
from . import double_interval as di
from . import elementary as el
from . import kernel_helpers as kh
from .lazy_imports import lazy_module_proxy
from . import mat_common
from .kernel_helpers import scalarize_binary_complex, scalarize_unary_complex, vmap_complex_scalar

hypgeom = lazy_module_proxy("hypgeom", package=__package__)

_LAZY_FAMILY_EXPORTS = {
    "acb_dirichlet_zeta_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_zeta_point"),
    "acb_dirichlet_eta_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_eta_point"),
    "acb_modular_j_point": ("point_wrappers_dirichlet_modular", "acb_modular_j_point"),
    "acb_dirichlet_zeta_batch_fixed_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_zeta_batch_fixed_point"),
    "acb_dirichlet_zeta_batch_padded_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_zeta_batch_padded_point"),
    "acb_dirichlet_eta_batch_fixed_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_eta_batch_fixed_point"),
    "acb_dirichlet_eta_batch_padded_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_eta_batch_padded_point"),
    "acb_modular_j_batch_fixed_point": ("point_wrappers_dirichlet_modular", "acb_modular_j_batch_fixed_point"),
    "acb_modular_j_batch_padded_point": ("point_wrappers_dirichlet_modular", "acb_modular_j_batch_padded_point"),
    "acb_elliptic_k_point": ("point_wrappers_elliptic", "acb_elliptic_k_point"),
    "acb_elliptic_e_point": ("point_wrappers_elliptic", "acb_elliptic_e_point"),
    "acb_elliptic_k_batch_fixed_point": ("point_wrappers_elliptic", "acb_elliptic_k_batch_fixed_point"),
    "acb_elliptic_k_batch_padded_point": ("point_wrappers_elliptic", "acb_elliptic_k_batch_padded_point"),
    "acb_elliptic_e_batch_fixed_point": ("point_wrappers_elliptic", "acb_elliptic_e_batch_fixed_point"),
    "acb_elliptic_e_batch_padded_point": ("point_wrappers_elliptic", "acb_elliptic_e_batch_padded_point"),
}



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

__all__ = sorted(
    name
    for name, value in globals().items()
    if not name.startswith('_') and callable(value) and name.startswith(('arb_mat_', 'acb_mat_'))
)
