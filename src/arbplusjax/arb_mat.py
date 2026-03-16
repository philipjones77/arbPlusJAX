from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import checks
from . import double_interval as di
from . import kernel_helpers as kh
from . import mat_common

jax.config.update("jax_enable_x64", True)


def arb_mat_as_matrix(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_matrix(x, "arb_mat.as_matrix")


def arb_mat_as_vector(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_vector(x, "arb_mat.as_vector")


def arb_mat_as_rhs(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_rhs(x, "arb_mat.as_rhs")


def arb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = arb_mat_as_matrix(a)
    return tuple(int(x) for x in arr.shape)


def arb_mat_zero(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    zeros = jnp.zeros((n, n), dtype=dtype)
    return mat_common.interval_from_point(zeros)


def arb_mat_identity(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    eye = jnp.eye(n, dtype=dtype)
    return mat_common.interval_from_point(eye)


def _as_mat_2x2(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_mat_2x2(x, "arb_mat._as_mat_2x2")


def _mid_matrix(a: jax.Array) -> jax.Array:
    return di.midpoint(arb_mat_as_matrix(a))


def _mid_vector(x: jax.Array) -> jax.Array:
    return di.midpoint(arb_mat_as_vector(x))


def _mid_rhs(x: jax.Array) -> jax.Array:
    arr = arb_mat_as_rhs(x)
    return di.midpoint(arr)


def _rhs_is_vector(x: jax.Array) -> bool:
    return int(jnp.asarray(x).ndim) == 2


def _rhs_rows_like(rhs: jax.Array, a: jax.Array) -> int:
    arr = jnp.asarray(rhs)
    return int(arr.shape[-2] if arr.ndim == a.ndim - 1 else arr.shape[-3])


def _finite_mask_from_point(x: jax.Array) -> jax.Array:
    if x.ndim >= 2:
        return jnp.all(jnp.isfinite(x), axis=tuple(range(x.ndim - 2, x.ndim)))
    return jnp.all(jnp.isfinite(x), axis=-1)


def _apply_perm_rhs(p: jax.Array, b: jax.Array) -> jax.Array:
    p_mid = di.midpoint(arb_mat_as_matrix(p))
    b_mid = _mid_rhs(b)
    if b.ndim == p.ndim - 1:
        return mat_common.interval_from_point(jnp.einsum("...ij,...j->...i", p_mid, b_mid))
    return mat_common.interval_from_point(jnp.matmul(p_mid, b_mid))


def _abs_interval(x: jax.Array) -> jax.Array:
    lo = x[..., 0]
    hi = x[..., 1]
    mag = jnp.maximum(jnp.abs(lo), jnp.abs(hi))
    return di.interval(jnp.zeros_like(mag), di._above(mag))


def _band_mask(rows: int, cols: int, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    i = jnp.arange(rows)[:, None]
    j = jnp.arange(cols)[None, :]
    return (i - j <= lower_bandwidth) & (j - i <= upper_bandwidth)


def _apply_band_mask_interval(a: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    mask = _band_mask(a.shape[-2], a.shape[-2], lower_bandwidth, upper_bandwidth)
    zero = di.interval(jnp.zeros(a.shape[:-1], dtype=a.dtype), jnp.zeros(a.shape[:-1], dtype=a.dtype))
    return jnp.where(mask[..., None], a, zero)


def arb_mat_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "arb_mat.matmul.inner")
    c = jnp.matmul(_mid_matrix(a), _mid_matrix(b))
    out = mat_common.interval_from_point(c)
    finite = jnp.all(jnp.isfinite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_matmul_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "arb_mat.matmul_basic.inner")
    prods = di.fast_mul(a[..., :, :, None, :], b[..., None, :, :, :])
    out = mat_common.interval_sum(prods, axis=-2)
    finite = jnp.all(jnp.isfinite(out), axis=(-3, -2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_matvec(a: jax.Array, x: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "arb_mat.matvec.inner")
    y = jnp.einsum("...ij,...j->...i", _mid_matrix(a), _mid_vector(x))
    out = mat_common.interval_from_point(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_matvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "arb_mat.matvec_basic.inner")
    prods = di.fast_mul(a, x[..., None, :, :])
    out = mat_common.interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_matvec_cached_prepare(a: jax.Array) -> jax.Array:
    return arb_mat_as_matrix(a)


def arb_mat_matvec_cached_prepare_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_prepare(a), prec_bits)


def arb_mat_dense_matvec_plan_prepare(a: jax.Array) -> mat_common.DenseMatvecPlan:
    matrix = arb_mat_as_matrix(a)
    return mat_common.DenseMatvecPlan(matrix=matrix, rows=int(matrix.shape[-3]), cols=int(matrix.shape[-2]), algebra="arb")


def arb_mat_dense_matvec_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    matrix = di.round_interval_outward(arb_mat_as_matrix(a), prec_bits)
    return mat_common.DenseMatvecPlan(matrix=matrix, rows=int(matrix.shape[-3]), cols=int(matrix.shape[-2]), algebra="arb")


def arb_mat_matvec_cached_apply(cache: jax.Array, x: jax.Array) -> jax.Array:
    cache = mat_common.as_dense_matvec_plan(cache, algebra="arb", label="arb_mat.matvec_cached_apply")
    matrix = arb_mat_as_matrix(cache.matrix)
    x = arb_mat_as_vector(x)
    checks.check_equal(cache.rows, x.shape[-2], "arb_mat.matvec_cached_apply.inner")
    prods = di.fast_mul(matrix, x[..., None, :, :])
    out = mat_common.interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_dense_matvec_plan_apply(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply(plan, x)


def arb_mat_permutation_matrix(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    p = jnp.eye(perm.shape[-1], dtype=dtype)[perm]
    return mat_common.interval_from_point(p)


def arb_mat_transpose(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(arb_mat_as_matrix(a), -3, -2)


def arb_mat_submatrix(a: jax.Array, row_start: int, row_stop: int, col_start: int, col_stop: int) -> jax.Array:
    a = arb_mat_as_matrix(a)
    return a[..., row_start:row_stop, col_start:col_stop, :]


def arb_mat_diag(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    idx = jnp.arange(a.shape[-2])
    return a[..., idx, idx, :]


def arb_mat_diag_matrix(d: jax.Array) -> jax.Array:
    d = arb_mat_as_vector(d)
    n = d.shape[-2]
    zeros = jnp.zeros(d.shape[:-2] + (n, n), dtype=d.dtype)
    out = di.interval(zeros, zeros)
    idx = jnp.arange(n)
    return out.at[..., idx, idx, :].set(d)


def _arb_block_zero(rows: int, cols: int, *, dtype: jnp.dtype) -> jax.Array:
    return mat_common.interval_from_point(jnp.zeros((rows, cols), dtype=dtype))


def _arb_block_sizes_from_matrix_blocks(block_rows) -> tuple[tuple[int, ...], tuple[int, ...]]:
    checks.check(len(block_rows) > 0, "arb_mat.block_rows_nonempty")
    row_count = len(block_rows)
    col_count = len(block_rows[0])
    checks.check(col_count > 0, "arb_mat.block_cols_nonempty")
    row_sizes: list[int] = []
    col_sizes: list[int] = [0] * col_count
    for i, row in enumerate(block_rows):
        checks.check_equal(len(row), col_count, "arb_mat.block_row_width")
        row_height = None
        for j, block in enumerate(row):
            block = arb_mat_as_matrix(block)
            if row_height is None:
                row_height = int(block.shape[-3])
            else:
                checks.check_equal(int(block.shape[-3]), row_height, "arb_mat.block_row_height")
            if i == 0:
                col_sizes[j] = int(block.shape[-2])
            else:
                checks.check_equal(int(block.shape[-2]), col_sizes[j], "arb_mat.block_col_width")
        row_sizes.append(int(row_height))
    return tuple(row_sizes), tuple(col_sizes)


def _arb_offsets(sizes) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for size in sizes:
        total += int(size)
        offsets.append(total)
    return tuple(offsets)


def arb_mat_block_assemble(block_rows) -> jax.Array:
    row_sizes, col_sizes = _arb_block_sizes_from_matrix_blocks(block_rows)
    del row_sizes, col_sizes
    row_chunks = []
    for row in block_rows:
        row_chunks.append(jnp.concatenate([arb_mat_as_matrix(block) for block in row], axis=-2))
    return jnp.concatenate(row_chunks, axis=-3)


def arb_mat_block_diag(blocks) -> jax.Array:
    checks.check(len(blocks) > 0, "arb_mat.block_diag_nonempty")
    matrices = [arb_mat_as_matrix(block) for block in blocks]
    row_sizes = [int(block.shape[-3]) for block in matrices]
    col_sizes = [int(block.shape[-2]) for block in matrices]
    dtype = matrices[0].dtype
    block_rows = []
    for i, row_block in enumerate(matrices):
        row_entries = []
        for j, _ in enumerate(matrices):
            if i == j:
                row_entries.append(row_block)
            else:
                row_entries.append(_arb_block_zero(row_sizes[i], col_sizes[j], dtype=dtype))
        block_rows.append(tuple(row_entries))
    return arb_mat_block_assemble(tuple(block_rows))


def arb_mat_block_extract(a: jax.Array, row_block_sizes, col_block_sizes, row_block: int, col_block: int) -> jax.Array:
    a = arb_mat_as_matrix(a)
    row_offsets = _arb_offsets(row_block_sizes)
    col_offsets = _arb_offsets(col_block_sizes)
    return arb_mat_submatrix(a, row_offsets[row_block], row_offsets[row_block + 1], col_offsets[col_block], col_offsets[col_block + 1])


def arb_mat_block_row(a: jax.Array, row_block_sizes, row_block: int) -> jax.Array:
    a = arb_mat_as_matrix(a)
    row_offsets = _arb_offsets(row_block_sizes)
    return a[..., row_offsets[row_block] : row_offsets[row_block + 1], :, :]


def arb_mat_block_col(a: jax.Array, col_block_sizes, col_block: int) -> jax.Array:
    a = arb_mat_as_matrix(a)
    col_offsets = _arb_offsets(col_block_sizes)
    return a[..., :, col_offsets[col_block] : col_offsets[col_block + 1], :]


def arb_mat_block_matmul(a_blocks, b_blocks) -> jax.Array:
    a_row_sizes, a_col_sizes = _arb_block_sizes_from_matrix_blocks(a_blocks)
    b_row_sizes, b_col_sizes = _arb_block_sizes_from_matrix_blocks(b_blocks)
    checks.check_equal(a_col_sizes, b_row_sizes, "arb_mat.block_matmul.partition_inner")
    out_rows = []
    for i, a_row in enumerate(a_blocks):
        out_row = []
        for j in range(len(b_col_sizes)):
            total = None
            for k in range(len(a_col_sizes)):
                prod = arb_mat_matmul(a_row[k], b_blocks[k][j])
                total = prod if total is None else di.fast_add(total, prod)
            if total is None:
                total = _arb_block_zero(a_row_sizes[i], b_col_sizes[j], dtype=arb_mat_as_matrix(a_row[0]).dtype)
            out_row.append(total)
        out_rows.append(tuple(out_row))
    return arb_mat_block_assemble(tuple(out_rows))


def arb_mat_banded_matvec(a: jax.Array, x: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    a = arb_mat_as_matrix(a)
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "arb_mat.banded_matvec.inner")
    mid = _mid_matrix(a)
    mask = _band_mask(mid.shape[-2], mid.shape[-1], lower_bandwidth, upper_bandwidth)
    y = jnp.einsum("...ij,...j->...i", jnp.where(mask, mid, jnp.zeros_like(mid)), _mid_vector(x))
    out = mat_common.interval_from_point(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_banded_matvec_basic(a: jax.Array, x: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    a = arb_mat_as_matrix(a)
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "arb_mat.banded_matvec_basic.inner")
    masked = _apply_band_mask_interval(a, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)
    prods = di.fast_mul(masked, x[..., None, :, :])
    out = mat_common.interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_solve(a: jax.Array, b: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "arb_mat.solve.inner")
    a_mid = _mid_matrix(a)
    b_mid = _mid_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    x = jnp.linalg.solve(a_mid, b_mid[..., None] if vector_rhs else b_mid)
    if vector_rhs:
        x = x[..., 0]
    out = mat_common.interval_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))


def arb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_solve(a, b)


def arb_mat_inv(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    inv = jnp.linalg.inv(_mid_matrix(a))
    out = mat_common.interval_from_point(inv)
    finite = jnp.all(jnp.isfinite(inv), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_inv_basic(a: jax.Array) -> jax.Array:
    return arb_mat_inv(a)


def arb_mat_sqr(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    c = jnp.matmul(_mid_matrix(a), _mid_matrix(a))
    out = mat_common.interval_from_point(c)
    finite = jnp.all(jnp.isfinite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_sqr_basic(a: jax.Array) -> jax.Array:
    return arb_mat_matmul_basic(a, a)


def arb_mat_det(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    det = jnp.linalg.det(_mid_matrix(a))
    out = mat_common.interval_from_point(det)
    finite = jnp.isfinite(det)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_det_basic(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    n = a.shape[-2]
    if n == 1:
        out = a[..., 0, 0, :]
    elif n == 2:
        out = mat_common.interval_det_2x2(a)
    elif n == 3:
        out = mat_common.interval_det_3x3(a)
    else:
        out = arb_mat_det(a)
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_trace(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    tr = jnp.trace(_mid_matrix(a), axis1=-2, axis2=-1)
    out = mat_common.interval_from_point(tr)
    finite = jnp.isfinite(tr)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_trace_basic(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    out = mat_common.interval_trace(a)
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_norm_fro(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    value = jnp.linalg.norm(_mid_matrix(a), ord="fro", axis=(-2, -1))
    out = mat_common.interval_from_point(value)
    finite = jnp.isfinite(value)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_norm_fro_basic(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    sq = di.fast_mul(a, a)
    total = mat_common.interval_sum(mat_common.interval_sum(sq, axis=-2), axis=-1)
    out = di.sqrt(total)
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_norm_1(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    value = jnp.linalg.norm(_mid_matrix(a), ord=1, axis=(-2, -1))
    out = mat_common.interval_from_point(value)
    finite = jnp.isfinite(value)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_norm_1_basic(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    abs_a = _abs_interval(a)
    col_sums = mat_common.interval_sum(abs_a, axis=-2)
    value = jnp.max(di.midpoint(col_sums), axis=-1)
    out = mat_common.interval_from_point(value)
    finite = jnp.isfinite(value)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_norm_inf(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    value = jnp.linalg.norm(_mid_matrix(a), ord=jnp.inf, axis=(-2, -1))
    out = mat_common.interval_from_point(value)
    finite = jnp.isfinite(value)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_norm_inf_basic(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    abs_a = _abs_interval(a)
    row_sums = mat_common.interval_sum(abs_a, axis=-1)
    value = jnp.max(di.midpoint(row_sums), axis=-1)
    out = mat_common.interval_from_point(value)
    finite = jnp.isfinite(value)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_det_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_det_basic(a)


def arb_mat_trace_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_trace_basic(a)


def arb_mat_norm_fro_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_norm_fro_basic(a)


def arb_mat_norm_1_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_norm_1_basic(a)


def arb_mat_norm_inf_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_norm_inf_basic(a)


def arb_mat_triangular_solve(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "arb_mat.triangular_solve.inner")
    a_mid = _mid_matrix(a)
    b_mid = _mid_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    x = lax.linalg.triangular_solve(
        a_mid,
        b_mid[..., None] if vector_rhs else b_mid,
        left_side=True,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    if vector_rhs:
        x = x[..., 0]
    out = mat_common.interval_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))


def arb_mat_triangular_solve_basic(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal)


def arb_mat_lu(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    mid = _mid_matrix(a)
    lu, _, perm = lax.linalg.lu(mid)
    n = mid.shape[-1]
    eye = jnp.eye(n, dtype=mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return mat_common.interval_from_point(p), mat_common.interval_from_point(l), mat_common.interval_from_point(u)


def arb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return arb_mat_lu(a)


def arb_mat_dense_lu_solve_plan_prepare(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    p, l, u = arb_mat_lu(a)
    return mat_common.DenseLUSolvePlan(p=p, l=l, u=u, rows=int(arb_mat_as_matrix(a).shape[-3]), algebra="arb")


def arb_mat_dense_lu_solve_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    p, l, u = arb_mat_lu_prec(a, prec_bits=prec_bits)
    return mat_common.DenseLUSolvePlan(p=p, l=l, u=u, rows=int(arb_mat_as_matrix(a).shape[-3]), algebra="arb")


def arb_mat_lu_solve(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="arb", label="arb_mat.lu_solve")
    p = arb_mat_as_matrix(plan.p)
    l = arb_mat_as_matrix(plan.l)
    u = arb_mat_as_matrix(plan.u)
    b = arb_mat_as_rhs(b)
    checks.check_equal(plan.rows, _rhs_rows_like(b, p), "arb_mat.lu_solve.inner")
    pb = _apply_perm_rhs(p, b)
    y = arb_mat_triangular_solve(l, pb, lower=True, unit_diagonal=True)
    return arb_mat_triangular_solve(u, y, lower=False, unit_diagonal=False)


def arb_mat_lu_solve_basic(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return arb_mat_lu_solve(plan, b)


def arb_mat_dense_lu_solve_plan_apply(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return arb_mat_lu_solve(plan, b)


def arb_mat_qr(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    q, r = jnp.linalg.qr(_mid_matrix(a))
    return mat_common.interval_from_point(q), mat_common.interval_from_point(r)


def arb_mat_qr_basic(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_mat_qr(a)


def arb_mat_permutation_matrix_rigorous(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return arb_mat_permutation_matrix(perm, dtype=dtype)


def arb_mat_transpose_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_transpose(a)


def arb_mat_submatrix_rigorous(a: jax.Array, row_start: int, row_stop: int, col_start: int, col_stop: int) -> jax.Array:
    return arb_mat_submatrix(a, row_start, row_stop, col_start, col_stop)


def arb_mat_diag_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_diag(a)


def arb_mat_diag_matrix_rigorous(d: jax.Array) -> jax.Array:
    return arb_mat_diag_matrix(d)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matmul_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matmul(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower_bandwidth", "upper_bandwidth"))
def arb_mat_banded_matvec_prec(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        arb_mat_banded_matvec(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matvec_cached_apply_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_apply(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_inv_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_inv(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_sqr_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_sqr(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_det_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_det(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_trace_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_trace(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_norm_fro_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_fro(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_norm_1_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_1(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_norm_inf_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_inf(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower", "unit_diagonal"))
def arb_mat_triangular_solve_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_lu_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = arb_mat_lu(a)
    return (
        di.round_interval_outward(p, prec_bits),
        di.round_interval_outward(l, prec_bits),
        di.round_interval_outward(u, prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_qr_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    q, r = arb_mat_qr(a)
    return di.round_interval_outward(q, prec_bits), di.round_interval_outward(r, prec_bits)


arb_mat_matmul_jit = jax.jit(arb_mat_matmul)
arb_mat_matvec_jit = jax.jit(arb_mat_matvec)
arb_mat_banded_matvec_jit = jax.jit(arb_mat_banded_matvec, static_argnames=("lower_bandwidth", "upper_bandwidth"))
arb_mat_matvec_cached_apply_jit = jax.jit(arb_mat_matvec_cached_apply)
arb_mat_solve_jit = jax.jit(arb_mat_solve)
arb_mat_inv_jit = jax.jit(arb_mat_inv)
arb_mat_sqr_jit = jax.jit(arb_mat_sqr)
arb_mat_det_jit = jax.jit(arb_mat_det)
arb_mat_trace_jit = jax.jit(arb_mat_trace)
arb_mat_norm_fro_jit = jax.jit(arb_mat_norm_fro)
arb_mat_norm_1_jit = jax.jit(arb_mat_norm_1)
arb_mat_norm_inf_jit = jax.jit(arb_mat_norm_inf)
arb_mat_triangular_solve_jit = jax.jit(arb_mat_triangular_solve, static_argnames=("lower", "unit_diagonal"))
arb_mat_lu_jit = jax.jit(arb_mat_lu)
arb_mat_qr_jit = jax.jit(arb_mat_qr)


def arb_mat_matmul_batch_fixed(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_matmul(a, b)


def arb_mat_matmul_batch_padded(a: jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b), pad_to=pad_to)
    return arb_mat_matmul(*call_args)


def arb_mat_matvec_batch_fixed(a: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec(a, x)


def arb_mat_matvec_batch_padded(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return arb_mat_matvec(*call_args)


def arb_mat_banded_matvec_batch_fixed(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    return arb_mat_banded_matvec(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


def arb_mat_banded_matvec_batch_padded(
    a: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return arb_mat_banded_matvec(*call_args, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


def arb_mat_matvec_cached_apply_batch_fixed(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply(cache, x)


def arb_mat_matvec_cached_apply_batch_padded(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((cache, x), pad_to=pad_to)
    return arb_mat_matvec_cached_apply(*call_args)


def arb_mat_matvec_cached_prepare_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_prepare(a)


def arb_mat_matvec_cached_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_matvec_cached_prepare(*call_args)


def arb_mat_solve_batch_fixed(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_solve(a, b)


def arb_mat_solve_batch_padded(a: jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b), pad_to=pad_to)
    return arb_mat_solve(*call_args)


def arb_mat_inv_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_inv(a)


def arb_mat_inv_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_inv(*call_args)


def arb_mat_triangular_solve_batch_fixed(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return arb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal)


def arb_mat_triangular_solve_batch_padded(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b), pad_to=pad_to)
    return arb_mat_triangular_solve(*call_args, lower=lower, unit_diagonal=unit_diagonal)


def arb_mat_lu_batch_fixed(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return arb_mat_lu(a)


def arb_mat_lu_batch_padded(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_lu(*call_args)


def arb_mat_qr_batch_fixed(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_mat_qr(a)


def arb_mat_qr_batch_padded(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array]:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_qr(*call_args)


def arb_mat_det_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_det(a)


def arb_mat_det_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_det(*call_args)


def arb_mat_trace_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_trace(a)


def arb_mat_trace_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_trace(*call_args)


def arb_mat_sqr_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_sqr(a)


def arb_mat_sqr_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_sqr(*call_args)


def arb_mat_norm_fro_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_norm_fro(a)


def arb_mat_norm_fro_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_norm_fro(*call_args)


def arb_mat_norm_1_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_norm_1(a)


def arb_mat_norm_1_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_norm_1(*call_args)


def arb_mat_norm_inf_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_norm_inf(a)


def arb_mat_norm_inf_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_norm_inf(*call_args)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matmul_batch_fixed_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matmul_batch_fixed(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_matmul_batch_padded_prec(a: jax.Array, b: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matmul_batch_padded(a, b, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matvec_batch_fixed_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_batch_fixed(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_matvec_batch_padded_prec(a: jax.Array, x: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_batch_padded(a, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower_bandwidth", "upper_bandwidth"))
def arb_mat_banded_matvec_batch_fixed_prec(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        arb_mat_banded_matvec_batch_fixed(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits", "pad_to", "lower_bandwidth", "upper_bandwidth"))
def arb_mat_banded_matvec_batch_padded_prec(
    a: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        arb_mat_banded_matvec_batch_padded(
            a,
            x,
            pad_to=pad_to,
            lower_bandwidth=lower_bandwidth,
            upper_bandwidth=upper_bandwidth,
        ),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matvec_cached_apply_batch_fixed_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_apply_batch_fixed(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_matvec_cached_apply_batch_padded_prec(
    cache: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_apply_batch_padded(cache, x, pad_to=pad_to), prec_bits)


def arb_mat_matvec_cached_prepare_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_prepare_batch_fixed(a), prec_bits)


def arb_mat_matvec_cached_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_prepare_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_batch_fixed_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_batch_fixed(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_solve_batch_padded_prec(a: jax.Array, b: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_batch_padded(a, b, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_inv_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_inv_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_inv_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_inv_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower", "unit_diagonal"))
def arb_mat_triangular_solve_batch_fixed_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        arb_mat_triangular_solve_batch_fixed(a, b, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits", "pad_to", "lower", "unit_diagonal"))
def arb_mat_triangular_solve_batch_padded_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        arb_mat_triangular_solve_batch_padded(a, b, pad_to=pad_to, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_lu_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = arb_mat_lu_batch_fixed(a)
    return (
        di.round_interval_outward(p, prec_bits),
        di.round_interval_outward(l, prec_bits),
        di.round_interval_outward(u, prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_lu_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = arb_mat_lu_batch_padded(a, pad_to=pad_to)
    return (
        di.round_interval_outward(p, prec_bits),
        di.round_interval_outward(l, prec_bits),
        di.round_interval_outward(u, prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_qr_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    q, r = arb_mat_qr_batch_fixed(a)
    return di.round_interval_outward(q, prec_bits), di.round_interval_outward(r, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_qr_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    q, r = arb_mat_qr_batch_padded(a, pad_to=pad_to)
    return di.round_interval_outward(q, prec_bits), di.round_interval_outward(r, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_det_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_det_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_det_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_det_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_trace_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_trace_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_trace_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_trace_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_sqr_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_sqr_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_sqr_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_sqr_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_norm_fro_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_fro_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_norm_fro_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_fro_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_norm_1_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_1_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_norm_1_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_1_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_norm_inf_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_inf_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_norm_inf_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_norm_inf_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_permutation_matrix_prec(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_permutation_matrix(perm, dtype=dtype), prec_bits)


def arb_mat_transpose_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_transpose(a), prec_bits)


def arb_mat_submatrix_prec(
    a: jax.Array,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_submatrix(a, row_start, row_stop, col_start, col_stop), prec_bits)


def arb_mat_diag_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_diag(a), prec_bits)


def arb_mat_diag_matrix_prec(d: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_diag_matrix(d), prec_bits)


def arb_mat_lu_solve_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_lu_solve(plan, b), prec_bits)


def arb_mat_dense_lu_solve_plan_apply_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_lu_solve_plan_apply(plan, b), prec_bits)


def arb_mat_permutation_matrix_batch_fixed(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return arb_mat_permutation_matrix(perm, dtype=dtype)


def arb_mat_permutation_matrix_batch_padded(perm: jax.Array, *, pad_to: int, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((perm,), pad_to=pad_to)
    return arb_mat_permutation_matrix(*call_args, dtype=dtype)


def arb_mat_transpose_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_transpose(a)


def arb_mat_transpose_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_transpose(*call_args)


def arb_mat_diag_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_diag(a)


def arb_mat_diag_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_diag(*call_args)


def arb_mat_diag_matrix_batch_fixed(d: jax.Array) -> jax.Array:
    return arb_mat_diag_matrix(d)


def arb_mat_diag_matrix_batch_padded(d: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((d,), pad_to=pad_to)
    return arb_mat_diag_matrix(*call_args)


def arb_mat_lu_solve_batch_fixed(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return arb_mat_lu_solve(plan, b)


def arb_mat_lu_solve_batch_padded(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((plan, b), pad_to=pad_to)
    return arb_mat_lu_solve(*call_args)


def arb_mat_permutation_matrix_batch_fixed_prec(
    perm: jax.Array,
    *,
    dtype: jnp.dtype = jnp.float64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_permutation_matrix_batch_fixed(perm, dtype=dtype), prec_bits)


def arb_mat_permutation_matrix_batch_padded_prec(
    perm: jax.Array,
    *,
    pad_to: int,
    dtype: jnp.dtype = jnp.float64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_permutation_matrix_batch_padded(perm, pad_to=pad_to, dtype=dtype), prec_bits)


def arb_mat_transpose_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_transpose_batch_fixed(a), prec_bits)


def arb_mat_transpose_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_transpose_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_diag_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_diag_batch_fixed(a), prec_bits)


def arb_mat_diag_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_diag_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_diag_matrix_batch_fixed_prec(d: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_diag_matrix_batch_fixed(d), prec_bits)


def arb_mat_diag_matrix_batch_padded_prec(d: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_diag_matrix_batch_padded(d, pad_to=pad_to), prec_bits)


def arb_mat_lu_solve_batch_fixed_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_lu_solve_batch_fixed(plan, b), prec_bits)


def arb_mat_lu_solve_batch_padded_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_lu_solve_batch_padded(plan, b, pad_to=pad_to), prec_bits)


def arb_mat_2x2_det(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = di.midpoint(a[..., 0, 0, :])
    a01 = di.midpoint(a[..., 0, 1, :])
    a10 = di.midpoint(a[..., 1, 0, :])
    a11 = di.midpoint(a[..., 1, 1, :])
    det = a00 * a11 - a01 * a10
    finite = jnp.isfinite(det)
    out = mat_common.interval_from_point(det)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(a[..., 0, 0, :]))


def arb_mat_2x2_trace(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = di.midpoint(a[..., 0, 0, :])
    a11 = di.midpoint(a[..., 1, 1, :])
    tr = a00 + a11
    finite = jnp.isfinite(tr)
    out = mat_common.interval_from_point(tr)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(a[..., 0, 0, :]))


def arb_mat_2x2_det_rigorous(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    det = di.fast_sub(di.fast_mul(a00, a11), di.fast_mul(a01, a10))
    finite = mat_common.interval_is_finite(det)
    return jnp.where(finite[..., None], det, mat_common.full_interval_like(a[..., 0, 0, :]))


def arb_mat_2x2_trace_rigorous(a: jax.Array) -> jax.Array:
    a = _as_mat_2x2(a)
    tr = di.fast_add(a[..., 0, 0, :], a[..., 1, 1, :])
    finite = mat_common.interval_is_finite(tr)
    return jnp.where(finite[..., None], tr, mat_common.full_interval_like(a[..., 0, 0, :]))


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
    "arb_mat_as_matrix",
    "arb_mat_as_vector",
    "arb_mat_shape",
    "arb_mat_zero",
    "arb_mat_identity",
    "arb_mat_block_assemble",
    "arb_mat_block_diag",
    "arb_mat_block_extract",
    "arb_mat_block_row",
    "arb_mat_block_col",
    "arb_mat_block_matmul",
    "arb_mat_matmul",
    "arb_mat_matmul_basic",
    "arb_mat_matvec",
    "arb_mat_matvec_basic",
    "arb_mat_banded_matvec",
    "arb_mat_banded_matvec_basic",
    "arb_mat_matvec_cached_prepare",
    "arb_mat_dense_matvec_plan_prepare",
    "arb_mat_dense_matvec_plan_apply",
    "arb_mat_matvec_cached_apply",
    "arb_mat_solve",
    "arb_mat_solve_basic",
    "arb_mat_inv",
    "arb_mat_inv_basic",
    "arb_mat_sqr",
    "arb_mat_sqr_basic",
    "arb_mat_det",
    "arb_mat_det_basic",
    "arb_mat_trace",
    "arb_mat_trace_basic",
    "arb_mat_norm_fro",
    "arb_mat_norm_fro_basic",
    "arb_mat_norm_1",
    "arb_mat_norm_1_basic",
    "arb_mat_norm_inf",
    "arb_mat_norm_inf_basic",
    "arb_mat_det_rigorous",
    "arb_mat_trace_rigorous",
    "arb_mat_norm_fro_rigorous",
    "arb_mat_norm_1_rigorous",
    "arb_mat_norm_inf_rigorous",
    "arb_mat_triangular_solve",
    "arb_mat_triangular_solve_basic",
    "arb_mat_lu",
    "arb_mat_lu_basic",
    "arb_mat_qr",
    "arb_mat_qr_basic",
    "arb_mat_matmul_prec",
    "arb_mat_matvec_prec",
    "arb_mat_banded_matvec_prec",
    "arb_mat_matvec_cached_prepare_prec",
    "arb_mat_dense_matvec_plan_prepare_prec",
    "arb_mat_matvec_cached_apply_prec",
    "arb_mat_solve_prec",
    "arb_mat_inv_prec",
    "arb_mat_sqr_prec",
    "arb_mat_det_prec",
    "arb_mat_trace_prec",
    "arb_mat_norm_fro_prec",
    "arb_mat_norm_1_prec",
    "arb_mat_norm_inf_prec",
    "arb_mat_triangular_solve_prec",
    "arb_mat_lu_prec",
    "arb_mat_qr_prec",
    "arb_mat_matmul_jit",
    "arb_mat_matvec_jit",
    "arb_mat_banded_matvec_jit",
    "arb_mat_matvec_cached_apply_jit",
    "arb_mat_solve_jit",
    "arb_mat_inv_jit",
    "arb_mat_sqr_jit",
    "arb_mat_det_jit",
    "arb_mat_trace_jit",
    "arb_mat_norm_fro_jit",
    "arb_mat_norm_1_jit",
    "arb_mat_norm_inf_jit",
    "arb_mat_triangular_solve_jit",
    "arb_mat_lu_jit",
    "arb_mat_qr_jit",
    "arb_mat_matmul_batch_fixed",
    "arb_mat_matmul_batch_padded",
    "arb_mat_matvec_batch_fixed",
    "arb_mat_matvec_batch_padded",
    "arb_mat_banded_matvec_batch_fixed",
    "arb_mat_banded_matvec_batch_padded",
    "arb_mat_matvec_cached_prepare_batch_fixed",
    "arb_mat_matvec_cached_prepare_batch_padded",
    "arb_mat_matvec_cached_apply_batch_fixed",
    "arb_mat_matvec_cached_apply_batch_padded",
    "arb_mat_solve_batch_fixed",
    "arb_mat_solve_batch_padded",
    "arb_mat_inv_batch_fixed",
    "arb_mat_inv_batch_padded",
    "arb_mat_triangular_solve_batch_fixed",
    "arb_mat_triangular_solve_batch_padded",
    "arb_mat_lu_batch_fixed",
    "arb_mat_lu_batch_padded",
    "arb_mat_qr_batch_fixed",
    "arb_mat_qr_batch_padded",
    "arb_mat_det_batch_fixed",
    "arb_mat_det_batch_padded",
    "arb_mat_trace_batch_fixed",
    "arb_mat_trace_batch_padded",
    "arb_mat_sqr_batch_fixed",
    "arb_mat_sqr_batch_padded",
    "arb_mat_norm_fro_batch_fixed",
    "arb_mat_norm_fro_batch_padded",
    "arb_mat_norm_1_batch_fixed",
    "arb_mat_norm_1_batch_padded",
    "arb_mat_norm_inf_batch_fixed",
    "arb_mat_norm_inf_batch_padded",
    "arb_mat_matmul_batch_fixed_prec",
    "arb_mat_matmul_batch_padded_prec",
    "arb_mat_matvec_batch_fixed_prec",
    "arb_mat_matvec_batch_padded_prec",
    "arb_mat_banded_matvec_batch_fixed_prec",
    "arb_mat_banded_matvec_batch_padded_prec",
    "arb_mat_matvec_cached_prepare_batch_fixed_prec",
    "arb_mat_matvec_cached_prepare_batch_padded_prec",
    "arb_mat_matvec_cached_apply_batch_fixed_prec",
    "arb_mat_matvec_cached_apply_batch_padded_prec",
    "arb_mat_solve_batch_fixed_prec",
    "arb_mat_solve_batch_padded_prec",
    "arb_mat_inv_batch_fixed_prec",
    "arb_mat_inv_batch_padded_prec",
    "arb_mat_triangular_solve_batch_fixed_prec",
    "arb_mat_triangular_solve_batch_padded_prec",
    "arb_mat_lu_batch_fixed_prec",
    "arb_mat_lu_batch_padded_prec",
    "arb_mat_qr_batch_fixed_prec",
    "arb_mat_qr_batch_padded_prec",
    "arb_mat_det_batch_fixed_prec",
    "arb_mat_det_batch_padded_prec",
    "arb_mat_trace_batch_fixed_prec",
    "arb_mat_trace_batch_padded_prec",
    "arb_mat_sqr_batch_fixed_prec",
    "arb_mat_sqr_batch_padded_prec",
    "arb_mat_norm_fro_batch_fixed_prec",
    "arb_mat_norm_fro_batch_padded_prec",
    "arb_mat_norm_1_batch_fixed_prec",
    "arb_mat_norm_1_batch_padded_prec",
    "arb_mat_norm_inf_batch_fixed_prec",
    "arb_mat_norm_inf_batch_padded_prec",
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
