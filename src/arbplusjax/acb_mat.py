from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di
from . import kernel_helpers as kh
from . import mat_common



def acb_mat_as_matrix(x: jax.Array) -> jax.Array:
    return mat_common.as_box_matrix(x, "acb_mat.as_matrix")


def acb_mat_as_vector(x: jax.Array) -> jax.Array:
    return mat_common.as_box_vector(x, "acb_mat.as_vector")


def acb_mat_as_rhs(x: jax.Array) -> jax.Array:
    return mat_common.as_box_rhs(x, "acb_mat.as_rhs")


def acb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = acb_mat_as_matrix(a)
    return tuple(int(x) for x in arr.shape)


def acb_mat_zero(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    zeros = jnp.zeros((n, n), dtype=jnp.result_type(dtype, jnp.complex64))
    return mat_common.box_from_point(zeros)


def acb_mat_identity(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    eye = jnp.eye(n, dtype=jnp.result_type(dtype, jnp.complex64))
    return mat_common.box_from_point(eye)


def _band_mask(rows: int, cols: int, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    i = jnp.arange(rows)[:, None]
    j = jnp.arange(cols)[None, :]
    return (i - j <= lower_bandwidth) & (j - i <= upper_bandwidth)


def _apply_band_mask_box(a: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    mask = _band_mask(a.shape[-2], a.shape[-2], lower_bandwidth, upper_bandwidth)
    zero = mat_common.box_from_point(jnp.zeros(a.shape[:-1], dtype=jnp.complex128))
    return jnp.where(mask[..., None], a, zero)


def _as_mat_2x2(x: jax.Array) -> jax.Array:
    return mat_common.as_box_mat_2x2(x, "acb_mat._as_mat_2x2")


def _mid_matrix(a: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(acb_mat_as_matrix(a))


def _mid_vector(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(acb_mat_as_vector(x))


def _mid_rhs(x: jax.Array) -> jax.Array:
    arr = acb_mat_as_rhs(x)
    return acb_core.acb_midpoint(arr)


def _mid_hermitian_part(a: jax.Array) -> jax.Array:
    return mat_common.complex_midpoint_hermitian_part(_mid_matrix(a))


def _mid_is_hermitian(a: jax.Array) -> jax.Array:
    return mat_common.complex_midpoint_is_hermitian(_mid_matrix(a))


def _mid_cholesky(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    herm_mid = _mid_hermitian_part(a)
    chol = jnp.linalg.cholesky(herm_mid)
    ok = _mid_is_hermitian(a) & mat_common.lower_cholesky_finite(chol)
    return chol, ok


def _mid_hpd_solve(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol, ok = _mid_cholesky(a)
    x = mat_common.lower_cholesky_solve(chol, _mid_rhs(b))
    return x, ok


def _mid_hpd_inv(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol, ok = _mid_cholesky(a)
    eye = jnp.broadcast_to(jnp.eye(chol.shape[-1], dtype=chol.dtype), chol.shape)
    inv = mat_common.lower_cholesky_solve(chol, eye)
    return inv, ok


def _rhs_rows_like(rhs: jax.Array, a: jax.Array) -> int:
    arr = jnp.asarray(rhs)
    return int(arr.shape[-2] if arr.ndim == a.ndim - 1 else arr.shape[-3])


def _rhs_batch_shape(rhs: jax.Array, rows: int) -> tuple[int, ...]:
    arr = jnp.asarray(rhs)
    if arr.ndim >= 3 and int(arr.shape[-3]) == rows:
        return tuple(int(x) for x in arr.shape[:-3])
    return tuple(int(x) for x in arr.shape[:-2])


def _broadcast_box_matrix_batch(a: jax.Array, batch_shape: tuple[int, ...]) -> jax.Array:
    if not batch_shape or tuple(int(x) for x in a.shape[:-3]) == batch_shape:
        return a
    return jnp.broadcast_to(a, batch_shape + a.shape[-3:])


def _finite_mask_from_point(z: jax.Array) -> jax.Array:
    if z.ndim >= 2:
        return jnp.all(mat_common.complex_is_finite(z), axis=tuple(range(z.ndim - 2, z.ndim)))
    return jnp.all(mat_common.complex_is_finite(z), axis=-1)


def _apply_perm_rhs(p: jax.Array, b: jax.Array) -> jax.Array:
    p_mid = acb_core.acb_midpoint(acb_mat_as_matrix(p))
    b_mid = _mid_rhs(b)
    if b.ndim == p.ndim - 1:
        return mat_common.box_from_point(jnp.einsum("...ij,...j->...i", p_mid, b_mid))
    return mat_common.box_from_point(jnp.matmul(p_mid, b_mid))


def _apply_perm_transpose_rhs(p: jax.Array, b: jax.Array) -> jax.Array:
    p_mid = jnp.swapaxes(acb_core.acb_midpoint(acb_mat_as_matrix(p)), -2, -1)
    b_mid = _mid_rhs(b)
    if b.ndim == p.ndim - 1:
        return mat_common.box_from_point(jnp.einsum("...ij,...j->...i", p_mid, b_mid))
    return mat_common.box_from_point(jnp.matmul(p_mid, b_mid))


def _box_rad_parts(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    box = acb_mat_as_matrix(a)
    re_rad = di.ubound_radius(acb_core.acb_real(box))
    im_rad = di.ubound_radius(acb_core.acb_imag(box))
    return re_rad, im_rad


def _herm_rad_bound_matrix(a: jax.Array) -> jax.Array:
    re_rad, im_rad = _box_rad_parts(a)
    mag = di._above(jnp.sqrt(re_rad * re_rad + im_rad * im_rad))
    return 0.5 * (mag + jnp.swapaxes(mag, -2, -1))


def _herm_perturbation_bound(a: jax.Array) -> jax.Array:
    rad = _herm_rad_bound_matrix(a)
    return di._above(jnp.linalg.norm(rad, ord="fro", axis=(-2, -1)))


def _tightened_complex_eigvals_box(values: jax.Array, bound: jax.Array) -> jax.Array:
    real_iv = di.interval(di._below(jnp.real(values) - bound[..., None]), di._above(jnp.real(values) + bound[..., None]))
    imag_zero = di.interval(jnp.zeros_like(real_iv[..., 0]), jnp.zeros_like(real_iv[..., 1]))
    return acb_core.acb_box(real_iv, imag_zero)


def _tightened_complex_eigvecs_box(values: jax.Array, vectors: jax.Array, bound: jax.Array) -> jax.Array:
    n = vectors.shape[-1]
    if n == 1:
        radius = jnp.broadcast_to(bound[..., None], values.shape)
    else:
        diffs = jnp.abs(values[..., :, None] - values[..., None, :])
        huge = jnp.asarray(jnp.inf, dtype=jnp.real(values).dtype)
        diffs = diffs + jnp.eye(n, dtype=jnp.real(values).dtype) * huge
        gaps = jnp.min(diffs, axis=-1)
        tiny = jnp.asarray(64.0 * jnp.finfo(jnp.real(values).dtype).eps, dtype=jnp.real(values).dtype)
        radius = jnp.clip(bound[..., None] / jnp.maximum(gaps, tiny), 0.0, 1.0)
    vr = jnp.real(vectors)
    vi = jnp.imag(vectors)
    re_iv = di.interval(di._below(vr - radius[..., None, :]), di._above(vr + radius[..., None, :]))
    im_iv = di.interval(di._below(vi - radius[..., None, :]), di._above(vi + radius[..., None, :]))
    return acb_core.acb_box(re_iv, im_iv)


def acb_mat_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    a = mat_common.as_box_rect_matrix(a, "acb_mat.matmul.a")
    b = mat_common.as_box_rect_matrix(b, "acb_mat.matmul.b")
    checks.check_equal(a.shape[-2], b.shape[-3], "acb_mat.matmul.inner")
    c = jnp.matmul(acb_core.acb_midpoint(a), acb_core.acb_midpoint(b))
    out = mat_common.box_from_point(c)
    finite = jnp.all(mat_common.complex_is_finite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_matmul_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    a = mat_common.as_box_rect_matrix(a, "acb_mat.matmul_basic.a")
    b = mat_common.as_box_rect_matrix(b, "acb_mat.matmul_basic.b")
    checks.check_equal(a.shape[-2], b.shape[-3], "acb_mat.matmul_basic.inner")
    prods = acb_core.acb_mul(a[..., :, :, None, :], b[..., None, :, :, :])
    out = mat_common.box_sum(prods, axis=-2)
    finite = jnp.all(jnp.isfinite(out), axis=(-3, -2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_matvec(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_box_rect_matrix(a, "acb_mat.matvec.a")
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "acb_mat.matvec.inner")
    y = jnp.einsum("...ij,...j->...i", acb_core.acb_midpoint(a), _mid_vector(x))
    out = mat_common.box_from_point(y)
    finite = jnp.all(mat_common.complex_is_finite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_matvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_box_rect_matrix(a, "acb_mat.matvec_basic.a")
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "acb_mat.matvec_basic.inner")
    prods = acb_core.acb_mul(a, x[..., None, :, :])
    out = mat_common.box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_rmatvec(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_box_rect_matrix(a, "acb_mat.rmatvec.a")
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-3], x.shape[-2], "acb_mat.rmatvec.inner")
    y = jnp.einsum("...ji,...j->...i", acb_core.acb_midpoint(a), _mid_vector(x))
    out = mat_common.box_from_point(y)
    finite = jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_rmatvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_box_rect_matrix(a, "acb_mat.rmatvec_basic.a")
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-3], x.shape[-2], "acb_mat.rmatvec_basic.inner")
    prods = acb_core.acb_mul(jnp.swapaxes(a, -3, -2), x[..., None, :, :])
    out = mat_common.box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_matvec_cached_prepare(a: jax.Array) -> jax.Array:
    return mat_common.as_box_rect_matrix(a, "acb_mat.matvec_cached_prepare")


def acb_mat_rmatvec_cached_prepare(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(mat_common.as_box_rect_matrix(a, "acb_mat.rmatvec_cached_prepare"), -3, -2)


def acb_mat_matvec_cached_prepare_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_prepare(a), prec_bits)


def acb_mat_rmatvec_cached_prepare_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_cached_prepare(a), prec_bits)


def acb_mat_dense_matvec_plan_prepare(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return mat_common.dense_matvec_plan_from_matrix(a, algebra="acb", label="acb_mat.dense_matvec_plan_prepare")


def acb_mat_dense_matvec_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    matrix = acb_core.acb_box_round_prec(mat_common.as_box_rect_matrix(a, "acb_mat.dense_matvec_plan_prepare_prec"), prec_bits)
    return mat_common.dense_matvec_plan_from_matrix(matrix, algebra="acb", label="acb_mat.dense_matvec_plan_prepare_prec")


def acb_mat_matvec_cached_apply(cache: jax.Array, x: jax.Array) -> jax.Array:
    cache = mat_common.as_dense_matvec_plan(cache, algebra="acb", label="acb_mat.matvec_cached_apply")
    x = acb_mat_as_vector(x)
    matrix = _broadcast_box_matrix_batch(mat_common.as_box_rect_matrix(cache.matrix, "acb_mat.matvec_cached_apply.matrix"), tuple(int(v) for v in x.shape[:-2]))
    checks.check_equal(cache.cols, x.shape[-2], "acb_mat.matvec_cached_apply.inner")
    prods = acb_core.acb_mul(matrix, x[..., None, :, :])
    out = mat_common.box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_dense_matvec_plan_apply(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_apply(plan, x)


def acb_mat_rmatvec_cached_apply(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_apply(cache, x)


def acb_mat_permutation_matrix(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    p = jnp.eye(perm.shape[-1], dtype=jnp.result_type(dtype, jnp.complex64))[perm]
    return mat_common.box_from_point(p)


def acb_mat_transpose(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(mat_common.as_box_rect_matrix(a, "acb_mat.transpose"), -3, -2)


def acb_mat_conjugate_transpose(a: jax.Array) -> jax.Array:
    return acb_core.acb_conj(acb_mat_transpose(a))


def acb_mat_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_core.acb_add(acb_mat_as_matrix(a), acb_mat_as_matrix(b))


def acb_mat_sub(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_core.acb_sub(acb_mat_as_matrix(a), acb_mat_as_matrix(b))


def acb_mat_neg(a: jax.Array) -> jax.Array:
    return acb_core.acb_neg(acb_mat_as_matrix(a))


def acb_mat_mul_entrywise(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_core.acb_mul(acb_mat_as_matrix(a), acb_mat_as_matrix(b))


def acb_mat_conjugate(a: jax.Array) -> jax.Array:
    return acb_core.acb_conj(acb_mat_as_matrix(a))


def acb_mat_hermitian_part(a: jax.Array) -> jax.Array:
    return mat_common.box_from_point(_mid_hermitian_part(a))


def acb_mat_one(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return acb_mat_identity(n, dtype=dtype)


def acb_mat_ones(rows: int, cols: int | None = None, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    cols = rows if cols is None else cols
    return mat_common.box_from_point(jnp.ones((rows, cols), dtype=jnp.result_type(dtype, jnp.complex64)))


def acb_mat_companion(coeffs: jax.Array) -> jax.Array:
    coeffs = acb_mat_as_vector(coeffs)
    return mat_common.box_from_point(mat_common.companion_matrix(_mid_vector(coeffs)))


def acb_mat_hilbert(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return mat_common.box_from_point(mat_common.hilbert_matrix(n, dtype=jnp.result_type(dtype, jnp.complex64)))


def acb_mat_pascal(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return mat_common.box_from_point(mat_common.pascal_matrix(n, dtype=jnp.result_type(dtype, jnp.complex64)))


def acb_mat_stirling(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return mat_common.box_from_point(mat_common.stirling2_matrix(n, dtype=jnp.result_type(dtype, jnp.complex64)))


def acb_mat_nrows(a: jax.Array) -> int:
    return int(acb_mat_as_matrix(a).shape[-3])


def acb_mat_ncols(a: jax.Array) -> int:
    return int(acb_mat_as_matrix(a).shape[-2])


def acb_mat_entry(a: jax.Array, row: int, col: int) -> jax.Array:
    return acb_mat_as_matrix(a)[..., row, col, :]


def acb_mat_eq(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = acb_mat_as_matrix(a)
    bb = acb_mat_as_matrix(b)
    return jnp.all(mat_common.box_equal(aa, bb), axis=(-2, -1))


def acb_mat_equal(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_eq(a, b)


def acb_mat_ne(a: jax.Array, b: jax.Array) -> jax.Array:
    return ~acb_mat_eq(a, b)


def acb_mat_contains(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = acb_mat_as_matrix(a)
    bb = acb_mat_as_matrix(b)
    re_ok = di.contains(acb_core.acb_real(aa), acb_core.acb_real(bb))
    im_ok = di.contains(acb_core.acb_imag(aa), acb_core.acb_imag(bb))
    return jnp.all(re_ok & im_ok, axis=(-2, -1))


def acb_mat_overlaps(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = acb_mat_as_matrix(a)
    bb = acb_mat_as_matrix(b)
    return jnp.all(mat_common.box_overlaps(aa, bb), axis=(-2, -1))


def acb_mat_is_hermitian(a: jax.Array) -> jax.Array:
    return _mid_is_hermitian(a)


def acb_mat_is_hpd(a: jax.Array) -> jax.Array:
    _, ok = _mid_cholesky(a)
    lam_min = jnp.min(jnp.linalg.eigvalsh(_mid_hermitian_part(a)), axis=-1)
    return ok & (lam_min > _herm_perturbation_bound(a))


def acb_mat_is_square(a: jax.Array) -> jax.Array:
    aa = acb_mat_as_matrix(a)
    return aa.shape[-3] == aa.shape[-2]


def acb_mat_is_diag(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_diagonal(_mid_matrix(a))


def acb_mat_is_tril(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_mid_matrix(a), lower=True)


def acb_mat_is_triu(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_mid_matrix(a), lower=False)


def acb_mat_is_zero(a: jax.Array) -> jax.Array:
    return jnp.all(acb_core.acb_is_zero(acb_mat_as_matrix(a)), axis=(-2, -1))


def acb_mat_is_finite(a: jax.Array) -> jax.Array:
    return jnp.all(mat_common.box_is_finite(acb_mat_as_matrix(a)), axis=(-2, -1))


def acb_mat_is_exact(a: jax.Array) -> jax.Array:
    return jnp.all(mat_common.box_is_exact(acb_mat_as_matrix(a)), axis=(-2, -1))


def acb_mat_is_real(a: jax.Array) -> jax.Array:
    return jnp.all(mat_common.interval_is_zero(acb_core.acb_imag(acb_mat_as_matrix(a))), axis=(-2, -1))


def acb_mat_submatrix(a: jax.Array, row_start: int, row_stop: int, col_start: int, col_stop: int) -> jax.Array:
    a = acb_mat_as_matrix(a)
    return a[..., row_start:row_stop, col_start:col_stop, :]


def acb_mat_diag(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    idx = jnp.arange(a.shape[-2])
    return a[..., idx, idx, :]


def acb_mat_diag_matrix(d: jax.Array) -> jax.Array:
    d = acb_mat_as_vector(d)
    n = d.shape[-2]
    re = di.interval(jnp.zeros(d.shape[:-2] + (n, n), dtype=d.dtype), jnp.zeros(d.shape[:-2] + (n, n), dtype=d.dtype))
    im = di.interval(jnp.zeros(d.shape[:-2] + (n, n), dtype=d.dtype), jnp.zeros(d.shape[:-2] + (n, n), dtype=d.dtype))
    out = acb_core.acb_box(re, im)
    idx = jnp.arange(n)
    return out.at[..., idx, idx, :].set(d)


def _acb_block_zero(rows: int, cols: int, *, dtype: jnp.dtype) -> jax.Array:
    return mat_common.box_from_point(jnp.zeros((rows, cols), dtype=jnp.result_type(dtype, jnp.complex64)))


def _acb_block_sizes_from_matrix_blocks(block_rows) -> tuple[tuple[int, ...], tuple[int, ...]]:
    checks.check(len(block_rows) > 0, "acb_mat.block_rows_nonempty")
    row_count = len(block_rows)
    col_count = len(block_rows[0])
    checks.check(col_count > 0, "acb_mat.block_cols_nonempty")
    row_sizes: list[int] = []
    col_sizes: list[int] = [0] * col_count
    for i, row in enumerate(block_rows):
        checks.check_equal(len(row), col_count, "acb_mat.block_row_width")
        row_height = None
        for j, block in enumerate(row):
            block = mat_common.as_box_rect_matrix(block, "acb_mat.block_block")
            if row_height is None:
                row_height = int(block.shape[-3])
            else:
                checks.check_equal(int(block.shape[-3]), row_height, "acb_mat.block_row_height")
            if i == 0:
                col_sizes[j] = int(block.shape[-2])
            else:
                checks.check_equal(int(block.shape[-2]), col_sizes[j], "acb_mat.block_col_width")
        row_sizes.append(int(row_height))
    return tuple(row_sizes), tuple(col_sizes)


def _acb_offsets(sizes) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for size in sizes:
        total += int(size)
        offsets.append(total)
    return tuple(offsets)


def acb_mat_block_assemble(block_rows) -> jax.Array:
    row_sizes, col_sizes = _acb_block_sizes_from_matrix_blocks(block_rows)
    del row_sizes, col_sizes
    row_chunks = []
    for row in block_rows:
        row_chunks.append(jnp.concatenate([mat_common.as_box_rect_matrix(block, "acb_mat.block_assemble") for block in row], axis=-2))
    return jnp.concatenate(row_chunks, axis=-3)


def acb_mat_block_diag(blocks) -> jax.Array:
    checks.check(len(blocks) > 0, "acb_mat.block_diag_nonempty")
    matrices = [acb_mat_as_matrix(block) for block in blocks]
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
                row_entries.append(_acb_block_zero(row_sizes[i], col_sizes[j], dtype=dtype))
        block_rows.append(tuple(row_entries))
    return acb_mat_block_assemble(tuple(block_rows))


def acb_mat_block_extract(a: jax.Array, row_block_sizes, col_block_sizes, row_block: int, col_block: int) -> jax.Array:
    a = acb_mat_as_matrix(a)
    row_offsets = _acb_offsets(row_block_sizes)
    col_offsets = _acb_offsets(col_block_sizes)
    return acb_mat_submatrix(a, row_offsets[row_block], row_offsets[row_block + 1], col_offsets[col_block], col_offsets[col_block + 1])


def acb_mat_block_row(a: jax.Array, row_block_sizes, row_block: int) -> jax.Array:
    a = acb_mat_as_matrix(a)
    row_offsets = _acb_offsets(row_block_sizes)
    return a[..., row_offsets[row_block] : row_offsets[row_block + 1], :, :]


def acb_mat_block_col(a: jax.Array, col_block_sizes, col_block: int) -> jax.Array:
    a = acb_mat_as_matrix(a)
    col_offsets = _acb_offsets(col_block_sizes)
    return a[..., :, col_offsets[col_block] : col_offsets[col_block + 1], :]


def acb_mat_block_matmul(a_blocks, b_blocks) -> jax.Array:
    a_row_sizes, a_col_sizes = _acb_block_sizes_from_matrix_blocks(a_blocks)
    b_row_sizes, b_col_sizes = _acb_block_sizes_from_matrix_blocks(b_blocks)
    checks.check_equal(a_col_sizes, b_row_sizes, "acb_mat.block_matmul.partition_inner")
    out_rows = []
    for i, a_row in enumerate(a_blocks):
        out_row = []
        for j in range(len(b_col_sizes)):
            total = None
            for k in range(len(a_col_sizes)):
                prod = acb_mat_matmul(a_row[k], b_blocks[k][j])
                total = prod if total is None else acb_core.acb_add(total, prod)
            if total is None:
                total = _acb_block_zero(a_row_sizes[i], b_col_sizes[j], dtype=acb_mat_as_matrix(a_row[0]).dtype)
            out_row.append(total)
        out_rows.append(tuple(out_row))
    return acb_mat_block_assemble(tuple(out_rows))


def acb_mat_banded_matvec(a: jax.Array, x: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    a = acb_mat_as_matrix(a)
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "acb_mat.banded_matvec.inner")
    mid = _mid_matrix(a)
    mask = _band_mask(mid.shape[-2], mid.shape[-1], lower_bandwidth, upper_bandwidth)
    y = jnp.einsum("...ij,...j->...i", jnp.where(mask, mid, jnp.zeros_like(mid)), _mid_vector(x))
    out = mat_common.box_from_point(y)
    finite = jnp.all(mat_common.complex_is_finite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_banded_matvec_basic(a: jax.Array, x: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    a = acb_mat_as_matrix(a)
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "acb_mat.banded_matvec_basic.inner")
    masked = _apply_band_mask_box(a, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)
    prods = acb_core.acb_mul(masked, x[..., None, :, :])
    out = mat_common.box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_cho(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    chol, ok = _mid_cholesky(a)
    lam_min = jnp.min(jnp.linalg.eigvalsh(_mid_hermitian_part(a)), axis=-1)
    ok = ok & (lam_min > _herm_perturbation_bound(a))
    out = mat_common.box_from_point(chol)
    return jnp.where(ok[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_cho_basic(a: jax.Array) -> jax.Array:
    return acb_mat_cho(a)


def acb_mat_ldl(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = acb_mat_as_matrix(a)
    chol, ok = _mid_cholesky(a)
    diag = jnp.diagonal(chol, axis1=-2, axis2=-1)
    l = chol / diag[..., None, :]
    d = jnp.real(diag * jnp.conj(diag))
    l_out = mat_common.box_from_point(l)
    d_out = mat_common.box_from_point(d)
    mask_l = ok[..., None, None, None]
    mask_d = ok[..., None, None]
    return jnp.where(mask_l, l_out, mat_common.full_box_like(l_out)), jnp.where(mask_d, d_out, mat_common.full_box_like(d_out))


def acb_mat_ldl_basic(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_ldl(a)


def acb_mat_charpoly(a: jax.Array) -> jax.Array:
    mid = _mid_matrix(a)
    coeffs_general = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=False)
    coeffs_hermitian = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=True)
    coeffs = jnp.where(_mid_is_hermitian(a)[..., None], coeffs_hermitian, coeffs_general)
    return mat_common.box_from_point(coeffs)


def acb_mat_pow_ui(a: jax.Array, n: int) -> jax.Array:
    return mat_common.box_from_point(mat_common.matrix_power_ui(_mid_matrix(a), n))


def acb_mat_exp(a: jax.Array) -> jax.Array:
    mid = _mid_matrix(a)
    exp_general = mat_common.matrix_exp(mid, hermitian=False)
    exp_hermitian = mat_common.matrix_exp(mid, hermitian=True)
    exp_mid = jnp.where(_mid_is_hermitian(a)[..., None, None], exp_hermitian, exp_general)
    return mat_common.box_from_point(exp_mid)


def acb_mat_eigvalsh(a: jax.Array) -> jax.Array:
    values, _ = mat_common.complex_hermitian_eigh(_mid_matrix(a))
    out = _tightened_complex_eigvals_box(values, _herm_perturbation_bound(a))
    finite = mat_common.box_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_eigh(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    values, vectors = mat_common.complex_hermitian_eigh(_mid_matrix(a))
    bound = _herm_perturbation_bound(a)
    values_out = _tightened_complex_eigvals_box(values, bound)
    vectors_out = _tightened_complex_eigvecs_box(values, vectors, bound)
    val_finite = mat_common.box_is_finite(values_out)
    vec_finite = jnp.all(mat_common.box_is_finite(vectors_out), axis=(-2, -1))
    return (
        jnp.where(val_finite[..., None], values_out, mat_common.full_box_like(values_out)),
        jnp.where(vec_finite[..., None, None, None], vectors_out, mat_common.full_box_like(vectors_out)),
    )


def acb_mat_dense_hpd_solve_plan_prepare(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    factor = acb_mat_cho(a)
    return mat_common.dense_cholesky_solve_plan_from_factor(
        factor,
        algebra="acb",
        structure="hermitian",
        label="acb_mat.dense_hpd_solve_plan_prepare",
    )


def acb_mat_hpd_solve(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        plan = mat_common.as_dense_cholesky_solve_plan(
            a_or_plan,
            algebra="acb",
            structure="hermitian",
            label="acb_mat.hpd_solve",
        )
    else:
        plan = acb_mat_dense_hpd_solve_plan_prepare(a_or_plan)
    b = acb_mat_as_rhs(b)
    factor = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.factor), _rhs_batch_shape(b, plan.rows))
    checks.check_equal(plan.rows, _rhs_rows_like(b, factor), "acb_mat.hpd_solve.inner")
    x = mat_common.lower_cholesky_solve(_mid_matrix(factor), _mid_rhs(b))
    out = mat_common.box_from_point(x)
    finite = _finite_mask_from_point(x) & jnp.all(mat_common.box_is_finite(factor), axis=(-2, -1))
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_box_like(out))


def acb_mat_hpd_solve_basic(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return acb_mat_hpd_solve(a_or_plan, b)


def acb_mat_dense_hpd_solve_plan_apply(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return acb_mat_hpd_solve(plan, b)


def acb_mat_hpd_inv(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        plan = mat_common.as_dense_cholesky_solve_plan(
            a_or_plan,
            algebra="acb",
            structure="hermitian",
            label="acb_mat.hpd_inv",
        )
    else:
        plan = acb_mat_dense_hpd_solve_plan_prepare(a_or_plan)
    factor = acb_mat_as_matrix(plan.factor)
    inv_mid = mat_common.lower_cholesky_solve(
        _mid_matrix(factor),
        jnp.broadcast_to(jnp.eye(plan.rows, dtype=_mid_matrix(factor).dtype), _mid_matrix(factor).shape),
    )
    out = mat_common.box_from_point(inv_mid)
    finite = jnp.all(mat_common.complex_is_finite(inv_mid), axis=(-2, -1)) & jnp.all(mat_common.box_is_finite(factor), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_hpd_inv_basic(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return acb_mat_hpd_inv(a_or_plan)


def acb_mat_solve_tril(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return acb_mat_triangular_solve(a, b, lower=True, unit_diagonal=unit_diagonal)


def acb_mat_solve_triu(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return acb_mat_triangular_solve(a, b, lower=False, unit_diagonal=unit_diagonal)


def acb_mat_solve_lu(a_or_plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array] | jax.Array, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        return acb_mat_lu_solve(a_or_plan, b)
    return acb_mat_lu_solve(acb_mat_dense_lu_solve_plan_prepare(a_or_plan), b)


def acb_mat_solve_lu_precomp(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return acb_mat_lu_solve(plan, b)


def acb_mat_solve_transpose(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        b = acb_mat_as_rhs(b)
        factor = _broadcast_box_matrix_batch(acb_mat_as_matrix(a_or_plan.factor), _rhs_batch_shape(b, a_or_plan.rows))
        checks.check_equal(a_or_plan.rows, _rhs_rows_like(b, factor), "acb_mat.solve_transpose.inner_hpd")
        x = mat_common.lower_cholesky_solve_transpose(_mid_matrix(factor), _mid_rhs(b))
        out = mat_common.box_from_point(x)
        finite = _finite_mask_from_point(x)
        return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_box_like(out))
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        plan = mat_common.as_dense_lu_solve_plan(a_or_plan, algebra="acb", label="acb_mat.solve_transpose")
        b = acb_mat_as_rhs(b)
        batch_shape = _rhs_batch_shape(b, plan.rows)
        p = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.p), batch_shape)
        l = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.l), batch_shape)
        u = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.u), batch_shape)
        checks.check_equal(plan.rows, _rhs_rows_like(b, p), "acb_mat.solve_transpose.inner_lu")
        p_mid = _mid_matrix(p)
        l_mid = _mid_matrix(l)
        u_mid = _mid_matrix(u)
        b_mid = _mid_rhs(b)
        vector_rhs = b_mid.ndim == p_mid.ndim - 1
        y = lax.linalg.triangular_solve(
            u_mid,
            b_mid[..., None] if vector_rhs else b_mid,
            left_side=True,
            lower=False,
            transpose_a=True,
            conjugate_a=False,
        )
        z = lax.linalg.triangular_solve(
            l_mid,
            y,
            left_side=True,
            lower=True,
            unit_diagonal=True,
            transpose_a=True,
            conjugate_a=False,
        )
        x = jnp.einsum("...ij,...j->...i", jnp.swapaxes(p_mid, -2, -1), z[..., 0]) if vector_rhs else jnp.matmul(jnp.swapaxes(p_mid, -2, -1), z)
        out = mat_common.box_from_point(x)
        finite = _finite_mask_from_point(x)
        return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_box_like(out))
    a = acb_mat_as_matrix(a_or_plan)
    b = acb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "acb_mat.solve_transpose.inner")
    a_mid = _mid_matrix(a)
    b_mid = _mid_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    x = jnp.linalg.solve(jnp.swapaxes(a_mid, -2, -1), b_mid[..., None] if vector_rhs else b_mid)
    if vector_rhs:
        x = x[..., 0]
    out = mat_common.box_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_box_like(out))


def acb_mat_solve_add(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        solved = acb_mat_hpd_solve(a_or_plan, b)
    elif isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        solved = acb_mat_solve_lu(a_or_plan, b)
    else:
        solved = acb_mat_solve(a_or_plan, b)
    return acb_core.acb_add(acb_mat_as_rhs(y), solved)


def acb_mat_solve_transpose_add(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return acb_core.acb_add(acb_mat_as_rhs(y), acb_mat_solve_transpose(a_or_plan, b))


def acb_mat_mat_solve(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return acb_mat_hpd_solve(a_or_plan, b)
    return acb_mat_solve_lu(a_or_plan, b) if isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)) else acb_mat_solve(a_or_plan, b)


def acb_mat_mat_solve_transpose(a_or_plan, b: jax.Array) -> jax.Array:
    return acb_mat_solve_transpose(a_or_plan, b)


def acb_mat_solve(a: jax.Array, b: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    b = acb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "acb_mat.solve.inner")
    a_mid = _mid_matrix(a)
    b_mid = _mid_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    x_general = jnp.linalg.solve(a_mid, b_mid[..., None] if vector_rhs else b_mid)
    if vector_rhs:
        x_general = x_general[..., 0]
    x_hpd, hpd_ok = _mid_hpd_solve(a, b)
    x = jnp.where(hpd_ok[(...,) + (None,) * (x_general.ndim - hpd_ok.ndim)], x_hpd, x_general)
    out = mat_common.box_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_solve(a, b)


def acb_mat_inv(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    inv_general = jnp.linalg.inv(_mid_matrix(a))
    inv_hpd, hpd_ok = _mid_hpd_inv(a)
    inv = jnp.where(hpd_ok[..., None, None], inv_hpd, inv_general)
    out = mat_common.box_from_point(inv)
    finite = jnp.all(mat_common.complex_is_finite(inv), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_inv_basic(a: jax.Array) -> jax.Array:
    return acb_mat_inv(a)


def acb_mat_sqr(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    c = jnp.matmul(_mid_matrix(a), _mid_matrix(a))
    out = mat_common.box_from_point(c)
    finite = jnp.all(mat_common.complex_is_finite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_sqr_basic(a: jax.Array) -> jax.Array:
    return acb_mat_matmul_basic(a, a)


def acb_mat_det(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    det = jnp.linalg.det(_mid_matrix(a))
    out = mat_common.box_from_point(det)
    finite = mat_common.complex_is_finite(det)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_det_basic(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    n = a.shape[-2]
    if n == 1:
        out = a[..., 0, 0, :]
    elif n == 2:
        out = mat_common.box_det_2x2(a)
    elif n == 3:
        out = mat_common.box_det_3x3(a)
    else:
        out = acb_mat_det(a)
    finite = mat_common.box_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_trace(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    tr = jnp.trace(_mid_matrix(a), axis1=-2, axis2=-1)
    out = mat_common.box_from_point(tr)
    finite = mat_common.complex_is_finite(tr)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_trace_basic(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    out = mat_common.box_trace(a)
    finite = mat_common.box_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_norm_fro(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    value = jnp.linalg.norm(_mid_matrix(a), ord="fro", axis=(-2, -1))
    out = mat_common.box_from_point(value)
    finite = mat_common.complex_is_finite(value)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_norm_fro_basic(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    abs_sq = acb_core.acb_mul(a, acb_core.acb_conj(a))
    total = mat_common.box_sum(mat_common.box_sum(abs_sq, axis=-2), axis=-1)
    out = acb_core.acb_sqrt(total)
    finite = mat_common.box_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_norm_1(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    value = jnp.linalg.norm(_mid_matrix(a), ord=1, axis=(-2, -1))
    out = mat_common.box_from_point(value)
    finite = mat_common.complex_is_finite(value)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_norm_1_basic(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    abs_a = acb_core.acb_abs(a)
    col_sums = mat_common.interval_sum(abs_a, axis=-2)
    value = jnp.max(di.midpoint(col_sums), axis=-1)
    out = mat_common.box_from_point(value)
    finite = mat_common.box_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_norm_inf(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    value = jnp.linalg.norm(_mid_matrix(a), ord=jnp.inf, axis=(-2, -1))
    out = mat_common.box_from_point(value)
    finite = mat_common.complex_is_finite(value)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_norm_inf_basic(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    abs_a = acb_core.acb_abs(a)
    row_sums = mat_common.interval_sum(abs_a, axis=-1)
    value = jnp.max(di.midpoint(row_sums), axis=-1)
    out = mat_common.box_from_point(value)
    finite = mat_common.box_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_box_like(out))


def acb_mat_det_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_det_basic(a)


def acb_mat_trace_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_trace_basic(a)


def acb_mat_norm_fro_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_norm_fro_basic(a)


def acb_mat_norm_1_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_norm_1_basic(a)


def acb_mat_norm_inf_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_norm_inf_basic(a)


def acb_mat_triangular_solve(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    a = acb_mat_as_matrix(a)
    b = acb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "acb_mat.triangular_solve.inner")
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
    out = mat_common.box_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_box_like(out))


def acb_mat_triangular_solve_basic(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return acb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal)


def acb_mat_lu(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = acb_mat_as_matrix(a)
    mid = _mid_matrix(a)
    lu, _, perm = lax.linalg.lu(mid)
    n = mid.shape[-1]
    eye = jnp.eye(n, dtype=mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return mat_common.box_from_point(p), mat_common.box_from_point(l), mat_common.box_from_point(u)


def acb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return acb_mat_lu(a)


def acb_mat_dense_lu_solve_plan_prepare(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    p, l, u = acb_mat_lu(a)
    return mat_common.dense_lu_solve_plan_from_factors(p, l, u, algebra="acb", label="acb_mat.dense_lu_solve_plan_prepare")


def acb_mat_dense_lu_solve_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    p, l, u = acb_mat_lu_prec(a, prec_bits=prec_bits)
    return mat_common.dense_lu_solve_plan_from_factors(p, l, u, algebra="acb", label="acb_mat.dense_lu_solve_plan_prepare_prec")


def acb_mat_lu_solve(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="acb", label="acb_mat.lu_solve")
    b = acb_mat_as_rhs(b)
    batch_shape = _rhs_batch_shape(b, plan.rows)
    p = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.p), batch_shape)
    l = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.l), batch_shape)
    u = _broadcast_box_matrix_batch(acb_mat_as_matrix(plan.u), batch_shape)
    checks.check_equal(plan.rows, _rhs_rows_like(b, p), "acb_mat.lu_solve.inner")
    pb = _apply_perm_rhs(p, b)
    y = acb_mat_triangular_solve(l, pb, lower=True, unit_diagonal=True)
    return acb_mat_triangular_solve(u, y, lower=False, unit_diagonal=False)


def acb_mat_lu_solve_basic(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return acb_mat_lu_solve(plan, b)


def acb_mat_dense_lu_solve_plan_apply(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return acb_mat_lu_solve(plan, b)


def acb_mat_qr(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = acb_mat_as_matrix(a)
    q, r = jnp.linalg.qr(_mid_matrix(a))
    return mat_common.box_from_point(q), mat_common.box_from_point(r)


def acb_mat_qr_basic(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_qr(a)


def acb_mat_permutation_matrix_rigorous(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return acb_mat_permutation_matrix(perm, dtype=dtype)


def acb_mat_transpose_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_transpose(a)


def acb_mat_conjugate_transpose_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_conjugate_transpose(a)


def acb_mat_submatrix_rigorous(a: jax.Array, row_start: int, row_stop: int, col_start: int, col_stop: int) -> jax.Array:
    return acb_mat_submatrix(a, row_start, row_stop, col_start, col_stop)


def acb_mat_diag_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_diag(a)


def acb_mat_diag_matrix_rigorous(d: jax.Array) -> jax.Array:
    return acb_mat_diag_matrix(d)


def acb_mat_hermitian_part_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_hermitian_part(a)


def acb_mat_is_hermitian_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_hermitian(a)


def acb_mat_is_hpd_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_hpd(a)


def acb_mat_cho_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_cho(a)


def acb_mat_ldl_rigorous(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_ldl(a)


def acb_mat_eigvalsh_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_eigvalsh(a)


def acb_mat_eigh_rigorous(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_eigh(a)


def acb_mat_charpoly_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_charpoly(a)


def acb_mat_pow_ui_rigorous(a: jax.Array, n: int) -> jax.Array:
    return acb_mat_pow_ui(a, n)


def acb_mat_exp_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_exp(a)


def acb_mat_hpd_solve_rigorous(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_hpd_solve(a_or_plan, b)


def acb_mat_hpd_inv_rigorous(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return acb_mat_hpd_inv(a_or_plan)


def acb_mat_solve_tril_rigorous(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return acb_mat_solve_tril(a, b, unit_diagonal=unit_diagonal)


def acb_mat_solve_triu_rigorous(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return acb_mat_solve_triu(a, b, unit_diagonal=unit_diagonal)


def acb_mat_solve_lu_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    return acb_mat_solve_lu(a_or_plan, b)


def acb_mat_solve_transpose_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    return acb_mat_solve_transpose(a_or_plan, b)


def acb_mat_solve_add_rigorous(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return acb_mat_solve_add(a_or_plan, b, y)


def acb_mat_solve_transpose_add_rigorous(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return acb_mat_solve_transpose_add(a_or_plan, b, y)


def acb_mat_mat_solve_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    return acb_mat_mat_solve(a_or_plan, b)


def acb_mat_mat_solve_transpose_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    return acb_mat_mat_solve_transpose(a_or_plan, b)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matmul_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_rmatvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower_bandwidth", "upper_bandwidth"))
def acb_mat_banded_matvec_prec(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_mat_banded_matvec(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_cached_apply_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_apply(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_rmatvec_cached_apply_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_cached_apply(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_hermitian_part_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hermitian_part(a), prec_bits)


def acb_mat_is_hermitian_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_hermitian(a)


def acb_mat_is_hpd_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_hpd(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_cho_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_cho(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_ldl_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    l, d = acb_mat_ldl(a)
    return acb_core.acb_box_round_prec(l, prec_bits), acb_core.acb_box_round_prec(d, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_eigvalsh_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_eigvalsh(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_eigh_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    values, vectors = acb_mat_eigh(a)
    return acb_core.acb_box_round_prec(values, prec_bits), acb_core.acb_box_round_prec(vectors, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_charpoly_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_charpoly(a), prec_bits)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def acb_mat_pow_ui_prec(a: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_pow_ui(a, n), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_exp_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_exp(a), prec_bits)


def acb_mat_dense_hpd_solve_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseCholeskySolvePlan:
    factor = acb_mat_cho_prec(a, prec_bits=prec_bits)
    return mat_common.dense_cholesky_solve_plan_from_factor(
        factor,
        algebra="acb",
        structure="hermitian",
        label="acb_mat.dense_hpd_solve_plan_prepare_prec",
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_hpd_solve_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hpd_solve(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_hpd_inv_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hpd_inv(a_or_plan), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "unit_diagonal"))
def acb_mat_solve_tril_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_tril(a, b, unit_diagonal=unit_diagonal), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "unit_diagonal"))
def acb_mat_solve_triu_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_triu(a, b, unit_diagonal=unit_diagonal), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_lu_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_lu(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_transpose_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_transpose(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_add_prec(a_or_plan, b: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_add(a_or_plan, b, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_transpose_add_prec(a_or_plan, b: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_transpose_add(a_or_plan, b, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_mat_solve_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_mat_solve(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_mat_solve_transpose_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_mat_solve_transpose(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_inv_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_inv(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_sqr_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_sqr(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_det_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_det(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_trace_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_trace(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_norm_fro_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_fro(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_norm_1_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_1(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_norm_inf_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_inf(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower", "unit_diagonal"))
def acb_mat_triangular_solve_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_lu_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = acb_mat_lu(a)
    return (
        acb_core.acb_box_round_prec(p, prec_bits),
        acb_core.acb_box_round_prec(l, prec_bits),
        acb_core.acb_box_round_prec(u, prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_qr_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    q, r = acb_mat_qr(a)
    return acb_core.acb_box_round_prec(q, prec_bits), acb_core.acb_box_round_prec(r, prec_bits)


def acb_mat_add_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_add(a, b)


def acb_mat_sub_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_sub(a, b)


def acb_mat_neg_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_neg(a)


def acb_mat_mul_entrywise_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_mul_entrywise(a, b)


def acb_mat_conjugate_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_conjugate(a)


def acb_mat_is_diag_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_diag(a)


def acb_mat_is_tril_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_tril(a)


def acb_mat_is_triu_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_triu(a)


def acb_mat_is_zero_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_zero(a)


def acb_mat_is_finite_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_finite(a)


def acb_mat_is_exact_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_exact(a)


def acb_mat_is_real_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_is_real(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_add_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_add(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_sub_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_sub(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_neg_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_neg(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_mul_entrywise_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_mul_entrywise(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_conjugate_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_conjugate(a), prec_bits)


def acb_mat_is_diag_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_diag(a)


def acb_mat_is_tril_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_tril(a)


def acb_mat_is_triu_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_triu(a)


def acb_mat_is_zero_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_zero(a)


def acb_mat_is_finite_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_finite(a)


def acb_mat_is_exact_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_exact(a)


def acb_mat_is_real_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_real(a)


acb_mat_matmul_jit = jax.jit(acb_mat_matmul)
acb_mat_matvec_jit = jax.jit(acb_mat_matvec)
acb_mat_rmatvec_jit = jax.jit(acb_mat_rmatvec)
acb_mat_banded_matvec_jit = jax.jit(acb_mat_banded_matvec, static_argnames=("lower_bandwidth", "upper_bandwidth"))
acb_mat_matvec_cached_apply_jit = jax.jit(acb_mat_matvec_cached_apply)
acb_mat_rmatvec_cached_apply_jit = jax.jit(acb_mat_rmatvec_cached_apply)
acb_mat_hermitian_part_jit = jax.jit(acb_mat_hermitian_part)
acb_mat_solve_jit = jax.jit(acb_mat_solve)
acb_mat_inv_jit = jax.jit(acb_mat_inv)
acb_mat_cho_jit = jax.jit(acb_mat_cho)
acb_mat_ldl_jit = jax.jit(acb_mat_ldl)
acb_mat_hpd_solve_jit = jax.jit(acb_mat_hpd_solve)
acb_mat_hpd_inv_jit = jax.jit(acb_mat_hpd_inv)
acb_mat_sqr_jit = jax.jit(acb_mat_sqr)
acb_mat_det_jit = jax.jit(acb_mat_det)
acb_mat_trace_jit = jax.jit(acb_mat_trace)
acb_mat_norm_fro_jit = jax.jit(acb_mat_norm_fro)
acb_mat_norm_1_jit = jax.jit(acb_mat_norm_1)
acb_mat_norm_inf_jit = jax.jit(acb_mat_norm_inf)
acb_mat_triangular_solve_jit = jax.jit(acb_mat_triangular_solve, static_argnames=("lower", "unit_diagonal"))
acb_mat_lu_jit = jax.jit(acb_mat_lu)
acb_mat_qr_jit = jax.jit(acb_mat_qr)


def acb_mat_matmul_batch_fixed(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_matmul(a, b)


def acb_mat_matmul_batch_padded(a: jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b), pad_to=pad_to)
    return acb_mat_matmul(*call_args)


def acb_mat_matvec_batch_fixed(a: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec(a, x)


def acb_mat_rmatvec_batch_fixed(a: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_rmatvec(a, x)


def acb_mat_matvec_batch_padded(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return acb_mat_matvec(*call_args)


def acb_mat_rmatvec_batch_padded(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return acb_mat_rmatvec(*call_args)


def acb_mat_banded_matvec_batch_fixed(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    return acb_mat_banded_matvec(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


def acb_mat_banded_matvec_batch_padded(
    a: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return acb_mat_banded_matvec(*call_args, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth)


def acb_mat_matvec_cached_apply_batch_fixed(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_apply(cache, x)


def acb_mat_rmatvec_cached_apply_batch_fixed(cache: jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_rmatvec_cached_apply(cache, x)


def acb_mat_matvec_cached_apply_batch_padded(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((cache, x), pad_to=pad_to)
    return acb_mat_matvec_cached_apply(*call_args)


def acb_mat_rmatvec_cached_apply_batch_padded(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((cache, x), pad_to=pad_to)
    return acb_mat_rmatvec_cached_apply(*call_args)


def acb_mat_matvec_cached_prepare_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_matvec_cached_prepare(a)


def acb_mat_rmatvec_cached_prepare_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_rmatvec_cached_prepare(a)


def acb_mat_matvec_cached_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return acb_mat_matvec_cached_prepare(*call_args)


def acb_mat_rmatvec_cached_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return acb_mat_rmatvec_cached_prepare(*call_args)


def acb_mat_dense_matvec_plan_prepare_batch_fixed(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return acb_mat_dense_matvec_plan_prepare(a)


def acb_mat_dense_matvec_plan_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> mat_common.DenseMatvecPlan:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return acb_mat_dense_matvec_plan_prepare(*call_args)


def acb_mat_dense_matvec_plan_apply_batch_fixed(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return acb_mat_dense_matvec_plan_apply(plan, x)


def acb_mat_dense_matvec_plan_apply_batch_padded(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (x_pad,), _ = mat_common.pad_batch_repeat_last((x,), pad_to=pad_to)
    return acb_mat_dense_matvec_plan_apply(plan, x_pad)


def acb_mat_solve_batch_fixed(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_solve(a, b)


def acb_mat_solve_batch_padded(a: jax.Array, b: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b), pad_to=pad_to)
    return acb_mat_solve(*call_args)


def acb_mat_inv_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_inv(a)


def acb_mat_inv_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_inv(*call_args)


def acb_mat_triangular_solve_batch_fixed(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return acb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal)


def acb_mat_triangular_solve_batch_padded(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b), pad_to=pad_to)
    return acb_mat_triangular_solve(*call_args, lower=lower, unit_diagonal=unit_diagonal)


def acb_mat_lu_batch_fixed(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return acb_mat_lu(a)


def acb_mat_lu_batch_padded(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_lu(*call_args)


def acb_mat_qr_batch_fixed(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_qr(a)


def acb_mat_dense_lu_solve_plan_prepare_batch_fixed(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    return acb_mat_dense_lu_solve_plan_prepare(a)


def acb_mat_dense_lu_solve_plan_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> mat_common.DenseLUSolvePlan:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return acb_mat_dense_lu_solve_plan_prepare(*call_args)


def acb_mat_dense_lu_solve_plan_apply_batch_fixed(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return acb_mat_dense_lu_solve_plan_apply(plan, b)


def acb_mat_dense_lu_solve_plan_apply_batch_padded(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = mat_common.pad_batch_repeat_last((b,), pad_to=pad_to)
    return acb_mat_dense_lu_solve_plan_apply(plan, b_pad)


def acb_mat_qr_batch_padded(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array]:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_qr(*call_args)


def acb_mat_det_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_det(a)


def acb_mat_det_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_det(*call_args)


def acb_mat_trace_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_trace(a)


def acb_mat_trace_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_trace(*call_args)


def acb_mat_sqr_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_sqr(a)


def acb_mat_sqr_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_sqr(*call_args)


def acb_mat_norm_fro_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_norm_fro(a)


def acb_mat_norm_fro_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_norm_fro(*call_args)


def acb_mat_norm_1_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_norm_1(a)


def acb_mat_norm_1_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_norm_1(*call_args)


def acb_mat_norm_inf_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_norm_inf(a)


def acb_mat_norm_inf_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_norm_inf(*call_args)


def acb_mat_hermitian_part_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_hermitian_part(a)


def acb_mat_hermitian_part_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_hermitian_part(*call_args)


def acb_mat_is_hermitian_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_is_hermitian(a)


def acb_mat_is_hermitian_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_is_hermitian(*call_args)


def acb_mat_is_hpd_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_is_hpd(a)


def acb_mat_is_hpd_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_is_hpd(*call_args)


def acb_mat_cho_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_cho(a)


def acb_mat_cho_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_cho(*call_args)


def acb_mat_ldl_batch_fixed(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_ldl(a)


def acb_mat_ldl_batch_padded(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array]:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_ldl(*call_args)


def acb_mat_dense_hpd_solve_plan_prepare_batch_fixed(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    return acb_mat_dense_hpd_solve_plan_prepare(a)


def acb_mat_dense_hpd_solve_plan_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> mat_common.DenseCholeskySolvePlan:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return acb_mat_dense_hpd_solve_plan_prepare(*call_args)


def acb_mat_hpd_solve_batch_fixed(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_hpd_solve(a_or_plan, b)


def acb_mat_hpd_solve_batch_padded(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = mat_common.pad_batch_repeat_last((b,), pad_to=pad_to)
    return acb_mat_hpd_solve(a_or_plan, b_pad)


def acb_mat_dense_hpd_solve_plan_apply_batch_fixed(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return acb_mat_dense_hpd_solve_plan_apply(plan, b)


def acb_mat_dense_hpd_solve_plan_apply_batch_padded(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = mat_common.pad_batch_repeat_last((b,), pad_to=pad_to)
    return acb_mat_dense_hpd_solve_plan_apply(plan, b_pad)


def acb_mat_hpd_inv_batch_fixed(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return acb_mat_hpd_inv(a_or_plan)


def acb_mat_hpd_inv_batch_padded(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array, *, pad_to: int) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return acb_mat_hpd_inv(a_or_plan)
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a_or_plan,), pad_to=pad_to)
    return acb_mat_hpd_inv(*call_args)


def acb_mat_hermitian_part_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hermitian_part_batch_fixed(a), prec_bits)


def acb_mat_hermitian_part_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hermitian_part_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_is_hermitian_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_hermitian_batch_fixed(a)


def acb_mat_is_hermitian_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_hermitian_batch_padded(a, pad_to=pad_to)


def acb_mat_is_hpd_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_hpd_batch_fixed(a)


def acb_mat_is_hpd_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return acb_mat_is_hpd_batch_padded(a, pad_to=pad_to)


def acb_mat_cho_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_cho_batch_fixed(a), prec_bits)


def acb_mat_cho_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_cho_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_ldl_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    l, d = acb_mat_ldl_batch_fixed(a)
    return acb_core.acb_box_round_prec(l, prec_bits), acb_core.acb_box_round_prec(d, prec_bits)


def acb_mat_ldl_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    l, d = acb_mat_ldl_batch_padded(a, pad_to=pad_to)
    return acb_core.acb_box_round_prec(l, prec_bits), acb_core.acb_box_round_prec(d, prec_bits)


def acb_mat_dense_hpd_solve_plan_prepare_batch_fixed_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseCholeskySolvePlan:
    return acb_mat_dense_hpd_solve_plan_prepare_prec(a, prec_bits=prec_bits)


def acb_mat_dense_hpd_solve_plan_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseCholeskySolvePlan:
    plan = acb_mat_dense_hpd_solve_plan_prepare_batch_padded(a, pad_to=pad_to)
    return mat_common.dense_cholesky_solve_plan_from_factor(
        acb_core.acb_box_round_prec(plan.factor, prec_bits),
        algebra="acb",
        structure="hermitian",
        label="acb_mat.dense_hpd_solve_plan_prepare_batch_padded_prec",
    )


def acb_mat_hpd_solve_batch_fixed_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hpd_solve_batch_fixed(a_or_plan, b), prec_bits)


def acb_mat_hpd_solve_batch_padded_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hpd_solve_batch_padded(a_or_plan, b, pad_to=pad_to), prec_bits)


def acb_mat_dense_hpd_solve_plan_apply_prec(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_hpd_solve_plan_apply(plan, b), prec_bits)


def acb_mat_dense_hpd_solve_plan_apply_batch_fixed_prec(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_hpd_solve_plan_apply_batch_fixed(plan, b), prec_bits)


def acb_mat_dense_hpd_solve_plan_apply_batch_padded_prec(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_hpd_solve_plan_apply_batch_padded(plan, b, pad_to=pad_to), prec_bits)


def acb_mat_hpd_inv_batch_fixed_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hpd_inv_batch_fixed(a_or_plan), prec_bits)


def acb_mat_hpd_inv_batch_padded_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_hpd_inv_batch_padded(a_or_plan, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matmul_batch_fixed_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul_batch_fixed(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_matmul_batch_padded_prec(a: jax.Array, b: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul_batch_padded(a, b, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_batch_fixed_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_batch_fixed(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_rmatvec_batch_fixed_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_batch_fixed(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_matvec_batch_padded_prec(a: jax.Array, x: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_batch_padded(a, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_rmatvec_batch_padded_prec(a: jax.Array, x: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_batch_padded(a, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower_bandwidth", "upper_bandwidth"))
def acb_mat_banded_matvec_batch_fixed_prec(
    a: jax.Array,
    x: jax.Array,
    *,
    lower_bandwidth: int,
    upper_bandwidth: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_mat_banded_matvec_batch_fixed(a, x, lower_bandwidth=lower_bandwidth, upper_bandwidth=upper_bandwidth),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits", "pad_to", "lower_bandwidth", "upper_bandwidth"))
def acb_mat_banded_matvec_batch_padded_prec(
    a: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_mat_banded_matvec_batch_padded(
            a,
            x,
            pad_to=pad_to,
            lower_bandwidth=lower_bandwidth,
            upper_bandwidth=upper_bandwidth,
        ),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_cached_apply_batch_fixed_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_apply_batch_fixed(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_rmatvec_cached_apply_batch_fixed_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_cached_apply_batch_fixed(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_matvec_cached_apply_batch_padded_prec(
    cache: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_apply_batch_padded(cache, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_rmatvec_cached_apply_batch_padded_prec(
    cache: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_cached_apply_batch_padded(cache, x, pad_to=pad_to), prec_bits)


def acb_mat_matvec_cached_prepare_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_prepare_batch_fixed(a), prec_bits)


def acb_mat_rmatvec_cached_prepare_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_cached_prepare_batch_fixed(a), prec_bits)


def acb_mat_matvec_cached_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_prepare_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_rmatvec_cached_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_rmatvec_cached_prepare_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_dense_matvec_plan_prepare_batch_fixed_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    return acb_mat_dense_matvec_plan_prepare_prec(a, prec_bits=prec_bits)


def acb_mat_dense_matvec_plan_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    plan = acb_mat_dense_matvec_plan_prepare_batch_padded(a, pad_to=pad_to)
    return mat_common.dense_matvec_plan_from_matrix(
        acb_core.acb_box_round_prec(plan.matrix, prec_bits),
        algebra="acb",
        label="acb_mat.dense_matvec_plan_prepare_batch_padded_prec",
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_dense_matvec_plan_apply_prec(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_matvec_plan_apply(plan, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_dense_matvec_plan_apply_batch_fixed_prec(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_matvec_plan_apply_batch_fixed(plan, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_dense_matvec_plan_apply_batch_padded_prec(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_matvec_plan_apply_batch_padded(plan, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_batch_fixed_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_batch_fixed(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_solve_batch_padded_prec(a: jax.Array, b: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve_batch_padded(a, b, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_inv_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_inv_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_inv_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_inv_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower", "unit_diagonal"))
def acb_mat_triangular_solve_batch_fixed_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_mat_triangular_solve_batch_fixed(a, b, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits", "pad_to", "lower", "unit_diagonal"))
def acb_mat_triangular_solve_batch_padded_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        acb_mat_triangular_solve_batch_padded(a, b, pad_to=pad_to, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_lu_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = acb_mat_lu_batch_fixed(a)
    return (
        acb_core.acb_box_round_prec(p, prec_bits),
        acb_core.acb_box_round_prec(l, prec_bits),
        acb_core.acb_box_round_prec(u, prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_lu_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = acb_mat_lu_batch_padded(a, pad_to=pad_to)
    return (
        acb_core.acb_box_round_prec(p, prec_bits),
        acb_core.acb_box_round_prec(l, prec_bits),
        acb_core.acb_box_round_prec(u, prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_qr_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    q, r = acb_mat_qr_batch_fixed(a)
    return acb_core.acb_box_round_prec(q, prec_bits), acb_core.acb_box_round_prec(r, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_qr_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    q, r = acb_mat_qr_batch_padded(a, pad_to=pad_to)
    return acb_core.acb_box_round_prec(q, prec_bits), acb_core.acb_box_round_prec(r, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_det_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_det_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_det_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_det_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_trace_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_trace_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_trace_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_trace_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_sqr_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_sqr_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_sqr_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_sqr_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_norm_fro_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_fro_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_norm_fro_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_fro_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_norm_1_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_1_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_norm_1_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_1_batch_padded(a, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_norm_inf_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_inf_batch_fixed(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_norm_inf_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_norm_inf_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_permutation_matrix_prec(
    perm: jax.Array,
    *,
    dtype: jnp.dtype = jnp.float64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_permutation_matrix(perm, dtype=dtype), prec_bits)


def acb_mat_transpose_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_transpose(a), prec_bits)


def acb_mat_conjugate_transpose_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_conjugate_transpose(a), prec_bits)


def acb_mat_submatrix_prec(
    a: jax.Array,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_submatrix(a, row_start, row_stop, col_start, col_stop), prec_bits)


def acb_mat_diag_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_diag(a), prec_bits)


def acb_mat_diag_matrix_prec(d: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_diag_matrix(d), prec_bits)


def acb_mat_lu_solve_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_lu_solve(plan, b), prec_bits)


def acb_mat_dense_lu_solve_plan_apply_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_lu_solve_plan_apply(plan, b), prec_bits)


def acb_mat_dense_lu_solve_plan_prepare_batch_fixed_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    return acb_mat_dense_lu_solve_plan_prepare_prec(a, prec_bits=prec_bits)


def acb_mat_dense_lu_solve_plan_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    plan = acb_mat_dense_lu_solve_plan_prepare_batch_padded(a, pad_to=pad_to)
    return mat_common.dense_lu_solve_plan_from_factors(
        acb_core.acb_box_round_prec(plan.p, prec_bits),
        acb_core.acb_box_round_prec(plan.l, prec_bits),
        acb_core.acb_box_round_prec(plan.u, prec_bits),
        algebra="acb",
        label="acb_mat.dense_lu_solve_plan_prepare_batch_padded_prec",
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_dense_lu_solve_plan_apply_batch_fixed_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_lu_solve_plan_apply_batch_fixed(plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_dense_lu_solve_plan_apply_batch_padded_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_dense_lu_solve_plan_apply_batch_padded(plan, b, pad_to=pad_to), prec_bits)


def acb_mat_permutation_matrix_batch_fixed(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return acb_mat_permutation_matrix(perm, dtype=dtype)


def acb_mat_permutation_matrix_batch_padded(perm: jax.Array, *, pad_to: int, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((perm,), pad_to=pad_to)
    return acb_mat_permutation_matrix(*call_args, dtype=dtype)


def acb_mat_transpose_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_transpose(a)


def acb_mat_transpose_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_transpose(*call_args)


def acb_mat_conjugate_transpose_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_conjugate_transpose(a)


def acb_mat_conjugate_transpose_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_conjugate_transpose(*call_args)


def acb_mat_diag_batch_fixed(a: jax.Array) -> jax.Array:
    return acb_mat_diag(a)


def acb_mat_diag_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return acb_mat_diag(*call_args)


def acb_mat_diag_matrix_batch_fixed(d: jax.Array) -> jax.Array:
    return acb_mat_diag_matrix(d)


def acb_mat_diag_matrix_batch_padded(d: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((d,), pad_to=pad_to)
    return acb_mat_diag_matrix(*call_args)


def acb_mat_lu_solve_batch_fixed(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return acb_mat_lu_solve(plan, b)


def acb_mat_lu_solve_batch_padded(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((plan, b), pad_to=pad_to)
    return acb_mat_lu_solve(*call_args)


def acb_mat_permutation_matrix_batch_fixed_prec(
    perm: jax.Array,
    *,
    dtype: jnp.dtype = jnp.float64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_permutation_matrix_batch_fixed(perm, dtype=dtype), prec_bits)


def acb_mat_permutation_matrix_batch_padded_prec(
    perm: jax.Array,
    *,
    pad_to: int,
    dtype: jnp.dtype = jnp.float64,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_permutation_matrix_batch_padded(perm, pad_to=pad_to, dtype=dtype), prec_bits)


def acb_mat_transpose_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_transpose_batch_fixed(a), prec_bits)


def acb_mat_transpose_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_transpose_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_conjugate_transpose_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_conjugate_transpose_batch_fixed(a), prec_bits)


def acb_mat_conjugate_transpose_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_conjugate_transpose_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_diag_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_diag_batch_fixed(a), prec_bits)


def acb_mat_diag_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_diag_batch_padded(a, pad_to=pad_to), prec_bits)


def acb_mat_diag_matrix_batch_fixed_prec(d: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_diag_matrix_batch_fixed(d), prec_bits)


def acb_mat_diag_matrix_batch_padded_prec(d: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_diag_matrix_batch_padded(d, pad_to=pad_to), prec_bits)


def acb_mat_lu_solve_batch_fixed_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_lu_solve_batch_fixed(plan, b), prec_bits)


def acb_mat_lu_solve_batch_padded_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_lu_solve_batch_padded(plan, b, pad_to=pad_to), prec_bits)


def _make_acb_batch_fixed(name: str):
    fn = globals()[name]

    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapped.__name__ = f"{name}_batch_fixed"
    return wrapped


def _make_acb_batch_padded(name: str):
    fn = globals()[name]

    def wrapped(*args, pad_to: int, **kwargs):
        call_args = []
        for arg in args:
            if mat_common.is_dense_plan_like(arg):
                call_args.append(arg)
            elif not mat_common.is_batch_pad_candidate(arg):
                call_args.append(arg)
            else:
                (arg_pad,), _ = kh.pad_mixed_batch_args_repeat_last((arg,), pad_to=pad_to)
                call_args.append(arg_pad)
        return fn(*tuple(call_args), **kwargs)

    wrapped.__name__ = f"{name}_batch_padded"
    return wrapped


def _make_acb_batch_fixed_prec(name: str):
    fn = globals()[f"{name}_batch_fixed"]
    round_values = "eigh" in name
    passthrough = name.startswith("acb_mat_is_")

    def wrapped(*args, prec_bits: int = di.DEFAULT_PREC_BITS, **kwargs):
        out = fn(*args, **kwargs)
        if passthrough:
            return out
        if round_values:
            return tuple(acb_core.acb_box_round_prec(part, prec_bits) for part in out)
        return acb_core.acb_box_round_prec(out, prec_bits)

    wrapped.__name__ = f"{name}_batch_fixed_prec"
    return wrapped


def _make_acb_batch_padded_prec(name: str):
    fn = globals()[f"{name}_batch_padded"]
    round_values = "eigh" in name
    passthrough = name.startswith("acb_mat_is_")

    def wrapped(*args, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS, **kwargs):
        out = fn(*args, pad_to=pad_to, **kwargs)
        if passthrough:
            return out
        if round_values:
            return tuple(acb_core.acb_box_round_prec(part, prec_bits) for part in out)
        return acb_core.acb_box_round_prec(out, prec_bits)

    wrapped.__name__ = f"{name}_batch_padded_prec"
    return wrapped


for _acb_name in (
    "acb_mat_add",
    "acb_mat_sub",
    "acb_mat_neg",
    "acb_mat_mul_entrywise",
    "acb_mat_conjugate",
    "acb_mat_is_diag",
    "acb_mat_is_tril",
    "acb_mat_is_triu",
    "acb_mat_is_zero",
    "acb_mat_is_finite",
    "acb_mat_is_exact",
    "acb_mat_is_real",
    "acb_mat_charpoly",
    "acb_mat_pow_ui",
    "acb_mat_exp",
    "acb_mat_eigvalsh",
    "acb_mat_eigh",
    "acb_mat_solve_tril",
    "acb_mat_solve_triu",
    "acb_mat_solve_lu",
    "acb_mat_solve_transpose",
    "acb_mat_solve_add",
    "acb_mat_solve_transpose_add",
    "acb_mat_mat_solve",
    "acb_mat_mat_solve_transpose",
):
    globals()[f"{_acb_name}_batch_fixed"] = _make_acb_batch_fixed(_acb_name)
    globals()[f"{_acb_name}_batch_padded"] = _make_acb_batch_padded(_acb_name)
    globals()[f"{_acb_name}_batch_fixed_prec"] = _make_acb_batch_fixed_prec(_acb_name)
    globals()[f"{_acb_name}_batch_padded_prec"] = _make_acb_batch_padded_prec(_acb_name)


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


def acb_mat_operator_plan_prepare(a: jax.Array):
    from . import jcb_mat
    return jcb_mat.jcb_mat_dense_operator_plan_prepare(a)


def acb_mat_operator_rmatvec_plan_prepare(a: jax.Array):
    from . import jcb_mat
    return jcb_mat.jcb_mat_dense_operator_rmatvec_plan_prepare(a)


def acb_mat_operator_adjoint_plan_prepare(a: jax.Array):
    from . import jcb_mat
    return jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)


__all__ = [
    "acb_mat_as_matrix",
    "acb_mat_as_vector",
    "acb_mat_shape",
    "acb_mat_zero",
    "acb_mat_identity",
    "acb_mat_block_assemble",
    "acb_mat_block_diag",
    "acb_mat_block_extract",
    "acb_mat_block_row",
    "acb_mat_block_col",
    "acb_mat_block_matmul",
    "acb_mat_matmul",
    "acb_mat_matmul_basic",
    "acb_mat_matvec",
    "acb_mat_matvec_basic",
    "acb_mat_rmatvec",
    "acb_mat_rmatvec_basic",
    "acb_mat_banded_matvec",
    "acb_mat_banded_matvec_basic",
    "acb_mat_matvec_cached_prepare",
    "acb_mat_rmatvec_cached_prepare",
    "acb_mat_dense_matvec_plan_prepare",
    "acb_mat_dense_matvec_plan_apply",
    "acb_mat_matvec_cached_apply",
    "acb_mat_rmatvec_cached_apply",
    "acb_mat_hermitian_part",
    "acb_mat_is_hermitian",
    "acb_mat_is_hpd",
    "acb_mat_cho",
    "acb_mat_ldl",
    "acb_mat_dense_hpd_solve_plan_prepare",
    "acb_mat_hpd_solve",
    "acb_mat_dense_hpd_solve_plan_apply",
    "acb_mat_hpd_inv",
    "acb_mat_solve",
    "acb_mat_solve_basic",
    "acb_mat_inv",
    "acb_mat_inv_basic",
    "acb_mat_sqr",
    "acb_mat_sqr_basic",
    "acb_mat_det",
    "acb_mat_det_basic",
    "acb_mat_trace",
    "acb_mat_trace_basic",
    "acb_mat_norm_fro",
    "acb_mat_norm_fro_basic",
    "acb_mat_norm_1",
    "acb_mat_norm_1_basic",
    "acb_mat_norm_inf",
    "acb_mat_norm_inf_basic",
    "acb_mat_det_rigorous",
    "acb_mat_trace_rigorous",
    "acb_mat_norm_fro_rigorous",
    "acb_mat_norm_1_rigorous",
    "acb_mat_norm_inf_rigorous",
    "acb_mat_triangular_solve",
    "acb_mat_triangular_solve_basic",
    "acb_mat_lu",
    "acb_mat_lu_basic",
    "acb_mat_qr",
    "acb_mat_qr_basic",
    "acb_mat_matmul_prec",
    "acb_mat_matvec_prec",
    "acb_mat_rmatvec_prec",
    "acb_mat_banded_matvec_prec",
    "acb_mat_matvec_cached_prepare_prec",
    "acb_mat_rmatvec_cached_prepare_prec",
    "acb_mat_dense_matvec_plan_prepare_prec",
    "acb_mat_matvec_cached_apply_prec",
    "acb_mat_rmatvec_cached_apply_prec",
    "acb_mat_hermitian_part_prec",
    "acb_mat_is_hermitian_prec",
    "acb_mat_is_hpd_prec",
    "acb_mat_cho_prec",
    "acb_mat_ldl_prec",
    "acb_mat_dense_hpd_solve_plan_prepare_prec",
    "acb_mat_hpd_solve_prec",
    "acb_mat_dense_hpd_solve_plan_apply_prec",
    "acb_mat_hpd_inv_prec",
    "acb_mat_solve_prec",
    "acb_mat_inv_prec",
    "acb_mat_sqr_prec",
    "acb_mat_det_prec",
    "acb_mat_trace_prec",
    "acb_mat_norm_fro_prec",
    "acb_mat_norm_1_prec",
    "acb_mat_norm_inf_prec",
    "acb_mat_triangular_solve_prec",
    "acb_mat_lu_prec",
    "acb_mat_qr_prec",
    "acb_mat_matmul_jit",
    "acb_mat_matvec_jit",
    "acb_mat_rmatvec_jit",
    "acb_mat_banded_matvec_jit",
    "acb_mat_matvec_cached_apply_jit",
    "acb_mat_rmatvec_cached_apply_jit",
    "acb_mat_hermitian_part_jit",
    "acb_mat_solve_jit",
    "acb_mat_inv_jit",
    "acb_mat_cho_jit",
    "acb_mat_ldl_jit",
    "acb_mat_hpd_solve_jit",
    "acb_mat_hpd_inv_jit",
    "acb_mat_sqr_jit",
    "acb_mat_det_jit",
    "acb_mat_trace_jit",
    "acb_mat_norm_fro_jit",
    "acb_mat_norm_1_jit",
    "acb_mat_norm_inf_jit",
    "acb_mat_triangular_solve_jit",
    "acb_mat_lu_jit",
    "acb_mat_qr_jit",
    "acb_mat_matmul_batch_fixed",
    "acb_mat_matmul_batch_padded",
    "acb_mat_matvec_batch_fixed",
    "acb_mat_matvec_batch_padded",
    "acb_mat_banded_matvec_batch_fixed",
    "acb_mat_banded_matvec_batch_padded",
    "acb_mat_matvec_cached_prepare_batch_fixed",
    "acb_mat_matvec_cached_prepare_batch_padded",
    "acb_mat_matvec_cached_apply_batch_fixed",
    "acb_mat_matvec_cached_apply_batch_padded",
    "acb_mat_hermitian_part_batch_fixed",
    "acb_mat_hermitian_part_batch_padded",
    "acb_mat_is_hermitian_batch_fixed",
    "acb_mat_is_hermitian_batch_padded",
    "acb_mat_is_hpd_batch_fixed",
    "acb_mat_is_hpd_batch_padded",
    "acb_mat_cho_batch_fixed",
    "acb_mat_cho_batch_padded",
    "acb_mat_ldl_batch_fixed",
    "acb_mat_ldl_batch_padded",
    "acb_mat_dense_hpd_solve_plan_prepare_batch_fixed",
    "acb_mat_dense_hpd_solve_plan_prepare_batch_padded",
    "acb_mat_hpd_solve_batch_fixed",
    "acb_mat_hpd_solve_batch_padded",
    "acb_mat_dense_hpd_solve_plan_apply_batch_fixed",
    "acb_mat_dense_hpd_solve_plan_apply_batch_padded",
    "acb_mat_hpd_inv_batch_fixed",
    "acb_mat_hpd_inv_batch_padded",
    "acb_mat_solve_batch_fixed",
    "acb_mat_solve_batch_padded",
    "acb_mat_inv_batch_fixed",
    "acb_mat_inv_batch_padded",
    "acb_mat_triangular_solve_batch_fixed",
    "acb_mat_triangular_solve_batch_padded",
    "acb_mat_lu_batch_fixed",
    "acb_mat_lu_batch_padded",
    "acb_mat_qr_batch_fixed",
    "acb_mat_qr_batch_padded",
    "acb_mat_det_batch_fixed",
    "acb_mat_det_batch_padded",
    "acb_mat_trace_batch_fixed",
    "acb_mat_trace_batch_padded",
    "acb_mat_sqr_batch_fixed",
    "acb_mat_sqr_batch_padded",
    "acb_mat_norm_fro_batch_fixed",
    "acb_mat_norm_fro_batch_padded",
    "acb_mat_norm_1_batch_fixed",
    "acb_mat_norm_1_batch_padded",
    "acb_mat_norm_inf_batch_fixed",
    "acb_mat_norm_inf_batch_padded",
    "acb_mat_matmul_batch_fixed_prec",
    "acb_mat_matmul_batch_padded_prec",
    "acb_mat_matvec_batch_fixed_prec",
    "acb_mat_matvec_batch_padded_prec",
    "acb_mat_banded_matvec_batch_fixed_prec",
    "acb_mat_banded_matvec_batch_padded_prec",
    "acb_mat_matvec_cached_prepare_batch_fixed_prec",
    "acb_mat_matvec_cached_prepare_batch_padded_prec",
    "acb_mat_matvec_cached_apply_batch_fixed_prec",
    "acb_mat_matvec_cached_apply_batch_padded_prec",
    "acb_mat_solve_batch_fixed_prec",
    "acb_mat_solve_batch_padded_prec",
    "acb_mat_inv_batch_fixed_prec",
    "acb_mat_inv_batch_padded_prec",
    "acb_mat_triangular_solve_batch_fixed_prec",
    "acb_mat_triangular_solve_batch_padded_prec",
    "acb_mat_lu_batch_fixed_prec",
    "acb_mat_lu_batch_padded_prec",
    "acb_mat_qr_batch_fixed_prec",
    "acb_mat_qr_batch_padded_prec",
    "acb_mat_det_batch_fixed_prec",
    "acb_mat_det_batch_padded_prec",
    "acb_mat_trace_batch_fixed_prec",
    "acb_mat_trace_batch_padded_prec",
    "acb_mat_sqr_batch_fixed_prec",
    "acb_mat_sqr_batch_padded_prec",
    "acb_mat_norm_fro_batch_fixed_prec",
    "acb_mat_norm_fro_batch_padded_prec",
    "acb_mat_norm_1_batch_fixed_prec",
    "acb_mat_norm_1_batch_padded_prec",
    "acb_mat_norm_inf_batch_fixed_prec",
    "acb_mat_norm_inf_batch_padded_prec",
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
    "acb_mat_operator_plan_prepare",
    "acb_mat_operator_rmatvec_plan_prepare",
    "acb_mat_operator_adjoint_plan_prepare",
]
