from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import checks
from . import double_interval as di
from . import kernel_helpers as kh
from . import mat_common



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


def _mid_symmetric_part(a: jax.Array) -> jax.Array:
    return mat_common.real_midpoint_symmetric_part(_mid_matrix(a))


def _mid_is_symmetric(a: jax.Array) -> jax.Array:
    return mat_common.real_midpoint_is_symmetric(_mid_matrix(a))


def _mid_cholesky(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    sym_mid = _mid_symmetric_part(a)
    chol = jnp.linalg.cholesky(sym_mid)
    ok = _mid_is_symmetric(a) & mat_common.lower_cholesky_finite(chol)
    return chol, ok


def _mid_spd_solve(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol, ok = _mid_cholesky(a)
    x = mat_common.lower_cholesky_solve(chol, _mid_rhs(b))
    return x, ok


def _mid_spd_inv(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol, ok = _mid_cholesky(a)
    eye = jnp.broadcast_to(jnp.eye(chol.shape[-1], dtype=chol.dtype), chol.shape)
    inv = mat_common.lower_cholesky_solve(chol, eye)
    return inv, ok


def _rhs_is_vector(x: jax.Array) -> bool:
    return int(jnp.asarray(x).ndim) == 2


def _rhs_rows_like(rhs: jax.Array, a: jax.Array) -> int:
    arr = jnp.asarray(rhs)
    return int(arr.shape[-2] if arr.ndim == a.ndim - 1 else arr.shape[-3])


def _rhs_batch_shape(rhs: jax.Array, rows: int) -> tuple[int, ...]:
    arr = jnp.asarray(rhs)
    if arr.ndim >= 3 and int(arr.shape[-3]) == rows:
        return tuple(int(x) for x in arr.shape[:-3])
    return tuple(int(x) for x in arr.shape[:-2])


def _broadcast_interval_matrix_batch(a: jax.Array, batch_shape: tuple[int, ...]) -> jax.Array:
    if not batch_shape or tuple(int(x) for x in a.shape[:-3]) == batch_shape:
        return a
    return jnp.broadcast_to(a, batch_shape + a.shape[-3:])


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


def _apply_perm_transpose_rhs(p: jax.Array, b: jax.Array) -> jax.Array:
    p_mid = jnp.swapaxes(di.midpoint(arb_mat_as_matrix(p)), -2, -1)
    b_mid = _mid_rhs(b)
    if b.ndim == p.ndim - 1:
        return mat_common.interval_from_point(jnp.einsum("...ij,...j->...i", p_mid, b_mid))
    return mat_common.interval_from_point(jnp.matmul(p_mid, b_mid))


def _abs_interval(x: jax.Array) -> jax.Array:
    lo = x[..., 0]
    hi = x[..., 1]
    mag = jnp.maximum(jnp.abs(lo), jnp.abs(hi))
    return di.interval(jnp.zeros_like(mag), di._above(mag))


def _rad_matrix(a: jax.Array) -> jax.Array:
    return di.ubound_radius(arb_mat_as_matrix(a))


def _interval_abs_upper(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return di._above(jnp.maximum(jnp.abs(x[..., 0]), jnp.abs(x[..., 1])))


def _hadamard_abs_det_bound(a: jax.Array) -> jax.Array:
    mags = _interval_abs_upper(arb_mat_as_matrix(a))
    row_norms = di._above(jnp.sqrt(jnp.sum(mags * mags, axis=-1)))
    return di._above(jnp.prod(row_norms, axis=-1))


def _sym_rad_matrix(a: jax.Array) -> jax.Array:
    rad = _rad_matrix(a)
    return 0.5 * (rad + jnp.swapaxes(rad, -2, -1))


def _sym_perturbation_bound(a: jax.Array) -> jax.Array:
    sym_rad = _sym_rad_matrix(a)
    return di._above(jnp.linalg.norm(sym_rad, ord="fro", axis=(-2, -1)))


def _tightened_real_eigvals_interval(values: jax.Array, bound: jax.Array) -> jax.Array:
    return di.interval(di._below(values - bound[..., None]), di._above(values + bound[..., None]))


def _tightened_real_eigvecs_interval(values: jax.Array, vectors: jax.Array, bound: jax.Array) -> jax.Array:
    n = vectors.shape[-1]
    if n == 1:
        radius = jnp.broadcast_to(bound[..., None], values.shape)
    else:
        diffs = jnp.abs(values[..., :, None] - values[..., None, :])
        huge = jnp.asarray(jnp.inf, dtype=values.dtype)
        diffs = diffs + jnp.eye(n, dtype=values.dtype) * huge
        gaps = jnp.min(diffs, axis=-1)
        tiny = jnp.asarray(64.0 * jnp.finfo(values.dtype).eps, dtype=values.dtype)
        radius = jnp.clip(bound[..., None] / jnp.maximum(gaps, tiny), 0.0, 1.0)
    return di.interval(
        di._below(vectors - radius[..., None, :]),
        di._above(vectors + radius[..., None, :]),
    )


def _band_mask(rows: int, cols: int, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    i = jnp.arange(rows)[:, None]
    j = jnp.arange(cols)[None, :]
    return (i - j <= lower_bandwidth) & (j - i <= upper_bandwidth)


def _apply_band_mask_interval(a: jax.Array, *, lower_bandwidth: int, upper_bandwidth: int) -> jax.Array:
    mask = _band_mask(a.shape[-2], a.shape[-2], lower_bandwidth, upper_bandwidth)
    zero = di.interval(jnp.zeros(a.shape[:-1], dtype=a.dtype), jnp.zeros(a.shape[:-1], dtype=a.dtype))
    return jnp.where(mask[..., None], a, zero)


def _mul_interval_matrix_rhs(a: jax.Array, x: jax.Array) -> jax.Array:
    x_arr = arb_mat_as_rhs(x)
    if x_arr.ndim == arb_mat_as_matrix(a).ndim - 1:
        return arb_mat_matvec_basic(a, x_arr)
    return arb_mat_matmul_basic(a, x_arr)


def _rigorous_linear_solve_enclosure(a: jax.Array, b: jax.Array, x_mid: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_rhs(b)
    mid = _mid_matrix(a)
    inv_mid = jnp.linalg.inv(mid)
    inv_abs = jnp.abs(inv_mid)
    rad = _rad_matrix(a)
    beta = jnp.linalg.norm(inv_abs @ rad, ord=jnp.inf, axis=(-2, -1))
    residual = di.fast_sub(b, _mul_interval_matrix_rhs(a, mat_common.interval_from_point(x_mid)))
    residual_abs = _interval_abs_upper(residual)
    correction_abs = (
        jnp.einsum("...ij,...j->...i", inv_abs, residual_abs)
        if residual_abs.ndim == inv_abs.ndim - 1
        else jnp.matmul(inv_abs, residual_abs)
    )
    denom = jnp.maximum(1.0 - beta, 64.0 * jnp.finfo(mid.dtype).eps)
    width = di._above(correction_abs / denom[(...,) + (None,) * (correction_abs.ndim - beta.ndim)])
    out = di.interval(di._below(x_mid - width), di._above(x_mid + width))
    finite = jnp.all(jnp.isfinite(inv_mid), axis=(-2, -1))
    safe = finite & (beta < 1.0)
    return jnp.where(safe[(...,) + (None,) * (out.ndim - safe.ndim)], out, mat_common.full_interval_like(out))


def _det_lipschitz_radius(a: jax.Array, det_mid: jax.Array) -> jax.Array:
    mid = _mid_matrix(a)
    delta = jnp.linalg.norm(_rad_matrix(a), ord=jnp.inf, axis=(-2, -1))
    growth = jnp.maximum(
        jnp.linalg.norm(mid, ord=1, axis=(-2, -1)),
        jnp.linalg.norm(mid, ord=jnp.inf, axis=(-2, -1)),
    )
    n = jnp.asarray(float(mid.shape[-1]), dtype=mid.dtype)
    return di._above(n * jnp.power(growth + delta, jnp.maximum(n - 1.0, 0.0)) * delta)


def _det_cofactor_lipschitz_radius(a: jax.Array) -> jax.Array:
    mags = _interval_abs_upper(arb_mat_as_matrix(a))
    row_norms = di._above(jnp.sqrt(jnp.sum(mags * mags, axis=-1)))
    col_norms = di._above(jnp.sqrt(jnp.sum(mags * mags, axis=-2)))
    if mags.shape[-1] == 1:
        cofactor_bound = jnp.ones_like(row_norms[..., 0])
    else:
        row_bound = di._above(jnp.prod(jnp.sort(row_norms, axis=-1)[..., 1:], axis=-1))
        col_bound = di._above(jnp.prod(jnp.sort(col_norms, axis=-1)[..., 1:], axis=-1))
        cofactor_bound = jnp.minimum(row_bound, col_bound)
    delta = di._above(jnp.linalg.norm(_rad_matrix(a), ord="fro", axis=(-2, -1)))
    n = jnp.asarray(float(mags.shape[-1]), dtype=delta.dtype)
    return di._above(n * cofactor_bound * delta)


def _basic_det_enclosure(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    mid = _mid_matrix(a)
    det_mid = jnp.linalg.det(mid)
    inv_mid = jnp.linalg.inv(mid)
    delta = jnp.linalg.norm(_rad_matrix(a), ord=jnp.inf, axis=(-2, -1))
    inv_norm = jnp.linalg.norm(jnp.abs(inv_mid), ord=jnp.inf, axis=(-2, -1))
    beta = inv_norm * delta
    rel = jnp.expm1(jnp.asarray(float(mid.shape[-1]), dtype=mid.dtype) * jnp.log1p(beta))
    tightened = di._above(jnp.abs(det_mid) * rel)
    lipschitz = _det_lipschitz_radius(a, det_mid)
    cofactor = _det_cofactor_lipschitz_radius(a)
    fallback = di._above(jnp.abs(det_mid) + _hadamard_abs_det_bound(a))
    radius = jnp.where(
        jnp.isfinite(beta) & (beta < 1.0) & jnp.isfinite(tightened),
        jnp.minimum(
            jnp.minimum(tightened, jnp.where(jnp.isfinite(lipschitz), lipschitz, tightened)),
            jnp.where(jnp.isfinite(cofactor), cofactor, tightened),
        ),
        jnp.minimum(
            jnp.where(jnp.isfinite(lipschitz), lipschitz, fallback),
            jnp.where(jnp.isfinite(cofactor), cofactor, fallback),
        ),
    )
    out = di.interval(di._below(det_mid - radius), di._above(det_mid + radius))
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def _rigorous_det_enclosure(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    basic = _basic_det_enclosure(a)
    mid = _mid_matrix(a)
    det_mid = jnp.linalg.det(mid)
    fallback = jnp.minimum(
        di._above(jnp.abs(det_mid) + _hadamard_abs_det_bound(a)),
        jnp.where(jnp.isfinite(_det_cofactor_lipschitz_radius(a)), _det_cofactor_lipschitz_radius(a), di._above(jnp.abs(det_mid) + _hadamard_abs_det_bound(a))),
    )
    basic_radius = 0.5 * (basic[..., 1] - basic[..., 0])
    radius = jnp.maximum(basic_radius, fallback)
    out = di.interval(di._below(det_mid - radius), di._above(det_mid + radius))
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def _widen_factor_interval(mid: jax.Array, radius: jax.Array) -> jax.Array:
    width = di._above(radius[..., None, None] * jnp.maximum(jnp.abs(mid), 1.0))
    out = di.interval(di._below(mid - width), di._above(mid + width))
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def _widen_triangular_interval(mid: jax.Array, radius: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    out = _widen_factor_interval(mid, radius)
    rows, cols = mid.shape[-2], mid.shape[-1]
    mask = _band_mask(rows, cols, rows - 1 if lower else 0, 0 if lower else cols - 1)
    zero = di.interval(jnp.zeros_like(mid), jnp.zeros_like(mid))
    out = jnp.where(mask[..., None], out, zero)
    if unit_diagonal:
        diag_mask = jnp.eye(rows, cols, dtype=bool)
        diag = di.interval(jnp.ones_like(mid), jnp.ones_like(mid))
        out = jnp.where(diag_mask[..., None], diag, out)
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def _dense_factor_radius(a: jax.Array, recon_mid: jax.Array) -> jax.Array:
    mid = _mid_matrix(a)
    scale = jnp.maximum(jnp.linalg.norm(mid, ord=jnp.inf, axis=(-2, -1)), 1.0)
    residual = di._above(jnp.linalg.norm(mid - recon_mid, ord=jnp.inf, axis=(-2, -1)))
    delta = di._above(jnp.linalg.norm(_rad_matrix(a), ord=jnp.inf, axis=(-2, -1)))
    return di._above((delta + residual + 64.0 * jnp.finfo(mid.dtype).eps) / scale)


def arb_mat_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    a = mat_common.as_interval_rect_matrix(a, "arb_mat.matmul.a")
    b = mat_common.as_interval_rect_matrix(b, "arb_mat.matmul.b")
    checks.check_equal(a.shape[-2], b.shape[-3], "arb_mat.matmul.inner")
    c = jnp.matmul(di.midpoint(a), di.midpoint(b))
    out = mat_common.interval_from_point(c)
    finite = jnp.all(jnp.isfinite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_matmul_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    a = mat_common.as_interval_rect_matrix(a, "arb_mat.matmul_basic.a")
    b = mat_common.as_interval_rect_matrix(b, "arb_mat.matmul_basic.b")
    checks.check_equal(a.shape[-2], b.shape[-3], "arb_mat.matmul_basic.inner")
    prods = di.fast_mul(a[..., :, :, None, :], b[..., None, :, :, :])
    out = mat_common.interval_sum(prods, axis=-2)
    finite = jnp.all(jnp.isfinite(out), axis=(-3, -2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_matvec(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_interval_rect_matrix(a, "arb_mat.matvec.a")
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "arb_mat.matvec.inner")
    y = jnp.einsum("...ij,...j->...i", di.midpoint(a), _mid_vector(x))
    out = mat_common.interval_from_point(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_matvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_interval_rect_matrix(a, "arb_mat.matvec_basic.a")
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "arb_mat.matvec_basic.inner")
    prods = di.fast_mul(a, x[..., None, :, :])
    out = mat_common.interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_rmatvec(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_interval_rect_matrix(a, "arb_mat.rmatvec.a")
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-3], x.shape[-2], "arb_mat.rmatvec.inner")
    y = jnp.einsum("...ji,...j->...i", di.midpoint(a), _mid_vector(x))
    out = mat_common.interval_from_point(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_rmatvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = mat_common.as_interval_rect_matrix(a, "arb_mat.rmatvec_basic.a")
    x = arb_mat_as_vector(x)
    checks.check_equal(a.shape[-3], x.shape[-2], "arb_mat.rmatvec_basic.inner")
    prods = di.fast_mul(jnp.swapaxes(a, -3, -2), x[..., None, :, :])
    out = mat_common.interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_matvec_cached_prepare(a: jax.Array) -> jax.Array:
    return mat_common.as_interval_rect_matrix(a, "arb_mat.matvec_cached_prepare")


def arb_mat_rmatvec_cached_prepare(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(mat_common.as_interval_rect_matrix(a, "arb_mat.rmatvec_cached_prepare"), -3, -2)


def arb_mat_matvec_cached_prepare_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_prepare(a), prec_bits)


def arb_mat_rmatvec_cached_prepare_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_cached_prepare(a), prec_bits)


def arb_mat_dense_matvec_plan_prepare(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return mat_common.dense_matvec_plan_from_matrix(a, algebra="arb", label="arb_mat.dense_matvec_plan_prepare")


def arb_mat_dense_matvec_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    matrix = di.round_interval_outward(mat_common.as_interval_rect_matrix(a, "arb_mat.dense_matvec_plan_prepare_prec"), prec_bits)
    return mat_common.dense_matvec_plan_from_matrix(matrix, algebra="arb", label="arb_mat.dense_matvec_plan_prepare_prec")


def arb_mat_matvec_cached_apply(cache: jax.Array, x: jax.Array) -> jax.Array:
    cache = mat_common.as_dense_matvec_plan(cache, algebra="arb", label="arb_mat.matvec_cached_apply")
    x = arb_mat_as_vector(x)
    matrix = _broadcast_interval_matrix_batch(mat_common.as_interval_rect_matrix(cache.matrix, "arb_mat.matvec_cached_apply.matrix"), tuple(int(v) for v in x.shape[:-2]))
    checks.check_equal(cache.cols, x.shape[-2], "arb_mat.matvec_cached_apply.inner")
    prods = di.fast_mul(matrix, x[..., None, :, :])
    out = mat_common.interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_dense_matvec_plan_apply(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply(plan, x)


def arb_mat_rmatvec_cached_apply(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_apply(cache, x)


def arb_mat_permutation_matrix(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    p = jnp.eye(perm.shape[-1], dtype=dtype)[perm]
    return mat_common.interval_from_point(p)


def arb_mat_transpose(a: jax.Array) -> jax.Array:
    return jnp.swapaxes(mat_common.as_interval_rect_matrix(a, "arb_mat.transpose"), -3, -2)


def arb_mat_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return di.fast_add(arb_mat_as_matrix(a), arb_mat_as_matrix(b))


def arb_mat_sub(a: jax.Array, b: jax.Array) -> jax.Array:
    return di.fast_sub(arb_mat_as_matrix(a), arb_mat_as_matrix(b))


def arb_mat_neg(a: jax.Array) -> jax.Array:
    return di.neg(arb_mat_as_matrix(a))


def arb_mat_mul_entrywise(a: jax.Array, b: jax.Array) -> jax.Array:
    return di.fast_mul(arb_mat_as_matrix(a), arb_mat_as_matrix(b))


def arb_mat_scalar_mul_arb(a: jax.Array, x: jax.Array) -> jax.Array:
    return di.fast_mul(arb_mat_as_matrix(a), di.as_interval(x)[..., None, None, :])


def arb_mat_scalar_div_arb(a: jax.Array, x: jax.Array) -> jax.Array:
    return di.fast_div(arb_mat_as_matrix(a), di.as_interval(x)[..., None, None, :])


def arb_mat_scalar_mul_si(a: jax.Array, x: float | int) -> jax.Array:
    xx = jnp.asarray(x, dtype=arb_mat_as_matrix(a).dtype)
    return arb_mat_scalar_mul_arb(a, di.interval(xx, xx))


def arb_mat_scalar_div_si(a: jax.Array, x: float | int) -> jax.Array:
    xx = jnp.asarray(x, dtype=arb_mat_as_matrix(a).dtype)
    return arb_mat_scalar_div_arb(a, di.interval(xx, xx))


def arb_mat_symmetric_part(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    return di.fast_mul(di.interval(jnp.full(a.shape[:-1], 0.5, dtype=a.dtype), jnp.full(a.shape[:-1], 0.5, dtype=a.dtype)), a + arb_mat_transpose(a))


def arb_mat_one(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return arb_mat_identity(n, dtype=dtype)


def arb_mat_ones(rows: int, cols: int | None = None, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    cols = rows if cols is None else cols
    return mat_common.interval_from_point(jnp.ones((rows, cols), dtype=dtype))


def arb_mat_companion(coeffs: jax.Array) -> jax.Array:
    coeffs = arb_mat_as_vector(coeffs)
    return mat_common.interval_from_point(mat_common.companion_matrix(_mid_vector(coeffs)))


def arb_mat_hilbert(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return mat_common.interval_from_point(mat_common.hilbert_matrix(n, dtype=dtype))


def arb_mat_pascal(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return mat_common.interval_from_point(mat_common.pascal_matrix(n, dtype=dtype))


def arb_mat_stirling(n: int, *, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return mat_common.interval_from_point(mat_common.stirling2_matrix(n, dtype=dtype))


def arb_mat_nrows(a: jax.Array) -> int:
    return int(arb_mat_as_matrix(a).shape[-3])


def arb_mat_ncols(a: jax.Array) -> int:
    return int(arb_mat_as_matrix(a).shape[-2])


def arb_mat_entry(a: jax.Array, row: int, col: int) -> jax.Array:
    return arb_mat_as_matrix(a)[..., row, col, :]


def arb_mat_eq(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = arb_mat_as_matrix(a)
    bb = arb_mat_as_matrix(b)
    return jnp.all(mat_common.interval_equal(aa, bb), axis=(-2, -1))


def arb_mat_equal(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_eq(a, b)


def arb_mat_ne(a: jax.Array, b: jax.Array) -> jax.Array:
    return ~arb_mat_eq(a, b)


def arb_mat_contains(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = arb_mat_as_matrix(a)
    bb = arb_mat_as_matrix(b)
    return jnp.all(di.contains(aa, bb), axis=(-2, -1))


def arb_mat_overlaps(a: jax.Array, b: jax.Array) -> jax.Array:
    aa = arb_mat_as_matrix(a)
    bb = arb_mat_as_matrix(b)
    return jnp.all(mat_common.interval_overlaps(aa, bb), axis=(-2, -1))


def arb_mat_is_symmetric(a: jax.Array) -> jax.Array:
    return _mid_is_symmetric(a)


def arb_mat_is_spd(a: jax.Array) -> jax.Array:
    _, ok = _mid_cholesky(a)
    lam_min = jnp.min(jnp.linalg.eigvalsh(_mid_symmetric_part(a)), axis=-1)
    return ok & (lam_min > _sym_perturbation_bound(a))


def arb_mat_is_square(a: jax.Array) -> jax.Array:
    aa = arb_mat_as_matrix(a)
    return aa.shape[-3] == aa.shape[-2]


def arb_mat_is_diag(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_diagonal(_mid_matrix(a))


def arb_mat_is_tril(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_mid_matrix(a), lower=True)


def arb_mat_is_triu(a: jax.Array) -> jax.Array:
    return mat_common.midpoint_is_triangular(_mid_matrix(a), lower=False)


def arb_mat_is_zero(a: jax.Array) -> jax.Array:
    return jnp.all(mat_common.interval_is_zero(arb_mat_as_matrix(a)), axis=(-2, -1))


def arb_mat_is_finite(a: jax.Array) -> jax.Array:
    return jnp.all(mat_common.interval_is_finite(arb_mat_as_matrix(a)), axis=(-2, -1))


def arb_mat_is_exact(a: jax.Array) -> jax.Array:
    return jnp.all(mat_common.interval_is_exact(arb_mat_as_matrix(a)), axis=(-2, -1))


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
            block = mat_common.as_interval_rect_matrix(block, "arb_mat.block_block")
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
        row_chunks.append(jnp.concatenate([mat_common.as_interval_rect_matrix(block, "arb_mat.block_assemble") for block in row], axis=-2))
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


def arb_mat_cho(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    chol, ok = _mid_cholesky(a)
    lam_min = jnp.min(jnp.linalg.eigvalsh(_mid_symmetric_part(a)), axis=-1)
    ok = ok & (lam_min > _sym_perturbation_bound(a))
    out = mat_common.interval_from_point(chol)
    return jnp.where(ok[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_cho_basic(a: jax.Array) -> jax.Array:
    return arb_mat_cho(a)


def arb_mat_ldl(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    chol, ok = _mid_cholesky(a)
    diag = jnp.diagonal(chol, axis1=-2, axis2=-1)
    l = chol / diag[..., None, :]
    d = diag * diag
    l_out = mat_common.interval_from_point(l)
    d_out = mat_common.interval_from_point(d)
    mask_l = ok[..., None, None, None]
    mask_d = ok[..., None, None]
    return jnp.where(mask_l, l_out, mat_common.full_interval_like(l_out)), jnp.where(mask_d, d_out, mat_common.full_interval_like(d_out))


def arb_mat_ldl_basic(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_mat_ldl(a)


def arb_mat_charpoly(a: jax.Array) -> jax.Array:
    mid = _mid_matrix(a)
    coeffs_general = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=False)
    coeffs_symmetric = mat_common.characteristic_polynomial_from_matrix(mid, hermitian=True)
    mask = _mid_is_symmetric(a)
    coeffs = jnp.where(mask[..., None], coeffs_symmetric, coeffs_general)
    return mat_common.interval_from_point(jnp.real(coeffs))


def arb_mat_pow_ui(a: jax.Array, n: int) -> jax.Array:
    return mat_common.interval_from_point(mat_common.matrix_power_ui(_mid_matrix(a), n))


def arb_mat_exp(a: jax.Array) -> jax.Array:
    mid = _mid_matrix(a)
    exp_general = mat_common.matrix_exp(mid, hermitian=False)
    exp_symmetric = mat_common.matrix_exp(mid, hermitian=True)
    exp_mid = jnp.where(_mid_is_symmetric(a)[..., None, None], exp_symmetric, exp_general)
    return mat_common.interval_from_point(jnp.real(exp_mid))


def arb_mat_eigvalsh(a: jax.Array) -> jax.Array:
    values, _ = mat_common.real_symmetric_eigh(_mid_matrix(a))
    out = _tightened_real_eigvals_interval(values, _sym_perturbation_bound(a))
    finite = mat_common.interval_is_finite(out)
    return jnp.where(finite[..., None], out, mat_common.full_interval_like(out))


def arb_mat_eigh(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    values, vectors = mat_common.real_symmetric_eigh(_mid_matrix(a))
    bound = _sym_perturbation_bound(a)
    values_out = _tightened_real_eigvals_interval(values, bound)
    vectors_out = _tightened_real_eigvecs_interval(values, vectors, bound)
    val_finite = mat_common.interval_is_finite(values_out)
    vec_finite = jnp.all(mat_common.interval_is_finite(vectors_out), axis=(-2, -1))
    return (
        jnp.where(val_finite[..., None], values_out, mat_common.full_interval_like(values_out)),
        jnp.where(vec_finite[..., None, None, None], vectors_out, mat_common.full_interval_like(vectors_out)),
    )


def arb_mat_dense_spd_solve_plan_prepare(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    factor = arb_mat_cho(a)
    return mat_common.dense_cholesky_solve_plan_from_factor(
        factor,
        algebra="arb",
        structure="symmetric",
        label="arb_mat.dense_spd_solve_plan_prepare",
    )


def arb_mat_spd_solve(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        plan = mat_common.as_dense_cholesky_solve_plan(
            a_or_plan,
            algebra="arb",
            structure="symmetric",
            label="arb_mat.spd_solve",
        )
    else:
        plan = arb_mat_dense_spd_solve_plan_prepare(a_or_plan)
    b = arb_mat_as_rhs(b)
    factor = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.factor), _rhs_batch_shape(b, plan.rows))
    checks.check_equal(plan.rows, _rhs_rows_like(b, factor), "arb_mat.spd_solve.inner")
    x = mat_common.lower_cholesky_solve(_mid_matrix(factor), _mid_rhs(b))
    out = mat_common.interval_from_point(x)
    finite = _finite_mask_from_point(x) & jnp.all(mat_common.interval_is_finite(factor), axis=(-2, -1))
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))


def arb_mat_spd_solve_basic(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return arb_mat_spd_solve(a_or_plan, b)


def arb_mat_dense_spd_solve_plan_apply(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return arb_mat_spd_solve(plan, b)


def arb_mat_spd_inv(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        plan = mat_common.as_dense_cholesky_solve_plan(
            a_or_plan,
            algebra="arb",
            structure="symmetric",
            label="arb_mat.spd_inv",
        )
    else:
        plan = arb_mat_dense_spd_solve_plan_prepare(a_or_plan)
    factor = arb_mat_as_matrix(plan.factor)
    inv_mid = mat_common.lower_cholesky_solve(
        _mid_matrix(factor),
        jnp.broadcast_to(jnp.eye(plan.rows, dtype=factor.dtype), _mid_matrix(factor).shape),
    )
    out = mat_common.interval_from_point(inv_mid)
    finite = jnp.all(jnp.isfinite(inv_mid), axis=(-2, -1)) & jnp.all(mat_common.interval_is_finite(factor), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_spd_inv_basic(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return arb_mat_spd_inv(a_or_plan)


def arb_mat_solve_tril(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_triangular_solve(a, b, lower=True, unit_diagonal=unit_diagonal)


def arb_mat_solve_triu(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_triangular_solve(a, b, lower=False, unit_diagonal=unit_diagonal)


def arb_mat_solve_lu(a_or_plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array] | jax.Array, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        return arb_mat_lu_solve(a_or_plan, b)
    return arb_mat_lu_solve(arb_mat_dense_lu_solve_plan_prepare(a_or_plan), b)


def arb_mat_solve_lu_precomp(plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    return arb_mat_lu_solve(plan, b)


def arb_mat_solve_transpose(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        b = arb_mat_as_rhs(b)
        factor = _broadcast_interval_matrix_batch(arb_mat_as_matrix(a_or_plan.factor), _rhs_batch_shape(b, a_or_plan.rows))
        checks.check_equal(a_or_plan.rows, _rhs_rows_like(b, factor), "arb_mat.solve_transpose.inner_spd")
        x = mat_common.lower_cholesky_solve_transpose(_mid_matrix(factor), _mid_rhs(b))
        out = mat_common.interval_from_point(x)
        finite = _finite_mask_from_point(x)
        return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))
    if isinstance(a_or_plan, mat_common.DenseLUSolvePlan) or isinstance(a_or_plan, tuple):
        plan = mat_common.as_dense_lu_solve_plan(a_or_plan, algebra="arb", label="arb_mat.solve_transpose")
        b = arb_mat_as_rhs(b)
        batch_shape = _rhs_batch_shape(b, plan.rows)
        p = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.p), batch_shape)
        l = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.l), batch_shape)
        u = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.u), batch_shape)
        checks.check_equal(plan.rows, _rhs_rows_like(b, p), "arb_mat.solve_transpose.inner_lu")
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
        out = mat_common.interval_from_point(x)
        finite = _finite_mask_from_point(x)
        return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))
    a = arb_mat_as_matrix(a_or_plan)
    b = arb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "arb_mat.solve_transpose.inner")
    a_mid = _mid_matrix(a)
    b_mid = _mid_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    x = jnp.linalg.solve(jnp.swapaxes(a_mid, -2, -1), b_mid[..., None] if vector_rhs else b_mid)
    if vector_rhs:
        x = x[..., 0]
    out = mat_common.interval_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))


def arb_mat_solve_add(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        solved = arb_mat_spd_solve(a_or_plan, b)
    elif isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        solved = arb_mat_solve_lu(a_or_plan, b)
    else:
        solved = arb_mat_solve(a_or_plan, b)
    return di.fast_add(arb_mat_as_rhs(y), solved)


def arb_mat_solve_transpose_add(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return di.fast_add(arb_mat_as_rhs(y), arb_mat_solve_transpose(a_or_plan, b))


def arb_mat_mat_solve(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return arb_mat_spd_solve(a_or_plan, b)
    return arb_mat_solve_lu(a_or_plan, b) if isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)) else arb_mat_solve(a_or_plan, b)


def arb_mat_mat_solve_transpose(a_or_plan, b: jax.Array) -> jax.Array:
    return arb_mat_solve_transpose(a_or_plan, b)


def arb_mat_solve_cho_precomp(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_spd_solve(plan, b)


def arb_mat_solve_ldl_precomp(plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_spd_solve(plan, b)


def arb_mat_inv_cho_precomp(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return arb_mat_spd_inv(plan)


def arb_mat_inv_ldl_precomp(plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return arb_mat_spd_inv(plan)


def arb_mat_solve(a: jax.Array, b: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_rhs(b)
    checks.check_equal(a.shape[-2], _rhs_rows_like(b, a), "arb_mat.solve.inner")
    a_mid = _mid_matrix(a)
    b_mid = _mid_rhs(b)
    vector_rhs = b_mid.ndim == a_mid.ndim - 1
    x_general = jnp.linalg.solve(a_mid, b_mid[..., None] if vector_rhs else b_mid)
    if vector_rhs:
        x_general = x_general[..., 0]
    x_spd, spd_ok = _mid_spd_solve(a, b)
    x = jnp.where(spd_ok[(...,) + (None,) * (x_general.ndim - spd_ok.ndim)], x_spd, x_general)
    out = mat_common.interval_from_point(x)
    finite = _finite_mask_from_point(x)
    return jnp.where(finite[(...,) + (None,) * (out.ndim - finite.ndim)], out, mat_common.full_interval_like(out))


def arb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_solve(a, b)


def arb_mat_inv(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    inv_general = jnp.linalg.inv(_mid_matrix(a))
    inv_spd, spd_ok = _mid_spd_inv(a)
    inv = jnp.where(spd_ok[..., None, None], inv_spd, inv_general)
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
        out = _basic_det_enclosure(a)
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
    a = arb_mat_as_matrix(a)
    n = a.shape[-2]
    if n <= 3:
        return arb_mat_det_basic(a)
    return _rigorous_det_enclosure(a)


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


def arb_mat_lu_rigorous(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    mid = _mid_matrix(a)
    lu, _, perm = lax.linalg.lu(mid)
    n = mid.shape[-1]
    eye = jnp.eye(n, dtype=mid.dtype)
    p_mid = eye[perm]
    l_mid = jnp.tril(lu, k=-1) + eye
    u_mid = jnp.triu(lu)
    recon = jnp.swapaxes(p_mid, -2, -1) @ (l_mid @ u_mid)
    radius = _dense_factor_radius(a, recon)
    p_out = mat_common.interval_from_point(p_mid)
    u_scale = jnp.maximum(jnp.linalg.norm(u_mid, ord=jnp.inf, axis=(-2, -1)), 1.0)
    l_scale = jnp.maximum(jnp.linalg.norm(l_mid, ord=jnp.inf, axis=(-2, -1)), 1.0)
    l_out = _widen_triangular_interval(l_mid, di._above(radius / u_scale), lower=True, unit_diagonal=True)
    u_out = _widen_triangular_interval(u_mid, di._above(radius / l_scale), lower=False)
    return p_out, l_out, u_out


def arb_mat_dense_lu_solve_plan_prepare(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    p, l, u = arb_mat_lu(a)
    return mat_common.dense_lu_solve_plan_from_factors(p, l, u, algebra="arb", label="arb_mat.dense_lu_solve_plan_prepare")


def arb_mat_dense_lu_solve_plan_prepare_rigorous(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    p, l, u = arb_mat_lu_rigorous(a)
    return mat_common.dense_lu_solve_plan_from_factors(p, l, u, algebra="arb", label="arb_mat.dense_lu_solve_plan_prepare_rigorous")


def arb_mat_dense_lu_solve_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    p, l, u = arb_mat_lu_prec(a, prec_bits=prec_bits)
    return mat_common.dense_lu_solve_plan_from_factors(p, l, u, algebra="arb", label="arb_mat.dense_lu_solve_plan_prepare_prec")


def arb_mat_lu_solve(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    plan = mat_common.as_dense_lu_solve_plan(plan, algebra="arb", label="arb_mat.lu_solve")
    b = arb_mat_as_rhs(b)
    batch_shape = _rhs_batch_shape(b, plan.rows)
    p = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.p), batch_shape)
    l = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.l), batch_shape)
    u = _broadcast_interval_matrix_batch(arb_mat_as_matrix(plan.u), batch_shape)
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


def arb_mat_qr_rigorous(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    q_mid, r_mid = jnp.linalg.qr(_mid_matrix(a))
    recon = q_mid @ r_mid
    orth = q_mid.T @ q_mid - jnp.eye(q_mid.shape[-1], dtype=q_mid.dtype)
    radius = _dense_factor_radius(a, recon) + di._above(jnp.linalg.norm(orth, ord=jnp.inf, axis=(-2, -1)))
    q_out = _widen_factor_interval(q_mid, radius)
    r_scale = jnp.maximum(jnp.linalg.norm(r_mid, ord=jnp.inf, axis=(-2, -1)), 1.0)
    r_out = _widen_triangular_interval(r_mid, di._above(radius / r_scale), lower=False)
    return q_out, r_out


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


def arb_mat_symmetric_part_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_symmetric_part(a)


def arb_mat_is_symmetric_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_symmetric(a)


def arb_mat_is_spd_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_spd(a)


def arb_mat_cho_rigorous(a: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    chol, ok = _mid_cholesky(a)
    recon = chol @ jnp.swapaxes(chol, -2, -1)
    radius = _dense_factor_radius(a, recon)
    chol_scale = jnp.maximum(jnp.linalg.norm(chol, ord=jnp.inf, axis=(-2, -1)), 1.0)
    out = _widen_triangular_interval(chol, di._above(radius / chol_scale), lower=True)
    lam_min = jnp.min(jnp.linalg.eigvalsh(_mid_symmetric_part(a)), axis=-1)
    ok = ok & (lam_min > _sym_perturbation_bound(a))
    return jnp.where(ok[..., None, None, None], out, mat_common.full_interval_like(out))


def arb_mat_ldl_rigorous(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    chol, ok = _mid_cholesky(a)
    diag = jnp.diagonal(chol, axis1=-2, axis2=-1)
    l_mid = chol / diag[..., None, :]
    d_mid = diag * diag
    recon = l_mid @ (d_mid[..., None, :] * jnp.swapaxes(l_mid, -2, -1))
    radius = _dense_factor_radius(a, recon)
    l_scale = jnp.maximum(jnp.linalg.norm(l_mid, ord=jnp.inf, axis=(-2, -1)), 1.0)
    l_out = _widen_triangular_interval(l_mid, di._above(radius / l_scale), lower=True, unit_diagonal=True)
    d_width = di._above(radius[..., None] * jnp.maximum(jnp.abs(d_mid), 1.0))
    d_out = di.interval(di._below(d_mid - d_width), di._above(d_mid + d_width))
    mask_l = ok[..., None, None, None]
    mask_d = ok[..., None, None]
    return jnp.where(mask_l, l_out, mat_common.full_interval_like(l_out)), jnp.where(mask_d, d_out, mat_common.full_interval_like(d_out))


def arb_mat_eigvalsh_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_eigvalsh(a)


def arb_mat_eigh_rigorous(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_mat_eigh(a)


def arb_mat_charpoly_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_charpoly(a)


def arb_mat_pow_ui_rigorous(a: jax.Array, n: int) -> jax.Array:
    return arb_mat_pow_ui(a, n)


def arb_mat_exp_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_exp(a)


def arb_mat_spd_solve_rigorous(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    solved = arb_mat_spd_solve(a_or_plan, b)
    x_mid = _mid_rhs(solved)
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        plan = mat_common.as_dense_cholesky_solve_plan(
            a_or_plan,
            algebra="arb",
            structure="symmetric",
            label="arb_mat.spd_solve_rigorous",
        )
        factor = arb_mat_as_matrix(plan.factor)
        a_recon = arb_mat_matmul_basic(factor, arb_mat_transpose(factor))
        return _rigorous_linear_solve_enclosure(a_recon, b, x_mid)
    return _rigorous_linear_solve_enclosure(a_or_plan, b, x_mid)


def arb_mat_spd_inv_rigorous(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    inv = arb_mat_spd_inv(a_or_plan)
    x_mid = _mid_matrix(inv)
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        plan = mat_common.as_dense_cholesky_solve_plan(
            a_or_plan,
            algebra="arb",
            structure="symmetric",
            label="arb_mat.spd_inv_rigorous",
        )
        factor = arb_mat_as_matrix(plan.factor)
        a_recon = arb_mat_matmul_basic(factor, arb_mat_transpose(factor))
        return _rigorous_linear_solve_enclosure(a_recon, arb_mat_identity(plan.rows, dtype=factor.dtype), x_mid)
    a = arb_mat_as_matrix(a_or_plan)
    return _rigorous_linear_solve_enclosure(a, arb_mat_identity(a.shape[-2], dtype=a.dtype), x_mid)


def arb_mat_solve_tril_rigorous(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_solve_tril(a, b, unit_diagonal=unit_diagonal)


def arb_mat_solve_triu_rigorous(a: jax.Array, b: jax.Array, *, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_solve_triu(a, b, unit_diagonal=unit_diagonal)


def arb_mat_solve_lu_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    solved = arb_mat_solve_lu(a_or_plan, b)
    x_mid = _mid_rhs(solved)
    if isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        plan = mat_common.as_dense_lu_solve_plan(a_or_plan, algebra="arb", label="arb_mat.solve_lu_rigorous")
        lu = arb_mat_matmul_basic(plan.l, plan.u)
        a_recon = arb_mat_matmul_basic(arb_mat_transpose(plan.p), lu)
        return _rigorous_linear_solve_enclosure(a_recon, b, x_mid)
    return _rigorous_linear_solve_enclosure(a_or_plan, b, x_mid)


def arb_mat_solve_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    solved = arb_mat_solve(a, b)
    return _rigorous_linear_solve_enclosure(a, b, _mid_rhs(solved))


def arb_mat_inv_rigorous(a: jax.Array) -> jax.Array:
    inv = arb_mat_inv(a)
    a = arb_mat_as_matrix(a)
    return _rigorous_linear_solve_enclosure(a, arb_mat_identity(a.shape[-2], dtype=a.dtype), _mid_matrix(inv))


def arb_mat_solve_transpose_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    return arb_mat_solve_transpose(a_or_plan, b)


def arb_mat_solve_add_rigorous(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return arb_mat_solve_add(a_or_plan, b, y)


def arb_mat_solve_transpose_add_rigorous(a_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return arb_mat_solve_transpose_add(a_or_plan, b, y)


def arb_mat_mat_solve_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return arb_mat_spd_solve_rigorous(a_or_plan, b)
    if isinstance(a_or_plan, (mat_common.DenseLUSolvePlan, tuple)):
        return arb_mat_solve_lu_rigorous(a_or_plan, b)
    solved = arb_mat_mat_solve(a_or_plan, b)
    return _rigorous_linear_solve_enclosure(a_or_plan, b, _mid_rhs(solved))


def arb_mat_mat_solve_transpose_rigorous(a_or_plan, b: jax.Array) -> jax.Array:
    return arb_mat_mat_solve_transpose(a_or_plan, b)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matmul_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matmul(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_rmatvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec(a, x), prec_bits)


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
def arb_mat_rmatvec_cached_apply_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_cached_apply(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_symmetric_part_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_symmetric_part(a), prec_bits)


def arb_mat_is_symmetric_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_symmetric(a)


def arb_mat_is_spd_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_spd(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_cho_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_cho(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_ldl_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    l, d = arb_mat_ldl(a)
    return di.round_interval_outward(l, prec_bits), di.round_interval_outward(d, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_eigvalsh_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_eigvalsh(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_eigh_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    values, vectors = arb_mat_eigh(a)
    return di.round_interval_outward(values, prec_bits), di.round_interval_outward(vectors, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_charpoly_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_charpoly(a), prec_bits)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def arb_mat_pow_ui_prec(a: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_pow_ui(a, n), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_exp_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_exp(a), prec_bits)


def arb_mat_dense_spd_solve_plan_prepare_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseCholeskySolvePlan:
    factor = arb_mat_cho_prec(a, prec_bits=prec_bits)
    return mat_common.dense_cholesky_solve_plan_from_factor(
        factor,
        algebra="arb",
        structure="symmetric",
        label="arb_mat.dense_spd_solve_plan_prepare_prec",
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_spd_solve_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_spd_solve(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_spd_inv_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_spd_inv(a_or_plan), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "unit_diagonal"))
def arb_mat_solve_tril_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_tril(a, b, unit_diagonal=unit_diagonal), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "unit_diagonal"))
def arb_mat_solve_triu_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_triu(a, b, unit_diagonal=unit_diagonal), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_lu_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_lu(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_transpose_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_transpose(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_add_prec(a_or_plan, b: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_add(a_or_plan, b, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_transpose_add_prec(a_or_plan, b: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve_transpose_add(a_or_plan, b, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_mat_solve_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_mat_solve(a_or_plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_mat_solve_transpose_prec(a_or_plan, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_mat_solve_transpose(a_or_plan, b), prec_bits)


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


def arb_mat_add_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_add(a, b)


def arb_mat_sub_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_sub(a, b)


def arb_mat_neg_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_neg(a)


def arb_mat_mul_entrywise_rigorous(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_mul_entrywise(a, b)


def arb_mat_is_diag_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_diag(a)


def arb_mat_is_tril_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_tril(a)


def arb_mat_is_triu_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_triu(a)


def arb_mat_is_zero_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_zero(a)


def arb_mat_is_finite_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_finite(a)


def arb_mat_is_exact_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_is_exact(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_add_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_add(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_sub_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_sub(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_neg_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_neg(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_mul_entrywise_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_mul_entrywise(a, b), prec_bits)


def arb_mat_is_diag_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_diag(a)


def arb_mat_is_tril_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_tril(a)


def arb_mat_is_triu_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_triu(a)


def arb_mat_is_zero_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_zero(a)


def arb_mat_is_finite_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_finite(a)


def arb_mat_is_exact_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_exact(a)


arb_mat_matmul_jit = jax.jit(arb_mat_matmul)
arb_mat_matvec_jit = jax.jit(arb_mat_matvec)
arb_mat_rmatvec_jit = jax.jit(arb_mat_rmatvec)
arb_mat_banded_matvec_jit = jax.jit(arb_mat_banded_matvec, static_argnames=("lower_bandwidth", "upper_bandwidth"))
arb_mat_matvec_cached_apply_jit = jax.jit(arb_mat_matvec_cached_apply)
arb_mat_rmatvec_cached_apply_jit = jax.jit(arb_mat_rmatvec_cached_apply)
arb_mat_symmetric_part_jit = jax.jit(arb_mat_symmetric_part)
arb_mat_solve_jit = jax.jit(arb_mat_solve)
arb_mat_inv_jit = jax.jit(arb_mat_inv)
arb_mat_cho_jit = jax.jit(arb_mat_cho)
arb_mat_ldl_jit = jax.jit(arb_mat_ldl)
arb_mat_spd_solve_jit = jax.jit(arb_mat_spd_solve)
arb_mat_spd_inv_jit = jax.jit(arb_mat_spd_inv)
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


def arb_mat_rmatvec_batch_fixed(a: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_rmatvec(a, x)


def arb_mat_matvec_batch_padded(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return arb_mat_matvec(*call_args)


def arb_mat_rmatvec_batch_padded(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return arb_mat_rmatvec(*call_args)


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


def arb_mat_rmatvec_cached_apply_batch_fixed(cache: jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_rmatvec_cached_apply(cache, x)


def arb_mat_matvec_cached_apply_batch_padded(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((cache, x), pad_to=pad_to)
    return arb_mat_matvec_cached_apply(*call_args)


def arb_mat_rmatvec_cached_apply_batch_padded(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((cache, x), pad_to=pad_to)
    return arb_mat_rmatvec_cached_apply(*call_args)


def arb_mat_matvec_cached_prepare_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_matvec_cached_prepare(a)


def arb_mat_rmatvec_cached_prepare_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_rmatvec_cached_prepare(a)


def arb_mat_matvec_cached_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return arb_mat_matvec_cached_prepare(*call_args)


def arb_mat_rmatvec_cached_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return arb_mat_rmatvec_cached_prepare(*call_args)


def arb_mat_dense_matvec_plan_prepare_batch_fixed(a: jax.Array) -> mat_common.DenseMatvecPlan:
    return arb_mat_dense_matvec_plan_prepare(a)


def arb_mat_dense_matvec_plan_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> mat_common.DenseMatvecPlan:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return arb_mat_dense_matvec_plan_prepare(*call_args)


def arb_mat_dense_matvec_plan_apply_batch_fixed(plan: mat_common.DenseMatvecPlan | jax.Array, x: jax.Array) -> jax.Array:
    return arb_mat_dense_matvec_plan_apply(plan, x)


def arb_mat_dense_matvec_plan_apply_batch_padded(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (x_pad,), _ = mat_common.pad_batch_repeat_last((x,), pad_to=pad_to)
    return arb_mat_dense_matvec_plan_apply(plan, x_pad)


def arb_mat_symmetric_part_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_symmetric_part(a)


def arb_mat_symmetric_part_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_symmetric_part(*call_args)


def arb_mat_is_symmetric_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_is_symmetric(a)


def arb_mat_is_symmetric_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_is_symmetric(*call_args)


def arb_mat_is_spd_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_is_spd(a)


def arb_mat_is_spd_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_is_spd(*call_args)


def arb_mat_cho_batch_fixed(a: jax.Array) -> jax.Array:
    return arb_mat_cho(a)


def arb_mat_cho_batch_padded(a: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_cho(*call_args)


def arb_mat_ldl_batch_fixed(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_mat_ldl(a)


def arb_mat_ldl_batch_padded(a: jax.Array, *, pad_to: int) -> tuple[jax.Array, jax.Array]:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a,), pad_to=pad_to)
    return arb_mat_ldl(*call_args)


def arb_mat_dense_spd_solve_plan_prepare_batch_fixed(a: jax.Array) -> mat_common.DenseCholeskySolvePlan:
    return arb_mat_dense_spd_solve_plan_prepare(a)


def arb_mat_dense_spd_solve_plan_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> mat_common.DenseCholeskySolvePlan:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return arb_mat_dense_spd_solve_plan_prepare(*call_args)


def arb_mat_spd_solve_batch_fixed(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_spd_solve(a_or_plan, b)


def arb_mat_spd_solve_batch_padded(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = mat_common.pad_batch_repeat_last((b,), pad_to=pad_to)
    return arb_mat_spd_solve(a_or_plan, b_pad)


def arb_mat_dense_spd_solve_plan_apply_batch_fixed(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
) -> jax.Array:
    return arb_mat_dense_spd_solve_plan_apply(plan, b)


def arb_mat_dense_spd_solve_plan_apply_batch_padded(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = mat_common.pad_batch_repeat_last((b,), pad_to=pad_to)
    return arb_mat_dense_spd_solve_plan_apply(plan, b_pad)


def arb_mat_spd_inv_batch_fixed(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array) -> jax.Array:
    return arb_mat_spd_inv(a_or_plan)


def arb_mat_spd_inv_batch_padded(a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array, *, pad_to: int) -> jax.Array:
    if isinstance(a_or_plan, mat_common.DenseCholeskySolvePlan):
        return arb_mat_spd_inv(a_or_plan)
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a_or_plan,), pad_to=pad_to)
    return arb_mat_spd_inv(*call_args)


def arb_mat_symmetric_part_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_symmetric_part_batch_fixed(a), prec_bits)


def arb_mat_symmetric_part_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_symmetric_part_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_is_symmetric_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_symmetric_batch_fixed(a)


def arb_mat_is_symmetric_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_symmetric_batch_padded(a, pad_to=pad_to)


def arb_mat_is_spd_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_spd_batch_fixed(a)


def arb_mat_is_spd_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    del prec_bits
    return arb_mat_is_spd_batch_padded(a, pad_to=pad_to)


def arb_mat_cho_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_cho_batch_fixed(a), prec_bits)


def arb_mat_cho_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_cho_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_ldl_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    l, d = arb_mat_ldl_batch_fixed(a)
    return di.round_interval_outward(l, prec_bits), di.round_interval_outward(d, prec_bits)


def arb_mat_ldl_batch_padded_prec(a: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    l, d = arb_mat_ldl_batch_padded(a, pad_to=pad_to)
    return di.round_interval_outward(l, prec_bits), di.round_interval_outward(d, prec_bits)


def arb_mat_dense_spd_solve_plan_prepare_batch_fixed_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseCholeskySolvePlan:
    return arb_mat_dense_spd_solve_plan_prepare_prec(a, prec_bits=prec_bits)


def arb_mat_dense_spd_solve_plan_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseCholeskySolvePlan:
    plan = arb_mat_dense_spd_solve_plan_prepare_batch_padded(a, pad_to=pad_to)
    return mat_common.dense_cholesky_solve_plan_from_factor(
        di.round_interval_outward(plan.factor, prec_bits),
        algebra="arb",
        structure="symmetric",
        label="arb_mat.dense_spd_solve_plan_prepare_batch_padded_prec",
    )


def arb_mat_spd_solve_batch_fixed_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_spd_solve_batch_fixed(a_or_plan, b), prec_bits)


def arb_mat_spd_solve_batch_padded_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_spd_solve_batch_padded(a_or_plan, b, pad_to=pad_to), prec_bits)


def arb_mat_dense_spd_solve_plan_apply_prec(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_spd_solve_plan_apply(plan, b), prec_bits)


def arb_mat_dense_spd_solve_plan_apply_batch_fixed_prec(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_spd_solve_plan_apply_batch_fixed(plan, b), prec_bits)


def arb_mat_dense_spd_solve_plan_apply_batch_padded_prec(
    plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_spd_solve_plan_apply_batch_padded(plan, b, pad_to=pad_to), prec_bits)


def arb_mat_spd_inv_batch_fixed_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_spd_inv_batch_fixed(a_or_plan), prec_bits)


def arb_mat_spd_inv_batch_padded_prec(
    a_or_plan: mat_common.DenseCholeskySolvePlan | jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_spd_inv_batch_padded(a_or_plan, pad_to=pad_to), prec_bits)


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


def arb_mat_dense_lu_solve_plan_prepare_batch_fixed(a: jax.Array) -> mat_common.DenseLUSolvePlan:
    return arb_mat_dense_lu_solve_plan_prepare(a)


def arb_mat_dense_lu_solve_plan_prepare_batch_padded(a: jax.Array, *, pad_to: int) -> mat_common.DenseLUSolvePlan:
    call_args, _ = mat_common.pad_batch_repeat_last((a,), pad_to=pad_to)
    return arb_mat_dense_lu_solve_plan_prepare(*call_args)


def arb_mat_dense_lu_solve_plan_apply_batch_fixed(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
) -> jax.Array:
    return arb_mat_dense_lu_solve_plan_apply(plan, b)


def arb_mat_dense_lu_solve_plan_apply_batch_padded(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    (b_pad,), _ = mat_common.pad_batch_repeat_last((b,), pad_to=pad_to)
    return arb_mat_dense_lu_solve_plan_apply(plan, b_pad)


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


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_rmatvec_batch_fixed_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_batch_fixed(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_matvec_batch_padded_prec(a: jax.Array, x: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_batch_padded(a, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_rmatvec_batch_padded_prec(a: jax.Array, x: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_batch_padded(a, x, pad_to=pad_to), prec_bits)


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


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_rmatvec_cached_apply_batch_fixed_prec(cache: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_cached_apply_batch_fixed(cache, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_matvec_cached_apply_batch_padded_prec(
    cache: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_apply_batch_padded(cache, x, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_rmatvec_cached_apply_batch_padded_prec(
    cache: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_cached_apply_batch_padded(cache, x, pad_to=pad_to), prec_bits)


def arb_mat_matvec_cached_prepare_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_prepare_batch_fixed(a), prec_bits)


def arb_mat_rmatvec_cached_prepare_batch_fixed_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_cached_prepare_batch_fixed(a), prec_bits)


def arb_mat_matvec_cached_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec_cached_prepare_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_rmatvec_cached_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_rmatvec_cached_prepare_batch_padded(a, pad_to=pad_to), prec_bits)


def arb_mat_dense_matvec_plan_prepare_batch_fixed_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    return arb_mat_dense_matvec_plan_prepare_prec(a, prec_bits=prec_bits)


def arb_mat_dense_matvec_plan_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseMatvecPlan:
    plan = arb_mat_dense_matvec_plan_prepare_batch_padded(a, pad_to=pad_to)
    return mat_common.dense_matvec_plan_from_matrix(
        di.round_interval_outward(plan.matrix, prec_bits),
        algebra="arb",
        label="arb_mat.dense_matvec_plan_prepare_batch_padded_prec",
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_dense_matvec_plan_apply_prec(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_matvec_plan_apply(plan, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_dense_matvec_plan_apply_batch_fixed_prec(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_matvec_plan_apply_batch_fixed(plan, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_dense_matvec_plan_apply_batch_padded_prec(
    plan: mat_common.DenseMatvecPlan | jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_matvec_plan_apply_batch_padded(plan, x, pad_to=pad_to), prec_bits)


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


def arb_mat_dense_lu_solve_plan_prepare_batch_fixed_prec(
    a: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    return arb_mat_dense_lu_solve_plan_prepare_prec(a, prec_bits=prec_bits)


def arb_mat_dense_lu_solve_plan_prepare_batch_padded_prec(
    a: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> mat_common.DenseLUSolvePlan:
    plan = arb_mat_dense_lu_solve_plan_prepare_batch_padded(a, pad_to=pad_to)
    return mat_common.dense_lu_solve_plan_from_factors(
        di.round_interval_outward(plan.p, prec_bits),
        di.round_interval_outward(plan.l, prec_bits),
        di.round_interval_outward(plan.u, prec_bits),
        algebra="arb",
        label="arb_mat.dense_lu_solve_plan_prepare_batch_padded_prec",
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_dense_lu_solve_plan_apply_batch_fixed_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_lu_solve_plan_apply_batch_fixed(plan, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def arb_mat_dense_lu_solve_plan_apply_batch_padded_prec(
    plan: mat_common.DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array],
    b: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(arb_mat_dense_lu_solve_plan_apply_batch_padded(plan, b, pad_to=pad_to), prec_bits)


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


def _make_arb_batch_fixed(name: str):
    fn = globals()[name]

    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapped.__name__ = f"{name}_batch_fixed"
    return wrapped


def _make_arb_batch_padded(name: str):
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


def _make_arb_batch_fixed_prec(name: str):
    fn = globals()[f"{name}_batch_fixed"]
    round_values = "eigh" in name
    passthrough = name.startswith("arb_mat_is_")

    def wrapped(*args, prec_bits: int = di.DEFAULT_PREC_BITS, **kwargs):
        out = fn(*args, **kwargs)
        if passthrough:
            return out
        if round_values:
            return tuple(di.round_interval_outward(part, prec_bits) for part in out)
        return di.round_interval_outward(out, prec_bits)

    wrapped.__name__ = f"{name}_batch_fixed_prec"
    return wrapped


def _make_arb_batch_padded_prec(name: str):
    fn = globals()[f"{name}_batch_padded"]
    round_values = "eigh" in name
    passthrough = name.startswith("arb_mat_is_")

    def wrapped(*args, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS, **kwargs):
        out = fn(*args, pad_to=pad_to, **kwargs)
        if passthrough:
            return out
        if round_values:
            return tuple(di.round_interval_outward(part, prec_bits) for part in out)
        return di.round_interval_outward(out, prec_bits)

    wrapped.__name__ = f"{name}_batch_padded_prec"
    return wrapped


for _arb_name in (
    "arb_mat_add",
    "arb_mat_sub",
    "arb_mat_neg",
    "arb_mat_mul_entrywise",
    "arb_mat_is_diag",
    "arb_mat_is_tril",
    "arb_mat_is_triu",
    "arb_mat_is_zero",
    "arb_mat_is_finite",
    "arb_mat_is_exact",
    "arb_mat_charpoly",
    "arb_mat_pow_ui",
    "arb_mat_exp",
    "arb_mat_eigvalsh",
    "arb_mat_eigh",
    "arb_mat_solve_tril",
    "arb_mat_solve_triu",
    "arb_mat_solve_lu",
    "arb_mat_solve_transpose",
    "arb_mat_solve_add",
    "arb_mat_solve_transpose_add",
    "arb_mat_mat_solve",
    "arb_mat_mat_solve_transpose",
):
    globals()[f"{_arb_name}_batch_fixed"] = _make_arb_batch_fixed(_arb_name)
    globals()[f"{_arb_name}_batch_padded"] = _make_arb_batch_padded(_arb_name)
    globals()[f"{_arb_name}_batch_fixed_prec"] = _make_arb_batch_fixed_prec(_arb_name)
    globals()[f"{_arb_name}_batch_padded_prec"] = _make_arb_batch_padded_prec(_arb_name)


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


def arb_mat_operator_plan_prepare(a: jax.Array):
    from . import jrb_mat
    return jrb_mat.jrb_mat_dense_operator_plan_prepare(a)


def arb_mat_operator_rmatvec_plan_prepare(a: jax.Array):
    from . import jrb_mat
    return jrb_mat.jrb_mat_dense_operator_rmatvec_plan_prepare(a)


def arb_mat_operator_adjoint_plan_prepare(a: jax.Array):
    from . import jrb_mat
    return jrb_mat.jrb_mat_dense_operator_adjoint_plan_prepare(a)


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
    "arb_mat_rmatvec",
    "arb_mat_rmatvec_basic",
    "arb_mat_banded_matvec",
    "arb_mat_banded_matvec_basic",
    "arb_mat_matvec_cached_prepare",
    "arb_mat_rmatvec_cached_prepare",
    "arb_mat_dense_matvec_plan_prepare",
    "arb_mat_dense_matvec_plan_apply",
    "arb_mat_matvec_cached_apply",
    "arb_mat_rmatvec_cached_apply",
    "arb_mat_symmetric_part",
    "arb_mat_is_symmetric",
    "arb_mat_is_spd",
    "arb_mat_cho",
    "arb_mat_ldl",
    "arb_mat_dense_spd_solve_plan_prepare",
    "arb_mat_spd_solve",
    "arb_mat_dense_spd_solve_plan_apply",
    "arb_mat_spd_inv",
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
    "arb_mat_solve_rigorous",
    "arb_mat_inv_rigorous",
    "arb_mat_triangular_solve",
    "arb_mat_triangular_solve_basic",
    "arb_mat_lu",
    "arb_mat_lu_basic",
    "arb_mat_lu_rigorous",
    "arb_mat_dense_lu_solve_plan_prepare_rigorous",
    "arb_mat_qr",
    "arb_mat_qr_basic",
    "arb_mat_qr_rigorous",
    "arb_mat_matmul_prec",
    "arb_mat_matvec_prec",
    "arb_mat_rmatvec_prec",
    "arb_mat_banded_matvec_prec",
    "arb_mat_matvec_cached_prepare_prec",
    "arb_mat_rmatvec_cached_prepare_prec",
    "arb_mat_dense_matvec_plan_prepare_prec",
    "arb_mat_matvec_cached_apply_prec",
    "arb_mat_rmatvec_cached_apply_prec",
    "arb_mat_symmetric_part_prec",
    "arb_mat_is_symmetric_prec",
    "arb_mat_is_spd_prec",
    "arb_mat_cho_prec",
    "arb_mat_ldl_prec",
    "arb_mat_dense_spd_solve_plan_prepare_prec",
    "arb_mat_spd_solve_prec",
    "arb_mat_dense_spd_solve_plan_apply_prec",
    "arb_mat_spd_inv_prec",
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
    "arb_mat_rmatvec_jit",
    "arb_mat_banded_matvec_jit",
    "arb_mat_matvec_cached_apply_jit",
    "arb_mat_rmatvec_cached_apply_jit",
    "arb_mat_symmetric_part_jit",
    "arb_mat_solve_jit",
    "arb_mat_inv_jit",
    "arb_mat_cho_jit",
    "arb_mat_ldl_jit",
    "arb_mat_spd_solve_jit",
    "arb_mat_spd_inv_jit",
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
    "arb_mat_symmetric_part_batch_fixed",
    "arb_mat_symmetric_part_batch_padded",
    "arb_mat_is_symmetric_batch_fixed",
    "arb_mat_is_symmetric_batch_padded",
    "arb_mat_is_spd_batch_fixed",
    "arb_mat_is_spd_batch_padded",
    "arb_mat_cho_batch_fixed",
    "arb_mat_cho_batch_padded",
    "arb_mat_ldl_batch_fixed",
    "arb_mat_ldl_batch_padded",
    "arb_mat_dense_spd_solve_plan_prepare_batch_fixed",
    "arb_mat_dense_spd_solve_plan_prepare_batch_padded",
    "arb_mat_spd_solve_batch_fixed",
    "arb_mat_spd_solve_batch_padded",
    "arb_mat_dense_spd_solve_plan_apply_batch_fixed",
    "arb_mat_dense_spd_solve_plan_apply_batch_padded",
    "arb_mat_spd_inv_batch_fixed",
    "arb_mat_spd_inv_batch_padded",
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
    "arb_mat_operator_plan_prepare",
    "arb_mat_operator_rmatvec_plan_prepare",
    "arb_mat_operator_adjoint_plan_prepare",
]
