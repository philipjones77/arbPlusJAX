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

jax.config.update("jax_enable_x64", True)


def acb_mat_as_matrix(x: jax.Array) -> jax.Array:
    return mat_common.as_box_matrix(x, "acb_mat.as_matrix")


def acb_mat_as_vector(x: jax.Array) -> jax.Array:
    return mat_common.as_box_vector(x, "acb_mat.as_vector")


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


def acb_mat_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    b = acb_mat_as_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "acb_mat.matmul.inner")
    c = jnp.matmul(_mid_matrix(a), _mid_matrix(b))
    out = mat_common.box_from_point(c)
    finite = jnp.all(mat_common.complex_is_finite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_matmul_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    b = acb_mat_as_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "acb_mat.matmul_basic.inner")
    prods = acb_core.acb_mul(a[..., :, :, None, :], b[..., None, :, :, :])
    out = mat_common.box_sum(prods, axis=-2)
    finite = jnp.all(jnp.isfinite(out), axis=(-3, -2, -1))
    return jnp.where(finite[..., None, None, None], out, mat_common.full_box_like(out))


def acb_mat_matvec(a: jax.Array, x: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "acb_mat.matvec.inner")
    y = jnp.einsum("...ij,...j->...i", _mid_matrix(a), _mid_vector(x))
    out = mat_common.box_from_point(y)
    finite = jnp.all(mat_common.complex_is_finite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_matvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    x = acb_mat_as_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "acb_mat.matvec_basic.inner")
    prods = acb_core.acb_mul(a, x[..., None, :, :])
    out = mat_common.box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_matvec_cached_prepare(a: jax.Array) -> jax.Array:
    return acb_mat_as_matrix(a)


def acb_mat_matvec_cached_apply(cache: jax.Array, x: jax.Array) -> jax.Array:
    cache = acb_mat_as_matrix(cache)
    x = acb_mat_as_vector(x)
    checks.check_equal(cache.shape[-2], x.shape[-2], "acb_mat.matvec_cached_apply.inner")
    prods = acb_core.acb_mul(cache, x[..., None, :, :])
    out = mat_common.box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


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


def acb_mat_solve(a: jax.Array, b: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    b = acb_mat_as_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "acb_mat.solve.inner")
    x = jnp.linalg.solve(_mid_matrix(a), _mid_vector(b))
    out = mat_common.box_from_point(x)
    finite = jnp.all(mat_common.complex_is_finite(x), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


def acb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return acb_mat_solve(a, b)


def acb_mat_inv(a: jax.Array) -> jax.Array:
    a = acb_mat_as_matrix(a)
    inv = jnp.linalg.inv(_mid_matrix(a))
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
    b = acb_mat_as_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "acb_mat.triangular_solve.inner")
    x = lax.linalg.triangular_solve(
        _mid_matrix(a),
        _mid_vector(b),
        left_side=True,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    out = mat_common.box_from_point(x)
    finite = jnp.all(mat_common.complex_is_finite(x), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_box_like(out))


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


def acb_mat_qr(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = acb_mat_as_matrix(a)
    q, r = jnp.linalg.qr(_mid_matrix(a))
    return mat_common.box_from_point(q), mat_common.box_from_point(r)


def acb_mat_qr_basic(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_mat_qr(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matmul_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec(a, x), prec_bits)


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


acb_mat_matmul_jit = jax.jit(acb_mat_matmul)
acb_mat_matvec_jit = jax.jit(acb_mat_matvec)
acb_mat_banded_matvec_jit = jax.jit(acb_mat_banded_matvec, static_argnames=("lower_bandwidth", "upper_bandwidth"))
acb_mat_matvec_cached_apply_jit = jax.jit(acb_mat_matvec_cached_apply)
acb_mat_solve_jit = jax.jit(acb_mat_solve)
acb_mat_inv_jit = jax.jit(acb_mat_inv)
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


def acb_mat_matvec_batch_padded(a: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, x), pad_to=pad_to)
    return acb_mat_matvec(*call_args)


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


def acb_mat_matvec_cached_apply_batch_padded(cache: jax.Array, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((cache, x), pad_to=pad_to)
    return acb_mat_matvec_cached_apply(*call_args)


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


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matmul_batch_fixed_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul_batch_fixed(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_matmul_batch_padded_prec(a: jax.Array, b: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul_batch_padded(a, b, pad_to=pad_to), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_batch_fixed_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_batch_fixed(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_matvec_batch_padded_prec(a: jax.Array, x: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_batch_padded(a, x, pad_to=pad_to), prec_bits)


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


@partial(jax.jit, static_argnames=("prec_bits", "pad_to"))
def acb_mat_matvec_cached_apply_batch_padded_prec(
    cache: jax.Array,
    x: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec_cached_apply_batch_padded(cache, x, pad_to=pad_to), prec_bits)


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
    "acb_mat_as_matrix",
    "acb_mat_as_vector",
    "acb_mat_shape",
    "acb_mat_zero",
    "acb_mat_identity",
    "acb_mat_matmul",
    "acb_mat_matmul_basic",
    "acb_mat_matvec",
    "acb_mat_matvec_basic",
    "acb_mat_banded_matvec",
    "acb_mat_banded_matvec_basic",
    "acb_mat_matvec_cached_prepare",
    "acb_mat_matvec_cached_apply",
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
    "acb_mat_banded_matvec_prec",
    "acb_mat_matvec_cached_apply_prec",
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
    "acb_mat_banded_matvec_jit",
    "acb_mat_matvec_cached_apply_jit",
    "acb_mat_solve_jit",
    "acb_mat_inv_jit",
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
    "acb_mat_matvec_cached_apply_batch_fixed",
    "acb_mat_matvec_cached_apply_batch_padded",
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
    "acb_mat_matvec_cached_apply_batch_fixed_prec",
    "acb_mat_matvec_cached_apply_batch_padded_prec",
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
]
