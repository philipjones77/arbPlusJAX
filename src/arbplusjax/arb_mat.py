from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from . import checks
from . import double_interval as di
from . import mat_common

jax.config.update("jax_enable_x64", True)


def arb_mat_as_matrix(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_matrix(x, "arb_mat.as_matrix")


def arb_mat_as_vector(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_vector(x, "arb_mat.as_vector")


def arb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = arb_mat_as_matrix(a)
    return tuple(int(x) for x in arr.shape)

def _as_mat_2x2(x: jax.Array) -> jax.Array:
    return mat_common.as_interval_mat_2x2(x, "arb_mat._as_mat_2x2")


def _mid_matrix(a: jax.Array) -> jax.Array:
    return di.midpoint(arb_mat_as_matrix(a))


def _mid_vector(x: jax.Array) -> jax.Array:
    return di.midpoint(arb_mat_as_vector(x))


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


def arb_mat_solve(a: jax.Array, b: jax.Array) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "arb_mat.solve.inner")
    x = jnp.linalg.solve(_mid_matrix(a), _mid_vector(b))
    out = mat_common.interval_from_point(x)
    finite = jnp.all(jnp.isfinite(x), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return arb_mat_solve(a, b)


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


def arb_mat_det_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_det_basic(a)


def arb_mat_trace_rigorous(a: jax.Array) -> jax.Array:
    return arb_mat_trace_basic(a)


def arb_mat_triangular_solve(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    a = arb_mat_as_matrix(a)
    b = arb_mat_as_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "arb_mat.triangular_solve.inner")
    x = jsp_linalg.solve_triangular(_mid_matrix(a), _mid_vector(b), lower=lower, unit_diagonal=unit_diagonal)
    out = mat_common.interval_from_point(x)
    finite = jnp.all(jnp.isfinite(x), axis=-1)
    return jnp.where(finite[..., None, None], out, mat_common.full_interval_like(out))


def arb_mat_triangular_solve_basic(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return arb_mat_triangular_solve(a, b, lower=lower, unit_diagonal=unit_diagonal)


def arb_mat_lu(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = arb_mat_as_matrix(a)
    p, l, u = jsp_linalg.lu(_mid_matrix(a))
    return mat_common.interval_from_point(p), mat_common.interval_from_point(l), mat_common.interval_from_point(u)


def arb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return arb_mat_lu(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matmul_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matmul(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_matvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_matvec(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_solve_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_solve(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_det_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_det(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mat_trace_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mat_trace(a), prec_bits)


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


arb_mat_matmul_jit = jax.jit(arb_mat_matmul)
arb_mat_matvec_jit = jax.jit(arb_mat_matvec)
arb_mat_solve_jit = jax.jit(arb_mat_solve)
arb_mat_det_jit = jax.jit(arb_mat_det)
arb_mat_trace_jit = jax.jit(arb_mat_trace)
arb_mat_triangular_solve_jit = jax.jit(arb_mat_triangular_solve, static_argnames=("lower", "unit_diagonal"))
arb_mat_lu_jit = jax.jit(arb_mat_lu)


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
    "arb_mat_matmul",
    "arb_mat_matmul_basic",
    "arb_mat_matvec",
    "arb_mat_matvec_basic",
    "arb_mat_solve",
    "arb_mat_solve_basic",
    "arb_mat_det",
    "arb_mat_det_basic",
    "arb_mat_trace",
    "arb_mat_trace_basic",
    "arb_mat_det_rigorous",
    "arb_mat_trace_rigorous",
    "arb_mat_triangular_solve",
    "arb_mat_triangular_solve_basic",
    "arb_mat_lu",
    "arb_mat_lu_basic",
    "arb_mat_matmul_prec",
    "arb_mat_matvec_prec",
    "arb_mat_solve_prec",
    "arb_mat_det_prec",
    "arb_mat_trace_prec",
    "arb_mat_triangular_solve_prec",
    "arb_mat_lu_prec",
    "arb_mat_matmul_jit",
    "arb_mat_matvec_jit",
    "arb_mat_solve_jit",
    "arb_mat_det_jit",
    "arb_mat_trace_jit",
    "arb_mat_triangular_solve_jit",
    "arb_mat_lu_jit",
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
