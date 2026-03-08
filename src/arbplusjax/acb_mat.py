from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from . import acb_core
from . import checks
from . import double_interval as di
from . import mat_common

jax.config.update("jax_enable_x64", True)


def acb_mat_as_matrix(x: jax.Array) -> jax.Array:
    return mat_common.as_box_matrix(x, "acb_mat.as_matrix")


def acb_mat_as_vector(x: jax.Array) -> jax.Array:
    return mat_common.as_box_vector(x, "acb_mat.as_vector")


def acb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = acb_mat_as_matrix(a)
    return tuple(int(x) for x in arr.shape)

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


def acb_mat_det_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_det_basic(a)


def acb_mat_trace_rigorous(a: jax.Array) -> jax.Array:
    return acb_mat_trace_basic(a)


def acb_mat_triangular_solve(a: jax.Array, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    a = acb_mat_as_matrix(a)
    b = acb_mat_as_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "acb_mat.triangular_solve.inner")
    x = jsp_linalg.solve_triangular(_mid_matrix(a), _mid_vector(b), lower=lower, unit_diagonal=unit_diagonal)
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
    p, l, u = jsp_linalg.lu(_mid_matrix(a))
    return mat_common.box_from_point(p), mat_common.box_from_point(l), mat_common.box_from_point(u)


def acb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return acb_mat_lu(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matmul_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matmul(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_matvec_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_matvec(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_solve_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_solve(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_det_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_det(a), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mat_trace_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mat_trace(a), prec_bits)


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


acb_mat_matmul_jit = jax.jit(acb_mat_matmul)
acb_mat_matvec_jit = jax.jit(acb_mat_matvec)
acb_mat_solve_jit = jax.jit(acb_mat_solve)
acb_mat_det_jit = jax.jit(acb_mat_det)
acb_mat_trace_jit = jax.jit(acb_mat_trace)
acb_mat_triangular_solve_jit = jax.jit(acb_mat_triangular_solve, static_argnames=("lower", "unit_diagonal"))
acb_mat_lu_jit = jax.jit(acb_mat_lu)


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
    "acb_mat_matmul",
    "acb_mat_matmul_basic",
    "acb_mat_matvec",
    "acb_mat_matvec_basic",
    "acb_mat_solve",
    "acb_mat_solve_basic",
    "acb_mat_det",
    "acb_mat_det_basic",
    "acb_mat_trace",
    "acb_mat_trace_basic",
    "acb_mat_det_rigorous",
    "acb_mat_trace_rigorous",
    "acb_mat_triangular_solve",
    "acb_mat_triangular_solve_basic",
    "acb_mat_lu",
    "acb_mat_lu_basic",
    "acb_mat_matmul_prec",
    "acb_mat_matvec_prec",
    "acb_mat_solve_prec",
    "acb_mat_det_prec",
    "acb_mat_trace_prec",
    "acb_mat_triangular_solve_prec",
    "acb_mat_lu_prec",
    "acb_mat_matmul_jit",
    "acb_mat_matvec_jit",
    "acb_mat_solve_jit",
    "acb_mat_det_jit",
    "acb_mat_trace_jit",
    "acb_mat_triangular_solve_jit",
    "acb_mat_lu_jit",
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
