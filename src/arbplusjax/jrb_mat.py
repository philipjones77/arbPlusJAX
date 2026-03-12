from __future__ import annotations

"""Jones real matrix-function subsystem scaffold and substrate.

This module is a separate Jones-labeled subsystem for new matrix-function work.
It does not replace `arb_mat`; `arb_mat` remains the canonical Arb/FLINT-style
JAX extension surface for real interval matrices.

Current implemented substrate:
- layout contracts for interval matrices/vectors
- point/basic matmul
- point/basic matvec
- point/basic solve
- point/basic triangular_solve
- point/basic lu

Planned scope beyond this substrate:
- contour-integral matrix logarithm / roots
- rational-Krylov matrix-function actions
- AD-aware matrix-function kernels with repo-standard engineering constraints

Provenance:
- classification: new
- base_names: jrb_mat
- module lineage: Jones matrix-function subsystem for real interval matrices
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import checks
from . import double_interval as di

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "new",
    "base_names": ("jrb_mat",),
    "module_lineage": "Jones matrix-function subsystem for real interval matrices",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}


def jrb_mat_as_interval_matrix(a: jax.Array) -> jax.Array:
    """Canonical Jones real-matrix layout: (..., n, n, 2)."""
    arr = di.as_interval(a)
    checks.check(arr.ndim >= 3, "jrb_mat.as_interval_matrix.ndim")
    checks.check_equal(arr.shape[-1], 2, "jrb_mat.as_interval_matrix.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], "jrb_mat.as_interval_matrix.square")
    return arr


def jrb_mat_as_interval_vector(x: jax.Array) -> jax.Array:
    """Canonical Jones real-vector layout: (..., n, 2)."""
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 2, "jrb_mat.as_interval_vector.ndim")
    checks.check_equal(arr.shape[-1], 2, "jrb_mat.as_interval_vector.tail")
    return arr


def jrb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = jrb_mat_as_interval_matrix(a)
    return tuple(int(x) for x in arr.shape)


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def _jrb_mid_matrix(a: jax.Array) -> jax.Array:
    return di.midpoint(jrb_mat_as_interval_matrix(a))


def _jrb_mid_vector(x: jax.Array) -> jax.Array:
    return di.midpoint(jrb_mat_as_interval_vector(x))


def _jrb_point_interval(x: jax.Array) -> jax.Array:
    return di.interval(di._below(x), di._above(x))


def _jrb_interval_sum(xs: jax.Array, axis: int = -1) -> jax.Array:
    lo = jnp.sum(xs[..., 0], axis=axis)
    hi = jnp.sum(xs[..., 1], axis=axis)
    return di.interval(di._below(lo), di._above(hi))


def jrb_mat_matmul_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    b = jrb_mat_as_interval_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "jrb_mat.matmul.inner")
    c = jnp.matmul(_jrb_mid_matrix(a), _jrb_mid_matrix(b))
    out = _jrb_point_interval(c)
    finite = jnp.all(jnp.isfinite(c), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, _full_interval_like(out))


def jrb_mat_matmul_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    b = jrb_mat_as_interval_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "jrb_mat.matmul.inner")
    prods = di.fast_mul(a[..., :, :, None, :], b[..., None, :, :, :])
    out = _jrb_interval_sum(prods, axis=-2)
    finite = jnp.all(jnp.isfinite(out), axis=(-3, -2, -1))
    return jnp.where(finite[..., None, None, None], out, _full_interval_like(out))


def jrb_mat_matvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    x = jrb_mat_as_interval_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "jrb_mat.matvec.inner")
    y = jnp.einsum("...ij,...j->...i", _jrb_mid_matrix(a), _jrb_mid_vector(x))
    out = _jrb_point_interval(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_matvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    x = jrb_mat_as_interval_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "jrb_mat.matvec.inner")
    prods = di.fast_mul(a, x[..., None, :, :])
    out = _jrb_interval_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_solve_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    b = jrb_mat_as_interval_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "jrb_mat.solve.inner")
    x = jnp.linalg.solve(_jrb_mid_matrix(a), _jrb_mid_vector(b))
    out = _jrb_point_interval(x)
    finite = jnp.all(jnp.isfinite(x), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return jrb_mat_solve_point(a, b)


def jrb_mat_triangular_solve_point(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    b = jrb_mat_as_interval_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "jrb_mat.triangular_solve.inner")
    x = lax.linalg.triangular_solve(
        _jrb_mid_matrix(a),
        _jrb_mid_vector(b),
        left_side=True,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    out = _jrb_point_interval(x)
    finite = jnp.all(jnp.isfinite(x), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_triangular_solve_basic(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return jrb_mat_triangular_solve_point(a, b, lower=lower, unit_diagonal=unit_diagonal)


def jrb_mat_lu_point(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    lu, _, perm = lax.linalg.lu(mid)
    n = mid.shape[-1]
    eye = jnp.eye(n, dtype=mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return _jrb_point_interval(p), _jrb_point_interval(l), _jrb_point_interval(u)


def jrb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jrb_mat_lu_point(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def jrb_mat_matmul_basic_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(jrb_mat_matmul_basic(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def jrb_mat_matvec_basic_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(jrb_mat_matvec_basic(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def jrb_mat_solve_basic_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(jrb_mat_solve_basic(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower", "unit_diagonal"))
def jrb_mat_triangular_solve_basic_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return di.round_interval_outward(
        jrb_mat_triangular_solve_basic(a, b, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def jrb_mat_lu_basic_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = jrb_mat_lu_basic(a)
    return (
        di.round_interval_outward(p, prec_bits),
        di.round_interval_outward(l, prec_bits),
        di.round_interval_outward(u, prec_bits),
    )


jrb_mat_matmul_basic_jit = jax.jit(jrb_mat_matmul_basic)
jrb_mat_matvec_basic_jit = jax.jit(jrb_mat_matvec_basic)
jrb_mat_solve_basic_jit = jax.jit(jrb_mat_solve_basic)
jrb_mat_triangular_solve_basic_jit = jax.jit(jrb_mat_triangular_solve_basic, static_argnames=("lower", "unit_diagonal"))
jrb_mat_lu_basic_jit = jax.jit(jrb_mat_lu_basic)


def jrb_mat_logm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_logm is planned but not implemented yet.")


def jrb_mat_sqrtm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_sqrtm is planned but not implemented yet.")


def jrb_mat_rootm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_rootm is planned but not implemented yet.")


def jrb_mat_signm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_signm is planned but not implemented yet.")


__all__ = [
    "PROVENANCE",
    "jrb_mat_as_interval_matrix",
    "jrb_mat_as_interval_vector",
    "jrb_mat_shape",
    "jrb_mat_matmul_point",
    "jrb_mat_matmul_basic",
    "jrb_mat_matvec_point",
    "jrb_mat_matvec_basic",
    "jrb_mat_solve_point",
    "jrb_mat_solve_basic",
    "jrb_mat_triangular_solve_point",
    "jrb_mat_triangular_solve_basic",
    "jrb_mat_lu_point",
    "jrb_mat_lu_basic",
    "jrb_mat_matmul_basic_prec",
    "jrb_mat_matvec_basic_prec",
    "jrb_mat_solve_basic_prec",
    "jrb_mat_triangular_solve_basic_prec",
    "jrb_mat_lu_basic_prec",
    "jrb_mat_matmul_basic_jit",
    "jrb_mat_matvec_basic_jit",
    "jrb_mat_solve_basic_jit",
    "jrb_mat_triangular_solve_basic_jit",
    "jrb_mat_lu_basic_jit",
    "jrb_mat_logm",
    "jrb_mat_sqrtm",
    "jrb_mat_rootm",
    "jrb_mat_signm",
]
