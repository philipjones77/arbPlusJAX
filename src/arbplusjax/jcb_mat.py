from __future__ import annotations

"""Jones complex matrix-function subsystem scaffold and substrate.

This module is a separate Jones-labeled subsystem for new complex matrix-function
work. It does not replace `acb_mat`; `acb_mat` remains the canonical
Arb/FLINT-style JAX extension surface for complex box matrices.

Current implemented substrate:
- layout contracts for complex-box matrices/vectors
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
- base_names: jcb_mat
- module lineage: Jones matrix-function subsystem for complex box matrices
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from . import acb_core
from . import checks
from . import double_interval as di

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "new",
    "base_names": ("jcb_mat",),
    "module_lineage": "Jones matrix-function subsystem for complex box matrices",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}


def jcb_mat_as_box_matrix(a: jax.Array) -> jax.Array:
    """Canonical Jones complex-matrix layout: (..., n, n, 4)."""
    arr = acb_core.as_acb_box(a)
    checks.check(arr.ndim >= 3, "jcb_mat.as_box_matrix.ndim")
    checks.check_equal(arr.shape[-1], 4, "jcb_mat.as_box_matrix.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], "jcb_mat.as_box_matrix.square")
    return arr


def jcb_mat_as_box_vector(x: jax.Array) -> jax.Array:
    """Canonical Jones complex-vector layout: (..., n, 4)."""
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 2, "jcb_mat.as_box_vector.ndim")
    checks.check_equal(arr.shape[-1], 4, "jcb_mat.as_box_vector.tail")
    return arr


def jcb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = jcb_mat_as_box_matrix(a)
    return tuple(int(x) for x in arr.shape)


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def _jcb_mid_matrix(a: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(jcb_mat_as_box_matrix(a))


def _jcb_mid_vector(x: jax.Array) -> jax.Array:
    return acb_core.acb_midpoint(jcb_mat_as_box_vector(x))


def _jcb_point_box(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def _jcb_box_sum(xs: jax.Array, axis: int = -1) -> jax.Array:
    re = acb_core.acb_real(xs)
    im = acb_core.acb_imag(xs)
    re_out = di.interval(di._below(jnp.sum(re[..., 0], axis=axis)), di._above(jnp.sum(re[..., 1], axis=axis)))
    im_out = di.interval(di._below(jnp.sum(im[..., 0], axis=axis)), di._above(jnp.sum(im[..., 1], axis=axis)))
    return acb_core.acb_box(re_out, im_out)


def jcb_mat_matmul_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    b = jcb_mat_as_box_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "jcb_mat.matmul.inner")
    c = jnp.matmul(_jcb_mid_matrix(a), _jcb_mid_matrix(b))
    out = _jcb_point_box(c)
    finite = jnp.all(jnp.isfinite(jnp.real(c)) & jnp.isfinite(jnp.imag(c)), axis=(-2, -1))
    return jnp.where(finite[..., None, None, None], out, _full_box_like(out))


def jcb_mat_matmul_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    b = jcb_mat_as_box_matrix(b)
    checks.check_equal(a.shape[-2], b.shape[-3], "jcb_mat.matmul.inner")
    prods = acb_core.acb_mul(a[..., :, :, None, :], b[..., None, :, :, :])
    out = _jcb_box_sum(prods, axis=-2)
    finite = jnp.all(jnp.isfinite(out), axis=(-3, -2, -1))
    return jnp.where(finite[..., None, None, None], out, _full_box_like(out))


def jcb_mat_matvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    x = jcb_mat_as_box_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "jcb_mat.matvec.inner")
    y = jnp.einsum("...ij,...j->...i", _jcb_mid_matrix(a), _jcb_mid_vector(x))
    out = _jcb_point_box(y)
    finite = jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_matvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    x = jcb_mat_as_box_vector(x)
    checks.check_equal(a.shape[-2], x.shape[-2], "jcb_mat.matvec.inner")
    prods = acb_core.acb_mul(a, x[..., None, :, :])
    out = _jcb_box_sum(prods, axis=-1)
    finite = jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_solve_point(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    b = jcb_mat_as_box_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "jcb_mat.solve.inner")
    x = jnp.linalg.solve(_jcb_mid_matrix(a), _jcb_mid_vector(b))
    out = _jcb_point_box(x)
    finite = jnp.all(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_solve_basic(a: jax.Array, b: jax.Array) -> jax.Array:
    return jcb_mat_solve_point(a, b)


def jcb_mat_triangular_solve_point(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    b = jcb_mat_as_box_vector(b)
    checks.check_equal(a.shape[-2], b.shape[-2], "jcb_mat.triangular_solve.inner")
    x = jsp_linalg.solve_triangular(_jcb_mid_matrix(a), _jcb_mid_vector(b), lower=lower, unit_diagonal=unit_diagonal)
    out = _jcb_point_box(x)
    finite = jnp.all(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_triangular_solve_basic(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return jcb_mat_triangular_solve_point(a, b, lower=lower, unit_diagonal=unit_diagonal)


def jcb_mat_lu_point(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = jcb_mat_as_box_matrix(a)
    p, l, u = jsp_linalg.lu(_jcb_mid_matrix(a))
    return _jcb_point_box(p), _jcb_point_box(l), _jcb_point_box(u)


def jcb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jcb_mat_lu_point(a)


@partial(jax.jit, static_argnames=("prec_bits",))
def jcb_mat_matmul_basic_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(jcb_mat_matmul_basic(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def jcb_mat_matvec_basic_prec(a: jax.Array, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(jcb_mat_matvec_basic(a, x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def jcb_mat_solve_basic_prec(a: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(jcb_mat_solve_basic(a, b), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "lower", "unit_diagonal"))
def jcb_mat_triangular_solve_basic_prec(
    a: jax.Array,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return acb_core.acb_box_round_prec(
        jcb_mat_triangular_solve_basic(a, b, lower=lower, unit_diagonal=unit_diagonal),
        prec_bits,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def jcb_mat_lu_basic_prec(a: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array]:
    p, l, u = jcb_mat_lu_basic(a)
    return (
        acb_core.acb_box_round_prec(p, prec_bits),
        acb_core.acb_box_round_prec(l, prec_bits),
        acb_core.acb_box_round_prec(u, prec_bits),
    )


jcb_mat_matmul_basic_jit = jax.jit(jcb_mat_matmul_basic)
jcb_mat_matvec_basic_jit = jax.jit(jcb_mat_matvec_basic)
jcb_mat_solve_basic_jit = jax.jit(jcb_mat_solve_basic)
jcb_mat_triangular_solve_basic_jit = jax.jit(jcb_mat_triangular_solve_basic, static_argnames=("lower", "unit_diagonal"))
jcb_mat_lu_basic_jit = jax.jit(jcb_mat_lu_basic)


def jcb_mat_logm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_logm is planned but not implemented yet.")


def jcb_mat_sqrtm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_sqrtm is planned but not implemented yet.")


def jcb_mat_rootm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_rootm is planned but not implemented yet.")


def jcb_mat_signm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_signm is planned but not implemented yet.")


__all__ = [
    "PROVENANCE",
    "jcb_mat_as_box_matrix",
    "jcb_mat_as_box_vector",
    "jcb_mat_shape",
    "jcb_mat_matmul_point",
    "jcb_mat_matmul_basic",
    "jcb_mat_matvec_point",
    "jcb_mat_matvec_basic",
    "jcb_mat_solve_point",
    "jcb_mat_solve_basic",
    "jcb_mat_triangular_solve_point",
    "jcb_mat_triangular_solve_basic",
    "jcb_mat_lu_point",
    "jcb_mat_lu_basic",
    "jcb_mat_matmul_basic_prec",
    "jcb_mat_matvec_basic_prec",
    "jcb_mat_solve_basic_prec",
    "jcb_mat_triangular_solve_basic_prec",
    "jcb_mat_lu_basic_prec",
    "jcb_mat_matmul_basic_jit",
    "jcb_mat_matvec_basic_jit",
    "jcb_mat_solve_basic_jit",
    "jcb_mat_triangular_solve_basic_jit",
    "jcb_mat_lu_basic_jit",
    "jcb_mat_logm",
    "jcb_mat_sqrtm",
    "jcb_mat_rootm",
    "jcb_mat_signm",
]
