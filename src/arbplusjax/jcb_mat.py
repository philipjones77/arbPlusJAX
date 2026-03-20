from __future__ import annotations

"""Jones complex matrix-free subsystem scaffold and substrate.

This module is a separate Jones-labeled subsystem for new complex matrix-free
work. It does not replace `acb_mat`; `acb_mat` remains the canonical
Arb/FLINT-style JAX extension surface for complex box matrices.

Current implemented substrate:
- layout contracts for complex-box matrices/vectors
- point/basic matmul
- point/basic matvec
- point/basic solve
- point/basic triangular_solve
- point/basic lu
- matrix-free point/basic action kernels for polynomial actions and expm-actions

Planned scope beyond this substrate:
- operator-first Arnoldi / Krylov matrix-function actions
- contour-integral matrix logarithm / roots
- AD-aware matrix-function kernels with repo-standard engineering constraints

Provenance:
- classification: new
- base_names: jcb_mat
- module lineage: Jones matrix-function subsystem for complex box matrices
- naming policy: see docs/standards/function_naming.md
- registry report: see docs/status/reports/function_implementation_index.md
"""

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di
from . import iterative_solvers
from . import mat_common
from . import matrix_free_basic
from . import sparse_common
from . import matrix_free_core


PROVENANCE = {
    "classification": "new",
    "base_names": ("jcb_mat",),
    "module_lineage": "Jones matrix-function subsystem for complex box matrices",
    "naming_policy": "docs/standards/function_naming.md",
    "registry_report": "docs/status/reports/function_implementation_index.md",
}


class JcbMatKrylovDiagnostics(NamedTuple):
    algorithm_code: jax.Array
    steps: jax.Array
    basis_dim: jax.Array
    restart_count: jax.Array
    beta0: jax.Array
    tail_norm: jax.Array
    breakdown: jax.Array
    used_adjoint: jax.Array
    gradient_supported: jax.Array
    probe_count: jax.Array


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


def _jcb_operator_vector(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    if arr.ndim >= 1 and arr.shape[-1] == 4:
        return _jcb_mid_vector(arr)
    return jnp.asarray(arr, dtype=jnp.complex128)


def _jcb_point_box(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def _jcb_round_basic(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(x, prec_bits)


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
    x = jnp.linalg.solve(_jcb_mid_matrix(a), _jcb_mid_vector(b)[..., None])[..., 0]
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
    x = lax.linalg.triangular_solve(
        _jcb_mid_matrix(a),
        _jcb_mid_vector(b),
        left_side=True,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
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
    mid = _jcb_mid_matrix(a)
    lu, _, perm = lax.linalg.lu(mid)
    n = mid.shape[-1]
    eye = jnp.eye(n, dtype=mid.dtype)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return _jcb_point_box(p), _jcb_point_box(l), _jcb_point_box(u)


def jcb_mat_lu_basic(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jcb_mat_lu_point(a)


def jcb_mat_det_point(a: jax.Array) -> jax.Array:
    """Determinant of dense complex-box matrix - point version."""
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    det_val = jnp.linalg.det(mid)
    return _jcb_point_box(det_val)


def jcb_mat_det_basic(a: jax.Array) -> jax.Array:
    """Determinant of dense complex-box matrix - basic interval version."""
    return jcb_mat_det_point(a)


def jcb_mat_inv_point(a: jax.Array) -> jax.Array:
    """Matrix inverse - point version."""
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    inv_val = jnp.linalg.inv(mid)
    return _jcb_point_box(inv_val)


def jcb_mat_inv_basic(a: jax.Array) -> jax.Array:
    """Matrix inverse - basic interval version."""
    return jcb_mat_inv_point(a)


def jcb_mat_sqr_point(a: jax.Array) -> jax.Array:
    """Matrix square - point version."""
    return jcb_mat_matmul_point(a, a)


def jcb_mat_sqr_basic(a: jax.Array) -> jax.Array:
    """Matrix square - basic interval version."""
    return jcb_mat_matmul_basic(a, a)


def jcb_mat_trace_point(a: jax.Array) -> jax.Array:
    """Trace of dense complex-box matrix - point version."""
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    trace_val = jnp.trace(mid, axis1=-2, axis2=-1)
    return _jcb_point_box(trace_val)


def jcb_mat_trace_basic(a: jax.Array) -> jax.Array:
    """Trace of dense complex-box matrix - basic interval version."""
    return jcb_mat_trace_point(a)


def jcb_mat_norm_fro_point(a: jax.Array) -> jax.Array:
    """Frobenius norm of dense complex-box matrix - point version."""
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    norm_val = jnp.linalg.norm(mid, ord='fro')
    return _jcb_point_box(norm_val)


def jcb_mat_norm_fro_basic(a: jax.Array) -> jax.Array:
    """Frobenius norm of dense complex-box matrix - basic interval version."""
    return jcb_mat_norm_fro_point(a)


def jcb_mat_norm_1_point(a: jax.Array) -> jax.Array:
    """1-norm of dense complex-box matrix - point version."""
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    norm_val = jnp.linalg.norm(mid, ord=1)
    return _jcb_point_box(norm_val)


def jcb_mat_norm_1_basic(a: jax.Array) -> jax.Array:
    """1-norm of dense complex-box matrix - basic interval version."""
    return jcb_mat_norm_1_point(a)


def jcb_mat_norm_inf_point(a: jax.Array) -> jax.Array:
    """Infinity norm of dense complex-box matrix - point version."""
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    norm_val = jnp.linalg.norm(mid, ord=jnp.inf)
    return _jcb_point_box(norm_val)


def jcb_mat_norm_inf_basic(a: jax.Array) -> jax.Array:
    """Infinity norm of dense complex-box matrix - basic interval version."""
    return jcb_mat_norm_inf_point(a)


def jcb_mat_dense_operator(a: jax.Array):
    """Return a matrix-free midpoint matvec closure for a dense complex-box matrix."""
    return matrix_free_core.dense_operator(_jcb_mid_matrix(a), midpoint_vector=_jcb_mid_vector)


def jcb_mat_dense_operator_adjoint(a: jax.Array):
    """Return the adjoint midpoint matvec closure for a dense complex-box matrix."""
    return matrix_free_core.dense_operator_adjoint(_jcb_mid_matrix(a), midpoint_vector=_jcb_mid_vector, conjugate=True)


def jcb_mat_dense_operator_rmatvec(a: jax.Array):
    return matrix_free_core.dense_operator_rmatvec(_jcb_mid_matrix(a), midpoint_vector=_jcb_mid_vector)


def jcb_mat_dense_operator_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jcb_mid_matrix(a), orientation="forward", algebra="jcb")


def jcb_mat_dense_operator_rmatvec_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jcb_mid_matrix(a), orientation="transpose", algebra="jcb")


def jcb_mat_dense_operator_adjoint_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jcb_mid_matrix(a), orientation="adjoint", algebra="jcb")


def _jcb_sparse_to_bcoo(x):
    return matrix_free_core.canonicalize_sparse_bcoo(
        x,
        algebra="jcb",
        sparse_common=sparse_common,
        label="jcb_mat.sparse_to_bcoo",
    )


def jcb_mat_bcoo_operator(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        midpoint_vector=_jcb_operator_vector,
        dtype=jnp.complex128,
        algebra="jcb",
        label="jcb_mat.bcoo_operator",
    )


def jcb_mat_sparse_operator(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jcb_mat_bcoo_operator(_jcb_sparse_to_bcoo(a))


def jcb_mat_bcoo_operator_adjoint(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_adjoint(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        midpoint_vector=_jcb_operator_vector,
        dtype=jnp.complex128,
        algebra="jcb",
        label="jcb_mat.bcoo_operator_adjoint",
        conjugate=True,
    )


def jcb_mat_sparse_operator_adjoint(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jcb_mat_bcoo_operator_adjoint(_jcb_sparse_to_bcoo(a))


def jcb_mat_bcoo_operator_rmatvec(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_rmatvec(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        midpoint_vector=_jcb_operator_vector,
        dtype=jnp.complex128,
        algebra="jcb",
        label="jcb_mat.bcoo_operator_rmatvec",
    )


def jcb_mat_sparse_operator_rmatvec(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jcb_mat_bcoo_operator_rmatvec(_jcb_sparse_to_bcoo(a))


def jcb_mat_bcoo_operator_plan_prepare(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        orientation="forward",
        algebra="jcb",
    )


def jcb_mat_sparse_operator_plan_prepare(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jcb_mat_bcoo_operator_plan_prepare(_jcb_sparse_to_bcoo(a))


def jcb_mat_bcoo_operator_rmatvec_plan_prepare(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        orientation="transpose",
        algebra="jcb",
    )


def jcb_mat_sparse_operator_rmatvec_plan_prepare(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jcb_mat_bcoo_operator_rmatvec_plan_prepare(_jcb_sparse_to_bcoo(a))


def jcb_mat_bcoo_operator_adjoint_plan_prepare(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        orientation="adjoint",
        algebra="jcb",
        conjugate_transpose=True,
    )


def jcb_mat_sparse_operator_adjoint_plan_prepare(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jcb_mat_bcoo_operator_adjoint_plan_prepare(_jcb_sparse_to_bcoo(a))


def jcb_mat_hermitian_operator(a: jax.Array):
    return jcb_mat_dense_operator(a)


def jcb_mat_hermitian_operator_plan_prepare(a: jax.Array):
    return jcb_mat_dense_operator_plan_prepare(a)


def jcb_mat_hpd_operator(a: jax.Array):
    return jcb_mat_dense_operator(a)


def jcb_mat_hpd_operator_plan_prepare(a: jax.Array):
    return jcb_mat_dense_operator_plan_prepare(a)


def jcb_mat_operator_plan_apply(plan: matrix_free_core.OperatorPlan, x: jax.Array) -> jax.Array:
    return jcb_mat_operator_apply_point(plan, x)


def _jcb_apply_operator_mid(operator, x: jax.Array) -> jax.Array:
    return matrix_free_core.operator_apply_midpoint(
        operator,
        x,
        midpoint_vector=_jcb_operator_vector,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
    )


def jcb_mat_rmatvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    return jcb_mat_operator_apply_point(jcb_mat_dense_operator_rmatvec(a), x)


def jcb_mat_rmatvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    return jcb_mat_rmatvec_point(a, x)


def jcb_mat_arnoldi_hessenberg_adjoint(matvec, *, krylov_depth: int, reortho: str = "full", custom_vjp: bool = True):
    return matrix_free_core.matfree_adjoints.arnoldi_hessenberg(
        matvec,
        krylov_depth=krylov_depth,
        reortho=reortho,
        custom_vjp=custom_vjp,
    )


def jcb_mat_cg_fixed_iterations(*, num_matvecs: int):
    return matrix_free_core.matfree_adjoints.cg_fixed_iterations(num_matvecs=num_matvecs)


def jcb_mat_solve_action_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
) -> jax.Array:
    value, _ = jcb_mat_solve_action_with_diagnostics_point(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
    )
    return value


def jcb_mat_solve_action_basic(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.solve_action_basic(
        jcb_mat_solve_action_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_solve_action_hermitian_point(matvec, b: jax.Array, **kwargs) -> jax.Array:
    return jcb_mat_solve_action_point(matvec, b, hermitian=True, **kwargs)


def jcb_mat_solve_action_hpd_point(matvec, b: jax.Array, **kwargs) -> jax.Array:
    return jcb_mat_solve_action_point(matvec, b, hermitian=True, **kwargs)


def jcb_mat_solve_action_with_diagnostics_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
):
    b = jcb_mat_as_box_vector(b)
    b_mid = _jcb_mid_vector(b)
    x0_mid = None if x0 is None else _jcb_mid_vector(x0)
    mv = lambda v: _jcb_apply_operator_mid(matvec, _jcb_point_box(v))
    precond = None if preconditioner is None else (lambda v: _jcb_apply_operator_mid(preconditioner, _jcb_point_box(v)))
    solver = iterative_solvers.cg if hermitian else iterative_solvers.gmres
    x_mid, info = solver(mv, b_mid, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
    out = _jcb_point_box(x_mid)
    finite = jnp.all(jnp.isfinite(jnp.real(x_mid)) & jnp.isfinite(jnp.imag(x_mid)), axis=-1)
    return jnp.where(finite[..., None], out, _full_box_like(out)), info


def jcb_mat_solve_action_with_diagnostics_basic(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.solve_action_with_diagnostics_basic(
        jcb_mat_solve_action_with_diagnostics_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_inverse_action_point(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
) -> jax.Array:
    return jcb_mat_solve_action_point(
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
    )


def jcb_mat_inverse_action_basic(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.inverse_action_basic(
        jcb_mat_inverse_action_point,
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_inverse_action_hermitian_point(matvec, x: jax.Array, **kwargs) -> jax.Array:
    return jcb_mat_inverse_action_point(matvec, x, hermitian=True, **kwargs)


def jcb_mat_inverse_action_hpd_point(matvec, x: jax.Array, **kwargs) -> jax.Array:
    return jcb_mat_inverse_action_point(matvec, x, hermitian=True, **kwargs)


def jcb_mat_inverse_action_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
):
    return jcb_mat_solve_action_with_diagnostics_point(
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
    )


def jcb_mat_inverse_action_with_diagnostics_basic(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.inverse_action_with_diagnostics_basic(
        jcb_mat_inverse_action_with_diagnostics_point,
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_operator_apply_point(matvec, x: jax.Array) -> jax.Array:
    return matrix_free_core.operator_apply_point(
        matvec,
        x,
        midpoint_apply=_jcb_apply_operator_mid,
        coerce_vector=jcb_mat_as_box_vector,
        point_from_midpoint=_jcb_point_box,
        full_like=_full_box_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1),
        dtype=jnp.complex128,
    )


def jcb_mat_operator_apply_basic(matvec, x: jax.Array) -> jax.Array:
    return matrix_free_basic.operator_apply_basic(
        jcb_mat_operator_apply_point,
        matvec,
        x,
        round_output=_jcb_round_basic,
        prec_bits=di.DEFAULT_PREC_BITS,
    )


def _jcb_leja_points_interval_point(
    lower: jax.Array,
    upper: jax.Array,
    degree: int,
    *,
    candidate_count: int = 64,
) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    if candidate_count < degree:
        raise ValueError("candidate_count must be >= degree")

    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    grid = center + radius * jnp.cos(jnp.linspace(0.0, jnp.pi, candidate_count, dtype=jnp.float64))
    eps = jnp.asarray(1e-30, dtype=jnp.float64)
    first = jnp.argmax(grid)
    selected = jnp.zeros((degree,), dtype=jnp.float64).at[0].set(grid[first])
    mask = jnp.zeros((candidate_count,), dtype=bool).at[first].set(True)
    scores = jnp.log(jnp.abs(grid - grid[first]) + eps)

    def body(k, carry):
        selected_k, mask_k, scores_k = carry
        idx = jnp.argmax(jnp.where(mask_k, -jnp.inf, scores_k))
        node = grid[idx]
        selected_k = selected_k.at[k].set(node)
        mask_k = mask_k.at[idx].set(True)
        scores_k = scores_k + jnp.log(jnp.abs(grid - node) + eps)
        return selected_k, mask_k, scores_k

    selected, _, _ = lax.fori_loop(1, degree, body, (selected, mask, scores))
    return selected


def _jcb_log_leja_scaling_params(
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    lower, upper = spectral_bounds
    eps = jnp.asarray(1e-12, dtype=jnp.float64)
    alpha_scale = jnp.maximum(jnp.asarray(lower, dtype=jnp.float64), eps)
    scaled_lower = jnp.asarray(1.0, dtype=jnp.float64)
    scaled_upper = jnp.maximum(jnp.asarray(upper, dtype=jnp.float64) / alpha_scale, scaled_lower + eps)
    center = 0.5 * (scaled_lower + scaled_upper)
    gamma = jnp.maximum(0.25 * (scaled_upper - scaled_lower), eps)
    return alpha_scale, center, gamma, scaled_upper


def _jcb_newton_divided_differences_point(nodes: jax.Array, scalar_fun) -> jax.Array:
    coeffs = jnp.asarray(scalar_fun(nodes), dtype=jnp.complex128)
    degree = int(nodes.shape[0])
    for j in range(1, degree):
        numer = coeffs[j:] - coeffs[j - 1:-1]
        denom = nodes[j:] - nodes[: degree - j]
        coeffs = coeffs.at[j:].set(numer / denom)
    return coeffs


def _jcb_log_leja_coefficients_point(
    nodes: jax.Array,
    center: jax.Array,
    gamma: jax.Array,
) -> jax.Array:
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    floor = jnp.asarray(1e-30, dtype=jnp.float64)
    return _jcb_newton_divided_differences_point(
        nodes,
        lambda t: jnp.log(jnp.maximum(center + gamma * t, floor)),
    )


def _jcb_log_leja_setup_point(
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    degree: int,
    *,
    candidate_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    alpha_scale, center, gamma, _ = _jcb_log_leja_scaling_params(spectral_bounds)
    nodes = _jcb_leja_points_interval_point(
        jnp.asarray(-2.0, dtype=jnp.float64),
        jnp.asarray(2.0, dtype=jnp.float64),
        degree,
        candidate_count=candidate_count,
    )
    coeffs = _jcb_log_leja_coefficients_point(nodes, center, gamma)
    return alpha_scale, center, gamma, nodes, coeffs


def jcb_mat_poly_action_point(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    return matrix_free_core.poly_action_point(
        matvec,
        x,
        coefficients,
        midpoint_apply=_jcb_apply_operator_mid,
        coerce_vector=jcb_mat_as_box_vector,
        midpoint_vector=_jcb_mid_vector,
        point_from_midpoint=_jcb_point_box,
        full_like=_full_box_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1),
        coeff_dtype=jnp.complex128,
    )


def jcb_mat_poly_action_basic(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_poly_action_point,
        matvec,
        x,
        coefficients,
        round_output=_jcb_round_basic,
        prec_bits=di.DEFAULT_PREC_BITS,
    )


def jcb_mat_expm_action_point(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    return matrix_free_core.expm_action_point(
        matvec,
        x,
        terms=terms,
        midpoint_apply=_jcb_apply_operator_mid,
        coerce_vector=jcb_mat_as_box_vector,
        midpoint_vector=_jcb_mid_vector,
        point_from_midpoint=_jcb_point_box,
        full_like=_full_box_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1),
        scalar_dtype=jnp.float64,
    )


def jcb_mat_expm_action_basic(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_expm_action_point,
        matvec,
        x,
        terms=terms,
        round_output=_jcb_round_basic,
        prec_bits=di.DEFAULT_PREC_BITS,
    )


def _jcb_mat_arnoldi_hessenberg_state_point(matvec, x: jax.Array, steps: int):
    x = jcb_mat_as_box_vector(x)
    if steps <= 0:
        raise ValueError("steps must be > 0")
    v0 = _jcb_mid_vector(x)
    beta0 = jnp.linalg.norm(v0)
    if steps > int(v0.shape[-1]):
        raise ValueError("steps must be <= vector dimension")
    q0 = v0 / jnp.maximum(beta0, jnp.asarray(1e-30, dtype=jnp.float64))
    dim = q0.shape[-1]

    def body(carry, _):
        q_curr, basis, H, k = carry
        basis = basis.at[k].set(q_curr)
        z = _jcb_apply_operator_mid(matvec, _jcb_point_box(q_curr))
        mask = (jnp.arange(steps + 1, dtype=jnp.int32) < (k + 1)).astype(jnp.complex128)
        h_col = (jnp.conjugate(basis) @ z) * mask
        r = z - h_col @ basis
        beta = jnp.linalg.norm(r)
        q_next = jnp.where(beta > 1e-30, r / beta, jnp.zeros_like(r))
        basis = basis.at[k + 1].set(q_next)
        H = H.at[:, k].set(h_col)
        H = H.at[k + 1, k].set(beta)
        return (q_next, basis, H, k + 1), None

    init_basis = jnp.zeros((steps + 1, dim), dtype=jnp.complex128)
    init_basis = init_basis.at[0].set(q0)
    init_H = jnp.zeros((steps + 1, steps), dtype=jnp.complex128)
    init = (q0, init_basis, init_H, jnp.asarray(0, dtype=jnp.int32))
    (_, basis, H, _), _ = lax.scan(body, init, xs=None, length=steps)
    return basis[:-1], H[:-1, :], beta0, H[-1, steps - 1]


def jcb_mat_arnoldi_hessenberg_point(matvec, x: jax.Array, steps: int):
    basis, H, beta0, _ = _jcb_mat_arnoldi_hessenberg_state_point(matvec, x, steps)
    return basis, H, beta0


def jcb_mat_arnoldi_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    used_adjoint: bool = False,
) -> JcbMatKrylovDiagnostics:
    basis, _, beta0, tail = _jcb_mat_arnoldi_hessenberg_state_point(matvec, x, steps)
    tail_norm = jnp.abs(tail)
    breakdown = tail_norm <= jnp.asarray(1e-30, dtype=jnp.float64)
    return matrix_free_core.krylov_diagnostics(
        JcbMatKrylovDiagnostics,
        algorithm_code=0,
        steps=steps,
        basis_dim=basis.shape[0],
        beta0=beta0,
        tail_norm=tail_norm,
        breakdown=breakdown,
        used_adjoint=used_adjoint,
    )


def _jcb_mat_funm_action_arnoldi_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    return matrix_free_core.projected_krylov_action_point(
        matvec,
        x,
        dense_funm,
        steps,
        krylov_decomp=jcb_mat_arnoldi_hessenberg_point,
        point_from_midpoint=_jcb_point_box,
        full_like=_full_box_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1),
        coeff_dtype=jnp.complex128,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3, 4))
def jcb_mat_funm_action_arnoldi_point(matvec, x: jax.Array, dense_funm, steps: int, adjoint_matvec=None):
    return _jcb_mat_funm_action_arnoldi_point_base(matvec, x, dense_funm, steps)


def _jcb_mat_funm_action_arnoldi_point_fwd(matvec, x, dense_funm, steps, adjoint_matvec):
    y = _jcb_mat_funm_action_arnoldi_point_base(matvec, x, dense_funm, steps)
    return y, x


def _jcb_mat_funm_action_arnoldi_point_bwd(matvec, dense_funm, steps, adjoint_matvec, x, cotangent):
    if adjoint_matvec is None:
        raise ValueError("adjoint_matvec is required for differentiating jcb_mat_funm_action_arnoldi_point.")
    adjoint = _jcb_mat_funm_action_arnoldi_point_base(
        adjoint_matvec,
        _jcb_point_box(acb_core.acb_midpoint(cotangent)),
        dense_funm,
        steps,
    )
    return (adjoint,)


jcb_mat_funm_action_arnoldi_point.defvjp(
    _jcb_mat_funm_action_arnoldi_point_fwd,
    _jcb_mat_funm_action_arnoldi_point_bwd,
)


def _jcb_mat_funm_integrand_arnoldi_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    return matrix_free_core.projected_krylov_integrand_point(
        matvec,
        x,
        dense_funm,
        steps,
        krylov_decomp=jcb_mat_arnoldi_hessenberg_point,
        coeff_dtype=jnp.complex128,
        scalar_dtype=jnp.complex128,
        scalar_postprocess=lambda value: value,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3, 4))
def jcb_mat_funm_integrand_arnoldi_point(matvec, x: jax.Array, dense_funm, steps: int, adjoint_matvec=None):
    return _jcb_mat_funm_integrand_arnoldi_point_base(matvec, x, dense_funm, steps)


def _jcb_mat_funm_integrand_arnoldi_point_fwd(matvec, x, dense_funm, steps, adjoint_matvec):
    y = _jcb_mat_funm_integrand_arnoldi_point_base(matvec, x, dense_funm, steps)
    return y, x


def _jcb_mat_funm_integrand_arnoldi_point_bwd(matvec, dense_funm, steps, adjoint_matvec, x, cotangent):
    if adjoint_matvec is None:
        raise ValueError("adjoint_matvec is required for differentiating jcb_mat_funm_integrand_arnoldi_point.")
    action = _jcb_mat_funm_action_arnoldi_point_base(matvec, x, dense_funm, steps)
    adjoint_action = _jcb_mat_funm_action_arnoldi_point_base(adjoint_matvec, x, dense_funm, steps)
    scale = jnp.asarray(cotangent, dtype=jnp.complex128)
    grad = jnp.conjugate(scale) * acb_core.acb_midpoint(action) + scale * acb_core.acb_midpoint(adjoint_action)
    return (_jcb_point_box(grad),)


jcb_mat_funm_integrand_arnoldi_point.defvjp(
    _jcb_mat_funm_integrand_arnoldi_point_fwd,
    _jcb_mat_funm_integrand_arnoldi_point_bwd,
)


def jcb_mat_dense_funm_general_eig_point(scalar_fun):
    return matrix_free_core.dense_funm_general_eig(scalar_fun, dtype=jnp.complex128)


def jcb_mat_dense_funm_hermitian_eigh_point(scalar_fun):
    return matrix_free_core.dense_funm_hermitian_eigh(
        scalar_fun,
        dtype=jnp.complex128,
        conjugate_right=True,
    )


def _jcb_dense_funm_point(a: jax.Array, scalar_fun) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    mid = _jcb_mid_matrix(a)
    vals, vecs = jnp.linalg.eig(mid)
    inv = jnp.linalg.inv(vecs)
    out = vecs @ jnp.diag(scalar_fun(vals)) @ inv
    return _jcb_point_box(out)


def _jcb_dense_funm_action_point(a: jax.Array, x: jax.Array, dense_funm) -> jax.Array:
    a = jcb_mat_as_box_matrix(a)
    x = jcb_mat_as_box_vector(x)
    y = dense_funm(_jcb_mid_matrix(a)) @ _jcb_mid_vector(x)
    return _jcb_point_box(y)


def jcb_mat_funm_action_arnoldi_dense_point(
    a: jax.Array,
    x: jax.Array,
    dense_funm,
    steps: int,
) -> jax.Array:
    @jax.custom_vjp
    def _apply(a_inner: jax.Array, x_inner: jax.Array) -> jax.Array:
        a_checked = jcb_mat_as_box_matrix(a_inner)
        checks.check(steps > 0, "jcb_mat.funm_action_arnoldi_dense.steps")
        checks.check(steps <= a_checked.shape[-2], "jcb_mat.funm_action_arnoldi_dense.steps_dim")
        return _jcb_dense_funm_action_point(a_checked, x_inner, dense_funm)

    def _apply_fwd(a_inner, x_inner):
        a_checked = jcb_mat_as_box_matrix(a_inner)
        x_checked = jcb_mat_as_box_vector(x_inner)
        y = _jcb_dense_funm_action_point(a_checked, x_checked, dense_funm)
        return y, (_jcb_mid_matrix(a_checked), _jcb_mid_vector(x_checked))

    def _apply_bwd(res, cotangent):
        a_mid, x_mid = res
        c_mid = acb_core.acb_midpoint(acb_core.as_acb_box(cotangent))
        funm_mid = dense_funm(a_mid)
        x_grad = _jcb_point_box(jnp.conjugate(funm_mid).T @ c_mid)

        n = int(a_mid.shape[-1])
        eps = jnp.asarray(1e-6, dtype=jnp.float64)
        a_grad_mid = jnp.zeros_like(a_mid)
        for i in range(n):
            for j in range(n):
                basis = jnp.zeros_like(a_mid).at[i, j].set(1.0 + 0.0j)
                dy_re = ((dense_funm(a_mid + eps * basis) - dense_funm(a_mid - eps * basis)) / (2.0 * eps)) @ x_mid
                dy_im = ((dense_funm(a_mid + 1j * eps * basis) - dense_funm(a_mid - 1j * eps * basis)) / (2.0 * eps)) @ x_mid
                grad_re = jnp.real(jnp.vdot(c_mid, dy_re))
                grad_im = jnp.real(jnp.vdot(c_mid, dy_im))
                a_grad_mid = a_grad_mid.at[i, j].set(grad_re + 1j * grad_im)

        return (_jcb_point_box(a_grad_mid), x_grad)

    _apply.defvjp(_apply_fwd, _apply_bwd)
    return _apply(a, x)


def jcb_mat_log_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.log), steps, adjoint_matvec)


def jcb_mat_log_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_log_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_sqrt_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.sqrt), steps, adjoint_matvec)


def jcb_mat_sqrt_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_sqrt_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_root_action_arnoldi_point(matvec, x: jax.Array, *, degree: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return jcb_mat_funm_action_arnoldi_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
        adjoint_matvec,
    )


def jcb_mat_root_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    steps: int,
    adjoint_matvec=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_root_action_arnoldi_point,
        matvec,
        x,
        degree=degree,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_sign_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.where(vals == 0, 0.0 + 0.0j, vals / jnp.sqrt(vals * vals))),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sign_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_sign_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_sin_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.sin), steps, adjoint_matvec)


def jcb_mat_sin_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_sin_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_cos_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.cos), steps, adjoint_matvec)


def jcb_mat_cos_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_cos_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_sinh_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.sinh), steps, adjoint_matvec)


def jcb_mat_sinh_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_sinh_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_cosh_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.cosh), steps, adjoint_matvec)


def jcb_mat_cosh_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_cosh_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_tanh_action_arnoldi_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.tanh), steps, adjoint_matvec)


def jcb_mat_tanh_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_tanh_action_arnoldi_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_log_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.log), steps, used_adjoint)


def jcb_mat_log_action_hermitian_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_log_action_hermitian_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_sqrt_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.sqrt), steps, used_adjoint)


def jcb_mat_sqrt_action_hermitian_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_sqrt_action_hermitian_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_root_action_hermitian_point(matvec, x: jax.Array, *, degree: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return jcb_mat_funm_action_arnoldi_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
        used_adjoint,
    )


def jcb_mat_root_action_hermitian_basic(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    steps: int,
    adjoint_matvec=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_root_action_hermitian_point,
        matvec,
        x,
        degree=degree,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_sign_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return jcb_mat_funm_action_arnoldi_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.sign), steps, used_adjoint)


def jcb_mat_sign_action_hermitian_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_sign_action_hermitian_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_pow_action_hermitian_point(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if exponent < 0:
        raise ValueError("exponent must be >= 0")
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return jcb_mat_funm_action_arnoldi_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(lambda vals: jnp.power(vals, exponent)),
        steps,
        used_adjoint,
    )


def jcb_mat_pow_action_hermitian_basic(
    matvec,
    x: jax.Array,
    *,
    exponent: int,
    steps: int,
    adjoint_matvec=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_pow_action_hermitian_point,
        matvec,
        x,
        exponent=exponent,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_log_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_log_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_sqrt_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_sqrt_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_root_action_hpd_point(matvec, x: jax.Array, *, degree: int, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_root_action_hermitian_point(matvec, x, degree=degree, steps=steps, adjoint_matvec=adjoint_matvec)


def jcb_mat_log_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_log_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_sqrt_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_sqrt_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_root_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, *, degree: int, steps: int) -> jax.Array:
    return jcb_mat_root_action_arnoldi_point(jcb_mat_dense_operator(a), x, degree=degree, steps=steps, adjoint_matvec=jcb_mat_dense_operator_adjoint(a))


def jcb_mat_sign_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_sign_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_sin_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_sin_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_cos_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_cos_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_sinh_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_sinh_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_cosh_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_cosh_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_tanh_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jcb_mat_tanh_action_arnoldi_point(jcb_mat_dense_operator(a), x, steps, jcb_mat_dense_operator_adjoint(a))


def jcb_mat_pow_action_arnoldi_point(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if exponent < 0:
        raise ValueError("exponent must be >= 0")
    return jcb_mat_funm_action_arnoldi_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.power(vals, exponent)),
        steps,
        adjoint_matvec,
    )


def jcb_mat_pow_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    *,
    exponent: int,
    steps: int,
    adjoint_matvec=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jcb_mat_pow_action_arnoldi_point,
        matvec,
        x,
        exponent=exponent,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_pow_action_arnoldi_dense_point(a: jax.Array, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    return jcb_mat_pow_action_arnoldi_point(
        jcb_mat_dense_operator(a),
        x,
        exponent=exponent,
        steps=steps,
        adjoint_matvec=jcb_mat_dense_operator_adjoint(a),
    )


def _jcb_funm_action_newton_point(
    x: jax.Array,
    nodes: jax.Array,
    coeffs: jax.Array,
    step_fn,
    *,
    constant_shift: jax.Array = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    basis = _jcb_mid_vector(x)
    acc = (constant_shift + coeffs[0]) * basis
    for k in range(1, int(coeffs.shape[0])):
        basis = step_fn(basis, nodes[k - 1])
        acc = acc + coeffs[k] * basis
    out = _jcb_point_box(acc)
    finite = jnp.all(jnp.isfinite(jnp.real(acc)) & jnp.isfinite(jnp.imag(acc)), axis=-1)
    return jnp.where(finite[..., None], out, _full_box_like(out))


def _jcb_funm_action_newton_adaptive_point(
    x: jax.Array,
    nodes: jax.Array,
    coeffs: jax.Array,
    step_fn,
    *,
    min_degree: int,
    rtol: float,
    atol: float,
    constant_shift: jax.Array = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = jcb_mat_as_box_vector(x)
    degree = int(coeffs.shape[0])
    if degree <= 0:
        raise ValueError("degree must be > 0")
    min_degree = max(1, min(int(min_degree), degree))
    basis0 = _jcb_mid_vector(x)
    acc0 = (constant_shift + coeffs[0]) * basis0
    tail0 = jnp.linalg.norm(acc0)
    local_rtol = jnp.asarray(rtol, dtype=jnp.float64)
    local_atol = jnp.asarray(atol, dtype=jnp.float64)

    def body(k, carry):
        basis, acc, used_degree, tail_norm, done = carry

        def compute(_):
            next_basis = step_fn(basis, nodes[k - 1])
            term = coeffs[k] * next_basis
            next_acc = acc + term
            next_tail = jnp.linalg.norm(term)
            next_used = jnp.asarray(k + 1, dtype=jnp.int32)
            acc_norm = jnp.linalg.norm(next_acc)
            converged = jnp.logical_and(
                next_used >= jnp.asarray(min_degree, dtype=jnp.int32),
                next_tail <= local_atol + local_rtol * jnp.maximum(acc_norm, jnp.asarray(1.0, dtype=jnp.float64)),
            )
            return next_basis, next_acc, next_used, next_tail, converged

        return lax.cond(done, lambda _: carry, compute, operand=None)

    init = (
        basis0,
        acc0,
        jnp.asarray(1, dtype=jnp.int32),
        tail0,
        jnp.asarray(False),
    )
    _, acc, used_degree, tail_norm, _ = lax.fori_loop(1, degree, body, init)
    out = _jcb_point_box(acc)
    finite = jnp.all(jnp.isfinite(jnp.real(acc)) & jnp.isfinite(jnp.imag(acc)), axis=-1)
    return jnp.where(finite[..., None], out, _full_box_like(out)), used_degree, tail_norm


def jcb_mat_log_action_leja_point(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    total_degree = int(max_degree) if max_degree is not None else int(degree)
    if total_degree <= 0:
        raise ValueError("degree must be > 0")
    alpha_scale, center, gamma, nodes, coeffs = _jcb_log_leja_setup_point(
        spectral_bounds,
        total_degree,
        candidate_count=candidate_count,
    )

    def step_fn(basis, node):
        applied = _jcb_apply_operator_mid(matvec, _jcb_point_box(basis)) / jnp.asarray(alpha_scale, dtype=jnp.complex128)
        return (applied - jnp.asarray(center, dtype=jnp.complex128) * basis) / jnp.asarray(gamma, dtype=jnp.complex128) - node * basis

    shift = jnp.log(jnp.asarray(alpha_scale, dtype=jnp.complex128))
    if max_degree is not None:
        value, _, _ = _jcb_funm_action_newton_adaptive_point(
            x,
            nodes,
            coeffs,
            step_fn,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
            constant_shift=shift,
        )
        return value
    return _jcb_funm_action_newton_point(x, nodes, coeffs, step_fn, constant_shift=shift)


def jcb_mat_log_action_leja_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    total_degree = int(max_degree) if max_degree is not None else int(degree)
    if total_degree <= 0:
        raise ValueError("degree must be > 0")
    alpha_scale, center, gamma, nodes, coeffs = _jcb_log_leja_setup_point(
        spectral_bounds,
        total_degree,
        candidate_count=candidate_count,
    )

    def step_fn(basis, node):
        applied = _jcb_apply_operator_mid(matvec, _jcb_point_box(basis)) / jnp.asarray(alpha_scale, dtype=jnp.complex128)
        return (applied - jnp.asarray(center, dtype=jnp.complex128) * basis) / jnp.asarray(gamma, dtype=jnp.complex128) - node * basis

    shift = jnp.log(jnp.asarray(alpha_scale, dtype=jnp.complex128))
    if max_degree is not None:
        value, used_degree, tail_norm = _jcb_funm_action_newton_adaptive_point(
            x,
            nodes,
            coeffs,
            step_fn,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
            constant_shift=shift,
        )
    else:
        value = _jcb_funm_action_newton_point(x, nodes, coeffs, step_fn, constant_shift=shift)
        used_degree = jnp.asarray(total_degree, dtype=jnp.int32)
        basis = _jcb_mid_vector(x)
        tail = (shift + coeffs[0]) * basis
        for k in range(1, total_degree):
            basis = step_fn(basis, nodes[k - 1])
            tail = coeffs[k] * basis
        tail_norm = jnp.linalg.norm(tail)

    diag = JcbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(3, dtype=jnp.int32),
        steps=jnp.asarray(used_degree, dtype=jnp.int32),
        basis_dim=jnp.asarray(used_degree, dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(0.0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(1, dtype=jnp.int32),
    )
    return value, diag


def _jcb_apply_block_action_point(action_fn, probes: jax.Array) -> jax.Array:
    coerced = acb_core.as_acb_box(probes)
    outputs = jax.vmap(action_fn)(coerced)
    return acb_core.acb_midpoint(outputs)


def jcb_mat_hutchpp_trace_point(action_fn, sketch_probes: jax.Array, residual_probes: jax.Array) -> jax.Array:
    sketch = acb_core.as_acb_box(sketch_probes)
    residual = acb_core.as_acb_box(residual_probes)
    n = int(sketch.shape[-2] if sketch.shape[0] > 0 else residual.shape[-2])

    if sketch.shape[0] > 0:
        y_cols = jnp.swapaxes(_jcb_apply_block_action_point(action_fn, sketch), 0, 1)
        q, _ = jnp.linalg.qr(y_cols, mode="reduced")
        fq_cols = jnp.swapaxes(_jcb_apply_block_action_point(action_fn, jax.vmap(_jcb_point_box)(q.T)), 0, 1)
        trace_lr = jnp.trace(jnp.conjugate(q).T @ fq_cols)
    else:
        q = jnp.zeros((n, 0), dtype=jnp.complex128)
        trace_lr = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    if residual.shape[0] > 0:
        z = acb_core.acb_midpoint(residual)
        z_proj = z - (z @ q) @ jnp.conjugate(q).T
        hz = _jcb_apply_block_action_point(action_fn, jax.vmap(_jcb_point_box)(z_proj))
        residual_est = jnp.mean(jnp.sum(jnp.conjugate(z_proj) * hz, axis=-1))
    else:
        residual_est = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    return jnp.asarray(trace_lr + residual_est, dtype=jnp.complex128)


def jcb_mat_expm_action_arnoldi_restarted_point(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    restarts: int,
    adjoint_matvec=None,
) -> jax.Array:
    dense_exp = jcb_mat_dense_funm_general_eig_point(jnp.exp)
    scale = complex(1.0 / restarts)
    scaled_matvec = matrix_free_core.scaled_operator(matvec, scale)

    scaled_adjoint = None
    if adjoint_matvec is not None:
        scaled_adjoint = matrix_free_core.scaled_operator(adjoint_matvec, scale)

    return matrix_free_core.restarted_action_point(
        lambda y: jcb_mat_funm_action_arnoldi_point(scaled_matvec, y, dense_exp, steps, scaled_adjoint),
        jcb_mat_as_box_vector(x),
        restarts=restarts,
    )


def jcb_mat_expm_action_arnoldi_block_point(
    matvec,
    xs: jax.Array,
    *,
    steps: int,
    restarts: int = 1,
    adjoint_matvec=None,
) -> jax.Array:
    return matrix_free_core.block_action_point(
        lambda x: jcb_mat_expm_action_arnoldi_restarted_point(
            matvec,
            x,
            steps=steps,
            restarts=restarts,
            adjoint_matvec=adjoint_matvec,
        ),
        acb_core.as_acb_box(xs),
    )


def jcb_mat_expm_action_arnoldi_restarted_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    restarts: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    y = jcb_mat_expm_action_arnoldi_restarted_point(
        matvec,
        x,
        steps=steps,
        restarts=restarts,
        adjoint_matvec=adjoint_matvec,
    )
    diag = jcb_mat_arnoldi_diagnostics_point(matvec, x, steps, used_adjoint=adjoint_matvec is not None)
    diag = diag._replace(restart_count=jnp.asarray(restarts, dtype=jnp.int32))
    return y, diag


def jcb_mat_logm(a: jax.Array) -> jax.Array:
    return _jcb_dense_funm_point(a, jnp.log)


def jcb_mat_sqrtm(a: jax.Array) -> jax.Array:
    return _jcb_dense_funm_point(a, jnp.sqrt)


def jcb_mat_rootm(a: jax.Array, *, degree: int) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return _jcb_dense_funm_point(a, lambda vals: jnp.power(vals, inv_degree))


def jcb_mat_signm(a: jax.Array) -> jax.Array:
    return _jcb_dense_funm_point(a, lambda vals: jnp.where(vals == 0, 0.0 + 0.0j, vals / jnp.sqrt(vals * vals)))


def _jcb_mat_trace_integrand_point_base(matvec, x: jax.Array) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    x_mid = _jcb_mid_vector(x)
    y = _jcb_apply_operator_mid(matvec, x)
    return jnp.vdot(x_mid, y)


@partial(jax.custom_vjp, nondiff_argnums=(0, 2))
def jcb_mat_trace_integrand_point(matvec, x: jax.Array, adjoint_matvec=None) -> jax.Array:
    return _jcb_mat_trace_integrand_point_base(matvec, x)


def _jcb_mat_trace_integrand_point_fwd(matvec, x, adjoint_matvec):
    y = _jcb_mat_trace_integrand_point_base(matvec, x)
    return y, x


def _jcb_mat_trace_integrand_point_bwd(matvec, adjoint_matvec, x, cotangent):
    if adjoint_matvec is None:
        raise ValueError("adjoint_matvec is required for differentiating jcb_mat_trace_integrand_point.")
    action = _jcb_apply_operator_mid(matvec, x)
    adjoint_action = _jcb_apply_operator_mid(adjoint_matvec, x)
    scale = jnp.asarray(cotangent, dtype=jnp.complex128)
    grad = jnp.conjugate(scale) * action + scale * adjoint_action
    return (_jcb_point_box(grad),)


jcb_mat_trace_integrand_point.defvjp(
    _jcb_mat_trace_integrand_point_fwd,
    _jcb_mat_trace_integrand_point_bwd,
)


def jcb_mat_funm_trace_integrand_arnoldi_point(matvec, x: jax.Array, scalar_fun, steps: int, adjoint_matvec=None):
    dense_funm = jcb_mat_dense_funm_general_eig_point(scalar_fun)
    return jcb_mat_funm_integrand_arnoldi_point(matvec, x, dense_funm, steps=steps, adjoint_matvec=adjoint_matvec)


def jcb_mat_funm_trace_integrand_hermitian_point(matvec, x: jax.Array, scalar_fun, steps: int, adjoint_matvec=None):
    dense_funm = jcb_mat_dense_funm_hermitian_eigh_point(scalar_fun)
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return jcb_mat_funm_integrand_arnoldi_point(matvec, x, dense_funm, steps=steps, adjoint_matvec=used_adjoint)


def jcb_mat_trace_estimator_point(matvec, probes: jax.Array, adjoint_matvec=None) -> jax.Array:
    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        lambda v: jcb_mat_trace_integrand_point(matvec, v, adjoint_matvec),
    )


def jcb_mat_logdet_slq_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        lambda v: jcb_mat_funm_trace_integrand_arnoldi_point(
            matvec,
            v,
            jnp.log,
            steps=steps,
            adjoint_matvec=adjoint_matvec,
        )
    )


def jcb_mat_logdet_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jcb_mat_logdet_slq_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb_point_box,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_logdet_slq_hermitian_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        lambda v: jcb_mat_funm_trace_integrand_hermitian_point(
            matvec,
            v,
            jnp.log,
            steps=steps,
            adjoint_matvec=used_adjoint,
        )
    )


def jcb_mat_logdet_slq_hpd_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_logdet_slq_hermitian_point(matvec, probes, steps, adjoint_matvec)


def jcb_mat_det_slq_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return matrix_free_core.det_from_logdet(jcb_mat_logdet_slq_point(matvec, probes, steps, adjoint_matvec))


def jcb_mat_det_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jcb_mat_det_slq_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb_point_box,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def _jcb_mat_logdet_slq_point_plan_kernel(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    dense_funm = jcb_mat_dense_funm_general_eig_point(jnp.log)
    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        lambda v: _jcb_mat_funm_integrand_arnoldi_point_base(matvec, v, dense_funm, steps),
    )


def _jcb_mat_det_slq_point_plan_kernel(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    del adjoint_matvec
    return matrix_free_core.det_from_logdet(_jcb_mat_logdet_slq_point_plan_kernel(matvec, probes, steps))


def jcb_mat_det_slq_hermitian_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return matrix_free_core.det_from_logdet(jcb_mat_logdet_slq_hermitian_point(matvec, probes, steps, adjoint_matvec))


def jcb_mat_det_slq_hpd_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return matrix_free_core.det_from_logdet(jcb_mat_logdet_slq_hpd_point(matvec, probes, steps, adjoint_matvec))


def jcb_mat_funm_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    used_adjoint = adjoint_matvec is not None
    return mat_common.action_with_diagnostics(
        lambda xx: jcb_mat_funm_action_arnoldi_point(matvec, xx, dense_funm, steps, adjoint_matvec),
        lambda xx: jcb_mat_arnoldi_diagnostics_point(matvec, xx, steps, used_adjoint=used_adjoint),
        x,
    )


def jcb_mat_log_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.log),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sqrt_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.sqrt),
        steps,
        adjoint_matvec,
    )


def jcb_mat_root_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sign_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.where(vals == 0, 0.0 + 0.0j, vals / jnp.sqrt(vals * vals))),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sin_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.sin),
        steps,
        adjoint_matvec,
    )


def jcb_mat_cos_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.cos),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sinh_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.sinh),
        steps,
        adjoint_matvec,
    )


def jcb_mat_cosh_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.cosh),
        steps,
        adjoint_matvec,
    )


def jcb_mat_tanh_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(jnp.tanh),
        steps,
        adjoint_matvec,
    )


def jcb_mat_pow_action_arnoldi_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    exponent: int,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    if exponent < 0:
        raise ValueError("exponent must be >= 0")
    return jcb_mat_funm_action_arnoldi_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.power(vals, exponent)),
        steps,
        adjoint_matvec,
    )


def jcb_mat_trace_estimator_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    coerced = acb_core.as_acb_box(probes)
    used_adjoint = adjoint_matvec is not None
    first_steps = int(coerced[0].shape[-2])
    return mat_common.estimator_with_diagnostics(
        probes,
        coerce_probes=acb_core.as_acb_box,
        estimator_fn=lambda xs: jcb_mat_trace_estimator_point(matvec, xs, adjoint_matvec),
        diagnostics_fn=lambda first: jcb_mat_arnoldi_diagnostics_point(
            matvec,
            first,
            first_steps,
            used_adjoint=used_adjoint,
        ),
        algorithm_code=1,
        steps=first_steps,
        basis_dim=first_steps,
    )


def jcb_mat_logdet_slq_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    used_adjoint = adjoint_matvec is not None
    return mat_common.estimator_with_diagnostics(
        probes,
        coerce_probes=acb_core.as_acb_box,
        estimator_fn=lambda xs: jcb_mat_logdet_slq_point(matvec, xs, steps, adjoint_matvec),
        diagnostics_fn=lambda first: jcb_mat_arnoldi_diagnostics_point(
            matvec,
            first,
            steps,
            used_adjoint=used_adjoint,
        ),
        algorithm_code=2,
    )


def jcb_mat_logdet_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        jcb_mat_logdet_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb_point_box,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_det_slq_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    value, diag = jcb_mat_logdet_slq_with_diagnostics_point(matvec, probes, steps, adjoint_matvec)
    return matrix_free_core.det_from_logdet(value), diag


def jcb_mat_det_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        jcb_mat_det_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb_point_box,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_logdet_leja_hutchpp_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    action_fn = lambda v: jcb_mat_log_action_leja_point(
        matvec,
        v,
        degree=degree,
        spectral_bounds=spectral_bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )
    return jcb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)


def jcb_mat_logdet_leja_hutchpp_with_diagnostics_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    value = jcb_mat_logdet_leja_hutchpp_point(
        matvec,
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=spectral_bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )
    reference = acb_core.as_acb_box(sketch_probes)
    if reference.shape[0] > 0:
        _, action_diag = jcb_mat_log_action_leja_with_diagnostics_point(
            matvec,
            reference[0],
            degree=degree,
            spectral_bounds=spectral_bounds,
            candidate_count=candidate_count,
            max_degree=max_degree,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
        used_steps = action_diag.steps
        tail_norm = action_diag.tail_norm
    else:
        used_steps = jnp.asarray(max_degree if max_degree is not None else degree, dtype=jnp.int32)
        tail_norm = jnp.asarray(0.0, dtype=jnp.float64)
    diag = JcbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(3, dtype=jnp.int32),
        steps=jnp.asarray(used_steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(used_steps, dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(0.0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(
            acb_core.as_acb_box(sketch_probes).shape[0] + acb_core.as_acb_box(residual_probes).shape[0],
            dtype=jnp.int32,
        ),
    )
    return value, diag


def jcb_mat_det_leja_hutchpp_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    return matrix_free_core.det_from_logdet(
        jcb_mat_logdet_leja_hutchpp_point(
            matvec,
            sketch_probes,
            residual_probes,
            degree=degree,
            spectral_bounds=spectral_bounds,
            candidate_count=candidate_count,
            max_degree=max_degree,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
    )


def jcb_mat_det_leja_hutchpp_with_diagnostics_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    value, diag = jcb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        matvec,
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=spectral_bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )
    return matrix_free_core.det_from_logdet(value), diag


def jcb_mat_bcoo_gershgorin_bounds(a: sparse_common.SparseBCOO, *, eps: float = 1e-12) -> tuple[jax.Array, jax.Array]:
    a = sparse_common.as_sparse_bcoo(a, algebra="jcb", label="jcb_mat.bcoo_gershgorin_bounds")
    checks.check_equal(a.rows, a.cols, "jcb_mat.bcoo_gershgorin_bounds.square")
    rows = jnp.asarray(a.indices[:, 0], dtype=jnp.int32)
    cols = jnp.asarray(a.indices[:, 1], dtype=jnp.int32)
    data = jnp.asarray(a.data, dtype=jnp.complex128)
    n = int(a.rows)
    abs_row_sum = jax.ops.segment_sum(jnp.abs(data), rows, n)
    diag = jax.ops.segment_sum(jnp.where(rows == cols, jnp.real(data), 0.0), rows, n)
    radii = abs_row_sum - jnp.abs(diag)
    lower = jnp.min(diag - radii)
    upper = jnp.max(diag + radii)
    return (
        jnp.maximum(jnp.asarray(eps, dtype=jnp.float64), jnp.asarray(lower, dtype=jnp.float64)),
        jnp.asarray(upper, dtype=jnp.float64),
    )


def jcb_mat_bcoo_logdet_leja_hutchpp_point(
    a: sparse_common.SparseBCOO,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int = 32,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    candidate_count: int = 96,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    bounds = jcb_mat_bcoo_gershgorin_bounds(a) if spectral_bounds is None else spectral_bounds
    return jcb_mat_logdet_leja_hutchpp_point(
        jcb_mat_bcoo_operator(a),
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )


def jcb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(
    a: sparse_common.SparseBCOO,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int = 32,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    candidate_count: int = 96,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    bounds = jcb_mat_bcoo_gershgorin_bounds(a) if spectral_bounds is None else spectral_bounds
    return jcb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        jcb_mat_bcoo_operator(a),
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )


def jcb_mat_bcoo_det_leja_hutchpp_point(
    a: sparse_common.SparseBCOO,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int = 32,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    candidate_count: int = 96,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    return matrix_free_core.det_from_logdet(
        jcb_mat_bcoo_logdet_leja_hutchpp_point(
            a,
            sketch_probes,
            residual_probes,
            degree=degree,
            spectral_bounds=spectral_bounds,
            candidate_count=candidate_count,
            max_degree=max_degree,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
    )


def jcb_mat_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    return matrix_free_core.rademacher_probes_complex(_jcb_point_box, x.shape[-2], key=key, num=num)


def jcb_mat_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    return matrix_free_core.normal_probes_complex(_jcb_point_box, x.shape[-2], key=key, num=num)


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

_jcb_mat_expm_action_basic_jit_callable = jax.jit(jcb_mat_expm_action_basic, static_argnames=("matvec", "terms"))
_jcb_mat_expm_action_basic_jit_plan = jax.jit(jcb_mat_expm_action_basic, static_argnames=("terms",))


def jcb_mat_expm_action_basic_jit(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_expm_action_basic_jit_plan(matvec, x, terms=terms)
    return _jcb_mat_expm_action_basic_jit_callable(matvec, x, terms=terms)


_jcb_mat_solve_action_point_jit_callable = jax.jit(
    jcb_mat_solve_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "hermitian", "preconditioner"),
)
_jcb_mat_solve_action_point_jit_plan = jax.jit(
    jcb_mat_solve_action_point,
    static_argnames=("tol", "atol", "maxiter", "hermitian", "preconditioner"),
)


def jcb_mat_solve_action_point_jit(matvec, b: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_solve_action_point_jit_plan(matvec, b, **kwargs)
    return _jcb_mat_solve_action_point_jit_callable(matvec, b, **kwargs)


_jcb_mat_inverse_action_point_jit_callable = jax.jit(
    jcb_mat_inverse_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "hermitian", "preconditioner"),
)
_jcb_mat_inverse_action_point_jit_plan = jax.jit(
    jcb_mat_inverse_action_point,
    static_argnames=("tol", "atol", "maxiter", "hermitian", "preconditioner"),
)


def jcb_mat_inverse_action_point_jit(matvec, x: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_inverse_action_point_jit_plan(matvec, x, **kwargs)
    return _jcb_mat_inverse_action_point_jit_callable(matvec, x, **kwargs)


_jcb_mat_logdet_slq_point_jit_callable = jax.jit(jcb_mat_logdet_slq_point, static_argnames=("matvec", "steps", "adjoint_matvec"))
_jcb_mat_logdet_slq_point_jit_plan = jax.jit(_jcb_mat_logdet_slq_point_plan_kernel, static_argnames=("steps",))


def jcb_mat_logdet_slq_point_jit(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_logdet_slq_point_jit_plan(matvec, probes, steps=steps, adjoint_matvec=adjoint_matvec)
    return _jcb_mat_logdet_slq_point_jit_callable(matvec, probes, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_det_slq_point_jit_callable = jax.jit(jcb_mat_det_slq_point, static_argnames=("matvec", "steps", "adjoint_matvec"))
_jcb_mat_det_slq_point_jit_plan = jax.jit(_jcb_mat_det_slq_point_plan_kernel, static_argnames=("steps",))


def jcb_mat_det_slq_point_jit(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_det_slq_point_jit_plan(matvec, probes, steps=steps, adjoint_matvec=adjoint_matvec)
    return _jcb_mat_det_slq_point_jit_callable(matvec, probes, steps=steps, adjoint_matvec=adjoint_matvec)


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
    "jcb_mat_det_point",
    "jcb_mat_det_basic",
    "jcb_mat_inv_point",
    "jcb_mat_inv_basic",
    "jcb_mat_sqr_point",
    "jcb_mat_sqr_basic",
    "jcb_mat_trace_point",
    "jcb_mat_trace_basic",
    "jcb_mat_norm_fro_point",
    "jcb_mat_norm_fro_basic",
    "jcb_mat_norm_1_point",
    "jcb_mat_norm_1_basic",
    "jcb_mat_norm_inf_point",
    "jcb_mat_norm_inf_basic",
    "jcb_mat_dense_operator",
    "jcb_mat_dense_operator_adjoint",
    "jcb_mat_dense_operator_rmatvec",
    "jcb_mat_dense_operator_plan_prepare",
    "jcb_mat_dense_operator_rmatvec_plan_prepare",
    "jcb_mat_dense_operator_adjoint_plan_prepare",
    "jcb_mat_bcoo_operator",
    "jcb_mat_bcoo_operator_adjoint",
    "jcb_mat_bcoo_operator_rmatvec",
    "jcb_mat_bcoo_operator_plan_prepare",
    "jcb_mat_bcoo_operator_rmatvec_plan_prepare",
    "jcb_mat_bcoo_operator_adjoint_plan_prepare",
    "jcb_mat_operator_plan_apply",
    "jcb_mat_rmatvec_point",
    "jcb_mat_rmatvec_basic",
    "jcb_mat_arnoldi_hessenberg_adjoint",
    "jcb_mat_cg_fixed_iterations",
    "jcb_mat_solve_action_point",
    "jcb_mat_solve_action_basic",
    "jcb_mat_solve_action_hermitian_point",
    "jcb_mat_solve_action_hpd_point",
    "jcb_mat_solve_action_with_diagnostics_point",
    "jcb_mat_solve_action_with_diagnostics_basic",
    "jcb_mat_inverse_action_point",
    "jcb_mat_inverse_action_basic",
    "jcb_mat_inverse_action_hermitian_point",
    "jcb_mat_inverse_action_hpd_point",
    "jcb_mat_inverse_action_with_diagnostics_point",
    "jcb_mat_inverse_action_with_diagnostics_basic",
    "jcb_mat_operator_apply_point",
    "jcb_mat_operator_apply_basic",
    "jcb_mat_poly_action_point",
    "jcb_mat_poly_action_basic",
    "jcb_mat_expm_action_point",
    "jcb_mat_expm_action_basic",
    "jcb_mat_arnoldi_hessenberg_point",
    "jcb_mat_arnoldi_diagnostics_point",
    "jcb_mat_funm_action_arnoldi_point",
    "jcb_mat_funm_action_arnoldi_with_diagnostics_point",
    "jcb_mat_funm_integrand_arnoldi_point",
    "jcb_mat_dense_funm_general_eig_point",
    "jcb_mat_dense_funm_hermitian_eigh_point",
    "jcb_mat_funm_action_arnoldi_dense_point",
    "jcb_mat_log_action_arnoldi_point",
    "jcb_mat_log_action_arnoldi_basic",
    "jcb_mat_sqrt_action_arnoldi_point",
    "jcb_mat_sqrt_action_arnoldi_basic",
    "jcb_mat_root_action_arnoldi_point",
    "jcb_mat_root_action_arnoldi_basic",
    "jcb_mat_sign_action_arnoldi_point",
    "jcb_mat_sign_action_arnoldi_basic",
    "jcb_mat_sin_action_arnoldi_point",
    "jcb_mat_sin_action_arnoldi_basic",
    "jcb_mat_cos_action_arnoldi_point",
    "jcb_mat_cos_action_arnoldi_basic",
    "jcb_mat_sinh_action_arnoldi_point",
    "jcb_mat_sinh_action_arnoldi_basic",
    "jcb_mat_cosh_action_arnoldi_point",
    "jcb_mat_cosh_action_arnoldi_basic",
    "jcb_mat_tanh_action_arnoldi_point",
    "jcb_mat_tanh_action_arnoldi_basic",
    "jcb_mat_log_action_hermitian_point",
    "jcb_mat_log_action_hermitian_basic",
    "jcb_mat_sqrt_action_hermitian_point",
    "jcb_mat_sqrt_action_hermitian_basic",
    "jcb_mat_root_action_hermitian_point",
    "jcb_mat_root_action_hermitian_basic",
    "jcb_mat_sign_action_hermitian_point",
    "jcb_mat_sign_action_hermitian_basic",
    "jcb_mat_pow_action_hermitian_point",
    "jcb_mat_pow_action_hermitian_basic",
    "jcb_mat_log_action_hpd_point",
    "jcb_mat_sqrt_action_hpd_point",
    "jcb_mat_root_action_hpd_point",
    "jcb_mat_pow_action_arnoldi_point",
    "jcb_mat_pow_action_arnoldi_basic",
    "jcb_mat_log_action_arnoldi_dense_point",
    "jcb_mat_sqrt_action_arnoldi_dense_point",
    "jcb_mat_root_action_arnoldi_dense_point",
    "jcb_mat_sign_action_arnoldi_dense_point",
    "jcb_mat_sin_action_arnoldi_dense_point",
    "jcb_mat_cos_action_arnoldi_dense_point",
    "jcb_mat_sinh_action_arnoldi_dense_point",
    "jcb_mat_cosh_action_arnoldi_dense_point",
    "jcb_mat_tanh_action_arnoldi_dense_point",
    "jcb_mat_pow_action_arnoldi_dense_point",
    "jcb_mat_expm_action_arnoldi_restarted_point",
    "jcb_mat_expm_action_arnoldi_block_point",
    "jcb_mat_expm_action_arnoldi_restarted_with_diagnostics_point",
    "jcb_mat_trace_integrand_point",
    "jcb_mat_funm_trace_integrand_arnoldi_point",
    "jcb_mat_trace_estimator_point",
    "jcb_mat_trace_estimator_with_diagnostics_point",
    "jcb_mat_logdet_slq_point",
    "jcb_mat_logdet_slq_basic",
    "jcb_mat_logdet_slq_hermitian_point",
    "jcb_mat_logdet_slq_hpd_point",
    "jcb_mat_logdet_slq_with_diagnostics_point",
    "jcb_mat_logdet_slq_with_diagnostics_basic",
    "jcb_mat_det_slq_point",
    "jcb_mat_det_slq_basic",
    "jcb_mat_det_slq_hermitian_point",
    "jcb_mat_det_slq_hpd_point",
    "jcb_mat_det_slq_with_diagnostics_point",
    "jcb_mat_det_slq_with_diagnostics_basic",
    "jcb_mat_logdet_leja_hutchpp_point",
    "jcb_mat_logdet_leja_hutchpp_with_diagnostics_point",
    "jcb_mat_det_leja_hutchpp_point",
    "jcb_mat_det_leja_hutchpp_with_diagnostics_point",
    "jcb_mat_bcoo_gershgorin_bounds",
    "jcb_mat_bcoo_logdet_leja_hutchpp_point",
    "jcb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point",
    "jcb_mat_bcoo_det_leja_hutchpp_point",
    "jcb_mat_log_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sqrt_action_arnoldi_with_diagnostics_point",
    "jcb_mat_root_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sign_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sin_action_arnoldi_with_diagnostics_point",
    "jcb_mat_cos_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sinh_action_arnoldi_with_diagnostics_point",
    "jcb_mat_cosh_action_arnoldi_with_diagnostics_point",
    "jcb_mat_tanh_action_arnoldi_with_diagnostics_point",
    "jcb_mat_pow_action_arnoldi_with_diagnostics_point",
    "jcb_mat_log_action_leja_point",
    "jcb_mat_log_action_leja_with_diagnostics_point",
    "jcb_mat_hutchpp_trace_point",
    "jcb_mat_rademacher_probes_like",
    "jcb_mat_normal_probes_like",
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
    "jcb_mat_expm_action_basic_jit",
    "jcb_mat_logm",
    "jcb_mat_sqrtm",
    "jcb_mat_rootm",
    "jcb_mat_signm",
]
