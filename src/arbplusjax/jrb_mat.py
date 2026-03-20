from __future__ import annotations

"""Jones real matrix-free subsystem scaffold and substrate.

This module is a separate Jones-labeled subsystem for new matrix-free work.
It does not replace `arb_mat`; `arb_mat` remains the canonical Arb/FLINT-style
JAX extension surface for real interval matrices.

Current implemented substrate:
- layout contracts for interval matrices/vectors
- point/basic matmul
- point/basic matvec
- point/basic solve
- point/basic triangular_solve
- point/basic lu
- matrix-free point/basic action kernels for polynomial actions and expm-actions

Planned scope beyond this substrate:
- operator-first Lanczos / Krylov matrix-function actions
- contour-integral matrix logarithm / roots
- AD-aware matrix-function kernels with repo-standard engineering constraints

Provenance:
- classification: new
- base_names: jrb_mat
- module lineage: Jones matrix-function subsystem for real interval matrices
- naming policy: see docs/standards/function_naming.md
- registry report: see docs/status/reports/function_implementation_index.md
"""

from functools import partial
from typing import NamedTuple
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp

from . import checks
from . import double_interval as di
from . import iterative_solvers
from . import mat_common
from . import matrix_free_basic
from . import matrix_free_core
from . import sparse_common


PROVENANCE = {
    "classification": "new",
    "base_names": ("jrb_mat",),
    "module_lineage": "Jones matrix-function subsystem for real interval matrices",
    "naming_policy": "docs/standards/function_naming.md",
    "registry_report": "docs/status/reports/function_implementation_index.md",
}


class JrbMatKrylovDiagnostics(NamedTuple):
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


class JrbMatSelectedInverseDiagnostics(NamedTuple):
    algorithm_code: jax.Array
    partition_count: jax.Array
    overlap: jax.Array
    block_size: jax.Array
    max_local_size: jax.Array
    mean_local_size: jax.Array
    correction_probe_count: jax.Array
    correction_used: jax.Array
    converged_probe_count: jax.Array
    final_residual_max: jax.Array


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


def _jrb_operator_vector(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return _jrb_mid_vector(arr)
    return jnp.asarray(arr, dtype=jnp.float64)


def _jrb_point_interval(x: jax.Array) -> jax.Array:
    return di.interval(di._below(x), di._above(x))


def _jrb_round_basic(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(x, prec_bits)


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
    x = jnp.linalg.solve(_jrb_mid_matrix(a), _jrb_mid_vector(b)[..., None])[..., 0]
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


def jrb_mat_det_point(a: jax.Array) -> jax.Array:
    """Determinant of dense interval matrix - point version."""
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    det_val = jnp.linalg.det(mid)
    return _jrb_point_interval(det_val)


def jrb_mat_det_basic(a: jax.Array) -> jax.Array:
    """Determinant of dense interval matrix - basic interval version."""
    return jrb_mat_det_point(a)


def jrb_mat_inv_point(a: jax.Array) -> jax.Array:
    """Matrix inverse - point version."""
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    inv_val = jnp.linalg.inv(mid)
    return _jrb_point_interval(inv_val)


def jrb_mat_inv_basic(a: jax.Array) -> jax.Array:
    """Matrix inverse - basic interval version."""
    return jrb_mat_inv_point(a)


def jrb_mat_sqr_point(a: jax.Array) -> jax.Array:
    """Matrix square - point version."""
    return jrb_mat_matmul_point(a, a)


def jrb_mat_sqr_basic(a: jax.Array) -> jax.Array:
    """Matrix square - basic interval version."""
    return jrb_mat_matmul_basic(a, a)


def jrb_mat_trace_point(a: jax.Array) -> jax.Array:
    """Trace of dense interval matrix - point version."""
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    trace_val = jnp.trace(mid, axis1=-2, axis2=-1)
    return _jrb_point_interval(trace_val)


def jrb_mat_trace_basic(a: jax.Array) -> jax.Array:
    """Trace of dense interval matrix - basic interval version."""
    return jrb_mat_trace_point(a)


def jrb_mat_norm_fro_point(a: jax.Array) -> jax.Array:
    """Frobenius norm of dense interval matrix - point version."""
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    norm_val = jnp.linalg.norm(mid, ord='fro')
    return _jrb_point_interval(norm_val)


def jrb_mat_norm_fro_basic(a: jax.Array) -> jax.Array:
    """Frobenius norm of dense interval matrix - basic interval version."""
    return jrb_mat_norm_fro_point(a)


def jrb_mat_norm_1_point(a: jax.Array) -> jax.Array:
    """1-norm of dense interval matrix - point version."""
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    norm_val = jnp.linalg.norm(mid, ord=1)
    return _jrb_point_interval(norm_val)


def jrb_mat_norm_1_basic(a: jax.Array) -> jax.Array:
    """1-norm of dense interval matrix - basic interval version."""
    return jrb_mat_norm_1_point(a)


def jrb_mat_norm_inf_point(a: jax.Array) -> jax.Array:
    """Infinity norm of dense interval matrix - point version."""
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    norm_val = jnp.linalg.norm(mid, ord=jnp.inf)
    return _jrb_point_interval(norm_val)


def jrb_mat_norm_inf_basic(a: jax.Array) -> jax.Array:
    """Infinity norm of dense interval matrix - basic interval version."""
    return jrb_mat_norm_inf_point(a)


def jrb_mat_dense_operator(a: jax.Array):
    """Return a matrix-free midpoint matvec closure for a dense interval matrix."""
    return matrix_free_core.dense_operator(_jrb_mid_matrix(a), midpoint_vector=_jrb_mid_vector)


def jrb_mat_dense_operator_adjoint(a: jax.Array):
    """Return the adjoint midpoint matvec closure for a dense interval matrix."""
    return matrix_free_core.dense_operator_adjoint(_jrb_mid_matrix(a), midpoint_vector=_jrb_mid_vector, conjugate=False)


def jrb_mat_dense_operator_rmatvec(a: jax.Array):
    """Return the transpose midpoint matvec closure for right-vector products."""
    return matrix_free_core.dense_operator_rmatvec(_jrb_mid_matrix(a), midpoint_vector=_jrb_mid_vector)


def jrb_mat_dense_operator_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jrb_mid_matrix(a), orientation="forward", algebra="jrb")


def jrb_mat_dense_operator_rmatvec_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jrb_mid_matrix(a), orientation="transpose", algebra="jrb")


def jrb_mat_dense_operator_adjoint_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jrb_mid_matrix(a), orientation="transpose", algebra="jrb")


def _jrb_sparse_to_bcoo(x):
    return matrix_free_core.canonicalize_sparse_bcoo(
        x,
        algebra="jrb",
        sparse_common=sparse_common,
        label="jrb_mat.sparse_to_bcoo",
    )


def jrb_mat_bcoo_operator(a: sparse_common.SparseBCOO):
    """Return a matrix-free midpoint matvec closure for a real sparse BCOO matrix."""
    return matrix_free_core.sparse_bcoo_operator(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        midpoint_vector=_jrb_operator_vector,
        dtype=jnp.float64,
        algebra="jrb",
        label="jrb_mat.bcoo_operator",
    )


def jrb_mat_sparse_operator(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jrb_mat_bcoo_operator(_jrb_sparse_to_bcoo(a))


def jrb_mat_bcoo_operator_adjoint(a: sparse_common.SparseBCOO):
    """Return the adjoint matrix-free midpoint matvec closure for a real sparse BCOO matrix."""
    return matrix_free_core.sparse_bcoo_operator_adjoint(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        midpoint_vector=_jrb_operator_vector,
        dtype=jnp.float64,
        algebra="jrb",
        label="jrb_mat.bcoo_operator_adjoint",
        conjugate=False,
    )


def jrb_mat_sparse_operator_adjoint(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jrb_mat_bcoo_operator_adjoint(_jrb_sparse_to_bcoo(a))


def jrb_mat_bcoo_operator_rmatvec(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_rmatvec(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        midpoint_vector=_jrb_operator_vector,
        dtype=jnp.float64,
        algebra="jrb",
        label="jrb_mat.bcoo_operator_rmatvec",
    )


def jrb_mat_sparse_operator_rmatvec(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jrb_mat_bcoo_operator_rmatvec(_jrb_sparse_to_bcoo(a))


def jrb_mat_bcoo_operator_plan_prepare(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        orientation="forward",
        algebra="jrb",
    )


def jrb_mat_sparse_operator_plan_prepare(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jrb_mat_bcoo_operator_plan_prepare(_jrb_sparse_to_bcoo(a))


def jrb_mat_bcoo_operator_rmatvec_plan_prepare(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        orientation="transpose",
        algebra="jrb",
    )


def jrb_mat_sparse_operator_rmatvec_plan_prepare(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jrb_mat_bcoo_operator_rmatvec_plan_prepare(_jrb_sparse_to_bcoo(a))


def jrb_mat_bcoo_operator_adjoint_plan_prepare(a: sparse_common.SparseBCOO):
    return matrix_free_core.sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=sparse_common.as_sparse_bcoo,
        sparse_bcoo_cls=sparse_common.SparseBCOO,
        orientation="transpose",
        algebra="jrb",
    )


def jrb_mat_sparse_operator_adjoint_plan_prepare(a: sparse_common.SparseCOO | sparse_common.SparseCSR | sparse_common.SparseBCOO):
    return jrb_mat_bcoo_operator_adjoint_plan_prepare(_jrb_sparse_to_bcoo(a))


def jrb_mat_symmetric_operator(a: jax.Array):
    return jrb_mat_dense_operator(a)


def jrb_mat_symmetric_operator_plan_prepare(a: jax.Array):
    return jrb_mat_dense_operator_plan_prepare(a)


def jrb_mat_spd_operator(a: jax.Array):
    return jrb_mat_dense_operator(a)


def jrb_mat_spd_operator_plan_prepare(a: jax.Array):
    return jrb_mat_dense_operator_plan_prepare(a)


def jrb_mat_operator_plan_apply(plan: matrix_free_core.OperatorPlan, x: jax.Array) -> jax.Array:
    return jrb_mat_operator_apply_point(plan, x)


def _jrb_apply_operator_mid(operator, x: jax.Array) -> jax.Array:
    return matrix_free_core.operator_apply_midpoint(
        operator,
        x,
        midpoint_vector=_jrb_operator_vector,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )


def jrb_mat_rmatvec_point(a: jax.Array, x: jax.Array) -> jax.Array:
    return jrb_mat_operator_apply_point(jrb_mat_dense_operator_rmatvec(a), x)


def jrb_mat_rmatvec_basic(a: jax.Array, x: jax.Array) -> jax.Array:
    return jrb_mat_rmatvec_point(a, x)


def jrb_mat_lanczos_tridiag_adjoint(matvec, *, krylov_depth: int, reortho: str = "full", custom_vjp: bool = True):
    return matrix_free_core.matfree_adjoints.lanczos_tridiag(
        matvec,
        krylov_depth=krylov_depth,
        reortho=reortho,
        custom_vjp=custom_vjp,
    )


def jrb_mat_cg_fixed_iterations(*, num_matvecs: int):
    return matrix_free_core.matfree_adjoints.cg_fixed_iterations(num_matvecs=num_matvecs)


def jrb_mat_solve_action_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
) -> jax.Array:
    value, _ = jrb_mat_solve_action_with_diagnostics_point(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
    )
    return value


def jrb_mat_solve_action_basic(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.solve_action_basic(
        jrb_mat_solve_action_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_solve_action_with_diagnostics_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
):
    b = jrb_mat_as_interval_vector(b)
    b_mid = _jrb_mid_vector(b)
    x0_mid = None if x0 is None else _jrb_mid_vector(x0)
    mv = lambda v: _jrb_apply_operator_mid(matvec, _jrb_point_interval(v))
    precond = None if preconditioner is None else (lambda v: _jrb_apply_operator_mid(preconditioner, _jrb_point_interval(v)))
    solver = iterative_solvers.cg if symmetric else iterative_solvers.gmres
    x_mid, info = solver(mv, b_mid, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
    out = _jrb_point_interval(x_mid)
    finite = jnp.all(jnp.isfinite(x_mid), axis=-1)
    return jnp.where(finite[..., None], out, _full_interval_like(out)), info


def jrb_mat_solve_action_with_diagnostics_basic(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.solve_action_with_diagnostics_basic(
        jrb_mat_solve_action_with_diagnostics_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_inverse_action_point(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
) -> jax.Array:
    return jrb_mat_solve_action_point(
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
    )


def jrb_mat_inverse_action_basic(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.inverse_action_basic(
        jrb_mat_inverse_action_point,
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_inverse_action_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
):
    return jrb_mat_solve_action_with_diagnostics_point(
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
    )


def jrb_mat_inverse_action_with_diagnostics_basic(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.inverse_action_with_diagnostics_basic(
        jrb_mat_inverse_action_with_diagnostics_point,
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_bcoo_parametric_operator(indices: jax.Array, *, shape: tuple[int, int]):
    """Return a differentiable matrix-free operator closure for a fixed BCOO sparsity pattern."""
    idx = jnp.asarray(indices, dtype=jnp.int32)
    checks.check(idx.ndim == 2 and idx.shape[-1] == 2, "jrb_mat.bcoo_parametric_operator.indices")
    rows, cols = int(shape[0]), int(shape[1])
    row_ids = idx[:, 0]
    col_ids = idx[:, 1]

    def matvec(v: jax.Array, data: jax.Array) -> jax.Array:
        vv = _jrb_operator_vector(v)
        vals = jnp.asarray(data, dtype=jnp.float64) * vv[col_ids]
        return jax.ops.segment_sum(vals, row_ids, rows)

    return matvec


def jrb_mat_scipy_csr_operator(csr):
    """Return a matrix-free midpoint matvec closure for a SciPy CSR matrix via SparseBCOO."""
    bcoo = sparse_common.scipy_csr_to_sparse_bcoo(csr, algebra="jrb", dtype=jnp.float64)
    return jrb_mat_bcoo_operator(bcoo)


def jrb_mat_bcoo_gershgorin_bounds(a: sparse_common.SparseBCOO, *, eps: float = 1e-12) -> tuple[jax.Array, jax.Array]:
    """Return conservative Gershgorin spectral bounds for a real sparse matrix."""
    a = sparse_common.as_sparse_bcoo(a, algebra="jrb", label="jrb_mat.bcoo_gershgorin_bounds")
    checks.check_equal(a.rows, a.cols, "jrb_mat.bcoo_gershgorin_bounds.square")
    rows = jnp.asarray(a.indices[:, 0], dtype=jnp.int32)
    cols = jnp.asarray(a.indices[:, 1], dtype=jnp.int32)
    data = jnp.asarray(a.data, dtype=jnp.float64)
    n = int(a.rows)
    abs_row_sum = jax.ops.segment_sum(jnp.abs(data), rows, n)
    diag = jax.ops.segment_sum(jnp.where(rows == cols, data, 0.0), rows, n)
    radii = abs_row_sum - jnp.abs(diag)
    lower = jnp.min(diag - radii)
    upper = jnp.max(diag + radii)
    return (
        jnp.maximum(jnp.asarray(eps, dtype=jnp.float64), jnp.asarray(lower, dtype=jnp.float64)),
        jnp.asarray(upper, dtype=jnp.float64),
    )


def jrb_mat_bcoo_spectral_bounds_adaptive(
    a: sparse_common.SparseBCOO,
    *,
    steps: int = 16,
    safety_margin: float = 1.25,
    eps: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """Return a heuristic sparse-SPD spectral interval from Gershgorin plus short Lanczos.

    This estimator is intended for the point-mode Leja path. It is narrower than raw
    Gershgorin in many cases, but it is not a rigorous enclosure certificate.
    """
    a = sparse_common.as_sparse_bcoo(a, algebra="jrb", label="jrb_mat.bcoo_spectral_bounds_adaptive")
    checks.check_equal(a.rows, a.cols, "jrb_mat.bcoo_spectral_bounds_adaptive.square")
    g_lower, g_upper = jrb_mat_bcoo_gershgorin_bounds(a, eps=eps)
    n = int(a.rows)
    k = max(1, min(int(steps), n))
    scale = jnp.sqrt(jnp.asarray(float(n), dtype=jnp.float64))
    starts = jnp.stack(
        [
            jnp.ones((n,), dtype=jnp.float64) / scale,
            jnp.linspace(1.0, 2.0, n, dtype=jnp.float64) / jnp.linalg.norm(jnp.linspace(1.0, 2.0, n, dtype=jnp.float64)),
            jax.random.normal(jax.random.PRNGKey(0), (n,), dtype=jnp.float64),
        ],
        axis=0,
    )
    starts = starts / jnp.linalg.norm(starts, axis=1, keepdims=True)
    matvec = jrb_mat_bcoo_operator(a)

    def estimate_from_start(start):
        _, t, _, betas = _jrb_mat_lanczos_tridiag_state_point(matvec, _jrb_point_interval(start), k)
        ritz = jnp.linalg.eigvalsh(t)
        tail = jnp.asarray(betas[k - 1], dtype=jnp.float64)
        return ritz[0], ritz[-1], tail

    ritz_lowers, ritz_uppers, tails = jax.vmap(estimate_from_start)(starts)
    tail = jnp.max(tails)
    local_eps = jnp.asarray(eps, dtype=jnp.float64)
    heuristic_lower = jnp.maximum(
        local_eps,
        jnp.min(ritz_lowers) - jnp.asarray(safety_margin, dtype=jnp.float64) * tail,
    )
    heuristic_upper = jnp.maximum(
        heuristic_lower + local_eps,
        jnp.max(ritz_uppers) + jnp.asarray(safety_margin, dtype=jnp.float64) * tail,
    )
    lower = jnp.maximum(local_eps, jnp.minimum(g_lower, heuristic_lower))
    upper = jnp.maximum(lower + local_eps, jnp.minimum(g_upper, heuristic_upper))
    return lower, upper


def _jrb_partition_contiguous_blocks(n: int, block_size: int) -> list[np.ndarray]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    return [
        np.arange(start, min(start + block_size, n), dtype=np.int32)
        for start in range(0, n, block_size)
    ]


def _jrb_bcoo_host_graph_payload(a: sparse_common.SparseBCOO) -> tuple[list[dict[int, float]], list[set[int]]]:
    idx = np.asarray(jax.device_get(a.indices), dtype=np.int32)
    data = np.asarray(jax.device_get(a.data), dtype=np.float64)
    rows = [dict() for _ in range(int(a.rows))]
    adj = [set((i,)) for i in range(int(a.rows))]
    for (r, c), val in zip(idx, data, strict=False):
        if val == 0.0:
            continue
        rows[int(r)][int(c)] = rows[int(r)].get(int(c), 0.0) + float(val)
        adj[int(r)].add(int(c))
        adj[int(c)].add(int(r))
    return rows, adj


def _jrb_expand_overlap(seed_rows: np.ndarray, adjacency: list[set[int]], overlap: int) -> np.ndarray:
    visited = set(int(i) for i in np.asarray(seed_rows, dtype=np.int32))
    frontier = set(visited)
    for _ in range(int(overlap)):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(adjacency[node])
        next_frontier.difference_update(visited)
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    return np.asarray(sorted(visited), dtype=np.int32)


def _jrb_prepare_selected_inverse_rows_bcoo(
    a: sparse_common.SparseBCOO,
    *,
    overlap: int,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, int, float]:
    rows_payload, adjacency = _jrb_bcoo_host_graph_payload(a)
    partitions = _jrb_partition_contiguous_blocks(int(a.rows), int(block_size))
    local_nodes_per_part = [_jrb_expand_overlap(part, adjacency, int(overlap)) for part in partitions]
    max_local_size = max(int(nodes.shape[0]) for nodes in local_nodes_per_part)
    mean_local_size = float(np.mean([int(nodes.shape[0]) for nodes in local_nodes_per_part]))
    num_parts = len(partitions)

    local_blocks = np.tile(np.eye(max_local_size, dtype=np.float64), (num_parts, 1, 1))
    local_nodes_padded = np.zeros((num_parts, max_local_size), dtype=np.int32)
    seed_local_pos_padded = -np.ones((num_parts, int(block_size)), dtype=np.int32)

    for part_idx, (seed_rows, local_nodes) in enumerate(zip(partitions, local_nodes_per_part, strict=False)):
        k_local = int(local_nodes.shape[0])
        local_pos = {int(g): i for i, g in enumerate(local_nodes.tolist())}
        block = np.zeros((k_local, k_local), dtype=np.float64)
        for global_row in local_nodes.tolist():
            row_pos = local_pos[int(global_row)]
            for global_col, value in rows_payload[int(global_row)].items():
                col_pos = local_pos.get(int(global_col))
                if col_pos is not None:
                    block[row_pos, col_pos] += float(value)
        local_blocks[part_idx, :k_local, :k_local] = block
        local_nodes_padded[part_idx, :k_local] = local_nodes
        for seed_pos, global_row in enumerate(seed_rows.tolist()):
            seed_local_pos_padded[part_idx, seed_pos] = local_pos[int(global_row)]

    batched_blocks = jnp.asarray(local_blocks, dtype=jnp.float64)
    inv_blocks = jax.vmap(jnp.linalg.inv)(batched_blocks)
    inv_blocks_host = np.asarray(jax.device_get(inv_blocks), dtype=np.float64)

    row_cols = np.zeros((int(a.rows), max_local_size), dtype=np.int32)
    row_vals = np.zeros((int(a.rows), max_local_size), dtype=np.float64)
    diag_est = np.zeros((int(a.rows),), dtype=np.float64)

    for part_idx, (seed_rows, local_nodes) in enumerate(zip(partitions, local_nodes_per_part, strict=False)):
        k_local = int(local_nodes.shape[0])
        local_cols = local_nodes_padded[part_idx, :k_local]
        for seed_offset, global_row in enumerate(seed_rows.tolist()):
            local_row = int(seed_local_pos_padded[part_idx, seed_offset])
            vals = inv_blocks_host[part_idx, local_row, :k_local]
            row_cols[int(global_row), :k_local] = local_cols
            row_vals[int(global_row), :k_local] = vals
            diag_est[int(global_row)] = vals[local_row]

    return (
        jnp.asarray(row_cols, dtype=jnp.int32),
        jnp.asarray(row_vals, dtype=jnp.float64),
        jnp.asarray(diag_est, dtype=jnp.float64),
        max_local_size,
        mean_local_size,
    )


def _jrb_selected_inverse_row_apply(row_cols: jax.Array, row_vals: jax.Array, v: jax.Array) -> jax.Array:
    vv = jnp.asarray(v, dtype=jnp.float64)
    return jnp.sum(row_vals * vv[row_cols], axis=1)


def jrb_mat_bcoo_inverse_diagonal_point(
    a: sparse_common.SparseBCOO,
    *,
    overlap: int = 1,
    block_size: int = 16,
    correction_probes: int = 0,
    key: jax.Array | None = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> jax.Array:
    value, _ = jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(
        a,
        overlap=overlap,
        block_size=block_size,
        correction_probes=correction_probes,
        key=key,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    return value


def jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(
    a: sparse_common.SparseBCOO,
    *,
    overlap: int = 1,
    block_size: int = 16,
    correction_probes: int = 0,
    key: jax.Array | None = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> tuple[jax.Array, JrbMatSelectedInverseDiagnostics]:
    """Estimate diag(A^{-1}) for sparse SPD matrices via local selected inverse rows plus stochastic correction."""
    a = sparse_common.as_sparse_bcoo(a, algebra="jrb", label="jrb_mat.bcoo_inverse_diagonal")
    checks.check_equal(a.rows, a.cols, "jrb_mat.bcoo_inverse_diagonal.square")
    checks.check(int(overlap) >= 0, "jrb_mat.bcoo_inverse_diagonal.overlap")
    checks.check(int(block_size) > 0, "jrb_mat.bcoo_inverse_diagonal.block_size")
    checks.check(int(correction_probes) >= 0, "jrb_mat.bcoo_inverse_diagonal.correction_probes")

    row_cols, row_vals, local_diag, max_local_size, mean_local_size = _jrb_prepare_selected_inverse_rows_bcoo(
        a,
        overlap=int(overlap),
        block_size=int(block_size),
    )
    matvec = jrb_mat_bcoo_operator(a)
    approx_apply = lambda v: _jrb_selected_inverse_row_apply(row_cols, row_vals, v)
    n = int(a.rows)

    correction_used = int(correction_probes) > 0
    converged_probe_count = 0
    final_residual_max = 0.0
    if correction_used:
        solve_key = jax.random.PRNGKey(0) if key is None else key
        probes = jax.random.rademacher(solve_key, (int(correction_probes), n), dtype=jnp.float64)

        def solve_probe(rhs):
            x, info = iterative_solvers.cg(
                matvec,
                rhs,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
                M=approx_apply,
            )
            return x, info["converged"], info["residuals"][-1]

        xs, converged_flags, final_residuals = jax.vmap(solve_probe)(probes)
        approx_xs = jax.vmap(approx_apply)(probes)
        correction = jnp.mean(probes * (xs - approx_xs), axis=0)
        diag_est = local_diag + correction
        converged_probe_count = int(jnp.sum(converged_flags).item())
        final_residual_max = float(jnp.max(final_residuals).item())
    else:
        diag_est = local_diag

    partition_count = (n + int(block_size) - 1) // int(block_size)
    diagnostics = JrbMatSelectedInverseDiagnostics(
        algorithm_code=jnp.asarray(5, dtype=jnp.int32),
        partition_count=jnp.asarray(partition_count, dtype=jnp.int32),
        overlap=jnp.asarray(int(overlap), dtype=jnp.int32),
        block_size=jnp.asarray(int(block_size), dtype=jnp.int32),
        max_local_size=jnp.asarray(max_local_size, dtype=jnp.int32),
        mean_local_size=jnp.asarray(mean_local_size, dtype=jnp.float64),
        correction_probe_count=jnp.asarray(int(correction_probes), dtype=jnp.int32),
        correction_used=jnp.asarray(correction_used),
        converged_probe_count=jnp.asarray(converged_probe_count, dtype=jnp.int32),
        final_residual_max=jnp.asarray(final_residual_max, dtype=jnp.float64),
    )
    return jnp.asarray(diag_est, dtype=jnp.float64), diagnostics


def jrb_mat_operator_apply_point(matvec, x: jax.Array) -> jax.Array:
    return matrix_free_core.operator_apply_point(
        matvec,
        x,
        midpoint_apply=_jrb_apply_operator_mid,
        coerce_vector=jrb_mat_as_interval_vector,
        point_from_midpoint=_jrb_point_interval,
        full_like=_full_interval_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(y), axis=-1),
        dtype=jnp.float64,
    )


def jrb_mat_operator_apply_basic(matvec, x: jax.Array) -> jax.Array:
    return matrix_free_basic.operator_apply_basic(
        jrb_mat_operator_apply_point,
        matvec,
        x,
        round_output=_jrb_round_basic,
        prec_bits=di.DEFAULT_PREC_BITS,
    )


def _jrb_leja_points_interval_point(
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


def _jrb_log_leja_scaling_params(
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


def _jrb_newton_divided_differences_point(nodes: jax.Array, scalar_fun) -> jax.Array:
    coeffs = jnp.asarray(scalar_fun(nodes), dtype=jnp.float64)
    degree = int(nodes.shape[0])
    for j in range(1, degree):
        numer = coeffs[j:] - coeffs[j - 1:-1]
        denom = nodes[j:] - nodes[: degree - j]
        coeffs = coeffs.at[j:].set(numer / denom)
    return coeffs


def _jrb_log_leja_coefficients_point(
    nodes: jax.Array,
    center: jax.Array,
    gamma: jax.Array,
) -> jax.Array:
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    floor = jnp.asarray(1e-30, dtype=jnp.float64)
    return _jrb_newton_divided_differences_point(
        nodes,
        lambda t: jnp.log(jnp.maximum(center + gamma * t, floor)),
    )


def _jrb_log_leja_setup_point(
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    degree: int,
    *,
    candidate_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    alpha_scale, center, gamma, _ = _jrb_log_leja_scaling_params(spectral_bounds)
    nodes = _jrb_leja_points_interval_point(
        jnp.asarray(-2.0, dtype=jnp.float64),
        jnp.asarray(2.0, dtype=jnp.float64),
        degree,
        candidate_count=candidate_count,
    )
    coeffs = _jrb_log_leja_coefficients_point(nodes, center, gamma)
    return alpha_scale, center, gamma, nodes, coeffs


def _jrb_log_action_coordinate_shortcut_point(
    matvec,
    x: jax.Array,
    *,
    rtol: float,
    atol: float,
) -> tuple[jax.Array, jax.Array]:
    x = jrb_mat_as_interval_vector(x)
    basis = _jrb_mid_vector(x)
    zeros = _jrb_point_interval(jnp.zeros_like(basis))
    tiny = jnp.asarray(1e-30, dtype=jnp.float64)
    local_rtol = jnp.asarray(rtol, dtype=jnp.float64)
    local_atol = jnp.asarray(atol, dtype=jnp.float64)
    abs_basis = jnp.abs(basis)
    pivot = jnp.argmax(abs_basis)
    dominant = abs_basis[pivot]
    off_norm = jnp.linalg.norm(basis.at[pivot].set(0.0))
    coordinate_scale = local_atol + local_rtol * jnp.maximum(dominant, jnp.asarray(1.0, dtype=jnp.float64))
    looks_coordinate = off_norm <= coordinate_scale

    def no_shortcut(_):
        return zeros, jnp.asarray(False)

    def maybe_shortcut(_):
        norm_sq = jnp.real(jnp.vdot(basis, basis))

        def zero_branch(__):
            return zeros, jnp.asarray(True)

        def eigen_branch(__):
            applied = jnp.asarray(matvec(_jrb_point_interval(basis)), dtype=jnp.float64)
            basis_norm = jnp.sqrt(norm_sq)
            rayleigh = jnp.real(jnp.vdot(basis, applied)) / norm_sq
            residual = applied - rayleigh * basis
            residual_norm = jnp.linalg.norm(residual)
            scale = local_atol + local_rtol * jnp.maximum(
                jnp.linalg.norm(applied),
                jnp.maximum(jnp.abs(rayleigh) * basis_norm, jnp.asarray(1.0, dtype=jnp.float64)),
            )
            use_shortcut = jnp.logical_and(rayleigh > 0.0, residual_norm <= scale)
            exact = _jrb_point_interval(jnp.log(jnp.maximum(rayleigh, tiny)) * basis)
            return exact, use_shortcut

        return lax.cond(norm_sq <= tiny, zero_branch, eigen_branch, operand=None)

    return lax.cond(looks_coordinate, maybe_shortcut, no_shortcut, operand=None)


def _jrb_funm_action_newton_point(
    x: jax.Array,
    nodes: jax.Array,
    coeffs: jax.Array,
    step_fn,
    *,
    constant_shift: jax.Array = jnp.asarray(0.0, dtype=jnp.float64),
) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    basis = _jrb_mid_vector(x)
    acc = (constant_shift + coeffs[0]) * basis
    for k in range(1, int(coeffs.shape[0])):
        basis = step_fn(basis, nodes[k - 1])
        acc = acc + coeffs[k] * basis
    out = _jrb_point_interval(acc)
    finite = jnp.all(jnp.isfinite(acc), axis=-1)
    return jnp.where(finite[..., None], out, _full_interval_like(out))


def _jrb_funm_action_newton_adaptive_point(
    x: jax.Array,
    nodes: jax.Array,
    coeffs: jax.Array,
    step_fn,
    *,
    min_degree: int,
    rtol: float,
    atol: float,
    constant_shift: jax.Array = jnp.asarray(0.0, dtype=jnp.float64),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = jrb_mat_as_interval_vector(x)
    degree = int(coeffs.shape[0])
    if degree <= 0:
        raise ValueError("degree must be > 0")
    min_degree = max(1, min(int(min_degree), degree))
    basis0 = _jrb_mid_vector(x)
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
    out = _jrb_point_interval(acc)
    finite = jnp.all(jnp.isfinite(acc), axis=-1)
    return jnp.where(finite[..., None], out, _full_interval_like(out)), used_degree, tail_norm


def jrb_mat_log_action_leja_point(
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
    """Approximate log(A) x for SPD operators using Newton interpolation on Leja points."""
    total_degree = int(max_degree) if max_degree is not None else int(degree)
    if total_degree <= 0:
        raise ValueError("degree must be > 0")
    exact_value, use_shortcut = _jrb_log_action_coordinate_shortcut_point(matvec, x, rtol=rtol, atol=atol)
    alpha_scale, center, gamma, nodes, coeffs = _jrb_log_leja_setup_point(
        spectral_bounds,
        total_degree,
        candidate_count=candidate_count,
    )

    def step_fn(basis, node):
        applied = jnp.asarray(matvec(_jrb_point_interval(basis)), dtype=jnp.float64) / alpha_scale
        return (applied - center * basis) / gamma - node * basis

    def leja_branch(_):
        shift = jnp.log(alpha_scale)
        if max_degree is not None:
            value, _, _ = _jrb_funm_action_newton_adaptive_point(
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
        return _jrb_funm_action_newton_point(x, nodes, coeffs, step_fn, constant_shift=shift)

    return lax.cond(use_shortcut, lambda _: exact_value, leja_branch, operand=None)


def jrb_mat_log_action_leja_with_diagnostics_point(
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
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    total_degree = int(max_degree) if max_degree is not None else int(degree)
    if total_degree <= 0:
        raise ValueError("degree must be > 0")
    exact_value, use_shortcut = _jrb_log_action_coordinate_shortcut_point(matvec, x, rtol=rtol, atol=atol)
    alpha_scale, center, gamma, nodes, coeffs = _jrb_log_leja_setup_point(
        spectral_bounds,
        total_degree,
        candidate_count=candidate_count,
    )

    def step_fn(basis, node):
        applied = jnp.asarray(matvec(_jrb_point_interval(basis)), dtype=jnp.float64) / alpha_scale
        return (applied - center * basis) / gamma - node * basis

    def exact_branch(_):
        return exact_value, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0.0, dtype=jnp.float64)

    def leja_branch(_):
        shift = jnp.log(alpha_scale)
        if max_degree is not None:
            return _jrb_funm_action_newton_adaptive_point(
                x,
                nodes,
                coeffs,
                step_fn,
                min_degree=min_degree,
                rtol=rtol,
                atol=atol,
                constant_shift=shift,
            )
        value = _jrb_funm_action_newton_point(x, nodes, coeffs, step_fn, constant_shift=shift)
        used_degree = jnp.asarray(total_degree, dtype=jnp.int32)
        basis = _jrb_mid_vector(x)
        tail = (shift + coeffs[0]) * basis
        for k in range(1, total_degree):
            basis = step_fn(basis, nodes[k - 1])
            tail = coeffs[k] * basis
        tail_norm = jnp.linalg.norm(tail)
        return value, used_degree, tail_norm

    value, used_degree, tail_norm = lax.cond(use_shortcut, exact_branch, leja_branch, operand=None)
    algorithm_code = jnp.where(
        use_shortcut,
        jnp.asarray(4, dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
    )
    diag = JrbMatKrylovDiagnostics(
        algorithm_code=algorithm_code,
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


def _jrb_apply_block_action_point(action_fn, probes: jax.Array) -> jax.Array:
    coerced = di.as_interval(probes)
    outputs = jax.vmap(action_fn)(coerced)
    return di.midpoint(outputs)


def jrb_mat_hutchpp_trace_point(action_fn, sketch_probes: jax.Array, residual_probes: jax.Array) -> jax.Array:
    """Estimate tr(F) from an action oracle using Hutch++ probe partitions."""
    sketch = di.as_interval(sketch_probes)
    residual = di.as_interval(residual_probes)
    n = int(sketch.shape[-2] if sketch.shape[0] > 0 else residual.shape[-2])

    if sketch.shape[0] > 0:
        y_cols = jnp.swapaxes(_jrb_apply_block_action_point(action_fn, sketch), 0, 1)
        q, _ = jnp.linalg.qr(y_cols, mode="reduced")
        fq_cols = jnp.swapaxes(_jrb_apply_block_action_point(action_fn, jax.vmap(_jrb_point_interval)(q.T)), 0, 1)
        trace_lr = jnp.trace(q.T @ fq_cols)
    else:
        q = jnp.zeros((n, 0), dtype=jnp.float64)
        trace_lr = jnp.asarray(0.0, dtype=jnp.float64)

    if residual.shape[0] > 0:
        z = di.midpoint(residual)
        z_proj = z - (z @ q) @ q.T
        hz = _jrb_apply_block_action_point(action_fn, jax.vmap(_jrb_point_interval)(z_proj))
        residual_est = jnp.mean(jnp.sum(z_proj * hz, axis=-1))
    else:
        residual_est = jnp.asarray(0.0, dtype=jnp.float64)

    return jnp.asarray(trace_lr + residual_est, dtype=jnp.float64)


def jrb_mat_logdet_leja_hutchpp_point(
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
    """Estimate logdet(A) for SPD operators using Leja log-action plus Hutch++."""
    action_fn = lambda v: jrb_mat_log_action_leja_point(
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
    return jrb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)


def jrb_mat_det_leja_hutchpp_point(
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
        jrb_mat_logdet_leja_hutchpp_point(
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


def jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
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
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    value = jrb_mat_logdet_leja_hutchpp_point(
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
    reference = di.as_interval(sketch_probes)
    if reference.shape[0] > 0:
        _, action_diag = jrb_mat_log_action_leja_with_diagnostics_point(
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
    diag = JrbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(
            action_diag.algorithm_code if reference.shape[0] > 0 else 3,
            dtype=jnp.int32,
        ),
        steps=jnp.asarray(used_steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(used_steps, dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(0.0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(
            di.as_interval(sketch_probes).shape[0] + di.as_interval(residual_probes).shape[0],
            dtype=jnp.int32,
        ),
    )
    return value, diag


def jrb_mat_det_leja_hutchpp_with_diagnostics_point(
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
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    value, diag = jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
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


def jrb_mat_bcoo_logdet_leja_hutchpp_point(
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
    bounds_steps: int = 16,
    bounds_safety_margin: float = 1.25,
) -> jax.Array:
    bounds = (
        jrb_mat_bcoo_spectral_bounds_adaptive(
            a,
            steps=bounds_steps,
            safety_margin=bounds_safety_margin,
        )
        if spectral_bounds is None
        else spectral_bounds
    )
    return jrb_mat_logdet_leja_hutchpp_point(
        jrb_mat_bcoo_operator(a),
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


def jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(
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
    bounds_steps: int = 16,
    bounds_safety_margin: float = 1.25,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    bounds = (
        jrb_mat_bcoo_spectral_bounds_adaptive(
            a,
            steps=bounds_steps,
            safety_margin=bounds_safety_margin,
        )
        if spectral_bounds is None
        else spectral_bounds
    )
    return jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        jrb_mat_bcoo_operator(a),
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


def jrb_mat_poly_action_point(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    return matrix_free_core.poly_action_point(
        matvec,
        x,
        coefficients,
        midpoint_apply=_jrb_apply_operator_mid,
        coerce_vector=jrb_mat_as_interval_vector,
        midpoint_vector=_jrb_mid_vector,
        point_from_midpoint=_jrb_point_interval,
        full_like=_full_interval_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(y), axis=-1),
        coeff_dtype=jnp.float64,
    )


def jrb_mat_poly_action_basic(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_poly_action_point,
        matvec,
        x,
        coefficients,
        round_output=_jrb_round_basic,
        prec_bits=di.DEFAULT_PREC_BITS,
    )


def jrb_mat_expm_action_point(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    return matrix_free_core.expm_action_point(
        matvec,
        x,
        terms=terms,
        midpoint_apply=_jrb_apply_operator_mid,
        coerce_vector=jrb_mat_as_interval_vector,
        midpoint_vector=_jrb_mid_vector,
        point_from_midpoint=_jrb_point_interval,
        full_like=_full_interval_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(y), axis=-1),
        scalar_dtype=jnp.float64,
    )


def jrb_mat_expm_action_basic(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_expm_action_point,
        matvec,
        x,
        terms=terms,
        round_output=_jrb_round_basic,
        prec_bits=di.DEFAULT_PREC_BITS,
    )


def _jrb_mat_lanczos_tridiag_state_point(matvec, x: jax.Array, steps: int):
    x = jrb_mat_as_interval_vector(x)
    if steps <= 0:
        raise ValueError("steps must be > 0")
    v0 = _jrb_mid_vector(x)
    beta0 = jnp.linalg.norm(v0)
    if steps > int(v0.shape[-1]):
        raise ValueError("steps must be <= vector dimension")
    q0 = v0 / jnp.maximum(beta0, jnp.asarray(1e-30, dtype=jnp.float64))
    dim = q0.shape[-1]

    def body(carry, _):
        q_prev, q_curr, beta_prev, basis, alphas, betas, k = carry
        z = _jrb_apply_operator_mid(matvec, _jrb_point_interval(q_curr))
        alpha = jnp.vdot(q_curr, z).real
        r = z - alpha * q_curr - beta_prev * q_prev
        # Full reorthogonalisation against accumulated basis.
        mask = (jnp.arange(steps, dtype=jnp.int32) < k).astype(jnp.float64)
        proj = (basis @ r) * mask
        r = r - proj @ basis
        beta = jnp.linalg.norm(r)
        q_next = jnp.where(beta > 1e-30, r / beta, jnp.zeros_like(r))
        basis = basis.at[k].set(q_curr)
        alphas = alphas.at[k].set(alpha)
        betas = betas.at[k].set(beta)
        return (q_curr, q_next, beta, basis, alphas, betas, k + 1), None

    init_basis = jnp.zeros((steps, dim), dtype=jnp.float64)
    init_alphas = jnp.zeros((steps,), dtype=jnp.float64)
    init_betas = jnp.zeros((steps,), dtype=jnp.float64)
    init = (
        jnp.zeros_like(q0),
        q0,
        jnp.asarray(0.0, dtype=jnp.float64),
        init_basis,
        init_alphas,
        init_betas,
        jnp.asarray(0, dtype=jnp.int32),
    )
    (_, _, _, basis, alphas, betas, _), _ = lax.scan(body, init, xs=None, length=steps)
    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    return basis, T, beta0, betas


def jrb_mat_lanczos_tridiag_point(matvec, x: jax.Array, steps: int):
    basis, T, beta0, _ = _jrb_mat_lanczos_tridiag_state_point(matvec, x, steps)
    return basis, T, beta0


def jrb_mat_lanczos_diagnostics_point(matvec, x: jax.Array, steps: int) -> JrbMatKrylovDiagnostics:
    basis, _, beta0, betas = _jrb_mat_lanczos_tridiag_state_point(matvec, x, steps)
    tail_norm = betas[-1]
    breakdown = tail_norm <= jnp.asarray(1e-30, dtype=jnp.float64)
    return matrix_free_core.krylov_diagnostics(
        JrbMatKrylovDiagnostics,
        algorithm_code=0,
        steps=steps,
        basis_dim=basis.shape[0],
        beta0=beta0,
        tail_norm=tail_norm,
        breakdown=breakdown,
    )


def _jrb_mat_funm_action_lanczos_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    return matrix_free_core.projected_krylov_action_point(
        matvec,
        x,
        dense_funm,
        steps,
        krylov_decomp=jrb_mat_lanczos_tridiag_point,
        point_from_midpoint=_jrb_point_interval,
        full_like=_full_interval_like,
        finite_mask_fn=lambda y: jnp.all(jnp.isfinite(y), axis=-1),
        coeff_dtype=jnp.float64,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3))
def jrb_mat_funm_action_lanczos_point(matvec, x: jax.Array, dense_funm, steps: int):
    return _jrb_mat_funm_action_lanczos_point_base(matvec, x, dense_funm, steps)


def _jrb_mat_funm_action_lanczos_point_fwd(matvec, x, dense_funm, steps):
    y = _jrb_mat_funm_action_lanczos_point_base(matvec, x, dense_funm, steps)
    return y, x


def _jrb_mat_funm_action_lanczos_point_bwd(matvec, dense_funm, steps, x, cotangent):
    adjoint = _jrb_mat_funm_action_lanczos_point_base(
        matvec,
        _jrb_point_interval(di.midpoint(cotangent)),
        dense_funm,
        steps,
    )
    return (adjoint,)


jrb_mat_funm_action_lanczos_point.defvjp(
    _jrb_mat_funm_action_lanczos_point_fwd,
    _jrb_mat_funm_action_lanczos_point_bwd,
)


def _jrb_mat_funm_integrand_lanczos_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    return matrix_free_core.projected_krylov_integrand_point(
        matvec,
        x,
        dense_funm,
        steps,
        krylov_decomp=jrb_mat_lanczos_tridiag_point,
        coeff_dtype=jnp.float64,
        scalar_dtype=jnp.float64,
        scalar_postprocess=lambda value: value.real,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3))
def jrb_mat_funm_integrand_lanczos_point(matvec, x: jax.Array, dense_funm, steps: int):
    return _jrb_mat_funm_integrand_lanczos_point_base(matvec, x, dense_funm, steps)


def _jrb_mat_funm_integrand_lanczos_point_fwd(matvec, x, dense_funm, steps):
    y = _jrb_mat_funm_integrand_lanczos_point_base(matvec, x, dense_funm, steps)
    return y, x


def _jrb_mat_funm_integrand_lanczos_point_bwd(matvec, dense_funm, steps, x, cotangent):
    action = _jrb_mat_funm_action_lanczos_point_base(matvec, x, dense_funm, steps)
    scale = jnp.asarray(cotangent, dtype=jnp.float64)
    return (_jrb_point_interval(2.0 * scale * di.midpoint(action)),)


jrb_mat_funm_integrand_lanczos_point.defvjp(
    _jrb_mat_funm_integrand_lanczos_point_fwd,
    _jrb_mat_funm_integrand_lanczos_point_bwd,
)


def jrb_mat_dense_funm_sym_eigh_point(scalar_fun):
    return matrix_free_core.dense_funm_hermitian_eigh(
        scalar_fun,
        dtype=jnp.float64,
        conjugate_right=False,
    )


def _jrb_dense_funm_point(a: jax.Array, scalar_fun) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    vals, vecs = jnp.linalg.eigh(mid)
    out = vecs @ jnp.diag(scalar_fun(vals)) @ vecs.T
    return _jrb_point_interval(out)


def jrb_mat_funm_action_lanczos_dense_point(a: jax.Array, x: jax.Array, dense_funm, steps: int) -> jax.Array:
    return _jrb_mat_funm_action_lanczos_point_base(jrb_mat_dense_operator(a), x, dense_funm, steps)


def jrb_mat_log_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.log), steps)


def jrb_mat_log_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_log_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_sqrt_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sqrt), steps)


def jrb_mat_sqrt_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_sqrt_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_root_action_lanczos_point(matvec, x: jax.Array, *, degree: int, steps: int) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return jrb_mat_funm_action_lanczos_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
    )


def jrb_mat_root_action_lanczos_basic(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    steps: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_root_action_lanczos_point,
        matvec,
        x,
        degree=degree,
        steps=steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_sign_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sign), steps)


def jrb_mat_sign_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_sign_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_sin_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sin), steps)


def jrb_mat_sin_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_sin_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_cos_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.cos), steps)


def jrb_mat_cos_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_cos_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_sinh_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sinh), steps)


def jrb_mat_sinh_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_sinh_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_cosh_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.cosh), steps)


def jrb_mat_cosh_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_cosh_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_tanh_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.tanh), steps)


def jrb_mat_tanh_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_tanh_action_lanczos_point,
        matvec,
        x,
        steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_log_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_log_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_sqrt_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sqrt_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_root_action_lanczos_dense_point(a: jax.Array, x: jax.Array, *, degree: int, steps: int) -> jax.Array:
    return jrb_mat_root_action_lanczos_point(jrb_mat_dense_operator(a), x, degree=degree, steps=steps)


def jrb_mat_sign_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sign_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_sin_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sin_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_cos_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_cos_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_sinh_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sinh_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_cosh_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_cosh_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_tanh_action_lanczos_dense_point(a: jax.Array, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_tanh_action_lanczos_point(jrb_mat_dense_operator(a), x, steps)


def jrb_mat_pow_action_lanczos_point(matvec, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    if exponent < 0:
        raise ValueError("exponent must be >= 0")
    return jrb_mat_funm_action_lanczos_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(lambda vals: jnp.power(vals, exponent)),
        steps,
    )


def jrb_mat_pow_action_lanczos_basic(
    matvec,
    x: jax.Array,
    *,
    exponent: int,
    steps: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_pow_action_lanczos_point,
        matvec,
        x,
        exponent=exponent,
        steps=steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_pow_action_lanczos_dense_point(a: jax.Array, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    return jrb_mat_pow_action_lanczos_point(jrb_mat_dense_operator(a), x, exponent=exponent, steps=steps)


def jrb_mat_pow_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    exponent: int,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    if exponent < 0:
        raise ValueError("exponent must be >= 0")
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(lambda vals: jnp.power(vals, exponent)),
        steps,
    )


def jrb_mat_expm_action_lanczos_restarted_point(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    restarts: int,
) -> jax.Array:
    dense_exp = jrb_mat_dense_funm_sym_eigh_point(jnp.exp)
    scaled_matvec = matrix_free_core.scaled_operator(
        matvec,
        float(1.0 / restarts),
    )

    return matrix_free_core.restarted_action_point(
        lambda y: jrb_mat_funm_action_lanczos_point(scaled_matvec, y, dense_exp, steps),
        jrb_mat_as_interval_vector(x),
        restarts=restarts,
    )


def jrb_mat_expm_action_lanczos_block_point(
    matvec,
    xs: jax.Array,
    *,
    steps: int,
    restarts: int = 1,
) -> jax.Array:
    return matrix_free_core.block_action_point(
        lambda x: jrb_mat_expm_action_lanczos_restarted_point(matvec, x, steps=steps, restarts=restarts),
        di.as_interval(xs),
    )


def jrb_mat_expm_action_lanczos_restarted_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    restarts: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    y = jrb_mat_expm_action_lanczos_restarted_point(matvec, x, steps=steps, restarts=restarts)
    diag = jrb_mat_lanczos_diagnostics_point(matvec, x, steps)
    diag = diag._replace(restart_count=jnp.asarray(restarts, dtype=jnp.int32))
    return y, diag


def jrb_mat_logm(a: jax.Array) -> jax.Array:
    return _jrb_dense_funm_point(a, jnp.log)


def jrb_mat_sqrtm(a: jax.Array) -> jax.Array:
    return _jrb_dense_funm_point(a, jnp.sqrt)


def jrb_mat_rootm(a: jax.Array, *, degree: int) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return _jrb_dense_funm_point(a, lambda vals: jnp.power(vals, inv_degree))


def jrb_mat_signm(a: jax.Array) -> jax.Array:
    return _jrb_dense_funm_point(a, jnp.sign)


def _jrb_mat_trace_integrand_point_base(matvec, x: jax.Array) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    x_mid = _jrb_mid_vector(x)
    y = _jrb_apply_operator_mid(matvec, x)
    return jnp.vdot(x_mid, y).real


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def jrb_mat_trace_integrand_point(matvec, x: jax.Array) -> jax.Array:
    return _jrb_mat_trace_integrand_point_base(matvec, x)


def _jrb_mat_trace_integrand_point_fwd(matvec, x):
    y = _jrb_mat_trace_integrand_point_base(matvec, x)
    return y, x


def _jrb_mat_trace_integrand_point_bwd(matvec, x, cotangent):
    action = _jrb_apply_operator_mid(matvec, x)
    scale = jnp.asarray(cotangent, dtype=jnp.float64)
    return (_jrb_point_interval(2.0 * scale * action),)


jrb_mat_trace_integrand_point.defvjp(
    _jrb_mat_trace_integrand_point_fwd,
    _jrb_mat_trace_integrand_point_bwd,
)


def jrb_mat_funm_trace_integrand_lanczos_point(matvec, x: jax.Array, scalar_fun, steps: int):
    dense_funm = jrb_mat_dense_funm_sym_eigh_point(scalar_fun)
    return jrb_mat_funm_integrand_lanczos_point(matvec, x, dense_funm, steps=steps)


def jrb_mat_trace_estimator_point(matvec, probes: jax.Array) -> jax.Array:
    return mat_common.estimator_mean(
        probes,
        di.as_interval,
        lambda v: jrb_mat_trace_integrand_point(matvec, v),
    )


def jrb_mat_logdet_slq_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return mat_common.estimator_mean(
        probes,
        di.as_interval,
        lambda v: jrb_mat_funm_trace_integrand_lanczos_point(matvec, v, jnp.log, steps=steps),
    )


def jrb_mat_logdet_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jrb_mat_logdet_slq_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb_point_interval,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_det_slq_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return matrix_free_core.det_from_logdet(jrb_mat_logdet_slq_point(matvec, probes, steps))


def jrb_mat_det_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jrb_mat_det_slq_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb_point_interval,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def _jrb_mat_logdet_slq_point_plan_kernel(matvec, probes: jax.Array, steps: int) -> jax.Array:
    dense_funm = jrb_mat_dense_funm_sym_eigh_point(jnp.log)
    return mat_common.estimator_mean(
        probes,
        di.as_interval,
        lambda v: _jrb_mat_funm_integrand_lanczos_point_base(matvec, v, dense_funm, steps),
    )


def _jrb_mat_det_slq_point_plan_kernel(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return matrix_free_core.det_from_logdet(_jrb_mat_logdet_slq_point_plan_kernel(matvec, probes, steps))


def jrb_mat_funm_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return mat_common.action_with_diagnostics(
        lambda xx: jrb_mat_funm_action_lanczos_point(matvec, xx, dense_funm, steps),
        lambda xx: jrb_mat_lanczos_diagnostics_point(matvec, xx, steps),
        x,
    )


def jrb_mat_log_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.log),
        steps,
    )


def jrb_mat_sqrt_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.sqrt),
        steps,
    )


def jrb_mat_root_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
    )


def jrb_mat_sign_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.sign),
        steps,
    )


def jrb_mat_sin_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.sin),
        steps,
    )


def jrb_mat_cos_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.cos),
        steps,
    )


def jrb_mat_sinh_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.sinh),
        steps,
    )


def jrb_mat_cosh_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.cosh),
        steps,
    )


def jrb_mat_tanh_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return jrb_mat_funm_action_lanczos_with_diagnostics_point(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(jnp.tanh),
        steps,
    )


def jrb_mat_logdet_slq_symmetric_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_logdet_slq_point(matvec, probes, steps)


def jrb_mat_det_slq_symmetric_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_det_slq_point(matvec, probes, steps)


def jrb_mat_logdet_slq_spd_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_logdet_slq_point(matvec, probes, steps)


def jrb_mat_det_slq_spd_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_det_slq_point(matvec, probes, steps)


def jrb_mat_trace_estimator_with_diagnostics_point(
    matvec,
    probes: jax.Array,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    coerced = di.as_interval(probes)
    first_steps = int(coerced[0].shape[-2])
    return mat_common.estimator_with_diagnostics(
        probes,
        coerce_probes=di.as_interval,
        estimator_fn=lambda xs: jrb_mat_trace_estimator_point(matvec, xs),
        diagnostics_fn=lambda first: jrb_mat_lanczos_diagnostics_point(matvec, first, first_steps),
        algorithm_code=1,
        steps=first_steps,
        basis_dim=first_steps,
    )


def jrb_mat_logdet_slq_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    return mat_common.estimator_with_diagnostics(
        probes,
        coerce_probes=di.as_interval,
        estimator_fn=lambda xs: jrb_mat_logdet_slq_point(matvec, xs, steps),
        diagnostics_fn=lambda first: jrb_mat_lanczos_diagnostics_point(matvec, first, steps),
        algorithm_code=2,
    )


def jrb_mat_logdet_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        jrb_mat_logdet_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb_point_interval,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_det_slq_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    value, diag = jrb_mat_logdet_slq_with_diagnostics_point(matvec, probes, steps)
    return matrix_free_core.det_from_logdet(value), diag


def jrb_mat_det_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        jrb_mat_det_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb_point_interval,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    return matrix_free_core.rademacher_probes_real(_jrb_point_interval, x.shape[-2], key=key, num=num)


def jrb_mat_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    return matrix_free_core.normal_probes_real(_jrb_point_interval, x.shape[-2], key=key, num=num)


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

_jrb_mat_expm_action_basic_jit_callable = jax.jit(jrb_mat_expm_action_basic, static_argnames=("matvec", "terms"))
_jrb_mat_expm_action_basic_jit_plan = jax.jit(jrb_mat_expm_action_basic, static_argnames=("terms",))


def jrb_mat_expm_action_basic_jit(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_expm_action_basic_jit_plan(matvec, x, terms=terms)
    return _jrb_mat_expm_action_basic_jit_callable(matvec, x, terms=terms)


_jrb_mat_solve_action_point_jit_callable = jax.jit(
    jrb_mat_solve_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "symmetric", "preconditioner"),
)
_jrb_mat_solve_action_point_jit_plan = jax.jit(
    jrb_mat_solve_action_point,
    static_argnames=("tol", "atol", "maxiter", "symmetric", "preconditioner"),
)


def jrb_mat_solve_action_point_jit(matvec, b: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_solve_action_point_jit_plan(matvec, b, **kwargs)
    return _jrb_mat_solve_action_point_jit_callable(matvec, b, **kwargs)


_jrb_mat_inverse_action_point_jit_callable = jax.jit(
    jrb_mat_inverse_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "symmetric", "preconditioner"),
)
_jrb_mat_inverse_action_point_jit_plan = jax.jit(
    jrb_mat_inverse_action_point,
    static_argnames=("tol", "atol", "maxiter", "symmetric", "preconditioner"),
)


def jrb_mat_inverse_action_point_jit(matvec, x: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_inverse_action_point_jit_plan(matvec, x, **kwargs)
    return _jrb_mat_inverse_action_point_jit_callable(matvec, x, **kwargs)


_jrb_mat_logdet_slq_point_jit_callable = jax.jit(jrb_mat_logdet_slq_point, static_argnames=("matvec", "steps"))
_jrb_mat_logdet_slq_point_jit_plan = jax.jit(_jrb_mat_logdet_slq_point_plan_kernel, static_argnames=("steps",))


def jrb_mat_logdet_slq_point_jit(matvec, probes: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_logdet_slq_point_jit_plan(matvec, probes, steps=steps)
    return _jrb_mat_logdet_slq_point_jit_callable(matvec, probes, steps=steps)


_jrb_mat_det_slq_point_jit_callable = jax.jit(jrb_mat_det_slq_point, static_argnames=("matvec", "steps"))
_jrb_mat_det_slq_point_jit_plan = jax.jit(_jrb_mat_det_slq_point_plan_kernel, static_argnames=("steps",))


def jrb_mat_det_slq_point_jit(matvec, probes: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_det_slq_point_jit_plan(matvec, probes, steps=steps)
    return _jrb_mat_det_slq_point_jit_callable(matvec, probes, steps=steps)


__all__ = [
    "PROVENANCE",
    "JrbMatKrylovDiagnostics",
    "JrbMatSelectedInverseDiagnostics",
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
    "jrb_mat_det_point",
    "jrb_mat_det_basic",
    "jrb_mat_inv_point",
    "jrb_mat_inv_basic",
    "jrb_mat_sqr_point",
    "jrb_mat_sqr_basic",
    "jrb_mat_trace_point",
    "jrb_mat_trace_basic",
    "jrb_mat_norm_fro_point",
    "jrb_mat_norm_fro_basic",
    "jrb_mat_norm_1_point",
    "jrb_mat_norm_1_basic",
    "jrb_mat_norm_inf_point",
    "jrb_mat_norm_inf_basic",
    "jrb_mat_dense_operator",
    "jrb_mat_dense_operator_adjoint",
    "jrb_mat_dense_operator_rmatvec",
    "jrb_mat_dense_operator_plan_prepare",
    "jrb_mat_dense_operator_rmatvec_plan_prepare",
    "jrb_mat_dense_operator_adjoint_plan_prepare",
    "jrb_mat_bcoo_operator",
    "jrb_mat_bcoo_operator_adjoint",
    "jrb_mat_bcoo_operator_rmatvec",
    "jrb_mat_bcoo_operator_plan_prepare",
    "jrb_mat_bcoo_operator_rmatvec_plan_prepare",
    "jrb_mat_bcoo_operator_adjoint_plan_prepare",
    "jrb_mat_operator_plan_apply",
    "jrb_mat_rmatvec_point",
    "jrb_mat_rmatvec_basic",
    "jrb_mat_lanczos_tridiag_adjoint",
    "jrb_mat_cg_fixed_iterations",
    "jrb_mat_solve_action_point",
    "jrb_mat_solve_action_basic",
    "jrb_mat_solve_action_with_diagnostics_point",
    "jrb_mat_solve_action_with_diagnostics_basic",
    "jrb_mat_inverse_action_point",
    "jrb_mat_inverse_action_basic",
    "jrb_mat_inverse_action_with_diagnostics_point",
    "jrb_mat_inverse_action_with_diagnostics_basic",
    "jrb_mat_bcoo_parametric_operator",
    "jrb_mat_scipy_csr_operator",
    "jrb_mat_bcoo_gershgorin_bounds",
    "jrb_mat_bcoo_spectral_bounds_adaptive",
    "jrb_mat_operator_apply_point",
    "jrb_mat_operator_apply_basic",
    "jrb_mat_poly_action_point",
    "jrb_mat_poly_action_basic",
    "jrb_mat_expm_action_point",
    "jrb_mat_expm_action_basic",
    "jrb_mat_lanczos_tridiag_point",
    "jrb_mat_lanczos_diagnostics_point",
    "jrb_mat_funm_action_lanczos_point",
    "jrb_mat_funm_action_lanczos_with_diagnostics_point",
    "jrb_mat_funm_integrand_lanczos_point",
    "jrb_mat_dense_funm_sym_eigh_point",
    "jrb_mat_funm_action_lanczos_dense_point",
    "jrb_mat_log_action_lanczos_point",
    "jrb_mat_log_action_lanczos_basic",
    "jrb_mat_sqrt_action_lanczos_point",
    "jrb_mat_sqrt_action_lanczos_basic",
    "jrb_mat_root_action_lanczos_point",
    "jrb_mat_root_action_lanczos_basic",
    "jrb_mat_sign_action_lanczos_point",
    "jrb_mat_sign_action_lanczos_basic",
    "jrb_mat_sin_action_lanczos_point",
    "jrb_mat_sin_action_lanczos_basic",
    "jrb_mat_cos_action_lanczos_point",
    "jrb_mat_cos_action_lanczos_basic",
    "jrb_mat_sinh_action_lanczos_point",
    "jrb_mat_sinh_action_lanczos_basic",
    "jrb_mat_cosh_action_lanczos_point",
    "jrb_mat_cosh_action_lanczos_basic",
    "jrb_mat_tanh_action_lanczos_point",
    "jrb_mat_tanh_action_lanczos_basic",
    "jrb_mat_log_action_lanczos_dense_point",
    "jrb_mat_sqrt_action_lanczos_dense_point",
    "jrb_mat_root_action_lanczos_dense_point",
    "jrb_mat_sign_action_lanczos_dense_point",
    "jrb_mat_sin_action_lanczos_dense_point",
    "jrb_mat_cos_action_lanczos_dense_point",
    "jrb_mat_sinh_action_lanczos_dense_point",
    "jrb_mat_cosh_action_lanczos_dense_point",
    "jrb_mat_tanh_action_lanczos_dense_point",
    "jrb_mat_expm_action_lanczos_restarted_point",
    "jrb_mat_expm_action_lanczos_block_point",
    "jrb_mat_expm_action_lanczos_restarted_with_diagnostics_point",
    "jrb_mat_trace_integrand_point",
    "jrb_mat_funm_trace_integrand_lanczos_point",
    "jrb_mat_trace_estimator_point",
    "jrb_mat_trace_estimator_with_diagnostics_point",
    "jrb_mat_logdet_slq_point",
    "jrb_mat_logdet_slq_basic",
    "jrb_mat_logdet_slq_with_diagnostics_point",
    "jrb_mat_logdet_slq_with_diagnostics_basic",
    "jrb_mat_det_slq_point",
    "jrb_mat_det_slq_basic",
    "jrb_mat_det_slq_with_diagnostics_point",
    "jrb_mat_det_slq_with_diagnostics_basic",
    "jrb_mat_log_action_lanczos_with_diagnostics_point",
    "jrb_mat_sqrt_action_lanczos_with_diagnostics_point",
    "jrb_mat_root_action_lanczos_with_diagnostics_point",
    "jrb_mat_sign_action_lanczos_with_diagnostics_point",
    "jrb_mat_sin_action_lanczos_with_diagnostics_point",
    "jrb_mat_cos_action_lanczos_with_diagnostics_point",
    "jrb_mat_sinh_action_lanczos_with_diagnostics_point",
    "jrb_mat_cosh_action_lanczos_with_diagnostics_point",
    "jrb_mat_tanh_action_lanczos_with_diagnostics_point",
    "jrb_mat_log_action_leja_point",
    "jrb_mat_log_action_leja_with_diagnostics_point",
    "jrb_mat_hutchpp_trace_point",
    "jrb_mat_logdet_leja_hutchpp_point",
    "jrb_mat_logdet_leja_hutchpp_with_diagnostics_point",
    "jrb_mat_det_leja_hutchpp_point",
    "jrb_mat_det_leja_hutchpp_with_diagnostics_point",
    "jrb_mat_bcoo_logdet_leja_hutchpp_point",
    "jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point",
    "jrb_mat_bcoo_inverse_diagonal_point",
    "jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point",
    "jrb_mat_rademacher_probes_like",
    "jrb_mat_normal_probes_like",
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
    "jrb_mat_expm_action_basic_jit",
    "jrb_mat_logm",
    "jrb_mat_sqrtm",
    "jrb_mat_rootm",
    "jrb_mat_signm",
    "jrb_mat_pow_action_lanczos_point",
    "jrb_mat_pow_action_lanczos_basic",
    "jrb_mat_pow_action_lanczos_dense_point",
    "jrb_mat_pow_action_lanczos_with_diagnostics_point",
]
