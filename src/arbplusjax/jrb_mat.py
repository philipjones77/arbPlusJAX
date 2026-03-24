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
- naming policy: see docs/standards/function_naming_standard.md
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
from . import srb_block_mat
from . import srb_vblock_mat
from . import sparse_common


PROVENANCE = {
    "classification": "new",
    "base_names": ("jrb_mat",),
    "module_lineage": "Jones matrix-function subsystem for real interval matrices",
    "naming_policy": "docs/standards/function_naming_standard.md",
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
    primal_residual: jax.Array = jnp.asarray(0.0, dtype=jnp.float64)
    adjoint_residual: jax.Array = jnp.asarray(0.0, dtype=jnp.float64)
    regime_code: jax.Array = jnp.asarray(-1, dtype=jnp.int32)
    method_code: jax.Array = jnp.asarray(-1, dtype=jnp.int32)
    solver_code: jax.Array = jnp.asarray(-1, dtype=jnp.int32)
    structure_code: jax.Array = jnp.asarray(-1, dtype=jnp.int32)
    converged: jax.Array = jnp.asarray(False)
    locked_count: jax.Array = jnp.asarray(0, dtype=jnp.int32)
    convergence_metric: jax.Array = jnp.asarray(jnp.inf, dtype=jnp.float64)
    residual_history: jax.Array = jnp.asarray([jnp.inf], dtype=jnp.float64)
    deflated_count: jax.Array = jnp.asarray(0, dtype=jnp.int32)


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


def _jrb_structure_tag(*, symmetric: bool = False, spd: bool = False) -> str:
    if spd:
        return "spd"
    if symmetric:
        return "symmetric"
    return "general"


def _jrb_attach_diag(diag: JrbMatKrylovDiagnostics, *, regime: str, method: str, structure: str, work_units, primal_residual=0.0, adjoint_residual=0.0, note: str = ""):
    return matrix_free_core.attach_krylov_metadata(
        diag,
        regime=regime,
        method=method,
        structure=structure,
        work_units=work_units,
        primal_residual=primal_residual,
        adjoint_residual=adjoint_residual,
        note=note,
    )


def _jrb_update_convergence(
    diag: JrbMatKrylovDiagnostics,
    *,
    converged,
    convergence_metric,
    locked_count: int | jax.Array = 0,
    residual_history=None,
    deflated_count: int | jax.Array | None = None,
):
    if residual_history is None:
        existing = jnp.asarray(getattr(diag, "residual_history", jnp.asarray([diag.tail_norm], dtype=jnp.float64)), dtype=jnp.float64)
        residual_history = jnp.where(jnp.all(jnp.isfinite(existing)), existing, jnp.asarray([convergence_metric], dtype=jnp.float64))
    if deflated_count is None:
        deflated_count = locked_count
    return diag._replace(
        converged=jnp.asarray(converged),
        locked_count=jnp.asarray(locked_count, dtype=jnp.int32),
        convergence_metric=jnp.asarray(convergence_metric, dtype=jnp.float64),
        residual_history=jnp.asarray(residual_history, dtype=jnp.float64),
        deflated_count=jnp.asarray(deflated_count, dtype=jnp.int32),
    )


def _jrb_eig_residuals(matvec, vals: jax.Array, vecs: jax.Array) -> jax.Array:
    if vecs.ndim != 2:
        return jnp.asarray([], dtype=jnp.float64)
    applied = jax.vmap(lambda col: _jrb_apply_operator_mid(matvec, _jrb_point_interval(col)), in_axes=1, out_axes=1)(vecs)
    residual = applied - vecs * jnp.asarray(vals, dtype=applied.dtype)[None, :]
    return jnp.linalg.norm(residual, axis=0)


def _jrb_eig_diagnostics(
    matvec,
    vals: jax.Array,
    vecs: jax.Array,
    *,
    algorithm_code: int,
    steps,
    basis_dim,
    restart_count=0,
    structure: str = "symmetric",
    method: str = "lanczos",
    tol: float = 1e-3,
):
    residuals = _jrb_eig_residuals(matvec, vals, vecs)
    max_residual = jnp.max(residuals) if residuals.size else jnp.asarray(0.0, dtype=jnp.float64)
    requested = vals.shape[-1] if vals.ndim > 0 else 1
    converged_mask = residuals <= jnp.asarray(tol, dtype=jnp.float64)
    converged_count = jnp.sum(converged_mask.astype(jnp.int32)) if residuals.size else jnp.asarray(0, dtype=jnp.int32)
    diag = matrix_free_core.krylov_diagnostics(
        JrbMatKrylovDiagnostics,
        algorithm_code=algorithm_code,
        steps=steps,
        basis_dim=basis_dim,
        beta0=1.0,
        tail_norm=max_residual,
        breakdown=False,
        used_adjoint=False,
        gradient_supported=True,
        probe_count=requested,
        restart_count=restart_count,
    )
    diag = _jrb_attach_diag(
        diag,
        regime="structured",
        method=method,
        structure=structure,
        work_units=steps,
        primal_residual=max_residual,
        note="matrix_free.eigsh",
    )
    return _jrb_update_convergence(
        diag,
        converged=converged_count >= requested,
        convergence_metric=max_residual,
        locked_count=requested,
        residual_history=residuals if residuals.size else jnp.asarray([max_residual], dtype=jnp.float64),
        deflated_count=converged_count,
    )


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


def jrb_mat_dense_parametric_operator_plan_prepare(a: jax.Array):
    return matrix_free_core.parametric_dense_operator_plan(_jrb_mid_matrix(a), orientation="forward", algebra="jrb")


def jrb_mat_shell_operator_plan_prepare(callback, *, context=None):
    return matrix_free_core.shell_operator_plan(callback, context=context, orientation="forward", algebra="jrb")


def jrb_mat_dense_operator_rmatvec_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jrb_mid_matrix(a), orientation="transpose", algebra="jrb")


def jrb_mat_dense_parametric_operator_rmatvec_plan_prepare(a: jax.Array):
    return matrix_free_core.parametric_dense_operator_plan(_jrb_mid_matrix(a), orientation="transpose", algebra="jrb")


def jrb_mat_dense_operator_adjoint_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jrb_mid_matrix(a), orientation="transpose", algebra="jrb")


def jrb_mat_dense_parametric_operator_adjoint_plan_prepare(a: jax.Array):
    return matrix_free_core.parametric_dense_operator_plan(_jrb_mid_matrix(a), orientation="transpose", algebra="jrb")


def jrb_mat_finite_difference_operator_plan_prepare(
    function,
    *,
    base_point,
    base_value=None,
    context=None,
    relative_error: float = 1e-7,
    umin: float = 1e-6,
):
    return matrix_free_core.finite_difference_operator_plan(
        function,
        base_point=_jrb_operator_vector(base_point),
        base_value=None if base_value is None else jnp.asarray(base_value, dtype=jnp.float64),
        context=context,
        algebra="jrb",
        relative_error=relative_error,
        umin=umin,
    )


def jrb_mat_finite_difference_operator_plan_set_base(plan, *, base_point, base_value=None):
    return matrix_free_core.finite_difference_operator_plan_set_base(
        plan,
        base_point=_jrb_operator_vector(base_point),
        base_value=None if base_value is None else jnp.asarray(base_value, dtype=jnp.float64),
    )


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


def jrb_mat_block_sparse_operator_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jrb",
        orientation="forward",
        forward_callback=lambda v, mat: srb_block_mat.srb_block_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: srb_block_mat.srb_block_mat_rmatvec(mat, v),
    )


def jrb_mat_block_sparse_operator_rmatvec_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jrb",
        orientation="transpose",
        forward_callback=lambda v, mat: srb_block_mat.srb_block_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: srb_block_mat.srb_block_mat_rmatvec(mat, v),
    )


def jrb_mat_block_sparse_operator_adjoint_plan_prepare(a):
    return jrb_mat_block_sparse_operator_rmatvec_plan_prepare(a)


def jrb_mat_vblock_sparse_operator_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jrb",
        orientation="forward",
        forward_callback=lambda v, mat: srb_vblock_mat.srb_vblock_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: srb_vblock_mat.srb_vblock_mat_rmatvec(mat, v),
    )


def jrb_mat_vblock_sparse_operator_rmatvec_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jrb",
        orientation="transpose",
        forward_callback=lambda v, mat: srb_vblock_mat.srb_vblock_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: srb_vblock_mat.srb_vblock_mat_rmatvec(mat, v),
    )


def jrb_mat_vblock_sparse_operator_adjoint_plan_prepare(a):
    return jrb_mat_vblock_sparse_operator_rmatvec_plan_prepare(a)


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


def jrb_mat_jacobi_preconditioner_plan_prepare(a):
    if isinstance(a, matrix_free_core.OperatorPlan):
        if a.kind == "dense":
            return matrix_free_core.dense_jacobi_preconditioner_plan(a.payload, algebra="jrb")
        if a.kind == "sparse_bcoo":
            return matrix_free_core.sparse_bcoo_jacobi_preconditioner_plan(
                a.payload,
                as_sparse_bcoo=sparse_common.as_sparse_bcoo,
                algebra="jrb",
            )
        if a.kind == "finite_difference":
            return matrix_free_core.finite_difference_jacobi_preconditioner_plan(
                a,
                midpoint_vector=di.midpoint,
                sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
                dtype=jnp.float64,
                algebra="jrb",
            )
        raise ValueError(f"unsupported operator plan kind for Jacobi preconditioner: {a.kind}")
    if isinstance(a, (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)):
        return matrix_free_core.sparse_bcoo_jacobi_preconditioner_plan(
            _jrb_sparse_to_bcoo(a),
            as_sparse_bcoo=sparse_common.as_sparse_bcoo,
            algebra="jrb",
        )
    return matrix_free_core.dense_jacobi_preconditioner_plan(_jrb_mid_matrix(a), algebra="jrb")


def jrb_mat_shell_preconditioner_plan_prepare(callback, *, context=None):
    return matrix_free_core.shell_preconditioner_plan(callback, context=context, orientation="forward", algebra="jrb")


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


def jrb_mat_solve_action_symmetric_point(matvec, b: jax.Array, **kwargs) -> jax.Array:
    return jrb_mat_solve_action_point(matvec, b, symmetric=True, **kwargs)


def jrb_mat_solve_action_spd_point(matvec, b: jax.Array, **kwargs) -> jax.Array:
    return jrb_mat_solve_action_point(matvec, b, symmetric=True, **kwargs)


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
    value, _ = matrix_free_basic.solve_action_with_diagnostics_basic(
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
        invalidate_output=_full_interval_like,
    )
    return value


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
    x_mid, info, residual, rhs_norm = matrix_free_core.krylov_solve_midpoint(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        solver="cg" if symmetric else "gmres",
        midpoint_vector=_jrb_mid_vector,
        lift_vector=_jrb_point_interval,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    out = _jrb_point_interval(x_mid)
    finite = jnp.all(jnp.isfinite(x_mid), axis=-1)
    diag = JrbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(5 if symmetric else 6, dtype=jnp.int32),
        steps=jnp.asarray(info["iterations"], dtype=jnp.int32),
        basis_dim=jnp.asarray(info["iterations"], dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(rhs_norm, dtype=jnp.float64),
        tail_norm=jnp.asarray(info["residuals"][-1], dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(1, dtype=jnp.int32),
    )
    diag = _jrb_attach_diag(
        diag,
        regime="structured" if symmetric else "iterative",
        method="cg" if symmetric else "gmres",
        structure=_jrb_structure_tag(symmetric=symmetric),
        work_units=info["iterations"],
        primal_residual=residual,
        adjoint_residual=0.0,
        note="matrix_free.solve_action",
    )
    diag = _jrb_update_convergence(
        diag,
        converged=info["converged"],
        convergence_metric=residual,
    )
    return jnp.where(finite[..., None], out, _full_interval_like(out)), diag


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
        invalidate_output=_full_interval_like,
    )


def jrb_mat_minres_solve_action_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
) -> jax.Array:
    value, _ = jrb_mat_minres_solve_action_with_diagnostics_point(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
    )
    return value


def jrb_mat_minres_solve_action_with_diagnostics_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
):
    b = jrb_mat_as_interval_vector(b)
    x_mid, info, residual, rhs_norm = matrix_free_core.krylov_solve_midpoint(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        solver="minres",
        midpoint_vector=_jrb_mid_vector,
        lift_vector=_jrb_point_interval,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    out = _jrb_point_interval(x_mid)
    finite = jnp.all(jnp.isfinite(x_mid), axis=-1)
    diag = JrbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(7, dtype=jnp.int32),
        steps=jnp.asarray(info["iterations"], dtype=jnp.int32),
        basis_dim=jnp.asarray(info["iterations"], dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(rhs_norm, dtype=jnp.float64),
        tail_norm=jnp.asarray(info["residuals"][-1], dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(1, dtype=jnp.int32),
    )
    diag = _jrb_attach_diag(
        diag,
        regime="structured",
        method="minres",
        structure="symmetric",
        work_units=info["iterations"],
        primal_residual=residual,
        adjoint_residual=0.0,
        note="matrix_free.minres_solve_action",
    )
    diag = _jrb_update_convergence(
        diag,
        converged=info["converged"],
        convergence_metric=residual,
    )
    return jnp.where(finite[..., None], out, _full_interval_like(out)), diag


def jrb_mat_minres_inverse_action_point(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
) -> jax.Array:
    return jrb_mat_minres_solve_action_point(matvec, x, x0=x0, tol=tol, atol=atol, maxiter=maxiter, preconditioner=preconditioner)


def jrb_mat_minres_solve_action_basic(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    value, _ = matrix_free_basic.solve_action_with_diagnostics_basic(
        jrb_mat_minres_solve_action_with_diagnostics_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_interval_like,
    )
    return value


def jrb_mat_minres_solve_action_with_diagnostics_basic(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.solve_action_with_diagnostics_basic(
        jrb_mat_minres_solve_action_with_diagnostics_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_interval_like,
    )


def jrb_mat_minres_inverse_action_basic(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    value, _ = matrix_free_basic.inverse_action_with_diagnostics_basic(
        jrb_mat_minres_solve_action_with_diagnostics_point,
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_interval_like,
    )
    return value


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


def jrb_mat_inverse_action_symmetric_point(matvec, x: jax.Array, **kwargs) -> jax.Array:
    return jrb_mat_inverse_action_point(matvec, x, symmetric=True, **kwargs)


def jrb_mat_inverse_action_spd_point(matvec, x: jax.Array, **kwargs) -> jax.Array:
    return jrb_mat_inverse_action_point(matvec, x, symmetric=True, **kwargs)


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
    value, _ = matrix_free_basic.inverse_action_with_diagnostics_basic(
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
        invalidate_output=_full_interval_like,
    )
    return value


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
        invalidate_output=_full_interval_like,
    )


def jrb_mat_multi_shift_solve_point(
    matvec,
    rhs: jax.Array,
    shifts: jax.Array,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
) -> jax.Array:
    rhs = jrb_mat_as_interval_vector(rhs)
    plan = matrix_free_core.make_shifted_solve_plan(
        matvec,
        shifts,
        preconditioner=preconditioner,
        solver="multi_shift_cg" if symmetric else "multi_shift_gmres",
        algebra="jrb",
        structured=_jrb_structure_tag(symmetric=symmetric),
    )
    mids = matrix_free_core.multi_shift_solve_point(
        plan,
        rhs,
        apply_operator=iterative_solvers,
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    out = jax.vmap(_jrb_point_interval)(mids)
    finite = jnp.all(jnp.isfinite(mids), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_multi_shift_solve_symmetric_point(matvec, rhs: jax.Array, shifts: jax.Array, **kwargs) -> jax.Array:
    return jrb_mat_multi_shift_solve_point(matvec, rhs, shifts, symmetric=True, **kwargs)


def jrb_mat_multi_shift_solve_spd_point(matvec, rhs: jax.Array, shifts: jax.Array, **kwargs) -> jax.Array:
    return jrb_mat_multi_shift_solve_point(matvec, rhs, shifts, symmetric=True, **kwargs)


def jrb_mat_multi_shift_solve_basic(
    matvec,
    rhs: jax.Array,
    shifts: jax.Array,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_round_basic(
        jrb_mat_multi_shift_solve_point(
            matvec,
            rhs,
            shifts,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            symmetric=symmetric,
            preconditioner=preconditioner,
        ),
        prec_bits,
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


def jrb_mat_bcoo_parametric_operator_plan_prepare(indices: jax.Array, data: jax.Array, *, shape: tuple[int, int]):
    return matrix_free_core.parametric_bcoo_operator_plan(indices, data, shape=shape, dtype=jnp.float64, algebra="jrb")


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
    tol: float = 1e-3,
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
    tol: float = 1e-3,
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


def jrb_mat_rational_action_point(
    matvec,
    x: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
) -> jax.Array:
    x_checked = jrb_mat_as_interval_vector(x)
    x_mid = _jrb_mid_vector(x_checked)

    def apply_operator(v_mid: jax.Array) -> jax.Array:
        return di.midpoint(jrb_mat_operator_apply_point(matvec, _jrb_point_interval(v_mid)))

    def solve_shifted(shift, v_mid: jax.Array) -> jax.Array:
        shift_arr = jnp.asarray(shift, dtype=jnp.float64)
        shifted = matrix_free_core.shell_operator_plan(
            lambda y, context: di.midpoint(jrb_mat_operator_apply_point(context["operator"], _jrb_point_interval(y)))
            - context["shift"] * jnp.asarray(y, dtype=jnp.float64),
            context={"operator": matvec, "shift": shift_arr},
            orientation="forward",
            algebra="jrb",
        )
        solved = jrb_mat_solve_action_point(
            shifted,
            _jrb_point_interval(v_mid),
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            symmetric=symmetric,
            preconditioner=preconditioner,
        )
        return di.midpoint(solved)

    out_mid = matrix_free_core.rational_spectral_action_midpoint(
        apply_operator,
        solve_shifted,
        x_mid,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        coeff_dtype=jnp.float64,
    )
    return _jrb_point_interval(out_mid)


def jrb_mat_rational_action_basic(
    matvec,
    x: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.action_basic(
        jrb_mat_rational_action_point,
        matvec,
        x,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
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


def _jrb_mat_lanczos_tridiag_state_bucketed_point(matvec, x: jax.Array, effective_steps: int, max_steps: int):
    x = jrb_mat_as_interval_vector(x)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    v0 = _jrb_mid_vector(x)
    beta0 = jnp.linalg.norm(v0)
    if max_steps > int(v0.shape[-1]):
        raise ValueError("max_steps must be <= vector dimension")
    q0 = v0 / jnp.maximum(beta0, jnp.asarray(1e-30, dtype=jnp.float64))
    dim = q0.shape[-1]
    effective_steps = jnp.clip(jnp.asarray(effective_steps, dtype=jnp.int32), 1, jnp.asarray(max_steps, dtype=jnp.int32))

    def body(carry, _):
        q_prev, q_curr, beta_prev, basis, alphas, betas, k = carry

        def do_step(state):
            q_prev, q_curr, beta_prev, basis, alphas, betas, k = state
            z = _jrb_apply_operator_mid(matvec, _jrb_point_interval(q_curr))
            alpha = jnp.vdot(q_curr, z).real
            r = z - alpha * q_curr - beta_prev * q_prev
            mask = (jnp.arange(max_steps, dtype=jnp.int32) < k).astype(jnp.float64)
            proj = (basis @ r) * mask
            r = r - proj @ basis
            beta = jnp.linalg.norm(r)
            q_next = jnp.where(beta > 1e-30, r / beta, jnp.zeros_like(r))
            basis = basis.at[k].set(q_curr)
            alphas = alphas.at[k].set(alpha)
            betas = betas.at[k].set(beta)
            return (q_curr, q_next, beta, basis, alphas, betas, k + 1)

        return lax.cond(k < effective_steps, do_step, lambda state: state, carry), None

    init_basis = jnp.zeros((max_steps, dim), dtype=jnp.float64)
    init_alphas = jnp.zeros((max_steps,), dtype=jnp.float64)
    init_betas = jnp.zeros((max_steps,), dtype=jnp.float64)
    init = (
        jnp.zeros_like(q0),
        q0,
        jnp.asarray(0.0, dtype=jnp.float64),
        init_basis,
        init_alphas,
        init_betas,
        jnp.asarray(0, dtype=jnp.int32),
    )
    (_, _, _, basis, alphas, betas, _), _ = lax.scan(body, init, xs=None, length=max_steps)
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


def _jrb_eigsh_start_vector(size: int) -> jax.Array:
    values = jnp.linspace(1.0, 2.0, int(size), dtype=jnp.float64)
    return _jrb_point_interval(values)


def _jrb_eigsh_select_indices(evals: jax.Array, k: int, which: str) -> jax.Array:
    code = which.lower()
    if code in {"largest", "la", "lm"}:
        return jnp.arange(evals.shape[0] - k, evals.shape[0], dtype=jnp.int32)
    if code in {"smallest", "sa", "sm"}:
        return jnp.arange(0, k, dtype=jnp.int32)
    raise ValueError("which must be one of {'largest', 'smallest', 'la', 'sa', 'lm', 'sm'}")


def _jrb_eigsh_start_block(size: int, block_size: int) -> jax.Array:
    base = jnp.linspace(1.0, 2.0, int(size), dtype=jnp.float64)[:, None]
    offsets = jnp.linspace(0.0, 1.0, int(block_size), dtype=jnp.float64)[None, :]
    return base + offsets


def _jrb_eigsh_mid_block(v0, *, size: int, block_size: int) -> jax.Array:
    if v0 is None:
        return _jrb_eigsh_start_block(size, block_size)
    arr = jnp.asarray(v0)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return di.midpoint(arr)
    return jnp.asarray(arr, dtype=jnp.float64)


def _jrb_apply_operator_block_mid(matvec, block: jax.Array) -> jax.Array:
    return jax.vmap(lambda col: _jrb_apply_operator_mid(matvec, _jrb_point_interval(col)), in_axes=1, out_axes=1)(block)


def _jrb_orthonormalize_columns(block: jax.Array) -> jax.Array:
    return matrix_free_core.orthonormalize_columns(block)


def _jrb_ritz_pairs_from_basis(matvec, basis: jax.Array, *, k: int, which: str) -> tuple[jax.Array, jax.Array]:
    return matrix_free_core.ritz_pairs_from_basis(
        lambda q: _jrb_apply_operator_block_mid(matvec, q),
        basis,
        k=k,
        which=which,
        hermitian=True,
    )


def _jrb_apply_preconditioner_mid(preconditioner, x: jax.Array) -> jax.Array:
    return matrix_free_core.preconditioner_apply_midpoint(
        preconditioner,
        x,
        midpoint_vector=_jrb_operator_vector,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )


def _jrb_expand_subspace_with_corrections(
    basis: jax.Array,
    vecs: jax.Array,
    residuals: jax.Array,
    *,
    vals: jax.Array | None = None,
    target_cols: int,
    which: str = "largest",
    lock_tol: float = 1e-4,
    preconditioner=None,
    jacobi_davidson: bool = False,
) -> jax.Array:
    max_new_cols = max(0, min(int(target_cols) - int(basis.shape[1]), int(residuals.shape[1])))
    if vals is not None and max_new_cols > 0:
        order = matrix_free_core.eig_expansion_column_order(vals, residuals, which=which, lock_tol=lock_tol)
        chosen = order[:max_new_cols]
        vecs = vecs[:, chosen]
        residuals = residuals[:, chosen]
    corrections = residuals
    residual_norms = jnp.linalg.norm(corrections, axis=0, keepdims=True)
    safe_norms = jnp.where(residual_norms > 1e-12, residual_norms, 1.0)
    corrections = corrections / safe_norms
    if preconditioner is not None:
        corrections = jax.vmap(lambda col: _jrb_apply_preconditioner_mid(preconditioner, col), in_axes=1, out_axes=1)(corrections)
    if jacobi_davidson:
        coeffs = jnp.sum(vecs * corrections, axis=0, keepdims=True)
        projected = corrections - vecs * coeffs
        proj_norms = jnp.linalg.norm(projected, axis=0, keepdims=True)
        corrections = jnp.where(
            proj_norms > 1e-12,
            projected / jnp.where(proj_norms > 1e-12, proj_norms, 1.0),
            0.0,
        )
    correction_norms = jnp.linalg.norm(corrections, axis=0, keepdims=True)
    corrections = jnp.where(correction_norms > 1e-10, corrections, 0.0)
    trial = jnp.concatenate([basis, corrections], axis=1)
    basis_next = _jrb_orthonormalize_columns(trial)
    if basis_next.shape[1] < target_cols:
        pad = basis[:, : target_cols - basis_next.shape[1]]
        basis_next = _jrb_orthonormalize_columns(jnp.concatenate([basis_next, pad], axis=1))
    return basis_next[:, :target_cols]


def _jrb_restart_basis_from_pairs(
    *,
    actual_block: int,
    vals: jax.Array,
    vecs: jax.Array,
    residuals: jax.Array,
    which: str,
    lock_tol: float,
    refill_basis: jax.Array | None = None,
) -> jax.Array:
    return matrix_free_core.eig_restart_basis_from_pairs(
        vecs,
        vals,
        residuals,
        target_cols=actual_block,
        which=which,
        lock_tol=lock_tol,
        refill_basis=refill_basis,
    )


def _jrb_shifted_solve_mid(matvec, rhs_mid: jax.Array, *, shift, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None) -> jax.Array:
    shift_arr = jnp.asarray(shift, dtype=jnp.complex128)

    if isinstance(matvec, matrix_free_core.OperatorPlan):
        def mv(v):
            base = matrix_free_core.operator_apply_midpoint(
                matvec,
                v,
                midpoint_vector=lambda y: jnp.asarray(y, dtype=jnp.complex128),
                sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
                dtype=jnp.complex128,
            )
            return base - shift_arr * jnp.asarray(v, dtype=jnp.complex128)
    else:
        def real_apply(v):
            return _jrb_apply_operator_mid(matvec, _jrb_point_interval(v))

        def mv(v):
            return matrix_free_core.complexify_real_linear_operator(real_apply, v) - shift_arr * jnp.asarray(v, dtype=jnp.complex128)

    precond = None
    if preconditioner is not None:
        precond = lambda v: jnp.asarray(_jrb_apply_preconditioner_mid(preconditioner, _jrb_point_interval(jnp.real(v))), dtype=jnp.float64) + 1j * jnp.asarray(_jrb_apply_preconditioner_mid(preconditioner, _jrb_point_interval(jnp.imag(v))), dtype=jnp.float64)
    out, _ = iterative_solvers.gmres(mv, jnp.asarray(rhs_mid, dtype=jnp.complex128), tol=tol, atol=atol, maxiter=maxiter, M=precond)
    return out


def _jrb_spd_solve_mid(matvec, rhs_mid: jax.Array, *, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None) -> jax.Array:
    x_mid, _info, _residual, _rhs_norm = matrix_free_core.krylov_solve_midpoint(
        matvec,
        _jrb_point_interval(rhs_mid),
        x0=None,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        solver="cg",
        midpoint_vector=_jrb_mid_vector,
        lift_vector=_jrb_point_interval,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    return x_mid


def _jrb_generalized_shifted_solve_mid(
    a_matvec,
    b_matvec,
    rhs_mid: jax.Array,
    *,
    shift,
    preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> jax.Array:
    shift_arr = jnp.asarray(shift, dtype=jnp.complex128)

    def mv(v):
        av = matrix_free_core.complexify_real_linear_operator(
            lambda x: _jrb_apply_operator_mid(a_matvec, _jrb_point_interval(x)),
            v,
        )
        bv = matrix_free_core.complexify_real_linear_operator(
            lambda x: _jrb_apply_operator_mid(b_matvec, _jrb_point_interval(x)),
            v,
        )
        return av - shift_arr * bv

    precond = None
    if preconditioner is not None:
        precond = lambda v: jnp.asarray(
            _jrb_apply_preconditioner_mid(preconditioner, _jrb_point_interval(jnp.real(v))),
            dtype=jnp.float64,
        ) + 1j * jnp.asarray(
            _jrb_apply_preconditioner_mid(preconditioner, _jrb_point_interval(jnp.imag(v))),
            dtype=jnp.float64,
        )
    out, _ = iterative_solvers.gmres(mv, jnp.asarray(rhs_mid, dtype=jnp.complex128), tol=tol, atol=atol, maxiter=maxiter, M=precond)
    return out


def _jrb_generalized_eig_residuals(a_matvec, b_matvec, vals: jax.Array, vecs: jax.Array) -> jax.Array:
    if vecs.ndim != 2:
        return jnp.asarray([], dtype=jnp.float64)
    applied_a = _jrb_apply_operator_block_mid(a_matvec, vecs)
    applied_b = _jrb_apply_operator_block_mid(b_matvec, vecs)
    residual = applied_a - applied_b * jnp.asarray(vals, dtype=applied_a.dtype)[None, :]
    return jnp.linalg.norm(residual, axis=0)


def _jrb_generalized_eig_diagnostics(
    a_matvec,
    b_matvec,
    vals: jax.Array,
    vecs: jax.Array,
    *,
    algorithm_code: int,
    steps,
    basis_dim,
    restart_count=0,
    tol: float = 1e-3,
):
    residuals = _jrb_generalized_eig_residuals(a_matvec, b_matvec, vals, vecs)
    max_residual = jnp.max(residuals) if residuals.size else jnp.asarray(0.0, dtype=jnp.float64)
    requested = vals.shape[-1] if vals.ndim > 0 else 1
    converged_mask = residuals <= jnp.asarray(tol, dtype=jnp.float64)
    converged_count = jnp.sum(converged_mask.astype(jnp.int32)) if residuals.size else jnp.asarray(0, dtype=jnp.int32)
    diag = matrix_free_core.krylov_diagnostics(
        JrbMatKrylovDiagnostics,
        algorithm_code=algorithm_code,
        steps=steps,
        basis_dim=basis_dim,
        beta0=1.0,
        tail_norm=max_residual,
        breakdown=False,
        used_adjoint=False,
        gradient_supported=True,
        probe_count=requested,
        restart_count=restart_count,
    )
    diag = _jrb_attach_diag(
        diag,
        regime="structured",
        method="lanczos",
        structure="spd",
        work_units=steps,
        primal_residual=max_residual,
        note="matrix_free.generalized_eigsh",
    )
    return _jrb_update_convergence(
        diag,
        converged=converged_count >= requested,
        convergence_metric=max_residual,
        locked_count=requested,
        residual_history=residuals if residuals.size else jnp.asarray([max_residual], dtype=jnp.float64),
        deflated_count=converged_count,
    )


def jrb_mat_generalized_operator_plan_prepare(
    a_matvec,
    b_matvec,
    *,
    b_preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
):
    return matrix_free_core.generalized_shell_operator_plan(
        lambda v, context: _jrb_apply_operator_mid(context["a_matvec"], _jrb_point_interval(v)),
        lambda rhs, context: _jrb_spd_solve_mid(
            context["b_matvec"],
            rhs,
            preconditioner=context["b_preconditioner"],
            tol=context["tol"],
            atol=context["atol"],
            maxiter=context["maxiter"],
        ),
        context={
            "a_matvec": a_matvec,
            "b_matvec": b_matvec,
            "b_preconditioner": b_preconditioner,
            "tol": float(tol),
            "atol": float(atol),
            "maxiter": maxiter,
        },
        orientation="forward",
        algebra="jrb",
    )


def jrb_mat_eigsh_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    if size <= 0:
        raise ValueError("size must be > 0")
    if k <= 0 or k > size:
        raise ValueError("k must satisfy 0 < k <= size")
    resolved_steps = min(size, max(int(k) + 2, 2 * int(k) + 8)) if steps is None else int(steps)
    if resolved_steps <= 0 or resolved_steps > size:
        raise ValueError("steps must satisfy 0 < steps <= size")
    if k > resolved_steps:
        raise ValueError("k must be <= steps")
    start = _jrb_eigsh_start_vector(size) if v0 is None else jrb_mat_as_interval_vector(v0)
    basis, projected, _ = jrb_mat_lanczos_tridiag_point(matvec, start, resolved_steps)
    evals, coeffs = jnp.linalg.eigh(projected)
    indices = _jrb_eigsh_select_indices(evals, k, which)
    selected_vals = evals[indices]
    selected_coeffs = coeffs[:, indices]
    vectors = basis.T @ selected_coeffs
    norms = jnp.maximum(jnp.linalg.norm(vectors, axis=0), jnp.asarray(1e-30, dtype=jnp.float64))
    vectors = vectors / norms[None, :]
    return selected_vals, vectors


def jrb_mat_eigsh_basic(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> tuple[jax.Array, jax.Array]:
    values, vectors = jrb_mat_eigsh_point(matvec, size=size, k=k, which=which, steps=steps, v0=v0)
    return _jrb_round_basic(_jrb_point_interval(values), prec_bits), _jrb_round_basic(_jrb_point_interval(vectors), prec_bits)


def jrb_mat_eigsh_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_point(matvec, size=size, k=k, which=which, steps=steps, v0=v0)
    resolved_steps = min(size, max(int(k) + 2, 2 * int(k) + 8)) if steps is None else int(steps)
    diag = _jrb_eig_diagnostics(matvec, vals, vecs, algorithm_code=8, steps=resolved_steps, basis_dim=resolved_steps, method="lanczos", tol=tol)
    return vals, vecs, diag


def jrb_mat_eigsh_block_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    block_size: int | None = None,
    subspace_iters: int = 4,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis = _jrb_orthonormalize_columns(_jrb_eigsh_mid_block(v0, size=size, block_size=actual_block))

    def body(q, _):
        return _jrb_orthonormalize_columns(_jrb_apply_operator_block_mid(matvec, q)), None

    basis, _ = lax.scan(body, basis, xs=None, length=int(subspace_iters))
    return _jrb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jrb_mat_eigsh_block_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    block_size: int | None = None,
    subspace_iters: int = 4,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_block_point(matvec, size=size, k=k, which=which, block_size=block_size, subspace_iters=subspace_iters, v0=v0)
    diag = _jrb_eig_diagnostics(matvec, vals, vecs, algorithm_code=9, steps=subspace_iters, basis_dim=int(k if block_size is None else block_size), method="lanczos", tol=tol)
    return vals, vecs, diag


def jrb_mat_eigsh_restarted_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int = 4,
    restarts: int = 2,
    block_size: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis = _jrb_orthonormalize_columns(_jrb_eigsh_mid_block(v0, size=size, block_size=actual_block))
    restart_tol = matrix_free_core.eig_restart_lock_tolerance(steps=steps, restarts=restarts)
    keep_count = min(actual_block, basis.shape[1])

    def iterate(q):
        def body(q_inner, _):
            return _jrb_orthonormalize_columns(_jrb_apply_operator_block_mid(matvec, q_inner)), None
        q_out, _ = lax.scan(body, q, xs=None, length=int(steps))
        vals, vecs = _jrb_ritz_pairs_from_basis(matvec, q_out, k=keep_count, which=which)
        residuals = _jrb_apply_operator_block_mid(matvec, vecs) - vecs * vals[None, :]
        return _jrb_restart_basis_from_pairs(
            actual_block=actual_block,
            vals=vals,
            vecs=vecs,
            residuals=residuals,
            which=which,
            lock_tol=restart_tol,
            refill_basis=q_out,
        )

    for _ in range(int(restarts)):
        basis = iterate(basis)

    return _jrb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jrb_mat_eigsh_restarted_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int = 4,
    restarts: int = 2,
    block_size: int | None = None,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_restarted_point(
        matvec,
        size=size,
        k=k,
        which=which,
        steps=steps,
        restarts=restarts,
        block_size=block_size,
        v0=v0,
    )
    diag = _jrb_eig_diagnostics(
        matvec,
        vals,
        vecs,
        algorithm_code=10,
        steps=steps,
        basis_dim=int(k if block_size is None else block_size),
        restart_count=restarts,
        method="lanczos",
        tol=tol,
    )
    return vals, vecs, diag


def jrb_mat_eigsh_krylov_schur_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int = 4,
    restarts: int = 2,
    block_size: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis0 = _jrb_eigsh_mid_block(v0, size=size, block_size=actual_block)
    basis = matrix_free_core.restarted_subspace_iteration_point(
        lambda q: _jrb_apply_operator_block_mid(matvec, q),
        basis0,
        subspace_iters=int(steps),
        restarts=int(restarts),
        k=k,
        which=which,
        hermitian=True,
    )
    keep_count = min(actual_block, basis.shape[1])
    vals, vecs = _jrb_ritz_pairs_from_basis(matvec, basis, k=keep_count, which=which)
    residuals = _jrb_apply_operator_block_mid(matvec, vecs) - vecs * vals[None, :]
    basis = _jrb_restart_basis_from_pairs(
        actual_block=actual_block,
        vals=vals,
        vecs=vecs,
        residuals=residuals,
        which=which,
        lock_tol=1e-4,
        refill_basis=basis,
    )
    return _jrb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jrb_mat_eigsh_krylov_schur_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int = 4,
    restarts: int = 2,
    block_size: int | None = None,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_krylov_schur_point(
        matvec,
        size=size,
        k=k,
        which=which,
        steps=steps,
        restarts=restarts,
        block_size=block_size,
        v0=v0,
    )
    diag = _jrb_eig_diagnostics(
        matvec,
        vals,
        vecs,
        algorithm_code=11,
        steps=steps,
        basis_dim=int(k if block_size is None else block_size),
        restart_count=restarts,
        method="lanczos",
        tol=tol,
    )
    return vals, vecs, diag


def jrb_mat_eigsh_davidson_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    subspace_iters: int = 4,
    block_size: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis = _jrb_orthonormalize_columns(_jrb_eigsh_mid_block(v0, size=size, block_size=actual_block))

    target_cols = min(size, max(actual_block, k))
    lock_tol = max(float(tol), matrix_free_core.eig_restart_lock_tolerance(steps=subspace_iters, restarts=1))
    for _ in range(int(subspace_iters)):
        keep_count = min(actual_block, basis.shape[1])
        vals, vecs = _jrb_ritz_pairs_from_basis(matvec, basis, k=keep_count, which=which)
        applied = _jrb_apply_operator_block_mid(matvec, vecs)
        residuals = applied - vecs * vals[None, :]
        basis_seed = _jrb_restart_basis_from_pairs(
            actual_block=min(actual_block, basis.shape[1]),
            vals=vals,
            vecs=vecs,
            residuals=residuals,
            which=which,
            lock_tol=lock_tol,
            refill_basis=basis,
        )
        target_cols = min(size, basis_seed.shape[1] + min(actual_block, residuals.shape[1]))
        basis = _jrb_expand_subspace_with_corrections(
            basis_seed,
            vecs,
            residuals,
            vals=vals,
            target_cols=target_cols,
            which=which,
            lock_tol=lock_tol,
            preconditioner=preconditioner,
            jacobi_davidson=False,
        )
    return _jrb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jrb_mat_eigsh_davidson_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    subspace_iters: int = 4,
    block_size: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_davidson_point(
        matvec,
        size=size,
        k=k,
        which=which,
        subspace_iters=subspace_iters,
        block_size=block_size,
        preconditioner=preconditioner,
        v0=v0,
        tol=tol,
    )
    diag = _jrb_eig_diagnostics(matvec, vals, vecs, algorithm_code=12, steps=subspace_iters, basis_dim=int(k if block_size is None else block_size), method="lanczos", tol=tol)
    return vals, vecs, diag


def jrb_mat_eigsh_jacobi_davidson_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    subspace_iters: int = 4,
    block_size: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis = _jrb_orthonormalize_columns(_jrb_eigsh_mid_block(v0, size=size, block_size=actual_block))

    target_cols = min(size, max(actual_block, k))
    lock_tol = max(float(tol), matrix_free_core.eig_restart_lock_tolerance(steps=subspace_iters, restarts=1))
    for _ in range(int(subspace_iters)):
        keep_count = min(actual_block, basis.shape[1])
        vals, vecs = _jrb_ritz_pairs_from_basis(matvec, basis, k=keep_count, which=which)
        applied = _jrb_apply_operator_block_mid(matvec, vecs)
        residuals = applied - vecs * vals[None, :]
        basis_seed = _jrb_restart_basis_from_pairs(
            actual_block=min(actual_block, basis.shape[1]),
            vals=vals,
            vecs=vecs,
            residuals=residuals,
            which=which,
            lock_tol=lock_tol,
            refill_basis=basis,
        )
        target_cols = min(size, basis_seed.shape[1] + min(actual_block, residuals.shape[1]))
        basis = _jrb_expand_subspace_with_corrections(
            basis_seed,
            vecs,
            residuals,
            vals=vals,
            target_cols=target_cols,
            which=which,
            lock_tol=lock_tol,
            preconditioner=preconditioner,
            jacobi_davidson=True,
        )
    return _jrb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jrb_mat_eigsh_jacobi_davidson_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    subspace_iters: int = 4,
    block_size: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-6,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_jacobi_davidson_point(
        matvec,
        size=size,
        k=k,
        which=which,
        subspace_iters=subspace_iters,
        block_size=block_size,
        preconditioner=preconditioner,
        v0=v0,
        tol=tol,
    )
    diag = _jrb_eig_diagnostics(matvec, vals, vecs, algorithm_code=13, steps=subspace_iters, basis_dim=int(k if block_size is None else block_size), method="lanczos", tol=tol)
    return vals, vecs, diag


def jrb_mat_geigsh_point(
    a_matvec,
    b_matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
    b_preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    plan = jrb_mat_generalized_operator_plan_prepare(
        a_matvec,
        b_matvec,
        b_preconditioner=b_preconditioner,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    return jrb_mat_eigsh_point(plan, size=size, k=k, which=which, steps=steps, v0=v0)


def jrb_mat_geigsh_with_diagnostics_point(
    a_matvec,
    b_matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
    b_preconditioner=None,
    tol: float = 1e-3,
    solve_tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_geigsh_point(
        a_matvec,
        b_matvec,
        size=size,
        k=k,
        which=which,
        steps=steps,
        v0=v0,
        b_preconditioner=b_preconditioner,
        tol=solve_tol,
        atol=atol,
        maxiter=maxiter,
    )
    resolved_steps = min(size, max(int(k) + 2, 2 * int(k) + 8)) if steps is None else int(steps)
    diag = _jrb_generalized_eig_diagnostics(
        a_matvec,
        b_matvec,
        vals,
        vecs,
        algorithm_code=16,
        steps=resolved_steps,
        basis_dim=resolved_steps,
        tol=tol,
    )
    return vals, vecs, diag


def jrb_mat_shift_invert_operator_plan_prepare(
    matvec,
    *,
    shift,
    preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
):
    return matrix_free_core.shell_operator_plan(
        lambda v, context: jnp.asarray(
            jnp.real(
                _jrb_shifted_solve_mid(
                    context["matvec"],
                    jnp.asarray(v, dtype=jnp.float64),
                    shift=context["shift"],
                    preconditioner=context["preconditioner"],
                    tol=context["tol"],
                    atol=context["atol"],
                    maxiter=context["maxiter"],
                )
            ),
            dtype=jnp.float64,
        ),
        context={
            "matvec": matvec,
            "shift": jnp.asarray(shift, dtype=jnp.float64),
            "preconditioner": preconditioner,
            "tol": float(tol),
            "atol": float(atol),
            "maxiter": maxiter,
        },
        orientation="forward",
        algebra="jrb",
    )


def jrb_mat_generalized_shift_invert_operator_plan_prepare(
    a_matvec,
    b_matvec,
    *,
    shift,
    preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
):
    return matrix_free_core.shell_operator_plan(
        lambda v, context: jnp.asarray(
            jnp.real(
                _jrb_generalized_shifted_solve_mid(
                    context["a_matvec"],
                    context["b_matvec"],
                    _jrb_apply_operator_mid(context["b_matvec"], _jrb_point_interval(jnp.asarray(v, dtype=jnp.float64))),
                    shift=context["shift"],
                    preconditioner=context["preconditioner"],
                    tol=context["tol"],
                    atol=context["atol"],
                    maxiter=context["maxiter"],
                )
            ),
            dtype=jnp.float64,
        ),
        context={
            "a_matvec": a_matvec,
            "b_matvec": b_matvec,
            "shift": jnp.asarray(shift, dtype=jnp.float64),
            "preconditioner": preconditioner,
            "tol": float(tol),
            "atol": float(atol),
            "maxiter": maxiter,
        },
        orientation="forward",
        algebra="jrb",
    )


def jrb_mat_eigsh_shift_invert_point(
    matvec,
    *,
    size: int,
    shift,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    plan = jrb_mat_shift_invert_operator_plan_prepare(matvec, shift=shift, preconditioner=preconditioner)
    vals, vecs = jrb_mat_eigsh_point(plan, size=size, k=k, which=which, steps=steps, v0=v0)
    mids = jnp.asarray(vals, dtype=jnp.complex128)
    mapped = jnp.asarray(shift, dtype=jnp.complex128) + 1.0 / mids
    return _jrb_point_interval(jnp.real(mapped)), _jrb_point_interval(jnp.real(vecs))


def jrb_mat_eigsh_shift_invert_with_diagnostics_point(
    matvec,
    *,
    size: int,
    shift,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-6,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_shift_invert_point(
        matvec,
        size=size,
        shift=shift,
        k=k,
        which=which,
        steps=steps,
        preconditioner=preconditioner,
        v0=v0,
    )
    resolved_steps = min(size, max(int(k) + 2, 2 * int(k) + 8)) if steps is None else int(steps)
    diag = _jrb_eig_diagnostics(matvec, di.midpoint(vals), di.midpoint(vecs), algorithm_code=14, steps=resolved_steps, basis_dim=resolved_steps, method="gmres", tol=tol)
    return vals, vecs, diag


def jrb_mat_geigsh_shift_invert_point(
    a_matvec,
    b_matvec,
    *,
    size: int,
    shift,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    plan = jrb_mat_generalized_shift_invert_operator_plan_prepare(
        a_matvec,
        b_matvec,
        shift=shift,
        preconditioner=preconditioner,
    )
    vals, vecs = jrb_mat_eigsh_point(plan, size=size, k=k, which=which, steps=steps, v0=v0)
    mids = jnp.asarray(vals, dtype=jnp.complex128)
    mapped = jnp.asarray(shift, dtype=jnp.complex128) + 1.0 / mids
    return _jrb_point_interval(jnp.real(mapped)), _jrb_point_interval(jnp.real(vecs))


def jrb_mat_geigsh_shift_invert_with_diagnostics_point(
    a_matvec,
    b_matvec,
    *,
    size: int,
    shift,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-6,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_geigsh_shift_invert_point(
        a_matvec,
        b_matvec,
        size=size,
        shift=shift,
        k=k,
        which=which,
        steps=steps,
        preconditioner=preconditioner,
        v0=v0,
    )
    resolved_steps = min(size, max(int(k) + 2, 2 * int(k) + 8)) if steps is None else int(steps)
    diag = _jrb_generalized_eig_diagnostics(
        a_matvec,
        b_matvec,
        di.midpoint(vals),
        di.midpoint(vecs),
        algorithm_code=17,
        steps=resolved_steps,
        basis_dim=resolved_steps,
        tol=tol,
    )
    return vals, vecs, diag


def _jrb_nep_scalar_residual(matvec, vec_mid: jax.Array) -> jax.Array:
    applied = _jrb_apply_operator_mid(matvec, _jrb_point_interval(vec_mid))
    denom = jnp.maximum(jnp.vdot(vec_mid, vec_mid), jnp.asarray(1e-30, dtype=jnp.float64))
    return jnp.asarray(jnp.vdot(vec_mid, applied) / denom, dtype=jnp.float64)


def _jrb_polynomial_operator_plan_prepare(coeff_matvecs, lam):
    coeffs = tuple(jnp.asarray(lam, dtype=jnp.float64) ** i for i in range(len(coeff_matvecs)))
    return matrix_free_core.shell_operator_plan(
        lambda v, context: sum(
            context["coeffs"][i] * _jrb_apply_operator_mid(context["ops"][i], _jrb_point_interval(jnp.asarray(v, dtype=jnp.float64)))
            for i in range(len(context["ops"]))
        ),
        context={"ops": tuple(coeff_matvecs), "coeffs": coeffs},
        orientation="forward",
        algebra="jrb",
    )


def _jrb_polynomial_derivative_operator_plan_prepare(coeff_matvecs, lam):
    coeffs = tuple(
        jnp.asarray(i, dtype=jnp.float64) * (jnp.asarray(lam, dtype=jnp.float64) ** (i - 1))
        for i in range(len(coeff_matvecs))
    )
    return matrix_free_core.shell_operator_plan(
        lambda v, context: sum(
            context["coeffs"][i] * _jrb_apply_operator_mid(context["ops"][i], _jrb_point_interval(jnp.asarray(v, dtype=jnp.float64)))
            for i in range(1, len(context["ops"]))
        ),
        context={"ops": tuple(coeff_matvecs), "coeffs": coeffs},
        orientation="forward",
        algebra="jrb",
    )


def jrb_mat_neigsh_point(
    matvec_builder,
    dmatvec_builder,
    *,
    size: int,
    lambda0,
    newton_iters: int = 4,
    eig_steps: int = 6,
    v0: jax.Array | None = None,
    preconditioner_builder=None,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    lam = jnp.asarray(lambda0, dtype=jnp.float64)
    vec = _jrb_mid_vector(_jrb_eigsh_start_vector(size) if v0 is None else jrb_mat_as_interval_vector(v0))
    for _ in range(int(newton_iters)):
        op = matvec_builder(lam)
        prec = None if preconditioner_builder is None else preconditioner_builder(lam)
        vals, vecs = jrb_mat_eigsh_shift_invert_point(
            op,
            size=size,
            shift=0.0,
            k=1,
            which="largest",
            steps=eig_steps,
            preconditioner=prec,
            v0=_jrb_point_interval(vec),
        )
        vec = di.midpoint(vecs)[:, 0]
        residual = di.midpoint(vals)[0]
        if jnp.abs(residual) <= jnp.asarray(tol, dtype=jnp.float64):
            break
        dop = dmatvec_builder(lam)
        derivative = _jrb_nep_scalar_residual(dop, vec)
        safe_derivative = jnp.where(jnp.abs(derivative) > tol, derivative, jnp.asarray(1.0, dtype=jnp.float64))
        lam = lam - residual / safe_derivative
    return _jrb_point_interval(jnp.asarray([lam], dtype=jnp.float64)), _jrb_point_interval(vec[:, None])


def jrb_mat_neigsh_with_diagnostics_point(
    matvec_builder,
    dmatvec_builder,
    *,
    size: int,
    lambda0,
    newton_iters: int = 4,
    eig_steps: int = 6,
    v0: jax.Array | None = None,
    preconditioner_builder=None,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    lam = jnp.asarray(lambda0, dtype=jnp.float64)
    vec = _jrb_mid_vector(_jrb_eigsh_start_vector(size) if v0 is None else jrb_mat_as_interval_vector(v0))
    residual_history = []
    for _ in range(int(newton_iters)):
        op = matvec_builder(lam)
        prec = None if preconditioner_builder is None else preconditioner_builder(lam)
        vals, vecs = jrb_mat_eigsh_shift_invert_point(
            op,
            size=size,
            shift=0.0,
            k=1,
            which="largest",
            steps=eig_steps,
            preconditioner=prec,
            v0=_jrb_point_interval(vec),
        )
        vec = di.midpoint(vecs)[:, 0]
        residual = di.midpoint(vals)[0]
        residual_history.append(jnp.abs(residual))
        if jnp.abs(residual) <= jnp.asarray(tol, dtype=jnp.float64):
            break
        dop = dmatvec_builder(lam)
        derivative = _jrb_nep_scalar_residual(dop, vec)
        safe_derivative = jnp.where(jnp.abs(derivative) > tol, derivative, jnp.asarray(1.0, dtype=jnp.float64))
        lam = lam - residual / safe_derivative
    vals_out = _jrb_point_interval(jnp.asarray([lam], dtype=jnp.float64))
    vecs_out = _jrb_point_interval(vec[:, None])
    history = jnp.asarray(residual_history if residual_history else [0.0], dtype=jnp.float64)
    diag = matrix_free_core.krylov_diagnostics(
        JrbMatKrylovDiagnostics,
        algorithm_code=18,
        steps=eig_steps,
        basis_dim=size,
        beta0=1.0,
        tail_norm=history[-1],
        breakdown=False,
        used_adjoint=False,
        gradient_supported=True,
        probe_count=1,
        restart_count=newton_iters,
    )
    diag = _jrb_attach_diag(
        diag,
        regime="structured",
        method="lanczos",
        structure="symmetric",
        work_units=eig_steps,
        primal_residual=history[-1],
        note="matrix_free.neigsh",
    )
    diag = _jrb_update_convergence(
        diag,
        converged=history[-1] <= jnp.asarray(tol, dtype=jnp.float64),
        convergence_metric=history[-1],
        locked_count=1,
        residual_history=history,
        deflated_count=1 if history[-1] <= jnp.asarray(tol, dtype=jnp.float64) else 0,
    )
    return vals_out, vecs_out, diag


def jrb_mat_peigsh_point(
    coeff_matvecs,
    *,
    size: int,
    lambda0,
    newton_iters: int = 4,
    eig_steps: int = 6,
    v0: jax.Array | None = None,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    return jrb_mat_neigsh_point(
        lambda lam: _jrb_polynomial_operator_plan_prepare(coeff_matvecs, lam),
        lambda lam: _jrb_polynomial_derivative_operator_plan_prepare(coeff_matvecs, lam),
        size=size,
        lambda0=lambda0,
        newton_iters=newton_iters,
        eig_steps=eig_steps,
        v0=v0,
        tol=tol,
    )


def jrb_mat_peigsh_with_diagnostics_point(
    coeff_matvecs,
    *,
    size: int,
    lambda0,
    newton_iters: int = 4,
    eig_steps: int = 6,
    v0: jax.Array | None = None,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs, diag = jrb_mat_neigsh_with_diagnostics_point(
        lambda lam: _jrb_polynomial_operator_plan_prepare(coeff_matvecs, lam),
        lambda lam: _jrb_polynomial_derivative_operator_plan_prepare(coeff_matvecs, lam),
        size=size,
        lambda0=lambda0,
        newton_iters=newton_iters,
        eig_steps=eig_steps,
        v0=v0,
        tol=tol,
    )
    return vals, vecs, diag._replace(algorithm_code=jnp.asarray(19, dtype=jnp.int32))


def jrb_mat_eigsh_contour_point(
    matvec,
    *,
    size: int,
    center,
    radius,
    k: int = 6,
    which: str = "largest",
    quadrature_order: int = 8,
    block_size: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis0 = jnp.asarray(_jrb_eigsh_mid_block(v0, size=size, block_size=actual_block), dtype=jnp.complex128)

    def solve_shifted_block(shift, block):
        return jax.vmap(
            lambda col: _jrb_shifted_solve_mid(matvec, col, shift=shift, preconditioner=preconditioner),
            in_axes=1,
            out_axes=1,
        )(block)

    filtered = matrix_free_core.contour_filter_subspace_point(
        solve_shifted_block,
        basis0,
        center=center,
        radius=radius,
        quadrature_order=int(quadrature_order),
    )
    vals, vecs = matrix_free_core.ritz_pairs_from_basis(
        lambda q: jax.vmap(
            lambda col: (
                matrix_free_core.operator_apply_midpoint(
                    matvec,
                    col,
                    midpoint_vector=lambda y: jnp.asarray(y, dtype=jnp.complex128),
                    sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
                    dtype=jnp.complex128,
                )
                if isinstance(matvec, matrix_free_core.OperatorPlan)
                else matrix_free_core.complexify_real_linear_operator(
                    lambda y: _jrb_apply_operator_mid(matvec, _jrb_point_interval(y)),
                    col,
                )
            ),
            in_axes=1,
            out_axes=1,
        )(q),
        filtered,
        k=k,
        which=which,
        hermitian=True,
    )
    return _jrb_point_interval(jnp.real(vals)), _jrb_point_interval(jnp.real(vecs))


def jrb_mat_eigsh_contour_with_diagnostics_point(
    matvec,
    *,
    size: int,
    center,
    radius,
    k: int = 6,
    which: str = "largest",
    quadrature_order: int = 8,
    block_size: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-6,
) -> tuple[jax.Array, jax.Array, JrbMatKrylovDiagnostics]:
    vals, vecs = jrb_mat_eigsh_contour_point(
        matvec,
        size=size,
        center=center,
        radius=radius,
        k=k,
        which=which,
        quadrature_order=quadrature_order,
        block_size=block_size,
        preconditioner=preconditioner,
        v0=v0,
    )
    diag = _jrb_eig_diagnostics(matvec, di.midpoint(vals), di.midpoint(vecs), algorithm_code=15, steps=quadrature_order, basis_dim=int(k if block_size is None else block_size), method="gmres", tol=tol)
    return vals, vecs, diag


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


def jrb_mat_log_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_log_action_lanczos_point(matvec, x, steps)


def jrb_mat_log_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_log_action_lanczos_point(matvec, x, steps)


def _jrb_action_basic_from_diagnostics(point_with_diagnostics_fn, matvec, x: jax.Array, *args, prec_bits: int, **kwargs) -> jax.Array:
    value, _ = matrix_free_basic.action_with_diagnostics_basic(
        point_with_diagnostics_fn,
        matvec,
        x,
        *args,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_interval_like,
        **kwargs,
    )
    return value


def jrb_mat_log_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_log_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


def jrb_mat_sqrt_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sqrt), steps)


def jrb_mat_sqrt_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sqrt_action_lanczos_point(matvec, x, steps)


def jrb_mat_sqrt_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sqrt_action_lanczos_point(matvec, x, steps)


def jrb_mat_sqrt_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_sqrt_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


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


def jrb_mat_root_action_symmetric_point(matvec, x: jax.Array, *, degree: int, steps: int) -> jax.Array:
    return jrb_mat_root_action_lanczos_point(matvec, x, degree=degree, steps=steps)


def jrb_mat_root_action_spd_point(matvec, x: jax.Array, *, degree: int, steps: int) -> jax.Array:
    return jrb_mat_root_action_lanczos_point(matvec, x, degree=degree, steps=steps)


def jrb_mat_root_action_lanczos_basic(
    matvec,
    x: jax.Array,
    *,
    degree: int,
    steps: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(
        jrb_mat_root_action_lanczos_with_diagnostics_point,
        matvec,
        x,
        degree=degree,
        steps=steps,
        prec_bits=prec_bits,
    )


def jrb_mat_sign_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sign), steps)


def jrb_mat_sign_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sign_action_lanczos_point(matvec, x, steps)


def jrb_mat_sign_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sign_action_lanczos_point(matvec, x, steps)


def jrb_mat_sign_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_sign_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


def jrb_mat_sin_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sin), steps)


def jrb_mat_sin_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sin_action_lanczos_point(matvec, x, steps)


def jrb_mat_sin_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sin_action_lanczos_point(matvec, x, steps)


def jrb_mat_sin_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_sin_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


def jrb_mat_cos_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.cos), steps)


def jrb_mat_cos_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_cos_action_lanczos_point(matvec, x, steps)


def jrb_mat_cos_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_cos_action_lanczos_point(matvec, x, steps)


def jrb_mat_cos_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_cos_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


def jrb_mat_sinh_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sinh), steps)


def jrb_mat_sinh_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sinh_action_lanczos_point(matvec, x, steps)


def jrb_mat_sinh_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_sinh_action_lanczos_point(matvec, x, steps)


def jrb_mat_sinh_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_sinh_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


def jrb_mat_cosh_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.cosh), steps)


def jrb_mat_cosh_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_cosh_action_lanczos_point(matvec, x, steps)


def jrb_mat_cosh_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_cosh_action_lanczos_point(matvec, x, steps)


def jrb_mat_cosh_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_cosh_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


def jrb_mat_tanh_action_lanczos_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_funm_action_lanczos_point(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.tanh), steps)


def jrb_mat_tanh_action_symmetric_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_tanh_action_lanczos_point(matvec, x, steps)


def jrb_mat_tanh_action_spd_point(matvec, x: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_tanh_action_lanczos_point(matvec, x, steps)


def jrb_mat_tanh_action_lanczos_basic(
    matvec,
    x: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(jrb_mat_tanh_action_lanczos_with_diagnostics_point, matvec, x, steps, prec_bits=prec_bits)


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


def jrb_mat_pow_action_symmetric_point(matvec, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    return jrb_mat_pow_action_lanczos_point(matvec, x, exponent=exponent, steps=steps)


def jrb_mat_pow_action_spd_point(matvec, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    return jrb_mat_pow_action_lanczos_point(matvec, x, exponent=exponent, steps=steps)


def jrb_mat_pow_action_lanczos_basic(
    matvec,
    x: jax.Array,
    *,
    exponent: int,
    steps: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jrb_action_basic_from_diagnostics(
        jrb_mat_pow_action_lanczos_with_diagnostics_point,
        matvec,
        x,
        exponent=exponent,
        steps=steps,
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


def jrb_mat_expm_action_symmetric_point(matvec, x: jax.Array, *, steps: int, restarts: int = 1) -> jax.Array:
    return jrb_mat_expm_action_lanczos_restarted_point(matvec, x, steps=steps, restarts=restarts)


def jrb_mat_expm_action_spd_point(matvec, x: jax.Array, *, steps: int, restarts: int = 1) -> jax.Array:
    return jrb_mat_expm_action_lanczos_restarted_point(matvec, x, steps=steps, restarts=restarts)


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
        probe_midpoint=di.midpoint,
    )


def jrb_mat_logdet_slq_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return mat_common.estimator_mean(
        probes,
        di.as_interval,
        lambda v: jrb_mat_funm_trace_integrand_lanczos_point(matvec, v, jnp.log, steps=steps),
        probe_midpoint=di.midpoint,
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
        probe_midpoint=di.midpoint,
    )


def _jrb_mat_logdet_slq_point_plan_kernel_bucketed(matvec, probes: jax.Array, *, effective_steps: int, max_steps: int) -> jax.Array:
    dense_funm = jrb_mat_dense_funm_sym_eigh_point(jnp.log)

    def integrand(v):
        basis, projected, beta0 = _jrb_mat_lanczos_tridiag_state_bucketed_point(matvec, v, effective_steps, max_steps)[:3]
        del basis
        active = jnp.clip(jnp.asarray(effective_steps, dtype=jnp.int32), 1, jnp.asarray(max_steps, dtype=jnp.int32))
        active_mask = jnp.arange(max_steps, dtype=jnp.int32) < active
        active_matrix = active_mask[:, None] & active_mask[None, :]
        masked_projected = jnp.where(active_matrix, projected, jnp.zeros_like(projected))
        masked_projected = masked_projected + jnp.diag(jnp.where(active_mask, 0.0, 1.0))
        e1 = jnp.zeros((max_steps,), dtype=jnp.float64).at[0].set(jnp.asarray(1.0, dtype=jnp.float64))
        value = (beta0**2) * jnp.vdot(e1, dense_funm(masked_projected) @ e1)
        return jnp.asarray(jnp.real(value), dtype=jnp.float64)

    return mat_common.estimator_mean(
        probes,
        di.as_interval,
        integrand,
        probe_midpoint=di.midpoint,
    )


def _jrb_mat_det_slq_point_plan_kernel(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return matrix_free_core.det_from_logdet(_jrb_mat_logdet_slq_point_plan_kernel(matvec, probes, steps))


def _jrb_mat_det_slq_point_plan_kernel_bucketed(matvec, probes: jax.Array, *, effective_steps: int, max_steps: int) -> jax.Array:
    return matrix_free_core.det_from_logdet(
        _jrb_mat_logdet_slq_point_plan_kernel_bucketed(
            matvec,
            probes,
            effective_steps=effective_steps,
            max_steps=max_steps,
        )
    )


def jrb_mat_funm_action_lanczos_with_diagnostics_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    value, diag = mat_common.action_with_diagnostics(
        lambda xx: jrb_mat_funm_action_lanczos_point(matvec, xx, dense_funm, steps),
        lambda xx: jrb_mat_lanczos_diagnostics_point(matvec, xx, steps),
        x,
    )
    diag = _jrb_update_convergence(diag, converged=jnp.isfinite(diag.tail_norm), convergence_metric=diag.tail_norm)
    return value, diag


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


def jrb_mat_log_action_lanczos_with_diagnostics_basic(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.action_with_diagnostics_basic(
        jrb_mat_log_action_lanczos_with_diagnostics_point,
        matvec,
        x,
        steps=steps,
        round_output=_jrb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_interval_like,
    )


def jrb_mat_log_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_log_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_log_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_log_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_sqrt_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sqrt_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_sqrt_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sqrt_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_root_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, *, degree: int, steps: int):
    return jrb_mat_root_action_lanczos_with_diagnostics_point(matvec, x, degree=degree, steps=steps)


def jrb_mat_root_action_spd_with_diagnostics_point(matvec, x: jax.Array, *, degree: int, steps: int):
    return jrb_mat_root_action_lanczos_with_diagnostics_point(matvec, x, degree=degree, steps=steps)


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


def jrb_mat_sign_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sign_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_sign_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sign_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_sin_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sin_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_sin_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sin_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_cos_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_cos_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_cos_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_cos_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_sinh_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sinh_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_sinh_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_sinh_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_cosh_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_cosh_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_cosh_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_cosh_action_lanczos_with_diagnostics_point(matvec, x, steps)


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


def jrb_mat_tanh_action_symmetric_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_tanh_action_lanczos_with_diagnostics_point(matvec, x, steps)


def jrb_mat_tanh_action_spd_with_diagnostics_point(matvec, x: jax.Array, steps: int):
    return jrb_mat_tanh_action_lanczos_with_diagnostics_point(matvec, x, steps)


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
    value, diag = mat_common.estimator_with_diagnostics(
        probes,
        coerce_probes=di.as_interval,
        estimator_fn=lambda xs: jrb_mat_logdet_slq_point(matvec, xs, steps),
        diagnostics_fn=lambda first: jrb_mat_lanczos_diagnostics_point(matvec, first, steps),
        algorithm_code=2,
    )
    diag = _jrb_attach_diag(
        diag,
        regime="structured",
        method="lanczos",
        structure="spd",
        work_units=steps,
        primal_residual=diag.tail_norm,
        adjoint_residual=0.0,
        note="matrix_free.logdet_slq",
    )
    diag = _jrb_update_convergence(
        diag,
        converged=jnp.isfinite(diag.tail_norm),
        convergence_metric=diag.tail_norm,
    )
    return value, diag


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
        invalidate_output=_full_interval_like,
    )


def jrb_mat_det_slq_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    steps: int,
) -> tuple[jax.Array, JrbMatKrylovDiagnostics]:
    value, diag = jrb_mat_logdet_slq_with_diagnostics_point(matvec, probes, steps)
    diag = _jrb_attach_diag(
        diag,
        regime="structured",
        method="lanczos",
        structure="spd",
        work_units=steps,
        primal_residual=diag.primal_residual,
        adjoint_residual=diag.adjoint_residual,
        note="matrix_free.det_slq",
    )
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
        invalidate_output=_full_interval_like,
    )


def jrb_mat_logdet_solve_point(
    matvec,
    rhs: jax.Array,
    probes: jax.Array,
    steps: int,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
) -> matrix_free_core.LogdetSolveResult:
    return matrix_free_core.combine_logdet_solve_point(
        operator=matvec,
        rhs=rhs,
        probes=probes,
        solve_with_diagnostics=lambda operator, rhs_value: jrb_mat_solve_action_with_diagnostics_point(
            operator,
            rhs_value,
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            symmetric=symmetric,
            preconditioner=preconditioner,
        ),
        logdet_with_diagnostics=lambda operator, probe_value: jrb_mat_logdet_slq_with_diagnostics_point(operator, probe_value, steps),
        preconditioner=preconditioner,
        structured=_jrb_structure_tag(symmetric=symmetric, spd=symmetric),
        algebra="jrb",
    )


def jrb_mat_logdet_solve_basic(
    matvec,
    rhs: jax.Array,
    probes: jax.Array,
    steps: int,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> matrix_free_core.LogdetSolveResult:
    result = jrb_mat_logdet_solve_point(
        matvec,
        rhs,
        probes,
        steps,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
    )
    return matrix_free_core.LogdetSolveResult(
        logdet=_jrb_round_basic(_jrb_point_interval(result.logdet), prec_bits),
        solve=_jrb_round_basic(result.solve, prec_bits),
        aux=result.aux,
    )


def jrb_mat_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    return matrix_free_core.rademacher_probes_real(_jrb_point_interval, x.shape[-2], key=key, num=num)


def jrb_mat_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    return matrix_free_core.normal_probes_real(_jrb_point_interval, x.shape[-2], key=key, num=num)


def jrb_mat_orthogonal_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    return matrix_free_core.orthogonal_rademacher_probe_block_real(_jrb_point_interval, x.shape[-2], key=key, num=num)


def jrb_mat_orthogonal_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    return matrix_free_core.orthogonal_normal_probe_block_real(_jrb_point_interval, x.shape[-2], key=key, num=num)


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
_jrb_mat_operator_apply_point_jit_callable = jax.jit(jrb_mat_operator_apply_point, static_argnames=("matvec",))
_jrb_mat_operator_apply_point_jit_plan = jax.jit(jrb_mat_operator_apply_point)


def jrb_mat_operator_apply_point_jit(matvec, x: jax.Array) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_operator_apply_point_jit_plan(matvec, x)
    return _jrb_mat_operator_apply_point_jit_callable(matvec, x)


jrb_mat_rmatvec_point_jit = jax.jit(jrb_mat_rmatvec_point)

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
    static_argnames=("tol", "atol", "maxiter", "symmetric"),
)


def jrb_mat_solve_action_point_jit(matvec, b: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_solve_action_point_jit_plan(matvec, b, **kwargs)
    return _jrb_mat_solve_action_point_jit_callable(matvec, b, **kwargs)


_jrb_mat_minres_solve_action_point_jit_callable = jax.jit(
    jrb_mat_minres_solve_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "preconditioner"),
)
_jrb_mat_minres_solve_action_point_jit_plan = jax.jit(
    jrb_mat_minres_solve_action_point,
    static_argnames=("tol", "atol", "maxiter"),
)


def jrb_mat_minres_solve_action_point_jit(matvec, b: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_minres_solve_action_point_jit_plan(matvec, b, **kwargs)
    return _jrb_mat_minres_solve_action_point_jit_callable(matvec, b, **kwargs)


_jrb_mat_inverse_action_point_jit_callable = jax.jit(
    jrb_mat_inverse_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "symmetric", "preconditioner"),
)
_jrb_mat_inverse_action_point_jit_plan = jax.jit(
    jrb_mat_inverse_action_point,
    static_argnames=("tol", "atol", "maxiter", "symmetric"),
)


def jrb_mat_inverse_action_point_jit(matvec, x: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_inverse_action_point_jit_plan(matvec, x, **kwargs)
    return _jrb_mat_inverse_action_point_jit_callable(matvec, x, **kwargs)


_jrb_mat_minres_inverse_action_point_jit_callable = jax.jit(
    jrb_mat_minres_inverse_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "preconditioner"),
)
_jrb_mat_minres_inverse_action_point_jit_plan = jax.jit(
    jrb_mat_minres_inverse_action_point,
    static_argnames=("tol", "atol", "maxiter"),
)


def jrb_mat_minres_inverse_action_point_jit(matvec, x: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_minres_inverse_action_point_jit_plan(matvec, x, **kwargs)
    return _jrb_mat_minres_inverse_action_point_jit_callable(matvec, x, **kwargs)


_jrb_mat_logdet_slq_point_jit_callable = jax.jit(jrb_mat_logdet_slq_point, static_argnames=("matvec", "steps"))
_jrb_mat_logdet_slq_point_jit_plan = jax.jit(_jrb_mat_logdet_slq_point_plan_kernel, static_argnames=("steps",))
_jrb_mat_logdet_slq_point_jit_plan_bucketed = jax.jit(
    _jrb_mat_logdet_slq_point_plan_kernel_bucketed,
    static_argnames=("max_steps",),
)


def jrb_mat_logdet_slq_point_jit(matvec, probes: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_logdet_slq_point_jit_plan(matvec, probes, steps=steps)
    return _jrb_mat_logdet_slq_point_jit_callable(matvec, probes, steps=steps)


_jrb_mat_det_slq_point_jit_callable = jax.jit(jrb_mat_det_slq_point, static_argnames=("matvec", "steps"))
_jrb_mat_det_slq_point_jit_plan = jax.jit(_jrb_mat_det_slq_point_plan_kernel, static_argnames=("steps",))
_jrb_mat_det_slq_point_jit_plan_bucketed = jax.jit(
    _jrb_mat_det_slq_point_plan_kernel_bucketed,
    static_argnames=("max_steps",),
)


def jrb_mat_det_slq_point_jit(matvec, probes: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_det_slq_point_jit_plan(matvec, probes, steps=steps)
    return _jrb_mat_det_slq_point_jit_callable(matvec, probes, steps=steps)


def _jrb_mat_log_action_lanczos_point_plan_kernel(matvec, x: jax.Array, *, steps: int) -> jax.Array:
    return _jrb_mat_funm_action_lanczos_point_base(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.log), steps)


_jrb_mat_log_action_lanczos_point_jit_callable = jax.jit(jrb_mat_log_action_lanczos_point, static_argnames=("matvec", "steps"))
_jrb_mat_log_action_lanczos_point_jit_plan = jax.jit(_jrb_mat_log_action_lanczos_point_plan_kernel, static_argnames=("steps",))


def jrb_mat_log_action_lanczos_point_jit(matvec, x: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_log_action_lanczos_point_jit_plan(matvec, x, steps=steps)
    return _jrb_mat_log_action_lanczos_point_jit_callable(matvec, x, steps=steps)


def _jrb_mat_sqrt_action_lanczos_point_plan_kernel(matvec, x: jax.Array, *, steps: int) -> jax.Array:
    return _jrb_mat_funm_action_lanczos_point_base(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sqrt), steps)


_jrb_mat_sqrt_action_lanczos_point_jit_callable = jax.jit(jrb_mat_sqrt_action_lanczos_point, static_argnames=("matvec", "steps"))
_jrb_mat_sqrt_action_lanczos_point_jit_plan = jax.jit(_jrb_mat_sqrt_action_lanczos_point_plan_kernel, static_argnames=("steps",))


def jrb_mat_sqrt_action_lanczos_point_jit(matvec, x: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_sqrt_action_lanczos_point_jit_plan(matvec, x, steps=steps)
    return _jrb_mat_sqrt_action_lanczos_point_jit_callable(matvec, x, steps=steps)


def _jrb_mat_sign_action_lanczos_point_plan_kernel(matvec, x: jax.Array, *, steps: int) -> jax.Array:
    return _jrb_mat_funm_action_lanczos_point_base(matvec, x, jrb_mat_dense_funm_sym_eigh_point(jnp.sign), steps)


_jrb_mat_sign_action_lanczos_point_jit_callable = jax.jit(jrb_mat_sign_action_lanczos_point, static_argnames=("matvec", "steps"))
_jrb_mat_sign_action_lanczos_point_jit_plan = jax.jit(_jrb_mat_sign_action_lanczos_point_plan_kernel, static_argnames=("steps",))


def jrb_mat_sign_action_lanczos_point_jit(matvec, x: jax.Array, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_sign_action_lanczos_point_jit_plan(matvec, x, steps=steps)
    return _jrb_mat_sign_action_lanczos_point_jit_callable(matvec, x, steps=steps)


def _jrb_mat_pow_action_lanczos_point_plan_kernel(matvec, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    return _jrb_mat_funm_action_lanczos_point_base(
        matvec,
        x,
        jrb_mat_dense_funm_sym_eigh_point(lambda vals: jnp.power(vals, exponent)),
        steps,
    )


_jrb_mat_pow_action_lanczos_point_jit_callable = jax.jit(jrb_mat_pow_action_lanczos_point, static_argnames=("matvec", "exponent", "steps"))
_jrb_mat_pow_action_lanczos_point_jit_plan = jax.jit(_jrb_mat_pow_action_lanczos_point_plan_kernel, static_argnames=("exponent", "steps"))


def jrb_mat_pow_action_lanczos_point_jit(matvec, x: jax.Array, *, exponent: int, steps: int) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_pow_action_lanczos_point_jit_plan(matvec, x, exponent=exponent, steps=steps)
    return _jrb_mat_pow_action_lanczos_point_jit_callable(matvec, x, exponent=exponent, steps=steps)


_jrb_mat_eigsh_point_jit_callable = jax.jit(
    jrb_mat_eigsh_point,
    static_argnames=("matvec", "size", "k", "which", "steps"),
)
_jrb_mat_eigsh_point_jit_plan = jax.jit(
    jrb_mat_eigsh_point,
    static_argnames=("size", "k", "which", "steps"),
)


def jrb_mat_eigsh_point_jit(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_eigsh_point_jit_plan(matvec, size=size, k=k, which=which, steps=steps, v0=v0)
    return _jrb_mat_eigsh_point_jit_callable(matvec, size=size, k=k, which=which, steps=steps, v0=v0)


_jrb_mat_multi_shift_solve_point_jit_callable = jax.jit(
    jrb_mat_multi_shift_solve_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "symmetric", "preconditioner"),
)
_jrb_mat_multi_shift_solve_point_jit_plan = jax.jit(
    jrb_mat_multi_shift_solve_point,
    static_argnames=("tol", "atol", "maxiter", "symmetric"),
)


def jrb_mat_multi_shift_solve_point_jit(matvec, rhs: jax.Array, shifts: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_multi_shift_solve_point_jit_plan(matvec, rhs, shifts, **kwargs)
    return _jrb_mat_multi_shift_solve_point_jit_callable(matvec, rhs, shifts, **kwargs)


_jrb_mat_eigsh_block_point_jit_callable = jax.jit(
    jrb_mat_eigsh_block_point,
    static_argnames=("matvec", "size", "k", "which", "block_size", "subspace_iters"),
)
_jrb_mat_eigsh_block_point_jit_plan = jax.jit(
    jrb_mat_eigsh_block_point,
    static_argnames=("size", "k", "which", "block_size", "subspace_iters"),
)


def jrb_mat_eigsh_block_point_jit(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    block_size: int | None = None,
    subspace_iters: int = 4,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_eigsh_block_point_jit_plan(matvec, size=size, k=k, which=which, block_size=block_size, subspace_iters=subspace_iters, v0=v0)
    return _jrb_mat_eigsh_block_point_jit_callable(matvec, size=size, k=k, which=which, block_size=block_size, subspace_iters=subspace_iters, v0=v0)


_jrb_mat_eigsh_restarted_point_jit_callable = jax.jit(
    jrb_mat_eigsh_restarted_point,
    static_argnames=("matvec", "size", "k", "which", "steps", "restarts", "block_size"),
)
_jrb_mat_eigsh_restarted_point_jit_plan = jax.jit(
    jrb_mat_eigsh_restarted_point,
    static_argnames=("size", "k", "which", "steps", "restarts", "block_size"),
)


def jrb_mat_eigsh_restarted_point_jit(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int = 4,
    restarts: int = 2,
    block_size: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jrb_mat_eigsh_restarted_point_jit_plan(matvec, size=size, k=k, which=which, steps=steps, restarts=restarts, block_size=block_size, v0=v0)
    return _jrb_mat_eigsh_restarted_point_jit_callable(matvec, size=size, k=k, which=which, steps=steps, restarts=restarts, block_size=block_size, v0=v0)


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
    "jrb_mat_dense_parametric_operator_plan_prepare",
    "jrb_mat_shell_operator_plan_prepare",
    "jrb_mat_dense_operator_rmatvec_plan_prepare",
    "jrb_mat_dense_parametric_operator_rmatvec_plan_prepare",
    "jrb_mat_dense_operator_adjoint_plan_prepare",
    "jrb_mat_dense_parametric_operator_adjoint_plan_prepare",
    "jrb_mat_finite_difference_operator_plan_prepare",
    "jrb_mat_finite_difference_operator_plan_set_base",
    "jrb_mat_bcoo_operator",
    "jrb_mat_bcoo_operator_adjoint",
    "jrb_mat_bcoo_operator_rmatvec",
    "jrb_mat_bcoo_operator_plan_prepare",
    "jrb_mat_bcoo_operator_rmatvec_plan_prepare",
    "jrb_mat_bcoo_operator_adjoint_plan_prepare",
    "jrb_mat_block_sparse_operator_plan_prepare",
    "jrb_mat_block_sparse_operator_rmatvec_plan_prepare",
    "jrb_mat_block_sparse_operator_adjoint_plan_prepare",
    "jrb_mat_vblock_sparse_operator_plan_prepare",
    "jrb_mat_vblock_sparse_operator_rmatvec_plan_prepare",
    "jrb_mat_vblock_sparse_operator_adjoint_plan_prepare",
    "jrb_mat_operator_plan_apply",
    "jrb_mat_operator_apply_point_jit",
    "jrb_mat_rmatvec_point",
    "jrb_mat_rmatvec_basic",
    "jrb_mat_rmatvec_point_jit",
    "jrb_mat_lanczos_tridiag_adjoint",
    "jrb_mat_cg_fixed_iterations",
    "jrb_mat_jacobi_preconditioner_plan_prepare",
    "jrb_mat_shell_preconditioner_plan_prepare",
    "jrb_mat_solve_action_point",
    "jrb_mat_solve_action_symmetric_point",
    "jrb_mat_solve_action_spd_point",
    "jrb_mat_minres_solve_action_point",
    "jrb_mat_solve_action_basic",
    "jrb_mat_solve_action_with_diagnostics_point",
    "jrb_mat_solve_action_with_diagnostics_basic",
    "jrb_mat_minres_solve_action_basic",
    "jrb_mat_minres_solve_action_with_diagnostics_point",
    "jrb_mat_minres_solve_action_with_diagnostics_basic",
    "jrb_mat_inverse_action_point",
    "jrb_mat_inverse_action_symmetric_point",
    "jrb_mat_inverse_action_spd_point",
    "jrb_mat_minres_inverse_action_point",
    "jrb_mat_minres_inverse_action_basic",
    "jrb_mat_inverse_action_basic",
    "jrb_mat_inverse_action_with_diagnostics_point",
    "jrb_mat_inverse_action_with_diagnostics_basic",
    "jrb_mat_multi_shift_solve_point",
    "jrb_mat_multi_shift_solve_symmetric_point",
    "jrb_mat_multi_shift_solve_spd_point",
    "jrb_mat_multi_shift_solve_basic",
    "jrb_mat_bcoo_parametric_operator",
    "jrb_mat_bcoo_parametric_operator_plan_prepare",
    "jrb_mat_scipy_csr_operator",
    "jrb_mat_bcoo_gershgorin_bounds",
    "jrb_mat_bcoo_spectral_bounds_adaptive",
    "jrb_mat_operator_apply_point",
    "jrb_mat_operator_apply_basic",
    "jrb_mat_poly_action_point",
    "jrb_mat_poly_action_basic",
    "jrb_mat_rational_action_point",
    "jrb_mat_rational_action_basic",
    "jrb_mat_expm_action_point",
    "jrb_mat_expm_action_basic",
    "jrb_mat_lanczos_tridiag_point",
    "jrb_mat_lanczos_diagnostics_point",
    "jrb_mat_eigsh_point",
    "jrb_mat_eigsh_with_diagnostics_point",
    "jrb_mat_eigsh_basic",
    "jrb_mat_eigsh_block_point",
    "jrb_mat_eigsh_block_with_diagnostics_point",
    "jrb_mat_eigsh_restarted_point",
    "jrb_mat_eigsh_restarted_with_diagnostics_point",
    "jrb_mat_eigsh_krylov_schur_point",
    "jrb_mat_eigsh_krylov_schur_with_diagnostics_point",
    "jrb_mat_eigsh_davidson_point",
    "jrb_mat_eigsh_davidson_with_diagnostics_point",
    "jrb_mat_eigsh_jacobi_davidson_point",
    "jrb_mat_eigsh_jacobi_davidson_with_diagnostics_point",
    "jrb_mat_generalized_operator_plan_prepare",
    "jrb_mat_geigsh_point",
    "jrb_mat_geigsh_with_diagnostics_point",
    "jrb_mat_generalized_shift_invert_operator_plan_prepare",
    "jrb_mat_geigsh_shift_invert_point",
    "jrb_mat_geigsh_shift_invert_with_diagnostics_point",
    "jrb_mat_neigsh_point",
    "jrb_mat_neigsh_with_diagnostics_point",
    "jrb_mat_peigsh_point",
    "jrb_mat_peigsh_with_diagnostics_point",
    "jrb_mat_shift_invert_operator_plan_prepare",
    "jrb_mat_eigsh_shift_invert_point",
    "jrb_mat_eigsh_shift_invert_with_diagnostics_point",
    "jrb_mat_eigsh_contour_point",
    "jrb_mat_eigsh_contour_with_diagnostics_point",
    "jrb_mat_funm_action_lanczos_point",
    "jrb_mat_funm_action_lanczos_with_diagnostics_point",
    "jrb_mat_funm_integrand_lanczos_point",
    "jrb_mat_dense_funm_sym_eigh_point",
    "jrb_mat_funm_action_lanczos_dense_point",
    "jrb_mat_log_action_lanczos_point",
    "jrb_mat_log_action_symmetric_point",
    "jrb_mat_log_action_spd_point",
    "jrb_mat_log_action_lanczos_basic",
    "jrb_mat_sqrt_action_lanczos_point",
    "jrb_mat_sqrt_action_symmetric_point",
    "jrb_mat_sqrt_action_spd_point",
    "jrb_mat_sqrt_action_lanczos_basic",
    "jrb_mat_root_action_lanczos_point",
    "jrb_mat_root_action_symmetric_point",
    "jrb_mat_root_action_spd_point",
    "jrb_mat_root_action_lanczos_basic",
    "jrb_mat_sign_action_lanczos_point",
    "jrb_mat_sign_action_symmetric_point",
    "jrb_mat_sign_action_spd_point",
    "jrb_mat_sign_action_lanczos_basic",
    "jrb_mat_sin_action_lanczos_point",
    "jrb_mat_sin_action_symmetric_point",
    "jrb_mat_sin_action_spd_point",
    "jrb_mat_sin_action_lanczos_basic",
    "jrb_mat_cos_action_lanczos_point",
    "jrb_mat_cos_action_symmetric_point",
    "jrb_mat_cos_action_spd_point",
    "jrb_mat_cos_action_lanczos_basic",
    "jrb_mat_sinh_action_lanczos_point",
    "jrb_mat_sinh_action_symmetric_point",
    "jrb_mat_sinh_action_spd_point",
    "jrb_mat_sinh_action_lanczos_basic",
    "jrb_mat_cosh_action_lanczos_point",
    "jrb_mat_cosh_action_symmetric_point",
    "jrb_mat_cosh_action_spd_point",
    "jrb_mat_cosh_action_lanczos_basic",
    "jrb_mat_tanh_action_lanczos_point",
    "jrb_mat_tanh_action_symmetric_point",
    "jrb_mat_tanh_action_spd_point",
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
    "jrb_mat_expm_action_symmetric_point",
    "jrb_mat_expm_action_spd_point",
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
    "jrb_mat_logdet_solve_point",
    "jrb_mat_logdet_solve_basic",
    "jrb_mat_log_action_lanczos_with_diagnostics_point",
    "jrb_mat_log_action_lanczos_with_diagnostics_basic",
    "jrb_mat_log_action_symmetric_with_diagnostics_point",
    "jrb_mat_log_action_spd_with_diagnostics_point",
    "jrb_mat_sqrt_action_lanczos_with_diagnostics_point",
    "jrb_mat_sqrt_action_symmetric_with_diagnostics_point",
    "jrb_mat_sqrt_action_spd_with_diagnostics_point",
    "jrb_mat_root_action_lanczos_with_diagnostics_point",
    "jrb_mat_root_action_symmetric_with_diagnostics_point",
    "jrb_mat_root_action_spd_with_diagnostics_point",
    "jrb_mat_sign_action_lanczos_with_diagnostics_point",
    "jrb_mat_sign_action_symmetric_with_diagnostics_point",
    "jrb_mat_sign_action_spd_with_diagnostics_point",
    "jrb_mat_sin_action_lanczos_with_diagnostics_point",
    "jrb_mat_sin_action_symmetric_with_diagnostics_point",
    "jrb_mat_sin_action_spd_with_diagnostics_point",
    "jrb_mat_cos_action_lanczos_with_diagnostics_point",
    "jrb_mat_cos_action_symmetric_with_diagnostics_point",
    "jrb_mat_cos_action_spd_with_diagnostics_point",
    "jrb_mat_sinh_action_lanczos_with_diagnostics_point",
    "jrb_mat_sinh_action_symmetric_with_diagnostics_point",
    "jrb_mat_sinh_action_spd_with_diagnostics_point",
    "jrb_mat_cosh_action_lanczos_with_diagnostics_point",
    "jrb_mat_cosh_action_symmetric_with_diagnostics_point",
    "jrb_mat_cosh_action_spd_with_diagnostics_point",
    "jrb_mat_tanh_action_lanczos_with_diagnostics_point",
    "jrb_mat_tanh_action_symmetric_with_diagnostics_point",
    "jrb_mat_tanh_action_spd_with_diagnostics_point",
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
    "jrb_mat_orthogonal_rademacher_probes_like",
    "jrb_mat_orthogonal_normal_probes_like",
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
    "jrb_mat_solve_action_point_jit",
    "jrb_mat_minres_solve_action_point_jit",
    "jrb_mat_inverse_action_point_jit",
    "jrb_mat_minres_inverse_action_point_jit",
    "jrb_mat_logdet_slq_point_jit",
    "jrb_mat_det_slq_point_jit",
    "jrb_mat_log_action_lanczos_point_jit",
    "jrb_mat_sqrt_action_lanczos_point_jit",
    "jrb_mat_sign_action_lanczos_point_jit",
    "jrb_mat_pow_action_lanczos_point_jit",
    "jrb_mat_eigsh_point_jit",
    "jrb_mat_multi_shift_solve_point_jit",
    "jrb_mat_eigsh_block_point_jit",
    "jrb_mat_eigsh_restarted_point_jit",
    "jrb_mat_logm",
    "jrb_mat_sqrtm",
    "jrb_mat_rootm",
    "jrb_mat_signm",
    "jrb_mat_pow_action_lanczos_point",
    "jrb_mat_pow_action_symmetric_point",
    "jrb_mat_pow_action_spd_point",
    "jrb_mat_pow_action_lanczos_basic",
    "jrb_mat_pow_action_lanczos_dense_point",
    "jrb_mat_pow_action_lanczos_with_diagnostics_point",
]
