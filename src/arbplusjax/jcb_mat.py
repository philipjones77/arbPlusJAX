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
- naming policy: see docs/standards/function_naming_standard.md
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
from . import scb_block_mat
from . import scb_vblock_mat
from . import sparse_common
from . import matrix_free_core


PROVENANCE = {
    "classification": "new",
    "base_names": ("jcb_mat",),
    "module_lineage": "Jones matrix-function subsystem for complex box matrices",
    "naming_policy": "docs/standards/function_naming_standard.md",
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
    if arr.ndim >= 2 and arr.shape[-1] == 4:
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


def _jcb_structure_tag(*, hermitian: bool = False, hpd: bool = False) -> str:
    if hpd:
        return "hpd"
    if hermitian:
        return "hermitian"
    return "general"


def _jcb_attach_diag(diag: JcbMatKrylovDiagnostics, *, regime: str, method: str, structure: str, work_units, primal_residual=0.0, adjoint_residual=0.0, note: str = ""):
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


def _jcb_update_convergence(
    diag: JcbMatKrylovDiagnostics,
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


def _jcb_eig_residuals(matvec, vals: jax.Array, vecs: jax.Array) -> jax.Array:
    if vecs.ndim != 2:
        return jnp.asarray([], dtype=jnp.float64)
    applied = _jcb_apply_operator_block_mid(matvec, vecs)
    residual = applied - vecs * jnp.asarray(vals, dtype=applied.dtype)[None, :]
    return jnp.linalg.norm(residual, axis=0)


def _jcb_eig_diagnostics(
    matvec,
    vals: jax.Array,
    vecs: jax.Array,
    *,
    algorithm_code: int,
    steps,
    basis_dim,
    restart_count=0,
    structure: str = "hermitian",
    method: str = "lanczos",
    tol: float = 1e-3,
):
    residuals = _jcb_eig_residuals(matvec, vals, vecs)
    max_residual = jnp.max(residuals) if residuals.size else jnp.asarray(0.0, dtype=jnp.float64)
    requested = vals.shape[-1] if vals.ndim > 0 else 1
    converged_mask = residuals <= jnp.asarray(tol, dtype=jnp.float64)
    converged_count = jnp.sum(converged_mask.astype(jnp.int32)) if residuals.size else jnp.asarray(0, dtype=jnp.int32)
    diag = matrix_free_core.krylov_diagnostics(
        JcbMatKrylovDiagnostics,
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
    diag = _jcb_attach_diag(
        diag,
        regime="structured",
        method=method,
        structure=structure,
        work_units=steps,
        primal_residual=max_residual,
        note="matrix_free.eigsh",
    )
    return _jcb_update_convergence(
        diag,
        converged=converged_count >= requested,
        convergence_metric=max_residual,
        locked_count=requested,
        residual_history=residuals if residuals.size else jnp.asarray([max_residual], dtype=jnp.float64),
        deflated_count=converged_count,
    )


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


def jcb_mat_dense_parametric_operator_plan_prepare(a: jax.Array):
    return matrix_free_core.parametric_dense_operator_plan(_jcb_mid_matrix(a), orientation="forward", algebra="jcb")


def jcb_mat_shell_operator_plan_prepare(callback, *, context=None):
    return matrix_free_core.shell_operator_plan(callback, context=context, orientation="forward", algebra="jcb")


def jcb_mat_dense_operator_rmatvec_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jcb_mid_matrix(a), orientation="transpose", algebra="jcb")


def jcb_mat_dense_parametric_operator_rmatvec_plan_prepare(a: jax.Array):
    return matrix_free_core.parametric_dense_operator_plan(_jcb_mid_matrix(a), orientation="transpose", algebra="jcb")


def jcb_mat_dense_operator_adjoint_plan_prepare(a: jax.Array):
    return matrix_free_core.dense_operator_plan(_jcb_mid_matrix(a), orientation="adjoint", algebra="jcb")


def jcb_mat_dense_parametric_operator_adjoint_plan_prepare(a: jax.Array):
    return matrix_free_core.parametric_dense_operator_plan(_jcb_mid_matrix(a), orientation="adjoint", algebra="jcb")


def jcb_mat_finite_difference_operator_plan_prepare(
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
        base_point=_jcb_operator_vector(base_point),
        base_value=None if base_value is None else jnp.asarray(base_value, dtype=jnp.complex128),
        context=context,
        algebra="jcb",
        relative_error=relative_error,
        umin=umin,
    )


def jcb_mat_finite_difference_operator_plan_set_base(plan, *, base_point, base_value=None):
    return matrix_free_core.finite_difference_operator_plan_set_base(
        plan,
        base_point=_jcb_operator_vector(base_point),
        base_value=None if base_value is None else jnp.asarray(base_value, dtype=jnp.complex128),
    )


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


def jcb_mat_bcoo_parametric_operator_plan_prepare(indices: jax.Array, data: jax.Array, *, shape: tuple[int, int]):
    return matrix_free_core.parametric_bcoo_operator_plan(indices, data, shape=shape, dtype=jnp.complex128, algebra="jcb")


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


def jcb_mat_block_sparse_operator_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jcb",
        orientation="forward",
        forward_callback=lambda v, mat: scb_block_mat.scb_block_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: scb_block_mat.scb_block_mat_rmatvec(mat, v),
        adjoint_callback=lambda v, mat: scb_block_mat.scb_block_mat_adjoint_matvec(mat, v),
    )


def jcb_mat_block_sparse_operator_rmatvec_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jcb",
        orientation="transpose",
        forward_callback=lambda v, mat: scb_block_mat.scb_block_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: scb_block_mat.scb_block_mat_rmatvec(mat, v),
        adjoint_callback=lambda v, mat: scb_block_mat.scb_block_mat_adjoint_matvec(mat, v),
    )


def jcb_mat_block_sparse_operator_adjoint_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jcb",
        orientation="adjoint",
        forward_callback=lambda v, mat: scb_block_mat.scb_block_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: scb_block_mat.scb_block_mat_rmatvec(mat, v),
        adjoint_callback=lambda v, mat: scb_block_mat.scb_block_mat_adjoint_matvec(mat, v),
    )


def jcb_mat_vblock_sparse_operator_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jcb",
        orientation="forward",
        forward_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_rmatvec(mat, v),
        adjoint_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_adjoint_matvec(mat, v),
    )


def jcb_mat_vblock_sparse_operator_rmatvec_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jcb",
        orientation="transpose",
        forward_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_rmatvec(mat, v),
        adjoint_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_adjoint_matvec(mat, v),
    )


def jcb_mat_vblock_sparse_operator_adjoint_plan_prepare(a):
    return matrix_free_core.oriented_shell_operator_plan(
        context=a,
        algebra="jcb",
        orientation="adjoint",
        forward_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_matvec(mat, v),
        transpose_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_rmatvec(mat, v),
        adjoint_callback=lambda v, mat: scb_vblock_mat.scb_vblock_mat_adjoint_matvec(mat, v),
    )


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


def jcb_mat_jacobi_preconditioner_plan_prepare(a):
    if isinstance(a, matrix_free_core.OperatorPlan):
        if a.kind == "dense":
            return matrix_free_core.dense_jacobi_preconditioner_plan(a.payload, algebra="jcb")
        if a.kind == "sparse_bcoo":
            return matrix_free_core.sparse_bcoo_jacobi_preconditioner_plan(
                a.payload,
                as_sparse_bcoo=sparse_common.as_sparse_bcoo,
                algebra="jcb",
            )
        if a.kind == "finite_difference":
            return matrix_free_core.finite_difference_jacobi_preconditioner_plan(
                a,
                midpoint_vector=acb_core.acb_midpoint,
                sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
                dtype=jnp.complex128,
                algebra="jcb",
            )
        raise ValueError(f"unsupported operator plan kind for Jacobi preconditioner: {a.kind}")
    if isinstance(a, (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)):
        return matrix_free_core.sparse_bcoo_jacobi_preconditioner_plan(
            _jcb_sparse_to_bcoo(a),
            as_sparse_bcoo=sparse_common.as_sparse_bcoo,
            algebra="jcb",
        )
    return matrix_free_core.dense_jacobi_preconditioner_plan(_jcb_mid_matrix(a), algebra="jcb")


def jcb_mat_shell_preconditioner_plan_prepare(callback, *, context=None):
    return matrix_free_core.shell_preconditioner_plan(callback, context=context, orientation="forward", algebra="jcb")


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
    value, _ = matrix_free_basic.solve_action_with_diagnostics_basic(
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
        invalidate_output=_full_box_like,
    )
    return value


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
    x_mid, info, residual, rhs_norm = matrix_free_core.krylov_solve_midpoint(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        solver="cg" if hermitian else "gmres",
        midpoint_vector=_jcb_mid_vector,
        lift_vector=_jcb_point_box,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
    )
    out = _jcb_point_box(x_mid)
    finite = jnp.all(jnp.isfinite(jnp.real(x_mid)) & jnp.isfinite(jnp.imag(x_mid)), axis=-1)
    diag = JcbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(5 if hermitian else 6, dtype=jnp.int32),
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
    diag = _jcb_attach_diag(
        diag,
        regime="structured" if hermitian else "iterative",
        method="cg" if hermitian else "gmres",
        structure=_jcb_structure_tag(hermitian=hermitian),
        work_units=info["iterations"],
        primal_residual=residual,
        adjoint_residual=0.0,
        note="matrix_free.solve_action",
    )
    diag = _jcb_update_convergence(
        diag,
        converged=info["converged"],
        convergence_metric=residual,
    )
    return jnp.where(finite[..., None], out, _full_box_like(out)), diag


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
        invalidate_output=_full_box_like,
    )


def jcb_mat_minres_solve_action_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
) -> jax.Array:
    value, _ = jcb_mat_minres_solve_action_with_diagnostics_point(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
    )
    return value


def jcb_mat_minres_solve_action_with_diagnostics_point(
    matvec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
):
    b = jcb_mat_as_box_vector(b)
    x_mid, info, residual, rhs_norm = matrix_free_core.krylov_solve_midpoint(
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        solver="minres",
        midpoint_vector=_jcb_mid_vector,
        lift_vector=_jcb_point_box,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
    )
    out = _jcb_point_box(x_mid)
    finite = jnp.all(jnp.isfinite(jnp.real(x_mid)) & jnp.isfinite(jnp.imag(x_mid)), axis=-1)
    diag = JcbMatKrylovDiagnostics(
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
    diag = _jcb_attach_diag(
        diag,
        regime="structured",
        method="minres",
        structure="hermitian",
        work_units=info["iterations"],
        primal_residual=residual,
        adjoint_residual=0.0,
        note="matrix_free.minres_solve_action",
    )
    diag = _jcb_update_convergence(
        diag,
        converged=info["converged"],
        convergence_metric=residual,
    )
    return jnp.where(finite[..., None], out, _full_box_like(out)), diag


def jcb_mat_minres_inverse_action_point(
    matvec,
    x: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
) -> jax.Array:
    return jcb_mat_minres_solve_action_point(matvec, x, x0=x0, tol=tol, atol=atol, maxiter=maxiter, preconditioner=preconditioner)


def jcb_mat_minres_solve_action_basic(
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
        jcb_mat_minres_solve_action_with_diagnostics_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_box_like,
    )
    return value


def jcb_mat_minres_solve_action_with_diagnostics_basic(
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
        jcb_mat_minres_solve_action_with_diagnostics_point,
        matvec,
        b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_box_like,
    )


def jcb_mat_minres_inverse_action_basic(
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
        jcb_mat_minres_solve_action_with_diagnostics_point,
        matvec,
        x,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_box_like,
    )
    return value


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
    value, _ = matrix_free_basic.inverse_action_with_diagnostics_basic(
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
        invalidate_output=_full_box_like,
    )
    return value


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
        invalidate_output=_full_box_like,
    )


def jcb_mat_multi_shift_solve_point(
    matvec,
    rhs: jax.Array,
    shifts: jax.Array,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = True,
    preconditioner=None,
) -> jax.Array:
    rhs = jcb_mat_as_box_vector(rhs)
    plan = matrix_free_core.make_shifted_solve_plan(
        matvec,
        shifts,
        preconditioner=preconditioner,
        solver="multi_shift_cg" if hermitian else "multi_shift_gmres",
        algebra="jcb",
        structured=_jcb_structure_tag(hermitian=hermitian),
    )
    mids = matrix_free_core.multi_shift_solve_point(
        plan,
        rhs,
        apply_operator=iterative_solvers,
        midpoint_vector=acb_core.acb_midpoint,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    out = jax.vmap(_jcb_point_box)(mids)
    finite = jnp.all(jnp.isfinite(jnp.real(mids)) & jnp.isfinite(jnp.imag(mids)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_multi_shift_solve_hermitian_point(matvec, rhs: jax.Array, shifts: jax.Array, **kwargs) -> jax.Array:
    return jcb_mat_multi_shift_solve_point(matvec, rhs, shifts, hermitian=True, **kwargs)


def jcb_mat_multi_shift_solve_hpd_point(matvec, rhs: jax.Array, shifts: jax.Array, **kwargs) -> jax.Array:
    return jcb_mat_multi_shift_solve_point(matvec, rhs, shifts, hermitian=True, **kwargs)


def jcb_mat_multi_shift_solve_basic(
    matvec,
    rhs: jax.Array,
    shifts: jax.Array,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jcb_round_basic(
        jcb_mat_multi_shift_solve_point(
            matvec,
            rhs,
            shifts,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            hermitian=hermitian,
            preconditioner=preconditioner,
        ),
        prec_bits,
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


def jcb_mat_rational_action_point(
    matvec,
    x: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    adjoint_matvec=None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
) -> jax.Array:
    x_checked = jcb_mat_as_box_vector(x)
    x_mid = _jcb_mid_vector(x_checked)

    def apply_operator(v_mid: jax.Array) -> jax.Array:
        return acb_core.acb_midpoint(jcb_mat_operator_apply_point(matvec, _jcb_point_box(v_mid)))

    def solve_shifted(shift, v_mid: jax.Array) -> jax.Array:
        shift_arr = jnp.asarray(shift, dtype=jnp.complex128)
        shifted = matrix_free_core.shell_operator_plan(
            lambda y, context: acb_core.acb_midpoint(jcb_mat_operator_apply_point(context["operator"], _jcb_point_box(y)))
            - context["shift"] * jnp.asarray(y, dtype=jnp.complex128),
            context={"operator": matvec, "shift": shift_arr},
            orientation="forward",
            algebra="jcb",
        )
        solved = jcb_mat_solve_action_point(
            shifted,
            _jcb_point_box(v_mid),
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            hermitian=hermitian,
            preconditioner=preconditioner,
        )
        return acb_core.acb_midpoint(solved)

    out_mid = matrix_free_core.rational_spectral_action_midpoint(
        apply_operator,
        solve_shifted,
        x_mid,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        coeff_dtype=jnp.complex128,
    )
    return _jcb_point_box(out_mid)


def jcb_mat_rational_action_basic(
    matvec,
    x: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    adjoint_matvec=None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    del adjoint_matvec
    return matrix_free_basic.action_basic(
        jcb_mat_rational_action_point,
        matvec,
        x,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
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


def _jcb_mat_arnoldi_hessenberg_state_bucketed_point(matvec, x: jax.Array, effective_steps: int, max_steps: int):
    x = jcb_mat_as_box_vector(x)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    v0 = _jcb_mid_vector(x)
    beta0 = jnp.linalg.norm(v0)
    if max_steps > int(v0.shape[-1]):
        raise ValueError("max_steps must be <= vector dimension")
    q0 = v0 / jnp.maximum(beta0, jnp.asarray(1e-30, dtype=jnp.float64))
    dim = q0.shape[-1]
    effective_steps = jnp.clip(jnp.asarray(effective_steps, dtype=jnp.int32), 1, jnp.asarray(max_steps, dtype=jnp.int32))

    def body(carry, _):
        q_curr, basis, H, k = carry

        def do_step(state):
            q_curr, basis, H, k = state
            basis = basis.at[k].set(q_curr)
            z = _jcb_apply_operator_mid(matvec, _jcb_point_box(q_curr))
            mask = (jnp.arange(max_steps + 1, dtype=jnp.int32) < (k + 1)).astype(jnp.complex128)
            h_col = (jnp.conjugate(basis) @ z) * mask
            r = z - h_col @ basis
            beta = jnp.linalg.norm(r)
            q_next = jnp.where(beta > 1e-30, r / beta, jnp.zeros_like(r))
            basis = basis.at[k + 1].set(q_next)
            H = H.at[:, k].set(h_col)
            H = H.at[k + 1, k].set(beta)
            return (q_next, basis, H, k + 1)

        return lax.cond(k < effective_steps, do_step, lambda state: state, carry), None

    init_basis = jnp.zeros((max_steps + 1, dim), dtype=jnp.complex128)
    init_basis = init_basis.at[0].set(q0)
    init_H = jnp.zeros((max_steps + 1, max_steps), dtype=jnp.complex128)
    init = (q0, init_basis, init_H, jnp.asarray(0, dtype=jnp.int32))
    (_, basis, H, _), _ = lax.scan(body, init, xs=None, length=max_steps)
    return basis[:-1], H[:-1, :], beta0, H[-1, max_steps - 1]


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


def _jcb_mat_lanczos_tridiag_state_point(matvec, x: jax.Array, steps: int):
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
        q_prev, q_curr, beta_prev, basis, alphas, betas, k = carry
        z = _jcb_apply_operator_mid(matvec, _jcb_point_box(q_curr))
        alpha = jnp.real(jnp.vdot(q_curr, z))
        r = z - alpha * q_curr - beta_prev * q_prev
        mask = (jnp.arange(steps, dtype=jnp.int32) < k).astype(jnp.complex128)
        proj = (jnp.conjugate(basis) @ r) * mask
        r = r - proj @ basis
        beta = jnp.linalg.norm(r)
        q_next = jnp.where(beta > 1e-30, r / beta, jnp.zeros_like(r))
        basis = basis.at[k].set(q_curr)
        alphas = alphas.at[k].set(alpha)
        betas = betas.at[k].set(beta)
        return (q_curr, q_next, beta, basis, alphas, betas, k + 1), None

    init_basis = jnp.zeros((steps, dim), dtype=jnp.complex128)
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
    offdiag = betas[:-1].astype(jnp.complex128)
    projected = jnp.diag(alphas.astype(jnp.complex128)) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)
    return basis, projected, beta0, betas


def jcb_mat_lanczos_tridiag_point(matvec, x: jax.Array, steps: int):
    basis, projected, beta0, _ = _jcb_mat_lanczos_tridiag_state_point(matvec, x, steps)
    return basis, projected, beta0


def jcb_mat_lanczos_diagnostics_point(matvec, x: jax.Array, steps: int) -> JcbMatKrylovDiagnostics:
    basis, _, beta0, betas = _jcb_mat_lanczos_tridiag_state_point(matvec, x, steps)
    tail_norm = betas[-1]
    breakdown = tail_norm <= jnp.asarray(1e-30, dtype=jnp.float64)
    return matrix_free_core.krylov_diagnostics(
        JcbMatKrylovDiagnostics,
        algorithm_code=0,
        steps=steps,
        basis_dim=basis.shape[0],
        beta0=beta0,
        tail_norm=tail_norm,
        breakdown=breakdown,
        used_adjoint=False,
    )


def _jcb_eigsh_start_vector(size: int) -> jax.Array:
    values = jnp.linspace(1.0, 2.0, int(size), dtype=jnp.float64) + 1j * jnp.linspace(0.25, 0.75, int(size), dtype=jnp.float64)
    return _jcb_point_box(values)


def _jcb_eigsh_select_indices(evals: jax.Array, k: int, which: str) -> jax.Array:
    code = which.lower()
    if code in {"largest", "la", "lm"}:
        return jnp.arange(evals.shape[0] - k, evals.shape[0], dtype=jnp.int32)
    if code in {"smallest", "sa", "sm"}:
        return jnp.arange(0, k, dtype=jnp.int32)
    raise ValueError("which must be one of {'largest', 'smallest', 'la', 'sa', 'lm', 'sm'}")


def _jcb_eigsh_start_block(size: int, block_size: int) -> jax.Array:
    base = jnp.linspace(1.0, 2.0, int(size), dtype=jnp.float64)[:, None]
    offsets = jnp.linspace(0.0, 1.0, int(block_size), dtype=jnp.float64)[None, :]
    imag = 0.25 * (1.0 + offsets)
    return base + 1j * imag


def _jcb_eigsh_mid_block(v0, *, size: int, block_size: int) -> jax.Array:
    if v0 is None:
        return _jcb_eigsh_start_block(size, block_size)
    arr = jnp.asarray(v0)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return acb_core.acb_midpoint(arr)
    return jnp.asarray(arr, dtype=jnp.complex128)


def _jcb_apply_operator_block_mid(matvec, block: jax.Array) -> jax.Array:
    return jax.vmap(lambda col: _jcb_apply_operator_mid(matvec, _jcb_point_box(col)), in_axes=1, out_axes=1)(block)


def _jcb_orthonormalize_columns(block: jax.Array) -> jax.Array:
    return matrix_free_core.orthonormalize_columns(block)


def _jcb_ritz_pairs_from_basis(matvec, basis: jax.Array, *, k: int, which: str) -> tuple[jax.Array, jax.Array]:
    return matrix_free_core.ritz_pairs_from_basis(
        lambda q: _jcb_apply_operator_block_mid(matvec, q),
        basis,
        k=k,
        which=which,
        hermitian=True,
    )


def _jcb_apply_preconditioner_mid(preconditioner, x: jax.Array) -> jax.Array:
    payload = _jcb_operator_vector(x)
    return matrix_free_core.preconditioner_apply_midpoint(
        preconditioner,
        payload,
        midpoint_vector=lambda y: jnp.asarray(y, dtype=jnp.complex128),
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
    )


def _jcb_expand_subspace_with_corrections(
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
        corrections = jax.vmap(lambda col: _jcb_apply_preconditioner_mid(preconditioner, col), in_axes=1, out_axes=1)(corrections)
    if jacobi_davidson:
        coeffs = jnp.sum(jnp.conj(vecs) * corrections, axis=0, keepdims=True)
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
    basis_next = _jcb_orthonormalize_columns(trial)
    if basis_next.shape[1] < target_cols:
        pad = basis[:, : target_cols - basis_next.shape[1]]
        basis_next = _jcb_orthonormalize_columns(jnp.concatenate([basis_next, pad], axis=1))
    return basis_next[:, :target_cols]


def _jcb_restart_basis_from_pairs(
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


def _jcb_shifted_solve_mid(matvec, rhs_mid: jax.Array, *, shift, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None) -> jax.Array:
    shift_arr = jnp.asarray(shift, dtype=jnp.complex128)

    def mv(v):
        return _jcb_apply_operator_mid(matvec, v) - shift_arr * jnp.asarray(v, dtype=jnp.complex128)

    precond = None
    if preconditioner is not None:
        precond = lambda v: _jcb_apply_preconditioner_mid(preconditioner, v)
    out, _ = iterative_solvers.gmres(mv, jnp.asarray(rhs_mid, dtype=jnp.complex128), tol=tol, atol=atol, maxiter=maxiter, M=precond)
    return out


def _jcb_hpd_solve_mid(matvec, rhs_mid: jax.Array, *, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None) -> jax.Array:
    x_mid, _info, _residual, _rhs_norm = matrix_free_core.krylov_solve_midpoint(
        matvec,
        _jcb_point_box(rhs_mid),
        x0=None,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        solver="cg",
        midpoint_vector=_jcb_mid_vector,
        lift_vector=_jcb_point_box,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
    )
    return x_mid


def _jcb_generalized_shifted_solve_mid(
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
        av = _jcb_apply_operator_mid(a_matvec, jnp.asarray(v, dtype=jnp.complex128))
        bv = _jcb_apply_operator_mid(b_matvec, jnp.asarray(v, dtype=jnp.complex128))
        return av - shift_arr * bv

    precond = None
    if preconditioner is not None:
        precond = lambda v: _jcb_apply_preconditioner_mid(preconditioner, v)
    out, _ = iterative_solvers.gmres(mv, jnp.asarray(rhs_mid, dtype=jnp.complex128), tol=tol, atol=atol, maxiter=maxiter, M=precond)
    return out


def _jcb_generalized_eig_residuals(a_matvec, b_matvec, vals: jax.Array, vecs: jax.Array) -> jax.Array:
    if vecs.ndim != 2:
        return jnp.asarray([], dtype=jnp.float64)
    applied_a = _jcb_apply_operator_block_mid(a_matvec, vecs)
    applied_b = _jcb_apply_operator_block_mid(b_matvec, vecs)
    residual = applied_a - applied_b * jnp.asarray(vals, dtype=applied_a.dtype)[None, :]
    return jnp.linalg.norm(residual, axis=0)


def _jcb_generalized_eig_diagnostics(
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
    residuals = _jcb_generalized_eig_residuals(a_matvec, b_matvec, vals, vecs)
    max_residual = jnp.max(residuals) if residuals.size else jnp.asarray(0.0, dtype=jnp.float64)
    requested = vals.shape[-1] if vals.ndim > 0 else 1
    converged_mask = residuals <= jnp.asarray(tol, dtype=jnp.float64)
    converged_count = jnp.sum(converged_mask.astype(jnp.int32)) if residuals.size else jnp.asarray(0, dtype=jnp.int32)
    diag = matrix_free_core.krylov_diagnostics(
        JcbMatKrylovDiagnostics,
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
    diag = _jcb_attach_diag(
        diag,
        regime="structured",
        method="lanczos",
        structure="hpd",
        work_units=steps,
        primal_residual=max_residual,
        note="matrix_free.generalized_eigsh",
    )
    return _jcb_update_convergence(
        diag,
        converged=converged_count >= requested,
        convergence_metric=max_residual,
        locked_count=requested,
        residual_history=residuals if residuals.size else jnp.asarray([max_residual], dtype=jnp.float64),
        deflated_count=converged_count,
    )


def jcb_mat_generalized_operator_plan_prepare(
    a_matvec,
    b_matvec,
    *,
    b_preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
):
    return matrix_free_core.generalized_shell_operator_plan(
        lambda v, context: _jcb_apply_operator_mid(context["a_matvec"], jnp.asarray(v, dtype=jnp.complex128)),
        lambda rhs, context: _jcb_hpd_solve_mid(
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
        algebra="jcb",
    )


def jcb_mat_eigsh_point(
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
    start = _jcb_eigsh_start_vector(size) if v0 is None else jcb_mat_as_box_vector(v0)
    basis, projected, _ = jcb_mat_lanczos_tridiag_point(matvec, start, resolved_steps)
    evals, coeffs = jnp.linalg.eigh(projected)
    indices = _jcb_eigsh_select_indices(evals, k, which)
    selected_vals = evals[indices]
    selected_coeffs = coeffs[:, indices]
    vectors = basis.T @ selected_coeffs
    norms = jnp.maximum(jnp.linalg.norm(vectors, axis=0), jnp.asarray(1e-30, dtype=jnp.float64))
    vectors = vectors / norms[None, :]
    return selected_vals, vectors


def jcb_mat_eigsh_basic(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> tuple[jax.Array, jax.Array]:
    values, vectors = jcb_mat_eigsh_point(matvec, size=size, k=k, which=which, steps=steps, v0=v0)
    return _jcb_round_basic(_jcb_point_box(values), prec_bits), _jcb_round_basic(_jcb_point_box(vectors), prec_bits)


def jcb_mat_eigsh_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_point(matvec, size=size, k=k, which=which, steps=steps, v0=v0)
    resolved_steps = min(size, max(int(k) + 2, 2 * int(k) + 8)) if steps is None else int(steps)
    diag = _jcb_eig_diagnostics(matvec, vals, vecs, algorithm_code=8, steps=resolved_steps, basis_dim=resolved_steps, tol=tol)
    return vals, vecs, diag


def jcb_mat_eigsh_block_point(
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
    basis = _jcb_orthonormalize_columns(_jcb_eigsh_mid_block(v0, size=size, block_size=actual_block))

    def body(q, _):
        return _jcb_orthonormalize_columns(_jcb_apply_operator_block_mid(matvec, q)), None

    basis, _ = lax.scan(body, basis, xs=None, length=int(subspace_iters))
    return _jcb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jcb_mat_eigsh_block_with_diagnostics_point(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    block_size: int | None = None,
    subspace_iters: int = 4,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_block_point(matvec, size=size, k=k, which=which, block_size=block_size, subspace_iters=subspace_iters, v0=v0)
    diag = _jcb_eig_diagnostics(matvec, vals, vecs, algorithm_code=9, steps=subspace_iters, basis_dim=int(k if block_size is None else block_size), tol=tol)
    return vals, vecs, diag


def jcb_mat_eigsh_restarted_point(
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
    basis = _jcb_orthonormalize_columns(_jcb_eigsh_mid_block(v0, size=size, block_size=actual_block))
    restart_tol = matrix_free_core.eig_restart_lock_tolerance(steps=steps, restarts=restarts)
    keep_count = min(actual_block, basis.shape[1])

    def iterate(q):
        def body(q_inner, _):
            return _jcb_orthonormalize_columns(_jcb_apply_operator_block_mid(matvec, q_inner)), None
        q_out, _ = lax.scan(body, q, xs=None, length=int(steps))
        vals, vecs = _jcb_ritz_pairs_from_basis(matvec, q_out, k=keep_count, which=which)
        residuals = _jcb_apply_operator_block_mid(matvec, vecs) - vecs * vals[None, :]
        return _jcb_restart_basis_from_pairs(
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

    return _jcb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jcb_mat_eigsh_restarted_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_restarted_point(
        matvec,
        size=size,
        k=k,
        which=which,
        steps=steps,
        restarts=restarts,
        block_size=block_size,
        v0=v0,
    )
    diag = _jcb_eig_diagnostics(matvec, vals, vecs, algorithm_code=10, steps=steps, basis_dim=int(k if block_size is None else block_size), restart_count=restarts, tol=tol)
    return vals, vecs, diag


def jcb_mat_eigsh_krylov_schur_point(
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
    basis0 = _jcb_eigsh_mid_block(v0, size=size, block_size=actual_block)
    basis = matrix_free_core.restarted_subspace_iteration_point(
        lambda q: _jcb_apply_operator_block_mid(matvec, q),
        basis0,
        subspace_iters=int(steps),
        restarts=int(restarts),
        k=k,
        which=which,
        hermitian=True,
    )
    keep_count = min(actual_block, basis.shape[1])
    vals, vecs = _jcb_ritz_pairs_from_basis(matvec, basis, k=keep_count, which=which)
    residuals = _jcb_apply_operator_block_mid(matvec, vecs) - vecs * vals[None, :]
    basis = _jcb_restart_basis_from_pairs(
        actual_block=actual_block,
        vals=vals,
        vecs=vecs,
        residuals=residuals,
        which=which,
        lock_tol=1e-4,
        refill_basis=basis,
    )
    return _jcb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jcb_mat_eigsh_krylov_schur_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_krylov_schur_point(
        matvec,
        size=size,
        k=k,
        which=which,
        steps=steps,
        restarts=restarts,
        block_size=block_size,
        v0=v0,
    )
    diag = _jcb_eig_diagnostics(matvec, vals, vecs, algorithm_code=11, steps=steps, basis_dim=int(k if block_size is None else block_size), restart_count=restarts, tol=tol)
    return vals, vecs, diag


def jcb_mat_eigsh_davidson_point(
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
    basis = _jcb_orthonormalize_columns(_jcb_eigsh_mid_block(v0, size=size, block_size=actual_block))
    target_cols = min(size, max(actual_block, k))
    lock_tol = max(float(tol), matrix_free_core.eig_restart_lock_tolerance(steps=subspace_iters, restarts=1))

    for _ in range(int(subspace_iters)):
        keep_count = min(actual_block, basis.shape[1])
        vals, vecs = _jcb_ritz_pairs_from_basis(matvec, basis, k=keep_count, which=which)
        applied = _jcb_apply_operator_block_mid(matvec, vecs)
        residuals = applied - vecs * vals[None, :]
        basis_seed = _jcb_restart_basis_from_pairs(
            actual_block=min(actual_block, basis.shape[1]),
            vals=vals,
            vecs=vecs,
            residuals=residuals,
            which=which,
            lock_tol=lock_tol,
            refill_basis=basis,
        )
        target_cols = min(size, basis_seed.shape[1] + min(actual_block, residuals.shape[1]))
        basis = _jcb_expand_subspace_with_corrections(
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
    return _jcb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jcb_mat_eigsh_davidson_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_davidson_point(
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
    diag = _jcb_eig_diagnostics(matvec, vals, vecs, algorithm_code=12, steps=subspace_iters, basis_dim=int(k if block_size is None else block_size), tol=tol)
    return vals, vecs, diag


def jcb_mat_eigsh_jacobi_davidson_point(
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
    basis = _jcb_orthonormalize_columns(_jcb_eigsh_mid_block(v0, size=size, block_size=actual_block))
    target_cols = min(size, max(actual_block, k))
    lock_tol = max(float(tol), matrix_free_core.eig_restart_lock_tolerance(steps=subspace_iters, restarts=1))

    for _ in range(int(subspace_iters)):
        keep_count = min(actual_block, basis.shape[1])
        vals, vecs = _jcb_ritz_pairs_from_basis(matvec, basis, k=keep_count, which=which)
        applied = _jcb_apply_operator_block_mid(matvec, vecs)
        residuals = applied - vecs * vals[None, :]
        basis_seed = _jcb_restart_basis_from_pairs(
            actual_block=min(actual_block, basis.shape[1]),
            vals=vals,
            vecs=vecs,
            residuals=residuals,
            which=which,
            lock_tol=lock_tol,
            refill_basis=basis,
        )
        target_cols = min(size, basis_seed.shape[1] + min(actual_block, residuals.shape[1]))
        basis = _jcb_expand_subspace_with_corrections(
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
    return _jcb_ritz_pairs_from_basis(matvec, basis, k=k, which=which)


def jcb_mat_eigsh_jacobi_davidson_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_jacobi_davidson_point(
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
    diag = _jcb_eig_diagnostics(matvec, vals, vecs, algorithm_code=13, steps=subspace_iters, basis_dim=int(k if block_size is None else block_size), tol=tol)
    return vals, vecs, diag


def jcb_mat_geigsh_point(
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
    plan = jcb_mat_generalized_operator_plan_prepare(
        a_matvec,
        b_matvec,
        b_preconditioner=b_preconditioner,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    return jcb_mat_eigsh_point(plan, size=size, k=k, which=which, steps=steps, v0=v0)


def jcb_mat_geigsh_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_geigsh_point(
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
    diag = _jcb_generalized_eig_diagnostics(
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


def jcb_mat_shift_invert_operator_plan_prepare(
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
            _jcb_shifted_solve_mid(
                context["matvec"],
                jnp.asarray(v, dtype=jnp.complex128),
                shift=context["shift"],
                preconditioner=context["preconditioner"],
                tol=context["tol"],
                atol=context["atol"],
                maxiter=context["maxiter"],
            ),
            dtype=jnp.complex128,
        ),
        context={
            "matvec": matvec,
            "shift": jnp.asarray(shift, dtype=jnp.complex128),
            "preconditioner": preconditioner,
            "tol": float(tol),
            "atol": float(atol),
            "maxiter": maxiter,
        },
        orientation="forward",
        algebra="jcb",
    )


def jcb_mat_generalized_shift_invert_operator_plan_prepare(
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
            _jcb_generalized_shifted_solve_mid(
                context["a_matvec"],
                context["b_matvec"],
                _jcb_apply_operator_mid(context["b_matvec"], jnp.asarray(v, dtype=jnp.complex128)),
                shift=context["shift"],
                preconditioner=context["preconditioner"],
                tol=context["tol"],
                atol=context["atol"],
                maxiter=context["maxiter"],
            ),
            dtype=jnp.complex128,
        ),
        context={
            "a_matvec": a_matvec,
            "b_matvec": b_matvec,
            "shift": jnp.asarray(shift, dtype=jnp.complex128),
            "preconditioner": preconditioner,
            "tol": float(tol),
            "atol": float(atol),
            "maxiter": maxiter,
        },
        orientation="forward",
        algebra="jcb",
    )


def jcb_mat_eigsh_shift_invert_point(
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
    plan = jcb_mat_shift_invert_operator_plan_prepare(matvec, shift=shift, preconditioner=preconditioner)
    vals, vecs = jcb_mat_eigsh_point(plan, size=size, k=k, which=which, steps=steps, v0=v0)
    mids = jnp.asarray(vals, dtype=jnp.complex128)
    mapped = jnp.asarray(shift, dtype=jnp.complex128) + 1.0 / mids
    return _jcb_point_box(mapped), _jcb_point_box(vecs)


def jcb_mat_eigsh_shift_invert_with_diagnostics_point(
    matvec,
    *,
    size: int,
    shift,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    preconditioner=None,
    v0: jax.Array | None = None,
    tol: float = 1e-3,
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_shift_invert_point(
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
    diag = _jcb_eig_diagnostics(matvec, acb_core.acb_midpoint(vals), acb_core.acb_midpoint(vecs), algorithm_code=14, steps=resolved_steps, basis_dim=resolved_steps, method="gmres", tol=tol)
    return vals, vecs, diag


def jcb_mat_geigsh_shift_invert_point(
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
    plan = jcb_mat_generalized_shift_invert_operator_plan_prepare(
        a_matvec,
        b_matvec,
        shift=shift,
        preconditioner=preconditioner,
    )
    vals, vecs = jcb_mat_eigsh_point(plan, size=size, k=k, which=which, steps=steps, v0=v0)
    mids = jnp.asarray(vals, dtype=jnp.complex128)
    mapped = jnp.asarray(shift, dtype=jnp.complex128) + 1.0 / mids
    return _jcb_point_box(mapped), _jcb_point_box(vecs)


def jcb_mat_geigsh_shift_invert_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_geigsh_shift_invert_point(
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
    diag = _jcb_generalized_eig_diagnostics(
        a_matvec,
        b_matvec,
        acb_core.acb_midpoint(vals),
        acb_core.acb_midpoint(vecs),
        algorithm_code=17,
        steps=resolved_steps,
        basis_dim=resolved_steps,
        tol=tol,
    )
    return vals, vecs, diag


def _jcb_nep_scalar_residual(matvec, vec_mid: jax.Array) -> jax.Array:
    applied = _jcb_apply_operator_mid(matvec, jnp.asarray(vec_mid, dtype=jnp.complex128))
    denom = jnp.maximum(jnp.vdot(vec_mid, vec_mid), jnp.asarray(1e-30, dtype=jnp.complex128))
    return jnp.asarray(jnp.real(jnp.vdot(vec_mid, applied) / denom), dtype=jnp.float64)


def _jcb_polynomial_operator_plan_prepare(coeff_matvecs, lam):
    coeffs = tuple(jnp.asarray(lam, dtype=jnp.complex128) ** i for i in range(len(coeff_matvecs)))
    return matrix_free_core.shell_operator_plan(
        lambda v, context: sum(
            context["coeffs"][i] * _jcb_apply_operator_mid(context["ops"][i], jnp.asarray(v, dtype=jnp.complex128))
            for i in range(len(context["ops"]))
        ),
        context={"ops": tuple(coeff_matvecs), "coeffs": coeffs},
        orientation="forward",
        algebra="jcb",
    )


def _jcb_polynomial_derivative_operator_plan_prepare(coeff_matvecs, lam):
    coeffs = tuple(
        jnp.asarray(i, dtype=jnp.complex128) * (jnp.asarray(lam, dtype=jnp.complex128) ** (i - 1))
        for i in range(len(coeff_matvecs))
    )
    return matrix_free_core.shell_operator_plan(
        lambda v, context: sum(
            context["coeffs"][i] * _jcb_apply_operator_mid(context["ops"][i], jnp.asarray(v, dtype=jnp.complex128))
            for i in range(1, len(context["ops"]))
        ),
        context={"ops": tuple(coeff_matvecs), "coeffs": coeffs},
        orientation="forward",
        algebra="jcb",
    )


def jcb_mat_neigsh_point(
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
    lam = jnp.asarray(lambda0, dtype=jnp.complex128)
    vec = _jcb_operator_vector(_jcb_eigsh_start_vector(size) if v0 is None else jcb_mat_as_box_vector(v0))
    for _ in range(int(newton_iters)):
        op = matvec_builder(lam)
        prec = None if preconditioner_builder is None else preconditioner_builder(lam)
        vals, vecs = jcb_mat_eigsh_shift_invert_point(
            op,
            size=size,
            shift=0.0,
            k=1,
            which="largest",
            steps=eig_steps,
            preconditioner=prec,
            v0=_jcb_point_box(vec),
        )
        vec = acb_core.acb_midpoint(vecs)[:, 0]
        residual = acb_core.acb_midpoint(vals)[0]
        if jnp.abs(residual) <= jnp.asarray(tol, dtype=jnp.float64):
            break
        dop = dmatvec_builder(lam)
        derivative = _jcb_nep_scalar_residual(dop, vec)
        safe_derivative = jnp.where(jnp.abs(derivative) > tol, derivative, jnp.asarray(1.0, dtype=jnp.float64))
        lam = lam - residual / safe_derivative
    return _jcb_point_box(jnp.asarray([lam], dtype=jnp.complex128)), _jcb_point_box(vec[:, None])


def jcb_mat_neigsh_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    lam = jnp.asarray(lambda0, dtype=jnp.complex128)
    vec = _jcb_operator_vector(_jcb_eigsh_start_vector(size) if v0 is None else jcb_mat_as_box_vector(v0))
    residual_history = []
    for _ in range(int(newton_iters)):
        op = matvec_builder(lam)
        prec = None if preconditioner_builder is None else preconditioner_builder(lam)
        vals, vecs = jcb_mat_eigsh_shift_invert_point(
            op,
            size=size,
            shift=0.0,
            k=1,
            which="largest",
            steps=eig_steps,
            preconditioner=prec,
            v0=_jcb_point_box(vec),
        )
        vec = acb_core.acb_midpoint(vecs)[:, 0]
        residual = acb_core.acb_midpoint(vals)[0]
        residual_history.append(jnp.abs(residual))
        if jnp.abs(residual) <= jnp.asarray(tol, dtype=jnp.float64):
            break
        dop = dmatvec_builder(lam)
        derivative = _jcb_nep_scalar_residual(dop, vec)
        safe_derivative = jnp.where(jnp.abs(derivative) > tol, derivative, jnp.asarray(1.0, dtype=jnp.float64))
        lam = lam - residual / safe_derivative
    vals_out = _jcb_point_box(jnp.asarray([lam], dtype=jnp.complex128))
    vecs_out = _jcb_point_box(vec[:, None])
    history = jnp.asarray(residual_history if residual_history else [0.0], dtype=jnp.float64)
    diag = matrix_free_core.krylov_diagnostics(
        JcbMatKrylovDiagnostics,
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
    diag = _jcb_attach_diag(
        diag,
        regime="structured",
        method="lanczos",
        structure="hermitian",
        work_units=eig_steps,
        primal_residual=history[-1],
        note="matrix_free.neigsh",
    )
    diag = _jcb_update_convergence(
        diag,
        converged=history[-1] <= jnp.asarray(tol, dtype=jnp.float64),
        convergence_metric=history[-1],
        locked_count=1,
        residual_history=history,
        deflated_count=1 if history[-1] <= jnp.asarray(tol, dtype=jnp.float64) else 0,
    )
    return vals_out, vecs_out, diag


def jcb_mat_peigsh_point(
    coeff_matvecs,
    *,
    size: int,
    lambda0,
    newton_iters: int = 4,
    eig_steps: int = 6,
    v0: jax.Array | None = None,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    return jcb_mat_neigsh_point(
        lambda lam: _jcb_polynomial_operator_plan_prepare(coeff_matvecs, lam),
        lambda lam: _jcb_polynomial_derivative_operator_plan_prepare(coeff_matvecs, lam),
        size=size,
        lambda0=lambda0,
        newton_iters=newton_iters,
        eig_steps=eig_steps,
        v0=v0,
        tol=tol,
    )


def jcb_mat_peigsh_with_diagnostics_point(
    coeff_matvecs,
    *,
    size: int,
    lambda0,
    newton_iters: int = 4,
    eig_steps: int = 6,
    v0: jax.Array | None = None,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs, diag = jcb_mat_neigsh_with_diagnostics_point(
        lambda lam: _jcb_polynomial_operator_plan_prepare(coeff_matvecs, lam),
        lambda lam: _jcb_polynomial_derivative_operator_plan_prepare(coeff_matvecs, lam),
        size=size,
        lambda0=lambda0,
        newton_iters=newton_iters,
        eig_steps=eig_steps,
        v0=v0,
        tol=tol,
    )
    return vals, vecs, diag._replace(algorithm_code=jnp.asarray(19, dtype=jnp.int32))


def jcb_mat_eigsh_contour_point(
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
    basis0 = jnp.asarray(_jcb_eigsh_mid_block(v0, size=size, block_size=actual_block), dtype=jnp.complex128)

    def solve_shifted_block(shift, block):
        return jax.vmap(
            lambda col: _jcb_shifted_solve_mid(matvec, col, shift=shift, preconditioner=preconditioner),
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
        lambda q: _jcb_apply_operator_block_mid(matvec, q),
        filtered,
        k=k,
        which=which,
        hermitian=True,
    )
    return _jcb_point_box(vals), _jcb_point_box(vecs)


def jcb_mat_eigsh_contour_with_diagnostics_point(
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
) -> tuple[jax.Array, jax.Array, JcbMatKrylovDiagnostics]:
    vals, vecs = jcb_mat_eigsh_contour_point(
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
    diag = _jcb_eig_diagnostics(matvec, acb_core.acb_midpoint(vals), acb_core.acb_midpoint(vecs), algorithm_code=15, steps=quadrature_order, basis_dim=int(k if block_size is None else block_size), method="gmres", tol=tol)
    return vals, vecs, diag


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


def _jcb_mat_funm_action_hermitian_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    return matrix_free_core.projected_krylov_action_point(
        matvec,
        x,
        dense_funm,
        steps,
        krylov_decomp=jcb_mat_lanczos_tridiag_point,
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
def jcb_mat_funm_action_hermitian_point(matvec, x: jax.Array, dense_funm, steps: int, adjoint_matvec=None):
    del adjoint_matvec
    return _jcb_mat_funm_action_hermitian_point_base(matvec, x, dense_funm, steps)


def _jcb_mat_funm_action_hermitian_point_fwd(matvec, x, dense_funm, steps, adjoint_matvec):
    del adjoint_matvec
    y = _jcb_mat_funm_action_hermitian_point_base(matvec, x, dense_funm, steps)
    return y, x


def _jcb_mat_funm_action_hermitian_point_bwd(matvec, dense_funm, steps, adjoint_matvec, x, cotangent):
    del adjoint_matvec, x
    adjoint = _jcb_mat_funm_action_hermitian_point_base(
        matvec,
        _jcb_point_box(acb_core.acb_midpoint(cotangent)),
        dense_funm,
        steps,
    )
    return (adjoint,)


jcb_mat_funm_action_hermitian_point.defvjp(
    _jcb_mat_funm_action_hermitian_point_fwd,
    _jcb_mat_funm_action_hermitian_point_bwd,
)


def _jcb_mat_funm_integrand_hermitian_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    return matrix_free_core.projected_krylov_integrand_point(
        matvec,
        x,
        dense_funm,
        steps,
        krylov_decomp=jcb_mat_lanczos_tridiag_point,
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


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3, 4))
def jcb_mat_funm_integrand_hermitian_point(matvec, x: jax.Array, dense_funm, steps: int, adjoint_matvec=None):
    del adjoint_matvec
    return _jcb_mat_funm_integrand_hermitian_point_base(matvec, x, dense_funm, steps)


def _jcb_mat_funm_integrand_hermitian_point_fwd(matvec, x, dense_funm, steps, adjoint_matvec):
    del adjoint_matvec
    y = _jcb_mat_funm_integrand_hermitian_point_base(matvec, x, dense_funm, steps)
    return y, x


def _jcb_mat_funm_integrand_hermitian_point_bwd(matvec, dense_funm, steps, adjoint_matvec, x, cotangent):
    del adjoint_matvec
    action = _jcb_mat_funm_action_hermitian_point_base(matvec, x, dense_funm, steps)
    scale = jnp.asarray(cotangent, dtype=jnp.complex128)
    grad = 2.0 * jnp.real(scale) * acb_core.acb_midpoint(action)
    return (_jcb_point_box(grad),)


jcb_mat_funm_integrand_hermitian_point.defvjp(
    _jcb_mat_funm_integrand_hermitian_point_fwd,
    _jcb_mat_funm_integrand_hermitian_point_bwd,
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


def _jcb_action_basic_from_diagnostics(point_with_diagnostics_fn, matvec, x: jax.Array, *args, prec_bits: int, **kwargs) -> jax.Array:
    value, _ = matrix_free_basic.action_with_diagnostics_basic(
        point_with_diagnostics_fn,
        matvec,
        x,
        *args,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_box_like,
        **kwargs,
    )
    return value


def jcb_mat_log_action_arnoldi_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_log_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_sqrt_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_root_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        degree=degree,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_sign_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_sin_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_cos_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_sinh_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_cosh_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_tanh_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_log_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.log), steps, adjoint_matvec)


def jcb_mat_log_action_hermitian_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_log_action_hermitian_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_sqrt_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.sqrt), steps, adjoint_matvec)


def jcb_mat_sqrt_action_hermitian_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_sqrt_action_hermitian_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_root_action_hermitian_point(matvec, x: jax.Array, *, degree: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return jcb_mat_funm_action_hermitian_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
        adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_root_action_hermitian_with_diagnostics_point,
        matvec,
        x,
        degree=degree,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_sign_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.sign), steps, adjoint_matvec)


def jcb_mat_sign_action_hermitian_basic(
    matvec,
    x: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_sign_action_hermitian_with_diagnostics_point,
        matvec,
        x,
        steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_sin_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.sin), steps, adjoint_matvec)


def jcb_mat_cos_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.cos), steps, adjoint_matvec)


def jcb_mat_sinh_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.sinh), steps, adjoint_matvec)


def jcb_mat_cosh_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.cosh), steps, adjoint_matvec)


def jcb_mat_tanh_action_hermitian_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_funm_action_hermitian_point(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.tanh), steps, adjoint_matvec)


def jcb_mat_pow_action_hermitian_point(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if exponent < 0:
        raise ValueError("exponent must be >= 0")
    return jcb_mat_funm_action_hermitian_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(lambda vals: jnp.power(vals, exponent)),
        steps,
        adjoint_matvec,
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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_pow_action_hermitian_with_diagnostics_point,
        matvec,
        x,
        exponent=exponent,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_log_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_log_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_sqrt_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_sqrt_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_root_action_hpd_point(matvec, x: jax.Array, *, degree: int, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_root_action_hermitian_point(matvec, x, degree=degree, steps=steps, adjoint_matvec=adjoint_matvec)


def jcb_mat_sign_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_sign_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_sin_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_sin_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_cos_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_cos_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_sinh_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_sinh_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_cosh_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_cosh_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_tanh_action_hpd_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_tanh_action_hermitian_point(matvec, x, steps, adjoint_matvec)


def jcb_mat_pow_action_hpd_point(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_pow_action_hermitian_point(matvec, x, exponent=exponent, steps=steps, adjoint_matvec=adjoint_matvec)


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
    return _jcb_action_basic_from_diagnostics(
        jcb_mat_pow_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        exponent=exponent,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
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
    return jcb_mat_funm_integrand_hermitian_point(matvec, x, dense_funm, steps=steps, adjoint_matvec=adjoint_matvec)


def jcb_mat_trace_estimator_point(matvec, probes: jax.Array, adjoint_matvec=None) -> jax.Array:
    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        lambda v: jcb_mat_trace_integrand_point(matvec, v, adjoint_matvec),
        probe_midpoint=acb_core.acb_midpoint,
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
        ),
        probe_midpoint=acb_core.acb_midpoint,
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
        ),
        probe_midpoint=acb_core.acb_midpoint,
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
    del adjoint_matvec
    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        lambda v: _jcb_mat_funm_integrand_arnoldi_point_base(matvec, v, dense_funm, steps),
        probe_midpoint=acb_core.acb_midpoint,
    )


def _jcb_mat_logdet_slq_point_plan_kernel_bucketed(
    matvec,
    probes: jax.Array,
    *,
    effective_steps: int,
    max_steps: int,
    adjoint_matvec=None,
) -> jax.Array:
    dense_funm = jcb_mat_dense_funm_general_eig_point(jnp.log)
    del adjoint_matvec

    def integrand(v):
        basis, projected, beta0 = _jcb_mat_arnoldi_hessenberg_state_bucketed_point(matvec, v, effective_steps, max_steps)[:3]
        del basis
        active = jnp.clip(jnp.asarray(effective_steps, dtype=jnp.int32), 1, jnp.asarray(max_steps, dtype=jnp.int32))
        active_mask = jnp.arange(max_steps, dtype=jnp.int32) < active
        active_matrix = active_mask[:, None] & active_mask[None, :]
        masked_projected = jnp.where(active_matrix, projected, jnp.zeros_like(projected))
        masked_projected = masked_projected + jnp.diag(jnp.where(active_mask, 0.0, 1.0)).astype(projected.dtype)
        e1 = jnp.zeros((max_steps,), dtype=jnp.complex128).at[0].set(jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128))
        value = (beta0**2) * jnp.vdot(e1, dense_funm(masked_projected) @ e1)
        return jnp.asarray(value, dtype=jnp.complex128)

    return mat_common.estimator_mean(
        probes,
        acb_core.as_acb_box,
        integrand,
        probe_midpoint=acb_core.acb_midpoint,
    )


def _jcb_mat_det_slq_point_plan_kernel(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    del adjoint_matvec
    return matrix_free_core.det_from_logdet(_jcb_mat_logdet_slq_point_plan_kernel(matvec, probes, steps))


def _jcb_mat_det_slq_point_plan_kernel_bucketed(
    matvec,
    probes: jax.Array,
    *,
    effective_steps: int,
    max_steps: int,
    adjoint_matvec=None,
) -> jax.Array:
    del adjoint_matvec
    return matrix_free_core.det_from_logdet(
        _jcb_mat_logdet_slq_point_plan_kernel_bucketed(
            matvec,
            probes,
            effective_steps=effective_steps,
            max_steps=max_steps,
        )
    )


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
    value, diag = mat_common.action_with_diagnostics(
        lambda xx: jcb_mat_funm_action_arnoldi_point(matvec, xx, dense_funm, steps, adjoint_matvec),
        lambda xx: jcb_mat_arnoldi_diagnostics_point(matvec, xx, steps, used_adjoint=used_adjoint),
        x,
    )
    diag = _jcb_update_convergence(diag, converged=jnp.isfinite(diag.tail_norm), convergence_metric=diag.tail_norm)
    return value, diag


def jcb_mat_funm_action_hermitian_with_diagnostics_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    del adjoint_matvec
    value, diag = mat_common.action_with_diagnostics(
        lambda xx: jcb_mat_funm_action_hermitian_point(matvec, xx, dense_funm, steps),
        lambda xx: jcb_mat_lanczos_diagnostics_point(matvec, xx, steps),
        x,
    )
    diag = _jcb_update_convergence(diag, converged=jnp.isfinite(diag.tail_norm), convergence_metric=diag.tail_norm)
    return value, diag


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


def jcb_mat_log_action_arnoldi_with_diagnostics_basic(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    adjoint_matvec=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.action_with_diagnostics_basic(
        jcb_mat_log_action_arnoldi_with_diagnostics_point,
        matvec,
        x,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_box_like,
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


def jcb_mat_log_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.log),
        steps,
        adjoint_matvec,
    )


def jcb_mat_log_action_hermitian_with_diagnostics_basic(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    adjoint_matvec=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.action_with_diagnostics_basic(
        jcb_mat_log_action_hermitian_with_diagnostics_point,
        matvec,
        x,
        steps=steps,
        adjoint_matvec=adjoint_matvec,
        round_output=_jcb_round_basic,
        prec_bits=prec_bits,
        invalidate_output=_full_box_like,
    )


def jcb_mat_sqrt_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.sqrt),
        steps,
        adjoint_matvec,
    )


def jcb_mat_root_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, *, degree: int, steps: int, adjoint_matvec=None):
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(lambda vals: jnp.power(vals, inv_degree)),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sign_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.sign),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sin_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.sin),
        steps,
        adjoint_matvec,
    )


def jcb_mat_cos_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.cos),
        steps,
        adjoint_matvec,
    )


def jcb_mat_sinh_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.sinh),
        steps,
        adjoint_matvec,
    )


def jcb_mat_cosh_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.cosh),
        steps,
        adjoint_matvec,
    )


def jcb_mat_tanh_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(jnp.tanh),
        steps,
        adjoint_matvec,
    )


def jcb_mat_pow_action_hermitian_with_diagnostics_point(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None):
    return jcb_mat_funm_action_hermitian_with_diagnostics_point(
        matvec,
        x,
        jcb_mat_dense_funm_hermitian_eigh_point(lambda vals: jnp.power(vals, exponent)),
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
    value, diag = mat_common.estimator_with_diagnostics(
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
    diag = _jcb_attach_diag(
        diag,
        regime="structured" if used_adjoint else "iterative",
        method="arnoldi",
        structure="hermitian" if adjoint_matvec is matvec else "general",
        work_units=steps,
        primal_residual=diag.tail_norm,
        adjoint_residual=0.0,
        note="matrix_free.logdet_slq",
    )
    diag = _jcb_update_convergence(
        diag,
        converged=jnp.isfinite(diag.tail_norm),
        convergence_metric=diag.tail_norm,
    )
    return value, diag


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
        invalidate_output=_full_box_like,
    )


def jcb_mat_det_slq_with_diagnostics_point(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
) -> tuple[jax.Array, JcbMatKrylovDiagnostics]:
    value, diag = jcb_mat_logdet_slq_with_diagnostics_point(matvec, probes, steps, adjoint_matvec)
    diag = _jcb_attach_diag(
        diag,
        regime="structured" if diag.structure_code == matrix_free_core.structure_code("hermitian") else "iterative",
        method="arnoldi",
        structure="hermitian" if diag.structure_code == matrix_free_core.structure_code("hermitian") else "general",
        work_units=steps,
        primal_residual=diag.primal_residual,
        adjoint_residual=diag.adjoint_residual,
        note="matrix_free.det_slq",
    )
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
        invalidate_output=_full_box_like,
    )


def jcb_mat_logdet_solve_point(
    matvec,
    rhs: jax.Array,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
) -> matrix_free_core.LogdetSolveResult:
    return matrix_free_core.combine_logdet_solve_point(
        operator=matvec,
        rhs=rhs,
        probes=probes,
        solve_with_diagnostics=lambda operator, rhs_value: jcb_mat_solve_action_with_diagnostics_point(
            operator,
            rhs_value,
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            hermitian=hermitian,
            preconditioner=preconditioner,
        ),
        logdet_with_diagnostics=lambda operator, probe_value: jcb_mat_logdet_slq_with_diagnostics_point(
            operator,
            probe_value,
            steps,
            adjoint_matvec,
        ),
        preconditioner=preconditioner,
        structured=_jcb_structure_tag(hermitian=hermitian, hpd=hermitian),
        algebra="jcb",
    )


def jcb_mat_logdet_solve_basic(
    matvec,
    rhs: jax.Array,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> matrix_free_core.LogdetSolveResult:
    result = jcb_mat_logdet_solve_point(
        matvec,
        rhs,
        probes,
        steps,
        adjoint_matvec,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
    )
    return matrix_free_core.LogdetSolveResult(
        logdet=_jcb_round_basic(_jcb_point_box(result.logdet), prec_bits),
        solve=_jcb_round_basic(result.solve, prec_bits),
        aux=result.aux,
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


def jcb_mat_orthogonal_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    return matrix_free_core.orthogonal_rademacher_probe_block_complex(_jcb_point_box, x.shape[-2], key=key, num=num)


def jcb_mat_orthogonal_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    return matrix_free_core.orthogonal_normal_probe_block_complex(_jcb_point_box, x.shape[-2], key=key, num=num)


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
_jcb_mat_operator_apply_point_jit_callable = jax.jit(jcb_mat_operator_apply_point, static_argnames=("matvec",))
_jcb_mat_operator_apply_point_jit_plan = jax.jit(jcb_mat_operator_apply_point)


def jcb_mat_operator_apply_point_jit(matvec, x: jax.Array) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_operator_apply_point_jit_plan(matvec, x)
    return _jcb_mat_operator_apply_point_jit_callable(matvec, x)


jcb_mat_rmatvec_point_jit = jax.jit(jcb_mat_rmatvec_point)

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
    static_argnames=("tol", "atol", "maxiter", "hermitian"),
)


def jcb_mat_solve_action_point_jit(matvec, b: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_solve_action_point_jit_plan(matvec, b, **kwargs)
    return _jcb_mat_solve_action_point_jit_callable(matvec, b, **kwargs)


_jcb_mat_minres_solve_action_point_jit_callable = jax.jit(
    jcb_mat_minres_solve_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "preconditioner"),
)
_jcb_mat_minres_solve_action_point_jit_plan = jax.jit(
    jcb_mat_minres_solve_action_point,
    static_argnames=("tol", "atol", "maxiter"),
)


def jcb_mat_minres_solve_action_point_jit(matvec, b: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_minres_solve_action_point_jit_plan(matvec, b, **kwargs)
    return _jcb_mat_minres_solve_action_point_jit_callable(matvec, b, **kwargs)


_jcb_mat_inverse_action_point_jit_callable = jax.jit(
    jcb_mat_inverse_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "hermitian", "preconditioner"),
)
_jcb_mat_inverse_action_point_jit_plan = jax.jit(
    jcb_mat_inverse_action_point,
    static_argnames=("tol", "atol", "maxiter", "hermitian"),
)


def jcb_mat_inverse_action_point_jit(matvec, x: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_inverse_action_point_jit_plan(matvec, x, **kwargs)
    return _jcb_mat_inverse_action_point_jit_callable(matvec, x, **kwargs)


_jcb_mat_minres_inverse_action_point_jit_callable = jax.jit(
    jcb_mat_minres_inverse_action_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "preconditioner"),
)
_jcb_mat_minres_inverse_action_point_jit_plan = jax.jit(
    jcb_mat_minres_inverse_action_point,
    static_argnames=("tol", "atol", "maxiter"),
)


def jcb_mat_minres_inverse_action_point_jit(matvec, x: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_minres_inverse_action_point_jit_plan(matvec, x, **kwargs)
    return _jcb_mat_minres_inverse_action_point_jit_callable(matvec, x, **kwargs)


_jcb_mat_logdet_slq_point_jit_callable = jax.jit(jcb_mat_logdet_slq_point, static_argnames=("matvec", "steps", "adjoint_matvec"))
_jcb_mat_logdet_slq_point_jit_plan = jax.jit(_jcb_mat_logdet_slq_point_plan_kernel, static_argnames=("steps",))
_jcb_mat_logdet_slq_point_jit_plan_bucketed = jax.jit(
    _jcb_mat_logdet_slq_point_plan_kernel_bucketed,
    static_argnames=("max_steps",),
)


def jcb_mat_logdet_slq_point_jit(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_logdet_slq_point_jit_plan(matvec, probes, steps=steps)
    return _jcb_mat_logdet_slq_point_jit_callable(matvec, probes, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_det_slq_point_jit_callable = jax.jit(jcb_mat_det_slq_point, static_argnames=("matvec", "steps", "adjoint_matvec"))
_jcb_mat_det_slq_point_jit_plan = jax.jit(_jcb_mat_det_slq_point_plan_kernel, static_argnames=("steps",))
_jcb_mat_det_slq_point_jit_plan_bucketed = jax.jit(
    _jcb_mat_det_slq_point_plan_kernel_bucketed,
    static_argnames=("max_steps",),
)


def jcb_mat_det_slq_point_jit(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_det_slq_point_jit_plan(matvec, probes, steps=steps)
    return _jcb_mat_det_slq_point_jit_callable(matvec, probes, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_log_action_arnoldi_point_jit_callable = jax.jit(
    jcb_mat_log_action_arnoldi_point,
    static_argnames=("matvec", "steps", "adjoint_matvec"),
)
def _jcb_mat_log_action_arnoldi_point_plan_kernel(matvec, x: jax.Array, *, steps: int, adjoint_matvec=None) -> jax.Array:
    del adjoint_matvec
    return _jcb_mat_funm_action_arnoldi_point_base(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.log), steps)

_jcb_mat_log_action_arnoldi_point_jit_plan = jax.jit(
    _jcb_mat_log_action_arnoldi_point_plan_kernel,
    static_argnames=("steps",),
)


def jcb_mat_log_action_arnoldi_point_jit(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_log_action_arnoldi_point_jit_plan(matvec, x, steps=steps, adjoint_matvec=adjoint_matvec)
    return _jcb_mat_log_action_arnoldi_point_jit_callable(matvec, x, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_log_action_hermitian_point_jit_callable = jax.jit(
    jcb_mat_log_action_hermitian_point,
    static_argnames=("matvec", "steps", "adjoint_matvec"),
)
def _jcb_mat_log_action_hermitian_point_plan_kernel(matvec, x: jax.Array, *, steps: int, adjoint_matvec=None) -> jax.Array:
    del adjoint_matvec
    return _jcb_mat_funm_action_hermitian_point_base(matvec, x, jcb_mat_dense_funm_hermitian_eigh_point(jnp.log), steps)

_jcb_mat_log_action_hermitian_point_jit_plan = jax.jit(
    _jcb_mat_log_action_hermitian_point_plan_kernel,
    static_argnames=("steps",),
)


def jcb_mat_log_action_hermitian_point_jit(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_log_action_hermitian_point_jit_plan(matvec, x, steps=steps, adjoint_matvec=adjoint_matvec)
    return _jcb_mat_log_action_hermitian_point_jit_callable(matvec, x, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_sign_action_arnoldi_point_jit_callable = jax.jit(
    jcb_mat_sign_action_arnoldi_point,
    static_argnames=("matvec", "steps", "adjoint_matvec"),
)
def _jcb_mat_sign_action_arnoldi_point_plan_kernel(matvec, x: jax.Array, *, steps: int, adjoint_matvec=None) -> jax.Array:
    del adjoint_matvec
    return _jcb_mat_funm_action_arnoldi_point_base(matvec, x, jcb_mat_dense_funm_general_eig_point(jnp.sign), steps)

_jcb_mat_sign_action_arnoldi_point_jit_plan = jax.jit(
    _jcb_mat_sign_action_arnoldi_point_plan_kernel,
    static_argnames=("steps",),
)


def jcb_mat_sign_action_arnoldi_point_jit(matvec, x: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_sign_action_arnoldi_point_jit_plan(matvec, x, steps=steps, adjoint_matvec=adjoint_matvec)
    return _jcb_mat_sign_action_arnoldi_point_jit_callable(matvec, x, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_pow_action_arnoldi_point_jit_callable = jax.jit(
    jcb_mat_pow_action_arnoldi_point,
    static_argnames=("matvec", "exponent", "steps", "adjoint_matvec"),
)
def _jcb_mat_pow_action_arnoldi_point_plan_kernel(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None) -> jax.Array:
    del adjoint_matvec
    return _jcb_mat_funm_action_arnoldi_point_base(
        matvec,
        x,
        jcb_mat_dense_funm_general_eig_point(lambda vals: jnp.power(vals, exponent)),
        steps,
    )

_jcb_mat_pow_action_arnoldi_point_jit_plan = jax.jit(
    _jcb_mat_pow_action_arnoldi_point_plan_kernel,
    static_argnames=("exponent", "steps"),
)


def jcb_mat_pow_action_arnoldi_point_jit(matvec, x: jax.Array, *, exponent: int, steps: int, adjoint_matvec=None) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_pow_action_arnoldi_point_jit_plan(matvec, x, exponent=exponent, steps=steps, adjoint_matvec=adjoint_matvec)
    return _jcb_mat_pow_action_arnoldi_point_jit_callable(matvec, x, exponent=exponent, steps=steps, adjoint_matvec=adjoint_matvec)


_jcb_mat_eigsh_point_jit_callable = jax.jit(
    jcb_mat_eigsh_point,
    static_argnames=("matvec", "size", "k", "which", "steps"),
)
_jcb_mat_eigsh_point_jit_plan = jax.jit(
    jcb_mat_eigsh_point,
    static_argnames=("size", "k", "which", "steps"),
)


def jcb_mat_eigsh_point_jit(
    matvec,
    *,
    size: int,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_eigsh_point_jit_plan(matvec, size=size, k=k, which=which, steps=steps, v0=v0)
    return _jcb_mat_eigsh_point_jit_callable(matvec, size=size, k=k, which=which, steps=steps, v0=v0)


_jcb_mat_multi_shift_solve_point_jit_callable = jax.jit(
    jcb_mat_multi_shift_solve_point,
    static_argnames=("matvec", "tol", "atol", "maxiter", "hermitian", "preconditioner"),
)
_jcb_mat_multi_shift_solve_point_jit_plan = jax.jit(
    jcb_mat_multi_shift_solve_point,
    static_argnames=("tol", "atol", "maxiter", "hermitian"),
)


def jcb_mat_multi_shift_solve_point_jit(matvec, rhs: jax.Array, shifts: jax.Array, **kwargs) -> jax.Array:
    if isinstance(matvec, matrix_free_core.OperatorPlan):
        return _jcb_mat_multi_shift_solve_point_jit_plan(matvec, rhs, shifts, **kwargs)
    return _jcb_mat_multi_shift_solve_point_jit_callable(matvec, rhs, shifts, **kwargs)


_jcb_mat_eigsh_block_point_jit_callable = jax.jit(
    jcb_mat_eigsh_block_point,
    static_argnames=("matvec", "size", "k", "which", "block_size", "subspace_iters"),
)
_jcb_mat_eigsh_block_point_jit_plan = jax.jit(
    jcb_mat_eigsh_block_point,
    static_argnames=("size", "k", "which", "block_size", "subspace_iters"),
)


def jcb_mat_eigsh_block_point_jit(
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
        return _jcb_mat_eigsh_block_point_jit_plan(matvec, size=size, k=k, which=which, block_size=block_size, subspace_iters=subspace_iters, v0=v0)
    return _jcb_mat_eigsh_block_point_jit_callable(matvec, size=size, k=k, which=which, block_size=block_size, subspace_iters=subspace_iters, v0=v0)


_jcb_mat_eigsh_restarted_point_jit_callable = jax.jit(
    jcb_mat_eigsh_restarted_point,
    static_argnames=("matvec", "size", "k", "which", "steps", "restarts", "block_size"),
)
_jcb_mat_eigsh_restarted_point_jit_plan = jax.jit(
    jcb_mat_eigsh_restarted_point,
    static_argnames=("size", "k", "which", "steps", "restarts", "block_size"),
)


def jcb_mat_eigsh_restarted_point_jit(
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
        return _jcb_mat_eigsh_restarted_point_jit_plan(matvec, size=size, k=k, which=which, steps=steps, restarts=restarts, block_size=block_size, v0=v0)
    return _jcb_mat_eigsh_restarted_point_jit_callable(matvec, size=size, k=k, which=which, steps=steps, restarts=restarts, block_size=block_size, v0=v0)


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
    "jcb_mat_dense_parametric_operator_plan_prepare",
    "jcb_mat_shell_operator_plan_prepare",
    "jcb_mat_dense_operator_rmatvec_plan_prepare",
    "jcb_mat_dense_parametric_operator_rmatvec_plan_prepare",
    "jcb_mat_dense_operator_adjoint_plan_prepare",
    "jcb_mat_dense_parametric_operator_adjoint_plan_prepare",
    "jcb_mat_finite_difference_operator_plan_prepare",
    "jcb_mat_finite_difference_operator_plan_set_base",
    "jcb_mat_bcoo_operator",
    "jcb_mat_bcoo_parametric_operator_plan_prepare",
    "jcb_mat_bcoo_operator_adjoint",
    "jcb_mat_bcoo_operator_rmatvec",
    "jcb_mat_bcoo_operator_plan_prepare",
    "jcb_mat_bcoo_operator_rmatvec_plan_prepare",
    "jcb_mat_bcoo_operator_adjoint_plan_prepare",
    "jcb_mat_block_sparse_operator_plan_prepare",
    "jcb_mat_block_sparse_operator_rmatvec_plan_prepare",
    "jcb_mat_block_sparse_operator_adjoint_plan_prepare",
    "jcb_mat_vblock_sparse_operator_plan_prepare",
    "jcb_mat_vblock_sparse_operator_rmatvec_plan_prepare",
    "jcb_mat_vblock_sparse_operator_adjoint_plan_prepare",
    "jcb_mat_operator_plan_apply",
    "jcb_mat_operator_apply_point_jit",
    "jcb_mat_rmatvec_point",
    "jcb_mat_rmatvec_basic",
    "jcb_mat_rmatvec_point_jit",
    "jcb_mat_arnoldi_hessenberg_adjoint",
    "jcb_mat_cg_fixed_iterations",
    "jcb_mat_jacobi_preconditioner_plan_prepare",
    "jcb_mat_shell_preconditioner_plan_prepare",
    "jcb_mat_solve_action_point",
    "jcb_mat_solve_action_basic",
    "jcb_mat_solve_action_hermitian_point",
    "jcb_mat_solve_action_hpd_point",
    "jcb_mat_minres_solve_action_point",
    "jcb_mat_solve_action_with_diagnostics_point",
    "jcb_mat_solve_action_with_diagnostics_basic",
    "jcb_mat_minres_solve_action_basic",
    "jcb_mat_minres_solve_action_with_diagnostics_point",
    "jcb_mat_minres_solve_action_with_diagnostics_basic",
    "jcb_mat_inverse_action_point",
    "jcb_mat_inverse_action_basic",
    "jcb_mat_inverse_action_hermitian_point",
    "jcb_mat_inverse_action_hpd_point",
    "jcb_mat_minres_inverse_action_point",
    "jcb_mat_minres_inverse_action_basic",
    "jcb_mat_inverse_action_with_diagnostics_point",
    "jcb_mat_inverse_action_with_diagnostics_basic",
    "jcb_mat_multi_shift_solve_point",
    "jcb_mat_multi_shift_solve_hermitian_point",
    "jcb_mat_multi_shift_solve_hpd_point",
    "jcb_mat_multi_shift_solve_basic",
    "jcb_mat_operator_apply_point",
    "jcb_mat_operator_apply_basic",
    "jcb_mat_poly_action_point",
    "jcb_mat_poly_action_basic",
    "jcb_mat_rational_action_point",
    "jcb_mat_rational_action_basic",
    "jcb_mat_expm_action_point",
    "jcb_mat_expm_action_basic",
    "jcb_mat_arnoldi_hessenberg_point",
    "jcb_mat_arnoldi_diagnostics_point",
    "jcb_mat_lanczos_tridiag_point",
    "jcb_mat_lanczos_diagnostics_point",
    "jcb_mat_eigsh_point",
    "jcb_mat_eigsh_with_diagnostics_point",
    "jcb_mat_eigsh_basic",
    "jcb_mat_eigsh_block_point",
    "jcb_mat_eigsh_block_with_diagnostics_point",
    "jcb_mat_eigsh_restarted_point",
    "jcb_mat_eigsh_restarted_with_diagnostics_point",
    "jcb_mat_eigsh_krylov_schur_point",
    "jcb_mat_eigsh_krylov_schur_with_diagnostics_point",
    "jcb_mat_eigsh_davidson_point",
    "jcb_mat_eigsh_davidson_with_diagnostics_point",
    "jcb_mat_eigsh_jacobi_davidson_point",
    "jcb_mat_eigsh_jacobi_davidson_with_diagnostics_point",
    "jcb_mat_generalized_operator_plan_prepare",
    "jcb_mat_geigsh_point",
    "jcb_mat_geigsh_with_diagnostics_point",
    "jcb_mat_generalized_shift_invert_operator_plan_prepare",
    "jcb_mat_geigsh_shift_invert_point",
    "jcb_mat_geigsh_shift_invert_with_diagnostics_point",
    "jcb_mat_neigsh_point",
    "jcb_mat_neigsh_with_diagnostics_point",
    "jcb_mat_peigsh_point",
    "jcb_mat_peigsh_with_diagnostics_point",
    "jcb_mat_shift_invert_operator_plan_prepare",
    "jcb_mat_eigsh_shift_invert_point",
    "jcb_mat_eigsh_shift_invert_with_diagnostics_point",
    "jcb_mat_eigsh_contour_point",
    "jcb_mat_eigsh_contour_with_diagnostics_point",
    "jcb_mat_funm_action_arnoldi_point",
    "jcb_mat_funm_action_hermitian_point",
    "jcb_mat_funm_action_arnoldi_with_diagnostics_point",
    "jcb_mat_funm_action_hermitian_with_diagnostics_point",
    "jcb_mat_funm_integrand_arnoldi_point",
    "jcb_mat_funm_integrand_hermitian_point",
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
    "jcb_mat_sin_action_hermitian_point",
    "jcb_mat_cos_action_hermitian_point",
    "jcb_mat_sinh_action_hermitian_point",
    "jcb_mat_cosh_action_hermitian_point",
    "jcb_mat_tanh_action_hermitian_point",
    "jcb_mat_pow_action_hermitian_point",
    "jcb_mat_pow_action_hermitian_basic",
    "jcb_mat_log_action_hpd_point",
    "jcb_mat_sqrt_action_hpd_point",
    "jcb_mat_root_action_hpd_point",
    "jcb_mat_sign_action_hpd_point",
    "jcb_mat_sin_action_hpd_point",
    "jcb_mat_cos_action_hpd_point",
    "jcb_mat_sinh_action_hpd_point",
    "jcb_mat_cosh_action_hpd_point",
    "jcb_mat_tanh_action_hpd_point",
    "jcb_mat_pow_action_hpd_point",
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
    "jcb_mat_logdet_solve_point",
    "jcb_mat_logdet_solve_basic",
    "jcb_mat_logdet_leja_hutchpp_point",
    "jcb_mat_logdet_leja_hutchpp_with_diagnostics_point",
    "jcb_mat_det_leja_hutchpp_point",
    "jcb_mat_det_leja_hutchpp_with_diagnostics_point",
    "jcb_mat_bcoo_gershgorin_bounds",
    "jcb_mat_bcoo_logdet_leja_hutchpp_point",
    "jcb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point",
    "jcb_mat_bcoo_det_leja_hutchpp_point",
    "jcb_mat_log_action_arnoldi_with_diagnostics_point",
    "jcb_mat_log_action_arnoldi_with_diagnostics_basic",
    "jcb_mat_sqrt_action_arnoldi_with_diagnostics_point",
    "jcb_mat_root_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sign_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sin_action_arnoldi_with_diagnostics_point",
    "jcb_mat_cos_action_arnoldi_with_diagnostics_point",
    "jcb_mat_sinh_action_arnoldi_with_diagnostics_point",
    "jcb_mat_cosh_action_arnoldi_with_diagnostics_point",
    "jcb_mat_tanh_action_arnoldi_with_diagnostics_point",
    "jcb_mat_pow_action_arnoldi_with_diagnostics_point",
    "jcb_mat_log_action_hermitian_with_diagnostics_point",
    "jcb_mat_log_action_hermitian_with_diagnostics_basic",
    "jcb_mat_sqrt_action_hermitian_with_diagnostics_point",
    "jcb_mat_root_action_hermitian_with_diagnostics_point",
    "jcb_mat_sign_action_hermitian_with_diagnostics_point",
    "jcb_mat_sin_action_hermitian_with_diagnostics_point",
    "jcb_mat_cos_action_hermitian_with_diagnostics_point",
    "jcb_mat_sinh_action_hermitian_with_diagnostics_point",
    "jcb_mat_cosh_action_hermitian_with_diagnostics_point",
    "jcb_mat_tanh_action_hermitian_with_diagnostics_point",
    "jcb_mat_pow_action_hermitian_with_diagnostics_point",
    "jcb_mat_log_action_leja_point",
    "jcb_mat_log_action_leja_with_diagnostics_point",
    "jcb_mat_hutchpp_trace_point",
    "jcb_mat_rademacher_probes_like",
    "jcb_mat_normal_probes_like",
    "jcb_mat_orthogonal_rademacher_probes_like",
    "jcb_mat_orthogonal_normal_probes_like",
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
    "jcb_mat_solve_action_point_jit",
    "jcb_mat_minres_solve_action_point_jit",
    "jcb_mat_inverse_action_point_jit",
    "jcb_mat_minres_inverse_action_point_jit",
    "jcb_mat_logdet_slq_point_jit",
    "jcb_mat_det_slq_point_jit",
    "jcb_mat_log_action_arnoldi_point_jit",
    "jcb_mat_log_action_hermitian_point_jit",
    "jcb_mat_sign_action_arnoldi_point_jit",
    "jcb_mat_pow_action_arnoldi_point_jit",
    "jcb_mat_eigsh_point_jit",
    "jcb_mat_multi_shift_solve_point_jit",
    "jcb_mat_eigsh_block_point_jit",
    "jcb_mat_eigsh_restarted_point_jit",
    "jcb_mat_logm",
    "jcb_mat_sqrtm",
    "jcb_mat_rootm",
    "jcb_mat_signm",
]
