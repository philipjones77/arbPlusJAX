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

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from . import checks
from . import double_interval as di
from . import mat_common

jax.config.update("jax_enable_x64", True)

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
    mid = _jrb_mid_matrix(a)

    def matvec(v: jax.Array) -> jax.Array:
        vv = _jrb_mid_vector(v)
        return jnp.einsum("...ij,...j->...i", mid, vv)

    return matvec


def jrb_mat_dense_operator_adjoint(a: jax.Array):
    """Return the adjoint midpoint matvec closure for a dense interval matrix."""
    mid = _jrb_mid_matrix(a)

    def matvec(v: jax.Array) -> jax.Array:
        vv = _jrb_mid_vector(v)
        return jnp.einsum("...ji,...j->...i", mid, vv)

    return matvec


def jrb_mat_bcoo_operator(a: jsparse.BCOO):
    """Return a matrix-free midpoint matvec closure for a real JAX BCOO matrix."""
    checks.check(len(a.shape) == 2, "jrb_mat.bcoo_operator.rank")

    def matvec(v: jax.Array) -> jax.Array:
        vv = _jrb_operator_vector(v)
        return jnp.asarray(a @ vv, dtype=jnp.float64)

    return matvec


def jrb_mat_bcoo_operator_adjoint(a: jsparse.BCOO):
    """Return the adjoint matrix-free midpoint matvec closure for a real JAX BCOO matrix."""
    checks.check(len(a.shape) == 2, "jrb_mat.bcoo_operator_adjoint.rank")
    at = a.transpose()

    def matvec(v: jax.Array) -> jax.Array:
        vv = _jrb_operator_vector(v)
        return jnp.asarray(at @ vv, dtype=jnp.float64)

    return matvec


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
    """Return a matrix-free midpoint matvec closure for a SciPy CSR matrix via JAX BCOO."""
    bcoo = jsparse.BCOO.from_scipy_sparse(csr)
    return jrb_mat_bcoo_operator(bcoo)


def jrb_mat_bcoo_gershgorin_bounds(a: jsparse.BCOO, *, eps: float = 1e-12) -> tuple[jax.Array, jax.Array]:
    """Return conservative Gershgorin spectral bounds for a real sparse matrix."""
    checks.check(len(a.shape) == 2, "jrb_mat.bcoo_gershgorin_bounds.rank")
    checks.check_equal(a.shape[0], a.shape[1], "jrb_mat.bcoo_gershgorin_bounds.square")
    rows = jnp.asarray(a.indices[:, 0], dtype=jnp.int32)
    cols = jnp.asarray(a.indices[:, 1], dtype=jnp.int32)
    data = jnp.asarray(a.data, dtype=jnp.float64)
    n = int(a.shape[0])
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
    a: jsparse.BCOO,
    *,
    steps: int = 16,
    safety_margin: float = 1.25,
    eps: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """Return a heuristic sparse-SPD spectral interval from Gershgorin plus short Lanczos.

    This estimator is intended for the point-mode Leja path. It is narrower than raw
    Gershgorin in many cases, but it is not a rigorous enclosure certificate.
    """
    checks.check(len(a.shape) == 2, "jrb_mat.bcoo_spectral_bounds_adaptive.rank")
    checks.check_equal(a.shape[0], a.shape[1], "jrb_mat.bcoo_spectral_bounds_adaptive.square")
    g_lower, g_upper = jrb_mat_bcoo_gershgorin_bounds(a, eps=eps)
    n = int(a.shape[0])
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


def jrb_mat_operator_apply_point(matvec, x: jax.Array) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    y = jnp.asarray(matvec(x), dtype=jnp.float64)
    out = _jrb_point_interval(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_operator_apply_basic(matvec, x: jax.Array) -> jax.Array:
    return jrb_mat_operator_apply_point(matvec, x)


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


def _jrb_newton_divided_differences_point(nodes: jax.Array, scalar_fun) -> jax.Array:
    coeffs = jnp.asarray(scalar_fun(nodes), dtype=jnp.float64)
    degree = int(nodes.shape[0])
    for j in range(1, degree):
        numer = coeffs[j:] - coeffs[j - 1:-1]
        denom = nodes[j:] - nodes[: degree - j]
        coeffs = coeffs.at[j:].set(numer / denom)
    return coeffs


def _jrb_funm_action_newton_point(matvec, x: jax.Array, nodes: jax.Array, coeffs: jax.Array) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    basis = _jrb_mid_vector(x)
    acc = coeffs[0] * basis
    for k in range(1, int(coeffs.shape[0])):
        basis = jnp.asarray(matvec(_jrb_point_interval(basis)), dtype=jnp.float64) - nodes[k - 1] * basis
        acc = acc + coeffs[k] * basis
    out = _jrb_point_interval(acc)
    finite = jnp.all(jnp.isfinite(acc), axis=-1)
    return jnp.where(finite[..., None], out, _full_interval_like(out))


def _jrb_funm_action_newton_adaptive_point(
    matvec,
    x: jax.Array,
    nodes: jax.Array,
    coeffs: jax.Array,
    *,
    min_degree: int,
    rtol: float,
    atol: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = jrb_mat_as_interval_vector(x)
    degree = int(coeffs.shape[0])
    if degree <= 0:
        raise ValueError("degree must be > 0")
    min_degree = max(1, min(int(min_degree), degree))
    basis0 = _jrb_mid_vector(x)
    acc0 = coeffs[0] * basis0
    tail0 = jnp.linalg.norm(acc0)
    local_rtol = jnp.asarray(rtol, dtype=jnp.float64)
    local_atol = jnp.asarray(atol, dtype=jnp.float64)

    def body(k, carry):
        basis, acc, used_degree, tail_norm, done = carry

        def compute(_):
            next_basis = jnp.asarray(matvec(_jrb_point_interval(basis)), dtype=jnp.float64) - nodes[k - 1] * basis
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
    lower, upper = spectral_bounds
    lower = jnp.maximum(jnp.asarray(lower, dtype=jnp.float64), jnp.asarray(1e-12, dtype=jnp.float64))
    upper = jnp.maximum(jnp.asarray(upper, dtype=jnp.float64), lower + jnp.asarray(1e-12, dtype=jnp.float64))
    nodes = _jrb_leja_points_interval_point(lower, upper, total_degree, candidate_count=candidate_count)
    coeffs = _jrb_newton_divided_differences_point(nodes, jnp.log)
    if max_degree is not None:
        value, _, _ = _jrb_funm_action_newton_adaptive_point(
            matvec,
            x,
            nodes,
            coeffs,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
        return value
    return _jrb_funm_action_newton_point(matvec, x, nodes, coeffs)


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
    lower, upper = spectral_bounds
    lower = jnp.maximum(jnp.asarray(lower, dtype=jnp.float64), jnp.asarray(1e-12, dtype=jnp.float64))
    upper = jnp.maximum(jnp.asarray(upper, dtype=jnp.float64), lower + jnp.asarray(1e-12, dtype=jnp.float64))
    nodes = _jrb_leja_points_interval_point(lower, upper, total_degree, candidate_count=candidate_count)
    coeffs = _jrb_newton_divided_differences_point(nodes, jnp.log)
    if max_degree is not None:
        value, used_degree, tail_norm = _jrb_funm_action_newton_adaptive_point(
            matvec,
            x,
            nodes,
            coeffs,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
    else:
        value = _jrb_funm_action_newton_point(matvec, x, nodes, coeffs)
        used_degree = jnp.asarray(total_degree, dtype=jnp.int32)
        basis = _jrb_mid_vector(x)
        tail = coeffs[0] * basis
        for k in range(1, total_degree):
            basis = jnp.asarray(matvec(_jrb_point_interval(basis)), dtype=jnp.float64) - nodes[k - 1] * basis
            tail = coeffs[k] * basis
        tail_norm = jnp.linalg.norm(tail)
    diag = JrbMatKrylovDiagnostics(
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
            di.as_interval(sketch_probes).shape[0] + di.as_interval(residual_probes).shape[0],
            dtype=jnp.int32,
        ),
    )
    return value, diag


def jrb_mat_bcoo_logdet_leja_hutchpp_point(
    a: jsparse.BCOO,
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
    a: jsparse.BCOO,
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
    x = jrb_mat_as_interval_vector(x)
    coeffs = jnp.asarray(coefficients, dtype=jnp.float64)
    x_mid = _jrb_mid_vector(x)

    def step(carry, coeff):
        term, acc = carry
        next_acc = acc + coeff * term
        next_term = matvec(_jrb_point_interval(term))
        return (next_term, next_acc), None

    if coeffs.ndim != 1:
        raise ValueError("coefficients must be rank-1")
    init = (x_mid, jnp.zeros_like(x_mid))
    (_, acc), _ = lax.scan(step, init, coeffs)
    out = _jrb_point_interval(acc)
    finite = jnp.all(jnp.isfinite(acc), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_poly_action_basic(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    return jrb_mat_poly_action_point(matvec, x, coefficients)


def jrb_mat_expm_action_point(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    if terms <= 0:
        raise ValueError("terms must be > 0")
    x_mid = _jrb_mid_vector(x)

    def step(carry, k):
        term, acc = carry
        next_term = matvec(_jrb_point_interval(term)) / jnp.asarray(k, dtype=jnp.float64)
        next_acc = acc + next_term
        return (next_term, next_acc), None

    init = (x_mid, x_mid)
    (_, acc), _ = lax.scan(step, init, jnp.arange(1, terms, dtype=jnp.int32))
    out = _jrb_point_interval(acc)
    finite = jnp.all(jnp.isfinite(acc), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_interval_like(out))


def jrb_mat_expm_action_basic(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    return jrb_mat_expm_action_point(matvec, x, terms=terms)


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
        z = jnp.asarray(matvec(_jrb_point_interval(q_curr)), dtype=jnp.float64)
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
    return JrbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(0, dtype=jnp.int32),
        steps=jnp.asarray(steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(basis.shape[0], dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(beta0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(breakdown),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(1, dtype=jnp.int32),
    )


def _jrb_mat_funm_action_lanczos_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    basis, T, beta0 = jrb_mat_lanczos_tridiag_point(matvec, x, steps)
    e1 = jnp.zeros((steps,), dtype=jnp.float64).at[0].set(1.0)
    y = beta0 * (basis.T @ (dense_funm(T) @ e1))
    out = _jrb_point_interval(y)
    finite = jnp.all(jnp.isfinite(y), axis=-1)
    return jnp.where(finite[..., None], out, _full_interval_like(out))


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
    basis, T, beta0 = jrb_mat_lanczos_tridiag_point(matvec, x, steps)
    del basis
    e1 = jnp.zeros((steps,), dtype=jnp.float64).at[0].set(1.0)
    value = (beta0**2) * jnp.vdot(e1, dense_funm(T) @ e1).real
    return jnp.asarray(value, dtype=jnp.float64)


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
    def apply(matrix: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eigh(jnp.asarray(matrix, dtype=jnp.float64))
        return evecs @ jnp.diag(scalar_fun(evals)) @ evecs.T

    return apply


def _jrb_dense_funm_point(a: jax.Array, scalar_fun) -> jax.Array:
    a = jrb_mat_as_interval_matrix(a)
    mid = _jrb_mid_matrix(a)
    vals, vecs = jnp.linalg.eigh(mid)
    out = vecs @ jnp.diag(scalar_fun(vals)) @ vecs.T
    return _jrb_point_interval(out)


def jrb_mat_funm_action_lanczos_dense_point(a: jax.Array, x: jax.Array, dense_funm, steps: int) -> jax.Array:
    return _jrb_mat_funm_action_lanczos_point_base(jrb_mat_dense_operator(a), x, dense_funm, steps)


def jrb_mat_expm_action_lanczos_restarted_point(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    restarts: int,
) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    if restarts <= 0:
        raise ValueError("restarts must be > 0")

    dense_exp = jrb_mat_dense_funm_sym_eigh_point(jnp.exp)

    def scaled_matvec(v):
        return matvec(v) / jnp.asarray(restarts, dtype=jnp.float64)

    def body(y, _):
        next_y = jrb_mat_funm_action_lanczos_point(scaled_matvec, y, dense_exp, steps)
        return next_y, None

    y, _ = lax.scan(body, x, xs=None, length=restarts)
    return y


def jrb_mat_expm_action_lanczos_block_point(
    matvec,
    xs: jax.Array,
    *,
    steps: int,
    restarts: int = 1,
) -> jax.Array:
    xs = di.as_interval(xs)
    return jax.vmap(
        lambda x: jrb_mat_expm_action_lanczos_restarted_point(
            matvec,
            x,
            steps=steps,
            restarts=restarts,
        )
    )(xs)


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
    y = jnp.asarray(matvec(x), dtype=jnp.float64)
    return jnp.vdot(x_mid, y).real


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def jrb_mat_trace_integrand_point(matvec, x: jax.Array) -> jax.Array:
    return _jrb_mat_trace_integrand_point_base(matvec, x)


def _jrb_mat_trace_integrand_point_fwd(matvec, x):
    y = _jrb_mat_trace_integrand_point_base(matvec, x)
    return y, x


def _jrb_mat_trace_integrand_point_bwd(matvec, x, cotangent):
    action = jnp.asarray(matvec(x), dtype=jnp.float64)
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


def jrb_mat_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    shape = (num,) + x.shape
    mids = jax.random.rademacher(key, shape=x.shape[:-1] + (x.shape[-2],), dtype=jnp.float64)
    mids = jax.random.rademacher(key, shape=(num, x.shape[-2]), dtype=jnp.float64)
    return jax.vmap(_jrb_point_interval)(mids)


def jrb_mat_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jrb_mat_as_interval_vector(x)
    mids = jax.random.normal(key, shape=(num, x.shape[-2]), dtype=jnp.float64)
    return jax.vmap(_jrb_point_interval)(mids)


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
jrb_mat_expm_action_basic_jit = jax.jit(jrb_mat_expm_action_basic, static_argnames=("matvec", "terms"))


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
    "jrb_mat_bcoo_operator",
    "jrb_mat_bcoo_operator_adjoint",
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
    "jrb_mat_expm_action_lanczos_restarted_point",
    "jrb_mat_expm_action_lanczos_block_point",
    "jrb_mat_expm_action_lanczos_restarted_with_diagnostics_point",
    "jrb_mat_trace_integrand_point",
    "jrb_mat_funm_trace_integrand_lanczos_point",
    "jrb_mat_trace_estimator_point",
    "jrb_mat_trace_estimator_with_diagnostics_point",
    "jrb_mat_logdet_slq_point",
    "jrb_mat_logdet_slq_with_diagnostics_point",
    "jrb_mat_log_action_leja_point",
    "jrb_mat_log_action_leja_with_diagnostics_point",
    "jrb_mat_hutchpp_trace_point",
    "jrb_mat_logdet_leja_hutchpp_point",
    "jrb_mat_logdet_leja_hutchpp_with_diagnostics_point",
    "jrb_mat_bcoo_logdet_leja_hutchpp_point",
    "jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point",
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
]
