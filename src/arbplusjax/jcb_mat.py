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
from . import mat_common

jax.config.update("jax_enable_x64", True)

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
    mid = _jcb_mid_matrix(a)

    def matvec(v: jax.Array) -> jax.Array:
        vv = _jcb_mid_vector(v)
        return jnp.einsum("...ij,...j->...i", mid, vv)

    return matvec


def jcb_mat_dense_operator_adjoint(a: jax.Array):
    """Return the adjoint midpoint matvec closure for a dense complex-box matrix."""
    mid = _jcb_mid_matrix(a)

    def matvec(v: jax.Array) -> jax.Array:
        vv = _jcb_mid_vector(v)
        return jnp.einsum("...ji,...j->...i", jnp.conjugate(mid), vv)

    return matvec


def jcb_mat_operator_apply_point(matvec, x: jax.Array) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    y = jnp.asarray(matvec(x), dtype=jnp.complex128)
    out = _jcb_point_box(y)
    finite = jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_operator_apply_basic(matvec, x: jax.Array) -> jax.Array:
    return jcb_mat_operator_apply_point(matvec, x)


def jcb_mat_poly_action_point(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    coeffs = jnp.asarray(coefficients, dtype=jnp.complex128)
    x_mid = _jcb_mid_vector(x)

    def step(carry, coeff):
        term, acc = carry
        next_acc = acc + coeff * term
        next_term = matvec(_jcb_point_box(term))
        return (next_term, next_acc), None

    if coeffs.ndim != 1:
        raise ValueError("coefficients must be rank-1")
    init = (x_mid, jnp.zeros_like(x_mid))
    (_, acc), _ = lax.scan(step, init, coeffs)
    out = _jcb_point_box(acc)
    finite = jnp.all(jnp.isfinite(jnp.real(acc)) & jnp.isfinite(jnp.imag(acc)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_poly_action_basic(matvec, x: jax.Array, coefficients: jax.Array) -> jax.Array:
    return jcb_mat_poly_action_point(matvec, x, coefficients)


def jcb_mat_expm_action_point(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    if terms <= 0:
        raise ValueError("terms must be > 0")
    x_mid = _jcb_mid_vector(x)

    def step(carry, k):
        term, acc = carry
        next_term = matvec(_jcb_point_box(term)) / jnp.asarray(k, dtype=jnp.float64)
        next_acc = acc + next_term
        return (next_term, next_acc), None

    init = (x_mid, x_mid)
    (_, acc), _ = lax.scan(step, init, jnp.arange(1, terms, dtype=jnp.int32))
    out = _jcb_point_box(acc)
    finite = jnp.all(jnp.isfinite(jnp.real(acc)) & jnp.isfinite(jnp.imag(acc)), axis=-1)
    return jnp.where(finite[..., None, None], out, _full_box_like(out))


def jcb_mat_expm_action_basic(matvec, x: jax.Array, terms: int = 16) -> jax.Array:
    return jcb_mat_expm_action_point(matvec, x, terms=terms)


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
        z = jnp.asarray(matvec(_jcb_point_box(q_curr)), dtype=jnp.complex128)
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
    return JcbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(0, dtype=jnp.int32),
        steps=jnp.asarray(steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(basis.shape[0], dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(beta0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(breakdown),
        used_adjoint=jnp.asarray(used_adjoint),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(1, dtype=jnp.int32),
    )


def _jcb_mat_funm_action_arnoldi_point_base(matvec, x: jax.Array, dense_funm, steps: int):
    basis, H, beta0 = jcb_mat_arnoldi_hessenberg_point(matvec, x, steps)
    e1 = jnp.zeros((steps,), dtype=jnp.complex128).at[0].set(1.0 + 0.0j)
    y = beta0 * (basis.T @ (dense_funm(H) @ e1))
    out = _jcb_point_box(y)
    finite = jnp.all(jnp.isfinite(jnp.real(y)) & jnp.isfinite(jnp.imag(y)), axis=-1)
    return jnp.where(finite[..., None], out, _full_box_like(out))


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
    basis, H, beta0 = jcb_mat_arnoldi_hessenberg_point(matvec, x, steps)
    del basis
    e1 = jnp.zeros((steps,), dtype=jnp.complex128).at[0].set(1.0 + 0.0j)
    value = (beta0**2) * jnp.vdot(e1, dense_funm(H) @ e1)
    return jnp.asarray(value, dtype=jnp.complex128)


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
    def apply(matrix: jax.Array) -> jax.Array:
        vals, vecs = jnp.linalg.eig(jnp.asarray(matrix, dtype=jnp.complex128))
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(scalar_fun(vals)) @ inv

    return apply


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


def jcb_mat_expm_action_arnoldi_restarted_point(
    matvec,
    x: jax.Array,
    *,
    steps: int,
    restarts: int,
    adjoint_matvec=None,
) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    if restarts <= 0:
        raise ValueError("restarts must be > 0")

    dense_exp = jcb_mat_dense_funm_general_eig_point(jnp.exp)

    def scaled_matvec(v):
        return matvec(v) / jnp.asarray(restarts, dtype=jnp.float64)

    scaled_adjoint = None
    if adjoint_matvec is not None:
        def scaled_adjoint(v):
            return adjoint_matvec(v) / jnp.asarray(restarts, dtype=jnp.float64)

    def body(y, _):
        next_y = jcb_mat_funm_action_arnoldi_point(scaled_matvec, y, dense_exp, steps, scaled_adjoint)
        return next_y, None

    y, _ = lax.scan(body, x, xs=None, length=restarts)
    return y


def jcb_mat_expm_action_arnoldi_block_point(
    matvec,
    xs: jax.Array,
    *,
    steps: int,
    restarts: int = 1,
    adjoint_matvec=None,
) -> jax.Array:
    xs = acb_core.as_acb_box(xs)
    return jax.vmap(
        lambda x: jcb_mat_expm_action_arnoldi_restarted_point(
            matvec,
            x,
            steps=steps,
            restarts=restarts,
            adjoint_matvec=adjoint_matvec,
        )
    )(xs)


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
    y = jnp.asarray(matvec(x), dtype=jnp.complex128)
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
    action = jnp.asarray(matvec(x), dtype=jnp.complex128)
    adjoint_action = jnp.asarray(adjoint_matvec(x), dtype=jnp.complex128)
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


def jcb_mat_rademacher_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    key_re, key_im = jax.random.split(key)
    re = jax.random.rademacher(key_re, shape=(num, x.shape[-2]), dtype=jnp.float64)
    im = jax.random.rademacher(key_im, shape=(num, x.shape[-2]), dtype=jnp.float64)
    mids = re + 1j * im
    return jax.vmap(_jcb_point_box)(mids)


def jcb_mat_normal_probes_like(x: jax.Array, *, key: jax.Array, num: int) -> jax.Array:
    x = jcb_mat_as_box_vector(x)
    key_re, key_im = jax.random.split(key)
    re = jax.random.normal(key_re, shape=(num, x.shape[-2]), dtype=jnp.float64)
    im = jax.random.normal(key_im, shape=(num, x.shape[-2]), dtype=jnp.float64)
    mids = (re + 1j * im) / jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return jax.vmap(_jcb_point_box)(mids)


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
jcb_mat_expm_action_basic_jit = jax.jit(jcb_mat_expm_action_basic, static_argnames=("matvec", "terms"))


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
    "jcb_mat_funm_action_arnoldi_dense_point",
    "jcb_mat_expm_action_arnoldi_restarted_point",
    "jcb_mat_expm_action_arnoldi_block_point",
    "jcb_mat_expm_action_arnoldi_restarted_with_diagnostics_point",
    "jcb_mat_trace_integrand_point",
    "jcb_mat_funm_trace_integrand_arnoldi_point",
    "jcb_mat_trace_estimator_point",
    "jcb_mat_trace_estimator_with_diagnostics_point",
    "jcb_mat_logdet_slq_point",
    "jcb_mat_logdet_slq_with_diagnostics_point",
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
