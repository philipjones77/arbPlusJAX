from __future__ import annotations

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import matrix_free_core
from . import sparse_common
from . import jrb_mat as _jrb


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
):
    size = int(size)
    actual_block = int(k if block_size is None else block_size)
    if actual_block < k:
        raise ValueError("block_size must be >= k")
    basis0 = jnp.asarray(_jrb._jrb_eigsh_mid_block(v0, size=size, block_size=actual_block), dtype=jnp.complex128)

    def solve_shifted_block(shift, block):
        return jax.vmap(
            lambda col: _jrb._jrb_shifted_solve_mid(matvec, col, shift=shift, preconditioner=preconditioner),
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
                    lambda y: _jrb._jrb_apply_operator_mid(matvec, _jrb._jrb_point_interval(y)),
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
    return _jrb._jrb_point_interval(jnp.real(vals)), _jrb._jrb_point_interval(jnp.real(vecs))


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
):
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
    diag = _jrb._jrb_eig_diagnostics(
        matvec,
        di.midpoint(vals),
        di.midpoint(vecs),
        algorithm_code=15,
        steps=quadrature_order,
        basis_dim=int(k if block_size is None else block_size),
        method="gmres",
        tol=tol,
    )
    return vals, vecs, diag


def _contour_action(matvec, x, *, center, radius, quadrature_order, preconditioner, tol, atol, maxiter, node_weight_fn):
    return _jrb._jrb_point_interval(
        matrix_free_core.contour_integral_action_point(
            lambda shift, v: _jrb._jrb_shifted_solve_mid(
                matvec, v, shift=shift, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter
            ),
            di.midpoint(_jrb.jrb_mat_as_interval_vector(x)),
            center=center,
            radius=radius,
            quadrature_order=quadrature_order,
            node_weight_fn=node_weight_fn,
        )
    )


def jrb_mat_log_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.log(node) / (2.0j * jnp.pi))


def jrb_mat_sqrt_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.sqrt(node) / (2.0j * jnp.pi))


def jrb_mat_root_action_contour_point(matvec, x: jax.Array, *, degree: int, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    if degree <= 0:
        raise ValueError("degree must be > 0")
    inv_degree = 1.0 / jnp.asarray(degree, dtype=jnp.float64)
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.power(node, inv_degree) / (2.0j * jnp.pi))


def jrb_mat_sign_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: (node / jnp.sqrt(node * node)) / (2.0j * jnp.pi))


def jrb_mat_sin_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.sin(node) / (2.0j * jnp.pi))


def jrb_mat_cos_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.cos(node) / (2.0j * jnp.pi))


def jrb_mat_sinh_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.sinh(node) / (2.0j * jnp.pi))


def jrb_mat_cosh_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.cosh(node) / (2.0j * jnp.pi))


def jrb_mat_tanh_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.tanh(node) / (2.0j * jnp.pi))


def jrb_mat_exp_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.exp(node) / (2.0j * jnp.pi))


def jrb_mat_tan_action_contour_point(matvec, x: jax.Array, *, center, radius, quadrature_order: int = 16, preconditioner=None, tol: float = 1e-8, atol: float = 0.0, maxiter: int | None = None):
    return _contour_action(matvec, x, center=center, radius=radius, quadrature_order=quadrature_order, preconditioner=preconditioner, tol=tol, atol=atol, maxiter=maxiter, node_weight_fn=lambda node: jnp.tan(node) / (2.0j * jnp.pi))
