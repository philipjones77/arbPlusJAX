"""Pure JAX implementations of Krylov subspace iterative solvers.

This module provides JAX-native implementations of iterative linear system solvers
without scipy dependencies, optimized for use with sparse and matrix-free operators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

from . import jax_precision



def _norm(x: jax.Array) -> jax.Array:
    """Compute L2 norm."""
    return jax_precision.safe_norm(x)


def _safe_divide(a: jax.Array, b: jax.Array, eps: float = 1e-30) -> jax.Array:
    """Safe division avoiding division by zero."""
    return jnp.where(jnp.abs(b) > eps, a / b, 0.0)


@partial(jax.jit, static_argnames=("matvec", "maxiter", "restart", "M"))
def gmres(
    matvec,
    b: jax.Array,
    x0: jax.Array | None = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
    M=None
) -> tuple[jax.Array, dict]:
    """GMRES iterative solver (Generalized Minimal Residual).
    
    Solves Ax = b using GMRES with restarts.
    
    Args:
        matvec: Function that computes A @ x
        b: Right-hand side vector
        x0: Initial guess (if None, uses zeros)
        tol: Relative tolerance
        atol: Absolute tolerance
        maxiter: Maximum iterations (if None, uses n)
        restart: Number of iterations before restart
        M: Preconditioner (not yet implemented)
    
    Returns:
        x: Solution vector
        info: Dictionary with convergence information
    """
    n = b.shape[0]
    if maxiter is None:
        maxiter = n
    
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    bnorm = _norm(b)
    tol_threshold = jnp.maximum(jnp.asarray(tol, dtype=jnp.float64) * bnorm, jnp.asarray(atol, dtype=jnp.float64))
    
    def arnoldi_step(carry, _):
        V, H, beta, k = carry
        w = matvec(V[:, k])
        
        # Modified Gram-Schmidt orthogonalization
        def orth_iter(i, state):
            w_curr, h_col = state
            h_ik = jnp.vdot(V[:, i].astype(jax_precision.reduction_dtype(V[:, i])), w_curr.astype(jax_precision.reduction_dtype(w_curr)))
            w_curr = w_curr - h_ik * V[:, i]
            h_col = h_col.at[i].set(h_ik)
            return w_curr, h_col
        
        h_col = jnp.zeros(restart + 1, dtype=w.dtype)
        w, h_col = jax.lax.fori_loop(0, k + 1, orth_iter, (w, h_col))
        
        h_kp1 = _norm(w)
        h_col = h_col.at[k + 1].set(h_kp1)
        v_new = _safe_divide(w, h_kp1)
        
        V = V.at[:, k + 1].set(v_new)
        H = H.at[:, k].set(h_col)
        
        return (V, H, beta, k + 1), None
    
    def gmres_cycle(x_in):
        r_in = b - matvec(x_in)
        beta = _norm(r_in)
        
        # Early exit if residual is small
        converged = beta < tol_threshold
        
        V = jnp.zeros((n, restart + 1), dtype=b.dtype)
        V = V.at[:, 0].set(_safe_divide(r_in, beta))
        H = jnp.zeros((restart + 1, restart), dtype=b.dtype)
        
        (V, H, beta, _), _ = jax.lax.scan(
            arnoldi_step,
            (V, H, beta, 0),
            None,
            length=restart
        )
        
        # Solve least squares problem: min ||beta*e1 - H*y||
        e1 = jnp.zeros(restart + 1, dtype=b.dtype)
        e1 = e1.at[0].set(beta)
        
        # Use QR decomposition for least squares
        Q, R = jnp.linalg.qr(H)
        qt_e1 = Q.T @ e1
        y = jnp.linalg.solve(R[:restart, :restart], qt_e1[:restart])
        
        # Update solution
        x_out = x_in + V[:, :restart] @ y
        
        return jnp.where(converged, x_in, x_out)
    
    # Run GMRES with restarts
    def body_fn(carry):
        x_curr, iter_count = carry
        x_new = gmres_cycle(x_curr)
        r_new = b - matvec(x_new)
        residual = _norm(r_new)
        converged = residual < tol_threshold
        return (x_new, iter_count + restart), (residual, converged)
    
    def cond_fn(carry):
        _, iter_count = carry
        return iter_count < maxiter
    
    num_cycles = (maxiter + restart - 1) // restart
    (x_final, iters), (residuals, _) = jax.lax.scan(
        lambda c, _: body_fn(c),
        (x, 0),
        None,
        length=num_cycles
    )
    
    info = {"residual": residuals[-1], "iterations": iters}
    return x_final, info


@partial(jax.jit, static_argnames=("matvec", "maxiter", "M"))
def cg(
    matvec,
    b: jax.Array,
    x0: jax.Array | None = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M=None
) -> tuple[jax.Array, dict]:
    """Conjugate Gradient iterative solver.
    
    Solves Ax = b where A is symmetric positive definite using CG.
    
    Args:
        matvec: Function that computes A @ x
        b: Right-hand side vector
        x0: Initial guess (if None, uses zeros)
        tol: Relative tolerance
        atol: Absolute tolerance
        maxiter: Maximum iterations (if None, uses n)
        M: Optional left preconditioner solve/apply. Should approximate A^{-1}.
    
    Returns:
        x: Solution vector
        info: Dictionary with convergence information
    """
    n = b.shape[0]
    if maxiter is None:
        maxiter = n
    
    x = jnp.zeros_like(b) if x0 is None else x0
    apply_precond = (lambda v: v) if M is None else M

    r = b - matvec(x)
    z = apply_precond(r)
    p = z
    rzold = jax_precision.safe_vdot_real(r, z)
    
    bnorm = _norm(b)
    tol_threshold = jnp.maximum(tol * bnorm, atol)
    
    def body_fn(carry):
        x_curr, r_curr, z_curr, p_curr, rzold_curr, iter_count = carry
        
        Ap = matvec(p_curr)
        alpha = _safe_divide(rzold_curr, jax_precision.safe_vdot_real(p_curr, Ap))
        x_new = x_curr + alpha * p_curr
        r_new = r_curr - alpha * Ap
        z_new = apply_precond(r_new)
        rznew = jax_precision.safe_vdot_real(r_new, z_new)
        
        beta = _safe_divide(rznew, rzold_curr)
        p_new = z_new + beta * p_curr
        
        residual = _norm(r_new)
        converged = residual < tol_threshold
        
        return (x_new, r_new, z_new, p_new, rznew, iter_count + 1), (residual, converged)
    
    (x_final, _, _, _, _, iters), (residuals, _) = jax.lax.scan(
        lambda c, _: body_fn(c),
        (x, r, z, p, rzold, 0),
        None,
        length=maxiter
    )
    
    info = {"residual": residuals[-1], "iterations": iters}
    return x_final, info


@partial(jax.jit, static_argnames=("matvec", "maxiter", "M"))
def bicgstab(
    matvec,
    b: jax.Array,
    x0: jax.Array | None = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M=None
) -> tuple[jax.Array, dict]:
    """BiCGSTAB iterative solver (Biconjugate Gradient Stabilized).
    
    Solves Ax = b using BiCGSTAB method.
    
    Args:
        matvec: Function that computes A @ x
        b: Right-hand side vector
        x0: Initial guess (if None, uses zeros)
        tol: Relative tolerance
        atol: Absolute tolerance
        maxiter: Maximum iterations (if None, uses 2*n)
        M: Preconditioner (not yet implemented)
    
    Returns:
        x: Solution vector
        info: Dictionary with convergence information
    """
    n = b.shape[0]
    if maxiter is None:
        maxiter = 2 * n
    
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    r_tilde = r
    rho = alpha = omega = 1.0
    v = p = jnp.zeros_like(b)
    
    bnorm = _norm(b)
    tol_threshold = jnp.maximum(tol * bnorm, atol)
    
    def body_fn(carry):
        x_curr, r_curr, r_tilde, p_curr, v_curr, rho_old, alpha_old, omega_old, iter_count = carry
        
        rho_new = jnp.vdot(r_tilde, r_curr)
        beta = _safe_divide(rho_new * alpha_old, rho_old * omega_old)
        
        p_new = r_curr + beta * (p_curr - omega_old * v_curr)
        v_new = matvec(p_new)
        alpha_new = _safe_divide(rho_new, jnp.vdot(r_tilde, v_new))
        
        s = r_curr - alpha_new * v_new
        t = matvec(s)
        omega_new = _safe_divide(jnp.vdot(t, s), jnp.vdot(t, t))
        
        x_new = x_curr + alpha_new * p_new + omega_new * s
        r_new = s - omega_new * t
        
        residual = _norm(r_new)
        converged = residual < tol_threshold
        
        return (x_new, r_new, r_tilde, p_new, v_new, rho_new, alpha_new, omega_new, iter_count + 1), (residual, converged)
    
    (x_final, _, _, _, _, _, _, _, iters), (residuals, _) = jax.lax.scan(
        lambda c, _: body_fn(c),
        (x, r, r_tilde, p, v, rho, alpha, omega, 0),
        None,
        length=maxiter
    )
    
    info = {"residual": residuals[-1], "iterations": iters}
    return x_final, info


__all__ = ["gmres", "cg", "bicgstab"]
