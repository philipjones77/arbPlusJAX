"""Pure JAX iterative solvers for sparse linear systems.

This module provides JIT-compatible iterative solvers implemented entirely in JAX,
without scipy dependencies. All implementations use JAX control flow primitives
for optimal performance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from . import jax_precision



def _cg_step(carry, _):
    """Single CG iteration step."""
    x, r, p, rsold, converged = carry
    
    def body(state):
        x, r, p, rsold = state
        Ap = carry[-1]["matvec"](p)  # Get matvec from frozen dict
        alpha = rsold / jnp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = jnp.dot(r, r)
        p = r + (rsnew / rsold) * p
        return x, r, p, rsnew
    
    x, r, p, rsold = lax.cond(
        converged,
        lambda s: s,
        body,
        (x, r, p, rsold)
    )
    
    residual_norm = jnp.sqrt(jnp.dot(r, r))
    converged = converged | (residual_norm < carry[-1]["tol"])
    
    return (x, r, p, jnp.dot(r, r), converged), residual_norm


def cg(matvec, b, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """Conjugate Gradient solver - pure JAX implementation.
    
    Args:
        matvec: Function that computes matrix-vector product
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Relative tolerance
        atol: Absolute tolerance
        maxiter: Maximum iterations (default: size of b)
        M: Optional left preconditioner solve/apply. Should approximate A^{-1}.
        
    Returns:
        Tuple of (solution, info_dict)
    """
    n = b.shape[0]
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = jnp.asarray(x0, dtype=b.dtype)
    
    if maxiter is None:
        maxiter = n
    
    threshold = jnp.maximum(jnp.asarray(tol, dtype=jnp.float64) * jax_precision.safe_norm(b), jnp.asarray(atol, dtype=jnp.float64))
    
    # Initial residual
    apply_precond = (lambda v: v) if M is None else M

    r = b - matvec(x)
    z = apply_precond(r)
    p = z
    rzold = jax_precision.safe_vdot_real(r, z)
    converged = jnp.sqrt(rzold) < threshold
    
    def step_fn(carry, _):
        x, r, z, p, rzold, converged = carry
        
        def body(state):
            x, r, z, p, rzold = state
            Ap = matvec(p)
            pAp = jax_precision.safe_vdot_real(p, Ap)
            alpha = rzold / jnp.maximum(pAp, 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            z = apply_precond(r)
            rznew = jax_precision.safe_vdot_real(r, z)
            beta = rznew / jnp.maximum(rzold, 1e-30)
            p = z + beta * p
            return x, r, z, p, rznew

        x, r, z, p, rzold = lax.cond(
            converged,
            lambda s: s,
            body,
            (x, r, z, p, rzold)
        )
        
        residual_norm = jax_precision.safe_norm(r)
        converged = converged | (residual_norm < threshold)
        
        return (x, r, z, p, rzold, converged), residual_norm

    (x, r, z, p, rzold, converged), residuals = lax.scan(
        step_fn,
        (x, r, z, p, rzold, converged),
        None,
        length=maxiter
    )
    
    info = {"converged": converged, "residuals": residuals, "iterations": maxiter}
    return x, info


def bicgstab(matvec, b, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """BiCGSTAB solver - pure JAX implementation.
    
    Args:
        matvec: Function that computes matrix-vector product
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Relative tolerance
        atol: Absolute tolerance
        maxiter: Maximum iterations (default: size of b)
        M: Preconditioner (not yet supported)
        
    Returns:
        Tuple of (solution, info_dict)
    """
    n = b.shape[0]
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = jnp.asarray(x0, dtype=b.dtype)
    
    if maxiter is None:
        maxiter = n
    
    threshold = jnp.maximum(jnp.asarray(tol, dtype=jnp.float64) * jax_precision.safe_norm(b), jnp.asarray(atol, dtype=jnp.float64))
    
    r = b - matvec(x)
    r_tld = r
    rho = jnp.dot(r_tld, r)
    converged = jax_precision.safe_norm(r) < threshold
    p = r
    
    def step_fn(carry, _):
        x, r, r_tld, p, rho, converged = carry
        
        def body(state):
            x, r, p, rho = state
            v = matvec(p)
            alpha = rho / jnp.maximum(jax_precision.safe_dot(r_tld, v), 1e-30)
            s = r - alpha * v
            t = matvec(s)
            omega = jax_precision.safe_dot(t, s) / jnp.maximum(jax_precision.safe_dot(t, t), 1e-30)
            x = x + alpha * p + omega * s
            r = s - omega * t
            rho_new = jax_precision.safe_dot(r_tld, r)
            beta = (rho_new / jnp.maximum(rho, 1e-30)) * (alpha / jnp.maximum(omega, 1e-30))
            p = r + beta * (p - omega * v)
            return x, r, p, rho_new
        
        x, r, p, rho = lax.cond(
            converged,
            lambda s: (s[0], s[1], s[2], s[3]),
            body,
            (x, r, p, rho)
        )
        
        residual_norm = jax_precision.safe_norm(r)
        converged = converged | (residual_norm < threshold)
        
        return (x, r, r_tld, p, rho, converged), residual_norm
    
    (x, r, r_tld, p, rho, converged), residuals = lax.scan(
        step_fn,
        (x, r, r_tld, p, rho, converged),
        None,
        length=maxiter
    )
    
    info = {"converged": converged, "residuals": residuals, "iterations": maxiter}
    return x, info


def gmres(matvec, b, x0=None, tol=1e-5, atol=0.0, restart=20, maxiter=None, M=None):
    """GMRES solver - pure JAX implementation.
    
    Args:
        matvec: Function that computes matrix-vector product
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Relative tolerance
        atol: Absolute tolerance
        restart: Restart parameter
        maxiter: Maximum iterations (default: size of b)
        M: Preconditioner (not yet supported)
        
    Returns:
        Tuple of (solution, info_dict)
    """
    n = b.shape[0]
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = jnp.asarray(x0, dtype=b.dtype)
    
    if maxiter is None:
        maxiter = n // restart + 1
    
    threshold = jnp.maximum(tol * jnp.linalg.norm(b), atol)
    
    def arnoldi_iteration(A_mv, v0, k):
        """Arnoldi iteration to build Krylov subspace."""
        m = v0.shape[0]
        V = jnp.zeros((m, k + 1), dtype=v0.dtype)
        H = jnp.zeros((k + 1, k), dtype=v0.dtype)
        
        V = V.at[:, 0].set(v0 / jnp.linalg.norm(v0))
        
        def step(j, state):
            V, H = state
            w = A_mv(V[:, j])
            
            # Modified Gram-Schmidt
            def gs_step(i, carry):
                w, H = carry
                h = jnp.dot(V[:, i], w)
                H = H.at[i, j].set(h)
                w = w - h * V[:, i]
                return w, H
            
            w, H = lax.fori_loop(0, j + 1, gs_step, (w, H))
            
            h_norm = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(h_norm)
            V = V.at[:, j + 1].set(w / jnp.maximum(h_norm, 1e-30))
            
            return V, H
        
        V, H = lax.fori_loop(0, k, step, (V, H))
        return V, H
    
    def solve_least_squares(H, beta):
        """Solve least squares problem via QR."""
        Q, R = jnp.linalg.qr(H)
        return jnp.linalg.solve(R, Q.T @ beta)
    
    r = b - matvec(x)
    beta = jnp.linalg.norm(r)
    converged = beta < threshold
    
    def restart_iteration(carry, _):
        x, converged = carry
        
        def body(x_in):
            r = b - matvec(x_in)
            beta = jnp.linalg.norm(r)
            
            V, H = arnoldi_iteration(matvec, r, restart)
            e1 = jnp.zeros(restart + 1, dtype=b.dtype).at[0].set(1.0)
            y = solve_least_squares(H, beta * e1)
            return x_in + V[:, :-1] @ y
        
        x = lax.cond(converged, lambda x: x, body, x)
        
        residual_norm = jnp.linalg.norm(b - matvec(x))
        converged = converged | (residual_norm < threshold)
        
        return (x, converged), residual_norm
    
    (x, converged), residuals = lax.scan(
        restart_iteration,
        (x, converged),
        None,
        length=maxiter
    )
    
    info = {"converged": converged, "residuals": residuals, "iterations": maxiter}
    return x, info


__all__ = ["cg", "bicgstab", "gmres"]
