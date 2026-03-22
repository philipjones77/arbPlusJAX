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
    
    restart = max(int(restart), 1)
    threshold = jnp.maximum(
        jnp.asarray(tol, dtype=jnp.float64) * jax_precision.safe_norm(b),
        jnp.asarray(atol, dtype=jnp.float64),
    )
    apply_precond = (lambda v: v) if M is None else M
    eps = jnp.asarray(1e-30, dtype=jnp.float64)

    def arnoldi_iteration(A_mv, v0, k):
        """Arnoldi iteration to build a Krylov basis for the preconditioned operator."""
        m = v0.shape[0]
        V = jnp.zeros((m, k + 1), dtype=v0.dtype)
        H = jnp.zeros((k + 1, k), dtype=v0.dtype)
        beta0 = jax_precision.safe_norm(v0)
        v_start = jnp.where(beta0 > eps, v0 / jnp.asarray(beta0, dtype=v0.dtype), jnp.zeros_like(v0))
        V = V.at[:, 0].set(v_start)

        def step(j, state):
            V, H = state
            w = A_mv(V[:, j])

            def gs_step(i, carry):
                w_local, H_local = carry
                h = jax_precision.safe_vdot(V[:, i], w_local)
                H_local = H_local.at[i, j].set(h)
                w_local = w_local - h * V[:, i]
                return w_local, H_local

            w, H = lax.fori_loop(0, j + 1, gs_step, (w, H))
            h_norm = jax_precision.safe_norm(w)
            H = H.at[j + 1, j].set(jnp.asarray(h_norm, dtype=H.dtype))
            v_next = jnp.where(h_norm > eps, w / jnp.asarray(h_norm, dtype=w.dtype), jnp.zeros_like(w))
            V = V.at[:, j + 1].set(v_next)
            return V, H

        V, H = lax.fori_loop(0, k, step, (V, H))
        return V, H, beta0

    def solve_least_squares(H, rhs_proj):
        return jnp.linalg.lstsq(H, rhs_proj, rcond=1e-12)[0]

    r = b - matvec(x)
    z = apply_precond(r)
    beta = jax_precision.safe_norm(z)
    converged = beta < threshold
    
    def restart_iteration(carry, _):
        x, converged = carry
        
        def body(x_in):
            r = b - matvec(x_in)
            z = apply_precond(r)
            V, H, beta_local = arnoldi_iteration(lambda v: apply_precond(matvec(v)), z, restart)
            e1 = jnp.zeros(restart + 1, dtype=b.dtype).at[0].set(jnp.asarray(beta_local, dtype=b.dtype))
            y = solve_least_squares(H, e1)
            return x_in + V[:, :-1] @ y
        
        x = lax.cond(converged, lambda x: x, body, x)
        
        residual_norm = jax_precision.safe_norm(b - matvec(x))
        converged = converged | (residual_norm < threshold)
        
        return (x, converged), residual_norm
    
    (x, converged), residuals = lax.scan(
        restart_iteration,
        (x, converged),
        None,
        length=maxiter
    )
    
    info = {"converged": converged, "residuals": residuals, "iterations": maxiter * restart}
    return x, info


def minres(matvec, b, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """MINRES solver for symmetric/Hermitian indefinite systems.

    When `M` is provided, this uses a left-preconditioned operator path
    `M(Ax) = M(b)`. This preserves the fully JAX-native execution model and
    works well for the current diagonal/Jacobi-style Jones preconditioners.
    """
    n = b.shape[0]
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = jnp.asarray(x0, dtype=b.dtype)

    if maxiter is None:
        maxiter = n

    apply_precond = (lambda v: v) if M is None else M

    def effective_matvec(v):
        Av = matvec(v)
        return apply_precond(Av)

    b_eff = apply_precond(b)
    r0 = b_eff - effective_matvec(x)
    beta1 = jax_precision.safe_norm(r0)
    threshold = jnp.maximum(jnp.asarray(tol, dtype=jnp.float64) * jax_precision.safe_norm(b_eff), jnp.asarray(atol, dtype=jnp.float64))
    eps = jnp.asarray(1e-30, dtype=jnp.float64)
    if maxiter is None:
        maxiter = n
    converged0 = beta1 < threshold
    v0 = jnp.where(beta1 > eps, r0 / jnp.asarray(beta1, dtype=b.dtype), jnp.zeros_like(r0))

    def lanczos_step(carry, _):
        v_prev, v, beta_prev = carry
        w = effective_matvec(v) - jnp.asarray(beta_prev, dtype=v.dtype) * v_prev
        alpha = jax_precision.safe_vdot_real(v, w)
        w = w - jnp.asarray(alpha, dtype=v.dtype) * v
        beta_next = jax_precision.safe_norm(w)
        v_next = jnp.where(beta_next > eps, w / jnp.asarray(beta_next, dtype=v.dtype), jnp.zeros_like(w))
        return (v, v_next, beta_next), (v, alpha, beta_next)

    (_, _, _), (basis_rows, alphas, betas) = lax.scan(
        lanczos_step,
        (jnp.zeros_like(r0), v0, jnp.asarray(0.0, dtype=jnp.float64)),
        None,
        length=maxiter,
    )

    basis = jnp.swapaxes(basis_rows, 0, 1)
    dtype = basis.dtype
    Tbar = jnp.zeros((maxiter + 1, maxiter), dtype=dtype)
    Tbar = Tbar.at[jnp.arange(maxiter), jnp.arange(maxiter)].set(alphas.astype(dtype))
    if maxiter > 1:
        off = betas[:-1].astype(dtype)
        Tbar = Tbar.at[jnp.arange(maxiter - 1), jnp.arange(1, maxiter)].set(off)
        Tbar = Tbar.at[jnp.arange(1, maxiter), jnp.arange(maxiter - 1)].set(off)
    Tbar = Tbar.at[maxiter, maxiter - 1].set(jnp.asarray(betas[-1], dtype=dtype))

    rhs_proj = jnp.zeros((maxiter + 1,), dtype=dtype).at[0].set(jnp.asarray(beta1, dtype=dtype))
    y = jnp.linalg.lstsq(Tbar, rhs_proj, rcond=1e-12)[0]
    x = x + basis @ y
    residual_proj = rhs_proj - Tbar @ y
    residual_norm = jax_precision.safe_norm(residual_proj)
    residuals = jnp.full((maxiter,), residual_norm, dtype=jnp.float64)
    info = {"converged": converged0 | (residual_norm < threshold), "residuals": residuals, "iterations": maxiter}
    return x, info


__all__ = ["cg", "bicgstab", "gmres", "minres"]
