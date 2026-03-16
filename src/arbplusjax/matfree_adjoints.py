"""Matrix-free adjoints for efficient backward differentiation.

This module implements efficient custom VJPs for Lanczos and Arnoldi iterations
based on the paper:

    Gradients of functions of large matrices.
    Nicholas Krämer, Pablo Moreno-Muñoz, Hrittik Roy, Søren Hauberg.
    2024. arXiv:2405.17277.
    https://arxiv.org/abs/2405.17277

The key insight is to reuse the Krylov decomposition from the forward pass
to compute gradients efficiently, rather than recomputing the entire
decomposition in the backward pass.

Features:
- Efficient Lanczos tridiagonalization with custom VJP (symmetric matrices)
- Efficient Arnoldi Hessenberg decomposition with custom VJP (general matrices)
- Reorthogonalization options: "none" or "full"
- Hutchinson trace estimator with custom VJP for variance reduction
- Integration with JAX's custom_linear_solve for iterative solvers

Provenance:
- classification: new
- base_names: matfree_adjoints
- module lineage: Matrix-free methods with efficient adjoints
"""

from __future__ import annotations

import functools
import warnings
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp


# ==============================================================================
# Lanczos Tridiagonalization with Efficient Adjoints
# ==============================================================================


def lanczos_tridiag(
    matvec: Callable,
    krylov_depth: int,
    /,
    *,
    reortho: str = "full",
    custom_vjp: bool = True,
) -> Callable:
    """Lanczos tridiagonalization with efficient custom VJP.
    
    For symmetric matrices A, computes the decomposition:
        A V = V T + beta * v_{k+1} e_k^T
    
    where V is orthonormal, T is tridiagonal.
    
    Args:
        matvec: Function (v, *params) -> A @ v
        krylov_depth: Number of Lanczos iterations
        reortho: Reorthogonalization strategy ("none" or "full")
        custom_vjp: Whether to use efficient custom VJP
        
    Returns:
        Function (vec, *params) -> ((basis, (diags, offdiags)), (remainder_vec, remainder_norm))
    """
    if reortho not in ["none", "full"]:
        msg = f"reortho={reortho} unsupported. Choose either 'none' or 'full'."
        raise ValueError(msg)

    def estimate(vec, *params):
        *values, _ = _lanczos_forward(matvec, krylov_depth, vec, *params, reortho=reortho)
        return values

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (jnp.linalg.norm(vec), *params))

    def estimate_bwd(cache, vjp_incoming):
        # Read incoming gradients and stack related quantities
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))

        # Read the cache and stack related quantities
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector_norm, *params) = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))

        # Compute the adjoints, discard the adjoint states, and return the gradients
        grads, _lambdas_and_mus_and_nus = _lanczos_adjoint(
            matvec=matvec,
            params=params,
            initvec_norm=vector_norm,
            alphas=alphas,
            betas=betas,
            xs=xs,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
            reortho=reortho,
        )
        return grads

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


def _lanczos_forward(matvec, krylov_depth, vec, *params, reortho: str):
    """Lanczos forward pass with optional reorthogonalization."""
    # Pre-allocate
    vectors = jnp.zeros((krylov_depth + 1, len(vec)), dtype=vec.dtype)
    offdiags = jnp.zeros((krylov_depth,), dtype=vec.dtype)
    diags = jnp.zeros((krylov_depth,), dtype=vec.dtype)

    # Normalize (not all Lanczos implementations do that)
    v0 = vec / jnp.linalg.norm(vec)
    vectors = vectors.at[0].set(v0)

    # Lanczos initialisation
    ((v1, offdiag), diag) = _lanczos_init(matvec, v0, *params)

    # Store results
    k = 0
    vectors = vectors.at[k + 1].set(v1)
    offdiags = offdiags.at[k].set(offdiag)
    diags = diags.at[k].set(diag)

    # Run Lanczos-loop
    if reortho == "full":
        step_fun = functools.partial(_lanczos_step_reortho, matvec, params)
    else:
        step_fun = functools.partial(_lanczos_step, matvec, params)
    
    init = (v1, offdiag, v0), (vectors, diags, offdiags)
    _, (vectors, diags, offdiags) = jax.lax.fori_loop(
        lower=1, upper=krylov_depth, body_fun=step_fun, init_val=init
    )

    # Reorganise the outputs
    decomposition = vectors[:-1], (diags, offdiags[:-1])
    remainder = vectors[-1], offdiags[-1]
    return decomposition, remainder, 1 / jnp.linalg.norm(vec)


def _lanczos_init(matvec, vec, *params):
    """Initialize Lanczos algorithm.
    
    Solve A x_{k} = a_k x_k + b_k x_{k+1}
    for x_{k+1}, a_k, and b_k, using orthogonality of the x_k.
    """
    a = vec @ (matvec(vec, *params))
    r = (matvec(vec, *params)) - a * vec
    b = jnp.linalg.norm(r)
    x = r / b
    return (x, b), a


def _lanczos_step(matvec, params, i, val):
    """Lanczos step without reorthogonalization."""
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _lanczos_step_apply(matvec, v1, offdiag, v0, *params), v1

    # Store results
    vectors = vectors.at[i + 1].set(v1)
    offdiags = offdiags.at[i].set(offdiag)
    diags = diags.at[i].set(diag)

    return (v1, offdiag, v0), (vectors, diags, offdiags)


def _lanczos_step_reortho(matvec, params, i, val):
    """Lanczos step with full reorthogonalization."""
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _lanczos_step_apply(matvec, v1, offdiag, v0, *params), v1
    
    # Full reorthogonalization
    v1 = v1 - vectors[:i+1].T @ (vectors[:i+1] @ v1)

    # Store results
    vectors = vectors.at[i + 1].set(v1)
    offdiags = offdiags.at[i].set(offdiag)
    diags = diags.at[i].set(diag)

    return (v1, offdiag, v0), (vectors, diags, offdiags)


def _lanczos_step_apply(matvec, vec, b, vec_previous, *params):
    """Apply Lanczos recurrence.
    
    Solve A x_{k} = b_{k-1} x_{k-1} + a_k x_k + b_k x_{k+1}
    for x_{k+1}, a_k, and b_k, using orthogonality of the x_k.
    """
    a = vec @ (matvec(vec, *params))
    r = matvec(vec, *params) - a * vec - b * vec_previous
    b = jnp.linalg.norm(r)
    x = r / b
    return (x, b), a


def _lanczos_adjoint(
    *,
    matvec,
    params,
    initvec_norm,
    alphas,
    betas,
    xs,
    dalphas,
    dbetas,
    dxs,
    reortho: str,
):
    """Compute the adjoint (backward pass) for Lanczos tridiagonalization.
    
    This efficiently reuses the forward pass basis rather than recomputing.
    """
    def adjoint_step(xi_and_lambda, inputs):
        return _lanczos_adjoint_step(
            *xi_and_lambda, matvec=matvec, params=params, reortho=reortho, **inputs
        )

    # Scan over all input gradients and output values
    xs0 = xs
    xs0 = xs0.at[-1, :].set(jnp.zeros_like(xs[-1, :]))

    loop_over = {
        "dx": dxs[:-1],
        "da": dalphas,
        "db": dbetas,
        "xs": (xs[1:], xs[:-1]),
        "a": alphas,
        "b": betas,
    }
    init_val = (xs0, -dxs[-1], jnp.zeros_like(dxs[-1]))
    (_, lambda_1, _lambda_2), (grad_summands, *other) = jax.lax.scan(
        adjoint_step, init=init_val, xs=loop_over, reverse=True
    )

    # Compute the gradients
    grad_matvec = jax.tree_util.tree_map(lambda s: jnp.sum(s, axis=0), grad_summands)
    grad_initvec = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / initvec_norm

    # Return values
    return (grad_initvec, grad_matvec), (lambda_1, *other)


def _lanczos_adjoint_step(xs_all, xi, lambda_plus, /, *, matvec, params, reortho, dx, da, db, xs, a, b):
    """Single step of the Lanczos adjoint computation."""
    # Read inputs
    (xplus, x) = xs

    # Apply formula
    xi /= b
    mu = db - lambda_plus.T @ x + xplus.T @ xi
    nu = da + x.T @ xi
    lambda_ = -xi + mu * xplus + nu * x

    # Value-and-grad of matrix-vector product
    matvec_lambda, vjp = jax.vjp(lambda *p: matvec(lambda_, *p), *params)
    (gradient_increment,) = vjp(x)

    # Reorthogonalization correction
    if reortho == "full":
        # Account for reorthogonalization in backward pass
        lambda_ = lambda_ - xs_all[:xs_all.shape[0]].T @ (xs_all[:xs_all.shape[0]] @ lambda_)

    # Prepare next step
    xi = -dx - matvec_lambda + a * lambda_ + b * lambda_plus - b * nu * xplus

    # Return values
    return (xs_all, xi, lambda_), (gradient_increment, lambda_, mu, nu, xi)


# ==============================================================================
# Arnoldi Hessenberg Decomposition with Efficient Adjoints
# ==============================================================================


def arnoldi_hessenberg(
    matvec: Callable,
    krylov_depth: int,
    /,
    *,
    reortho: str = "full",
    custom_vjp: bool = True,
) -> Callable:
    """Arnoldi Hessenberg decomposition with efficient custom VJP.
    
    For general matrices A, computes the decomposition:
        A V = V H + beta * v_{k+1} e_k^T
    
    where V is orthonormal, H is upper Hessenberg.
    
    Args:
        matvec: Function (v, *params) -> A @ v (can be complex)
        krylov_depth: Number of Arnoldi iterations
        reortho: Reorthogonalization strategy ("none" or "full")
        custom_vjp: Whether to use efficient custom VJP
        
    Returns:
        Function (vec, *params) -> (Q, H, v_out, norm_out)
    """
    if reortho not in ["none", "full"]:
        msg = f"reortho={reortho} unsupported. Choose either 'none' or 'full'."
        raise ValueError(msg)

    def estimate_public(v, *params):
        matvec_convert, aux_args = jax.closure_convert(matvec, v, *params)
        return estimate_backend(matvec_convert, v, *params, *aux_args)

    def estimate_backend(matvec_convert: Callable, v, *params):
        return _arnoldi_forward(matvec_convert, krylov_depth, v, *params, reortho=reortho)

    def estimate_fwd(matvec_convert: Callable, v, *params):
        outputs = estimate_backend(matvec_convert, v, *params)
        return outputs, (outputs, params)

    def estimate_bwd(matvec_convert: Callable, cache, vjp_incoming):
        (Q, H, r, c), params = cache
        dQ, dH, dr, dc = vjp_incoming

        return _arnoldi_adjoint(
            matvec_convert,
            *params,
            Q=Q,
            H=H,
            r=r,
            c=c,
            dQ=dQ,
            dH=dH,
            dr=dr,
            dc=dc,
            reortho=reortho,
        )

    if custom_vjp:
        estimate_backend = jax.custom_vjp(estimate_backend, nondiff_argnums=(0,))
        estimate_backend.defvjp(estimate_fwd, estimate_bwd)  # type: ignore
    
    return estimate_public


def _arnoldi_forward(matvec, krylov_depth, v, *params, reortho: str):
    """Arnoldi forward pass with optional reorthogonalization."""
    if krylov_depth < 1 or krylov_depth > len(v):
        msg = f"Parameter depth {krylov_depth} is outside the expected range"
        raise ValueError(msg)

    # Initialise the variables
    (n,), k = jnp.shape(v), krylov_depth
    Q = jnp.zeros((n, k), dtype=v.dtype)
    H = jnp.zeros((k, k), dtype=v.dtype)
    initlength = jnp.sqrt(jnp.dot(v.conj(), v))
    init = (Q, H, v, initlength)

    # Fix the step function
    def forward_step(i, val):
        return _arnoldi_forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    # Loop and return
    Q, H, v, _length = jax.lax.fori_loop(0, k, forward_step, init)
    return Q, H, v, 1 / initlength


def _arnoldi_forward_step(Q, H, v, length, matvec, *params, idx, reortho: str):
    """Single step of Arnoldi iteration."""
    # Save
    v /= length
    Q = Q.at[:, idx].set(v)

    # Evaluate
    v = matvec(v, *params)

    # Orthonormalise
    h = Q.T.conj() @ v
    v = v - Q @ h

    # Re-orthonormalise
    if reortho != "none":
        v = v - Q @ (Q.T.conj() @ v)

    # Read the length
    length = jnp.sqrt(jnp.dot(v.conj(), v))

    # Save
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)

    return Q, H, v, length


def _arnoldi_adjoint(matvec, *params, Q, H, r, c, dQ, dH, dr, dc, reortho: str):
    """Compute the adjoint (backward pass) for Arnoldi Hessenberg decomposition."""
    # Extract the matrix shapes from Q
    _, krylov_depth = jnp.shape(Q)

    # Prepare a bunch of auxiliary matrices
    def lower(m):
        m_tril = jnp.tril(m)
        return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))

    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]
    lower_mask = lower(jnp.ones((krylov_depth, krylov_depth)))

    # Initialise
    eta = dH @ e_K - Q.T @ dr
    lambda_k = dr + Q @ eta
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)
    dp = jax.tree_util.tree_map(jnp.zeros_like, params)

    # Prepare more auxiliary matrices
    Pi_xi = dQ.T + jnp.outer(eta, r)
    Pi_gamma = -dc * c * jnp.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)

    # Prepare reorthogonalisation:
    P = Q.T
    ps = dH.T
    ps_mask = jnp.tril(jnp.ones((krylov_depth, krylov_depth)), 1)

    # Loop over those values
    indices = jnp.arange(0, len(H), step=1)
    beta_minuses = jnp.concatenate([jnp.ones((1,), dtype=H.dtype), jnp.diag(H, -1)])
    alphas = jnp.diag(H)
    beta_pluses = H - jnp.diag(jnp.diag(H)) - jnp.diag(jnp.diag(H, -1), -1)
    scan_over = {
        "beta_minus": beta_minuses,
        "alpha": alphas,
        "beta_plus": beta_pluses,
        "idx": indices,
        "lower_mask": lower_mask,
        "Pi_gamma": Pi_gamma,
        "Pi_xi": Pi_xi,
        "p": ps,
        "p_mask": ps_mask,
        "q": Q.T,
    }

    # Fix the step function
    def adjoint_step(x, y):
        output = _arnoldi_adjoint_step(
            *x, **y, matvec=matvec, params=params, Q=Q, reortho=reortho
        )
        return output, ()

    # Scan
    init = (lambda_k, Lambda, Gamma, P, dp)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma, _P, dp) = result

    # Solve for the input gradient
    dv = lambda_k * c

    return dv, *dp


def _arnoldi_adjoint_step(
    # Running variables
    lambda_k,
    Lambda,
    Gamma,
    P,
    dp,
    *,
    # Matrix-vector product
    matvec,
    params,
    # Loop over: index
    idx,
    # Loop over: submatrices of H
    beta_minus,
    alpha,
    beta_plus,
    # Loop over: auxiliary variables for Gamma
    lower_mask,
    Pi_gamma,
    Pi_xi,
    q,
    # Loop over: reorthogonalisation
    p,
    p_mask,
    # Other parameters
    Q,
    reortho: str,
):
    """Single step of Arnoldi adjoint computation."""
    # Reorthogonalise
    if reortho == "full":
        P = p_mask[:, None] * P
        p = p_mask * p
        lambda_k = lambda_k - P.T @ (P @ lambda_k) + P.T @ p

    # Transposed matvec and parameter-gradient in a single matvec
    _, vjp = jax.vjp(lambda u, v: matvec(u, *v), q, params)
    vecmat_lambda, dp_increment = vjp(lambda_k)
    dp = jax.tree_util.tree_map(lambda g, h: g + h, dp, dp_increment)

    # Solve for (Gamma + Gamma.T) e_K
    tmp = lower_mask * (Pi_gamma - vecmat_lambda @ Q)
    Gamma = Gamma.at[idx, :].set(tmp)

    # Solve for the next lambda (backward substitution step)
    Lambda = Lambda.at[:, idx].set(lambda_k)
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    lambda_k = xi - (alpha * lambda_k - vecmat_lambda) - beta_plus @ Lambda.T
    lambda_k /= beta_minus
    
    return lambda_k, Lambda, Gamma, P, dp


# ==============================================================================
# Hutchinson Trace Estimator with Custom VJP
# ==============================================================================


def hutchinson_trace_estimator(
    integrand_fun: Callable,
    /,
    sample_fun: Callable,
    *,
    use_custom_vjp: bool = True,
) -> Callable:
    """Hutchinson trace estimator with custom VJP for variance reduction.
    
    Estimates tr(f(A)) using Monte Carlo sampling:
        tr(f(A)) ≈ E[z^T f(A) z]
    
    where z ~ sample_fun(key).
    
    Args:
        integrand_fun: Function (v, *params) -> f(A) @ v
        sample_fun: Function key -> samples (typically standard normal)
        use_custom_vjp: Use different random samples for forward/backward
        
    Returns:
        Function (key, *params) -> estimate
    """
    if use_custom_vjp:
        return _hutchinson_custom_vjp(integrand_fun, sample_fun)
    else:
        return _hutchinson_nograd(integrand_fun, sample_fun)


def _hutchinson_nograd(integrand_fun, sample_fun):
    """Hutchinson estimator with stopped gradients through samples."""
    def sample(key, *parameters):
        samples = sample_fun(key)
        samples = jax.lax.stop_gradient(samples)
        Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)

    return jax.jit(sample)


def _hutchinson_custom_vjp(integrand_fun, sample_fun):
    """Hutchinson estimator with different keys for forward/backward pass."""
    @jax.custom_vjp
    def sample(_key, *_parameters):
        # This function shall only be meaningful inside a VJP
        msg = "hutchinson_custom_vjp should only be called within a VJP context"
        raise RuntimeError(msg)

    def sample_fwd(key, *parameters):
        _key_fwd, key_bwd = jax.random.split(key, num=2)
        sampled = _sample(sample_fun, integrand_fun, key, *parameters)
        return sampled, {"key": key_bwd, "parameters": parameters}

    def sample_bwd(cache, vjp_incoming):
        def integrand_fun_new(v, *p):
            # Checkpoint the forward pass
            _fx, vjp = jax.vjp(integrand_fun, v, *p)
            return vjp(vjp_incoming)

        key = cache["key"]
        parameters = cache["parameters"]
        return _sample(sample_fun, integrand_fun_new, key, *parameters)

    sample.defvjp(sample_fwd, sample_bwd)
    return sample


def _sample(sample_fun, integrand_fun, key, *parameters):
    """Core Hutchinson sampling logic."""
    samples = sample_fun(key)
    Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
    return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)


# ==============================================================================
# Matrix Function Integrands (for use with Lanczos/Arnoldi)
# ==============================================================================


def lanczos_quadrature_spd(
    matfun: Callable,
    krylov_depth: int,
    matvec: Callable,
    /,
    *,
    reortho: str = "full",
    use_efficient_adjoint: bool = True,
) -> Callable:
    """Lanczos-based quadrature for v^T f(A) v where A is SPD.
    
    Useful for trace estimation via Hutchinson: tr(f(A)) ≈ E[z^T f(A) z].
    
    Args:
        matfun: Scalar function to apply to eigenvalues
        krylov_depth: Number of Lanczos iterations
        matvec: Function (v, *params) -> A @ v
        reortho: Reorthogonalization strategy
        use_efficient_adjoint: Use custom VJP from Krämer et al.
        
    Returns:
        Function (v0, *params) -> v0^T f(A) v0
    """
    def quadform(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)
        scale = jnp.linalg.norm(v0_flat)
        v0_flat /= scale

        @jax.tree_util.Partial
        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, unflatten = jax.flatten_util.ravel_pytree(av)
            return flat

        # Use the efficient VJP for tri-diagonalisation
        algorithm = lanczos_tridiag(
            matvec_flat,
            krylov_depth,
            custom_vjp=use_efficient_adjoint,
            reortho=reortho,
        )
        (basis, (diag, off_diag)), _remainder = algorithm(v0_flat, *parameters)

        # Diagonalize the tridiagonal matrix
        diag_mat = jnp.diag(diag)
        offdiag1 = jnp.diag(off_diag, -1)
        offdiag2 = jnp.diag(off_diag, 1)
        dense_matrix = diag_mat + offdiag1 + offdiag2

        eigvals, eigvecs = jnp.linalg.eigh(dense_matrix)

        # Since Q orthogonal to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        fx_eigvals = jax.vmap(matfun)(eigvals)
        return scale**2 * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


# ==============================================================================
# Enhanced CG Solver with Custom Linear Solve
# ==============================================================================


def cg_fixed_iterations(num_matvecs: int) -> Callable:
    """Conjugate gradient solver with custom_linear_solve integration.
    
    Uses jax.lax.custom_linear_solve for better autodiff support.
    
    Args:
        num_matvecs: Fixed number of CG iterations
        
    Returns:
        Function (A, b) -> (x, info)
    """
    def pcg(A: Callable, b: jax.Array, P: Callable):
        return jax.lax.custom_linear_solve(
            A, b, lambda a, r: pcg_impl(a, r, P=P), symmetric=True, has_aux=True
        )

    def pcg_impl(A: Callable, b, P):
        x = jnp.zeros_like(b)
        r = b - A(x)
        z = P(r)
        p = z

        body_fun = _make_cg_body(A, P)
        init = (x, p, r, z)
        x, p, r, z = jax.lax.fori_loop(0, num_matvecs, body_fun, init_val=init)
        return x, {"residual_abs": r, "residual_rel": r / jnp.abs(x)}

    def cg(A: Callable, b: jax.Array):
        return pcg(A, b, lambda v: v)

    return cg


def _make_cg_body(A, P):
    """Helper to create CG iteration body function."""
    def body_fun(_i, state):
        x, p, r, z = state

        Ap = A(p)
        a = _safe_divide(jnp.dot(r, z), (p.T @ Ap))
        x = x + a * p

        rold = r
        r = r - a * Ap

        zold = z
        z = P(r)

        b = _safe_divide(jnp.dot(r, z), jnp.dot(rold, zold))
        p = z + b * p
        return x, p, r, z

    return body_fun


def _safe_divide(num, denom, eps=1e-30):
    """Safe division avoiding NaN."""
    return num / (denom + eps)


# ==============================================================================
# Low-Rank Preconditioners (Partial Cholesky)
# ==============================================================================


def low_rank_preconditioner(cholesky: Callable, /) -> Callable:
    """Turn a low-rank Cholesky approximation into a preconditioner.
    
    For a matrix A ≈ s*I + L L^T, constructs a preconditioner that solves:
        (s*I + L L^T)^{-1} v
    
    using the Woodbury matrix identity:
        (s*I + L L^T)^{-1} = I/s - (1/s) L (I + L^T L/s)^{-1} L^T/s
    
    Args:
        cholesky: Function that computes low-rank Cholesky factorization
        
    Returns:
        Function (lazy_kernel, nrows) -> (solve_fn, info)
        
    Notes:
        Choose s (small_value) such that s*I + L L^T ≈ A holds.
        A good heuristic is s = sqrt(machine_epsilon / cond(A)).
        
    Example:
        >>> chol_fn = partial_cholesky(rank=50)
        >>> precond_fn = low_rank_preconditioner(chol_fn)
        >>> solve, info = precond_fn(kernel_fn, n=1000)
        >>> x = solve(b, s=1e-6)  # Solve (s*I + L L^T)^{-1} b
    """
    def solve_with_preconditioner(lazy_kernel, /, nrows: int):
        chol, info = cholesky(lazy_kernel, nrows)

        # Assert that the low-rank matrix is tall, not wide
        N, n = jnp.shape(chol)
        if n > N:
            raise ValueError(f"Low-rank matrix must be tall (N >= n), got {N} < {n}")

        @jax.custom_vjp
        def solve(v: jax.Array, s: float):
            """Solve (s*I + L L^T)^{-1} v using Woodbury identity."""
            # Scale
            U = chol / jnp.sqrt(s)
            V = chol.T / jnp.sqrt(s)
            v_scaled = v / s

            # Cholesky decompose the capacitance matrix and solve
            eye_n = jnp.eye(n, dtype=chol.dtype)
            chol_cap = jax.scipy.linalg.cho_factor(eye_n + V @ U)
            sol = jax.scipy.linalg.cho_solve(chol_cap, V @ v_scaled)
            return v_scaled - U @ sol

        # Block differentiation through preconditioner (non-differentiable)
        def fwd(v, s):
            return solve(v, s), None

        def bwd(_cache, _vjp_incoming):
            raise RuntimeError("Differentiation through preconditioner not supported")

        solve.defvjp(fwd, bwd)

        return solve, info

    return solve_with_preconditioner


def partial_cholesky(*, rank: int) -> Callable:
    """Compute a partial (low-rank) Cholesky factorization.
    
    For a positive semi-definite matrix A, computes L such that:
        A ≈ L L^T
    where L is N × rank (rank << N).
    
    Args:
        rank: Target rank for the approximation
        
    Returns:
        Function (lazy_kernel, n) -> (L, info)
        where lazy_kernel(i, j, *params) returns A[i,j]
        
    Example:
        >>> def kernel(i, j):
        ...     return jnp.exp(-0.5 * (i - j)**2)
        >>> chol_fn = partial_cholesky(rank=50)
        >>> L, info = chol_fn(kernel, n=1000)
        >>> # Now A ≈ L @ L.T
    """
    def cholesky(lazy_kernel: Callable, n: int, /):
        if rank > n:
            msg = f"Rank {rank} exceeds matrix size {n}"
            raise ValueError(msg)
        if rank < 1:
            msg = f"Rank must be positive, got {rank}"
            raise ValueError(msg)

        i, j = 0, 0
        element, aux_args = jax.closure_convert(lazy_kernel, i, j)
        return _cholesky(element, n, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0, 1])
    def _cholesky(lazy_kernel: Callable, n: int, *params):
        step = _cholesky_partial_body(lazy_kernel, n, *params)
        chol = jnp.zeros((n, rank))
        return jax.lax.fori_loop(0, rank, step, chol), {}

    # Block differentiation (non-differentiable factorization)
    def _fwd(*args):
        return _cholesky(*args), None

    def _bwd(*_args):
        raise RuntimeError("Differentiation through Cholesky not supported")

    _cholesky.defvjp(_fwd, _bwd)

    return cholesky


def _cholesky_partial_body(fn: Callable, n: int, *args):
    """Helper for partial Cholesky iteration."""
    idx = jnp.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_column(i):
        fun = jax.vmap(matrix_element, in_axes=(0, None))
        return fun(idx, i)

    def body(i, L):
        # Diagonal element
        element = matrix_element(i, i)
        l_ii = jnp.sqrt(element - jnp.dot(L[i], L[i]))

        # Column i
        column = matrix_column(i)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii

        return L.at[:, i].set(l_ji)

    return body


def partial_cholesky_pivoted(*, rank: int) -> Callable:
    """Compute a partial Cholesky factorization with pivoting.
    
    Similar to partial_cholesky, but uses pivoting to select the most
    important columns, improving numerical stability and approximation quality.
    
    Pivoting ensures that at each step, we choose the column with the
    largest residual diagonal element, which typically gives better
    low-rank approximations.
    
    Args:
        rank: Target rank for the approximation
        
    Returns:
        Function (lazy_kernel, n) -> (L, info)
        where L is the pivoted Cholesky factor (rows permuted)
        and info contains 'success' flag
    """
    def cholesky(matrix_element: Callable, n: int):
        if rank > n:
            msg = f"Rank {rank} exceeds matrix size {n}"
            raise ValueError(msg)
        if rank < 1:
            msg = f"Rank must be positive, got {rank}"
            raise ValueError(msg)

        i, j = 0, 0
        element, aux_args = jax.closure_convert(matrix_element, i, j)
        return call_backend(element, n, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0, 1])
    def call_backend(matrix_element: Callable, n: int, *params):
        body = _cholesky_partial_pivot_body(matrix_element, n, *params)

        L = jnp.zeros((n, rank))
        P = jnp.arange(n)

        init = (L, P, P, True)
        (L, P, _matrix, success) = jax.lax.fori_loop(0, rank, body, init)
        return _pivot_invert(L, P), {"success": success}

    # Block differentiation
    def fwd(*args):
        return call_backend(*args), None

    def bwd(*_args):
        raise RuntimeError("Differentiation through pivoted Cholesky not supported")

    call_backend.defvjp(fwd, bwd)

    return cholesky


def _cholesky_partial_pivot_body(fn: Callable, n: int, *args):
    """Helper for pivoted partial Cholesky iteration."""
    idx = jnp.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_element_p(i, j, *, permute):
        return matrix_element(permute[i], permute[j])

    def matrix_column_p(i, *, permute):
        fun = jax.vmap(matrix_element, in_axes=(0, None))
        return fun(permute[idx], permute[i])

    def matrix_diagonal_p(*, permute):
        fun = jax.vmap(matrix_element)
        return fun(permute[idx], permute[idx])

    def body(i, carry):
        L, P, P_matrix, success = carry

        # Access the matrix diagonal
        diagonal = matrix_diagonal_p(permute=P_matrix)

        # Find the largest residual diagonal entry
        residual_diag = diagonal - jax.vmap(jnp.dot)(L, L)
        res = jnp.abs(residual_diag)
        k = jnp.argmax(res)

        # Pivot (swap rows i and k)
        P_matrix = _swap_cols(P_matrix, i, k)
        L = _swap_rows(L, i, k)
        P = _swap_rows(P, i, k)

        # Access matrix elements
        element = matrix_element_p(i, i, permute=P_matrix)
        column = matrix_column_p(i, permute=P_matrix)

        # Perform Cholesky step
        l_ii_squared = element - jnp.dot(L[i], L[i])
        l_ii = jnp.sqrt(l_ii_squared)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii
        success = jnp.logical_and(success, l_ii_squared > 0.0)

        # Update
        L = L.at[:, i].set(l_ji)
        return L, P, P_matrix, success

    return body


def _swap_cols(arr, i, j):
    """Swap columns i and j."""
    return _swap_rows(arr.T, i, j).T


def _swap_rows(arr, i, j):
    """Swap rows i and j."""
    ai, aj = arr[i], arr[j]
    arr = arr.at[i].set(aj)
    return arr.at[j].set(ai)


def _pivot_invert(arr, pivot, /):
    """Invert and apply a pivoting array to a matrix."""
    return arr[jnp.argsort(pivot)]


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    "lanczos_tridiag",
    "arnoldi_hessenberg",
    "hutchinson_trace_estimator",
    "lanczos_quadrature_spd",
    "cg_fixed_iterations",
    "low_rank_preconditioner",
    "partial_cholesky",
    "partial_cholesky_pivoted",
]
