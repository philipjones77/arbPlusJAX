"""Decomposition adjoints for matrix-free Krylov methods."""

from __future__ import annotations

import functools
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp


def lanczos_tridiag(
    matvec: Callable,
    krylov_depth: int,
    /,
    *,
    reortho: str = "full",
    custom_vjp: bool = True,
) -> Callable:
    if reortho not in ["none", "full"]:
        raise ValueError(f"reortho={reortho} unsupported. Choose either 'none' or 'full'.")

    def raw_estimate(vec, *params):
        *values, _ = _lanczos_forward(matvec, krylov_depth, vec, *params, reortho=reortho)
        return values

    def estimate_standard(vec, *params):
        return raw_estimate(vec, *params)

    estimate = raw_estimate

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (jnp.linalg.norm(vec), *params))

    def estimate_bwd(cache, vjp_incoming):
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector_norm, *params) = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))
        (grad_initvec, grad_params), _ = _lanczos_adjoint(
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
        return (grad_initvec, *grad_params)

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    def estimate_public(vec, *params):
        if custom_vjp and len(params) == 0:
            return estimate_standard(vec, *params)
        return estimate(vec, *params)

    return estimate_public


def _lanczos_forward(matvec, krylov_depth, vec, *params, reortho: str):
    vectors = jnp.zeros((krylov_depth + 1, len(vec)), dtype=vec.dtype)
    offdiags = jnp.zeros((krylov_depth,), dtype=vec.dtype)
    diags = jnp.zeros((krylov_depth,), dtype=vec.dtype)
    v0 = vec / jnp.linalg.norm(vec)
    vectors = vectors.at[0].set(v0)
    ((v1, offdiag), diag) = _lanczos_init(matvec, v0, *params)
    vectors = vectors.at[1].set(v1)
    offdiags = offdiags.at[0].set(offdiag)
    diags = diags.at[0].set(diag)
    step_fun = functools.partial(_lanczos_step_reortho if reortho == "full" else _lanczos_step, matvec, params)
    init = (v1, offdiag, v0), (vectors, diags, offdiags)
    _, (vectors, diags, offdiags) = jax.lax.fori_loop(1, krylov_depth, step_fun, init)
    return (vectors[:-1], (diags, offdiags[:-1])), (vectors[-1], offdiags[-1]), 1 / jnp.linalg.norm(vec)


def _lanczos_init(matvec, vec, *params):
    a = vec @ (matvec(vec, *params))
    r = matvec(vec, *params) - a * vec
    b = jnp.linalg.norm(r)
    return (r / b, b), a


def _lanczos_step(matvec, params, i, val):
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _lanczos_step_apply(matvec, v1, offdiag, v0, *params), v1
    vectors = vectors.at[i + 1].set(v1)
    offdiags = offdiags.at[i].set(offdiag)
    diags = diags.at[i].set(diag)
    return (v1, offdiag, v0), (vectors, diags, offdiags)


def _lanczos_step_reortho(matvec, params, i, val):
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _lanczos_step_apply(matvec, v1, offdiag, v0, *params), v1
    active = (jnp.arange(vectors.shape[0]) <= i).astype(vectors.dtype)
    coeffs = (vectors @ v1) * active
    v1 = v1 - vectors.T @ coeffs
    vectors = vectors.at[i + 1].set(v1)
    offdiags = offdiags.at[i].set(offdiag)
    diags = diags.at[i].set(diag)
    return (v1, offdiag, v0), (vectors, diags, offdiags)


def _lanczos_step_apply(matvec, vec, b, vec_previous, *params):
    a = vec @ (matvec(vec, *params))
    r = matvec(vec, *params) - a * vec - b * vec_previous
    b = jnp.linalg.norm(r)
    return (r / b, b), a


def _lanczos_adjoint(*, matvec, params, initvec_norm, alphas, betas, xs, dalphas, dbetas, dxs, reortho: str):
    def adjoint_step(xi_and_lambda, inputs):
        return _lanczos_adjoint_step(*xi_and_lambda, matvec=matvec, params=params, reortho=reortho, **inputs)

    xs0 = xs.at[-1, :].set(jnp.zeros_like(xs[-1, :]))
    loop_over = {"dx": dxs[:-1], "da": dalphas, "db": dbetas, "xs": (xs[1:], xs[:-1]), "a": alphas, "b": betas}
    init_val = (xs0, -dxs[-1], jnp.zeros_like(dxs[-1]))
    (_, lambda_1, _), (grad_summands, *other) = jax.lax.scan(adjoint_step, init=init_val, xs=loop_over, reverse=True)
    grad_matvec = jax.tree_util.tree_map(lambda s: jnp.sum(s, axis=0), grad_summands)
    grad_initvec = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / initvec_norm
    return (grad_initvec, grad_matvec), (lambda_1, *other)


def _lanczos_adjoint_step(xs_all, xi, lambda_plus, /, *, matvec, params, reortho, dx, da, db, xs, a, b):
    xplus, x = xs
    xi /= b
    mu = db - lambda_plus.T @ x + xplus.T @ xi
    nu = da + x.T @ xi
    lambda_ = -xi + mu * xplus + nu * x
    matvec_lambda, vjp = jax.vjp(lambda *p: matvec(lambda_, *p), *params)
    gradient_increment = vjp(x)
    if reortho == "full":
        lambda_ = lambda_ - xs_all[: xs_all.shape[0]].T @ (xs_all[: xs_all.shape[0]] @ lambda_)
    xi = -dx - matvec_lambda + a * lambda_ + b * lambda_plus - b * nu * xplus
    return (xs_all, xi, lambda_), (gradient_increment, lambda_, mu, nu, xi)


def arnoldi_hessenberg(
    matvec: Callable,
    krylov_depth: int,
    /,
    *,
    reortho: str = "full",
    custom_vjp: bool = True,
) -> Callable:
    if reortho not in ["none", "full"]:
        raise ValueError(f"reortho={reortho} unsupported. Choose either 'none' or 'full'.")

    def estimate_public(v, *params):
        matvec_convert, aux_args = jax.closure_convert(matvec, v, *params)
        if custom_vjp and len(params) == 0:
            return _arnoldi_forward(matvec_convert, krylov_depth, v, *params, *aux_args, reortho=reortho)
        return estimate_backend(matvec_convert, v, *params, *aux_args)

    def estimate_backend(matvec_convert: Callable, v, *params):
        return _arnoldi_forward(matvec_convert, krylov_depth, v, *params, reortho=reortho)

    def estimate_fwd(matvec_convert: Callable, v, *params):
        outputs = estimate_backend(matvec_convert, v, *params)
        return outputs, (outputs, params)

    def estimate_bwd(matvec_convert: Callable, cache, vjp_incoming):
        (Q, H, r, c), params = cache
        dQ, dH, dr, dc = vjp_incoming
        return _arnoldi_adjoint(matvec_convert, *params, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=reortho)

    if custom_vjp:
        estimate_backend = jax.custom_vjp(estimate_backend, nondiff_argnums=(0,))
        estimate_backend.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate_public


def _arnoldi_forward(matvec, krylov_depth, v, *params, reortho: str):
    if krylov_depth < 1 or krylov_depth > len(v):
        raise ValueError(f"Parameter depth {krylov_depth} is outside the expected range")
    (n,), k = jnp.shape(v), krylov_depth
    Q = jnp.zeros((n, k), dtype=v.dtype)
    H = jnp.zeros((k, k), dtype=v.dtype)
    initlength = jnp.sqrt(jnp.dot(v.conj(), v))
    init = (Q, H, v, initlength)

    def forward_step(i, val):
        return _arnoldi_forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    Q, H, v, _ = jax.lax.fori_loop(0, k, forward_step, init)
    return Q, H, v, 1 / initlength


def _arnoldi_forward_step(Q, H, v, length, matvec, *params, idx, reortho: str):
    v /= length
    Q = Q.at[:, idx].set(v)
    v = matvec(v, *params)
    h = Q.T.conj() @ v
    v = v - Q @ h
    if reortho != "none":
        v = v - Q @ (Q.T.conj() @ v)
    length = jnp.sqrt(jnp.dot(v.conj(), v))
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)
    return Q, H, v, length


def _arnoldi_adjoint(matvec, *params, Q, H, r, c, dQ, dH, dr, dc, reortho: str):
    _, krylov_depth = jnp.shape(Q)

    def lower(m):
        m_tril = jnp.tril(m)
        return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))

    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]
    lower_mask = lower(jnp.ones((krylov_depth, krylov_depth)))
    eta = dH @ e_K - Q.T @ dr
    lambda_k = dr + Q @ eta
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)
    dp = jax.tree_util.tree_map(jnp.zeros_like, params)
    Pi_xi = dQ.T + jnp.outer(eta, r)
    Pi_gamma = -dc * c * jnp.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)
    P = Q.T
    ps = dH.T
    ps_mask = jnp.tril(jnp.ones((krylov_depth, krylov_depth)), 1)
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

    def adjoint_step(x, y):
        output = _arnoldi_adjoint_step(*x, **y, matvec=matvec, params=params, Q=Q, reortho=reortho)
        return output, ()

    init = (lambda_k, Lambda, Gamma, P, dp)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    lambda_k, _Lambda, _Gamma, _P, dp = result
    return lambda_k * c, *dp


def _arnoldi_adjoint_step(lambda_k, Lambda, Gamma, P, dp, *, matvec, params, idx, beta_minus, alpha, beta_plus, lower_mask, Pi_gamma, Pi_xi, q, p, p_mask, Q, reortho: str):
    if reortho == "full":
        P = p_mask[:, None] * P
        p = p_mask * p
        lambda_k = lambda_k - P.T @ (P @ lambda_k) + P.T @ p
    _, vjp = jax.vjp(lambda u, v: matvec(u, *v), q, params)
    vecmat_lambda, dp_increment = vjp(lambda_k)
    dp = jax.tree_util.tree_map(lambda g, h: g + h, dp, dp_increment)
    tmp = lower_mask * (Pi_gamma - vecmat_lambda @ Q)
    Gamma = Gamma.at[idx, :].set(tmp)
    Lambda = Lambda.at[:, idx].set(lambda_k)
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    lambda_k = xi - (alpha * lambda_k - vecmat_lambda) - beta_plus @ Lambda.T
    lambda_k /= beta_minus
    return lambda_k, Lambda, Gamma, P, dp
