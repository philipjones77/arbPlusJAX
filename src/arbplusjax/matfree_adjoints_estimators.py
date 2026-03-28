"""Estimator and preconditioner adjoints for matrix-free methods."""

from __future__ import annotations

import functools
from typing import Callable

import jax
import jax.flatten_util
from jax import lax
import jax.numpy as jnp

from . import matfree_adjoints_decompositions as decompositions


def hutchinson_trace_estimator(integrand_fun: Callable, /, sample_fun: Callable, *, use_custom_vjp: bool = True) -> Callable:
    return _hutchinson_custom_vjp(integrand_fun, sample_fun) if use_custom_vjp else _hutchinson_nograd(integrand_fun, sample_fun)


def _hutchinson_nograd(integrand_fun, sample_fun):
    def sample(key, *parameters):
        samples = jax.lax.stop_gradient(sample_fun(key))
        Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)

    return jax.jit(sample)


def _hutchinson_custom_vjp(integrand_fun, sample_fun):
    @jax.custom_vjp
    def sample(_key, *_parameters):
        raise RuntimeError("hutchinson_custom_vjp should only be called within a VJP context")

    def sample_fwd(key, *parameters):
        _key_fwd, key_bwd = jax.random.split(key, num=2)
        del _key_fwd
        sampled = _sample(sample_fun, integrand_fun, key, *parameters)
        return sampled, {"key": key_bwd, "parameters": parameters}

    def sample_bwd(cache, vjp_incoming):
        def integrand_fun_new(v, *p):
            _fx, vjp = jax.vjp(integrand_fun, v, *p)
            return vjp(vjp_incoming)

        return _sample(sample_fun, integrand_fun_new, cache["key"], *cache["parameters"])

    sample.defvjp(sample_fwd, sample_bwd)
    return sample


def _sample(sample_fun, integrand_fun, key, *parameters):
    samples = sample_fun(key)
    Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
    return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)


def lanczos_quadrature_spd(matfun: Callable, krylov_depth: int, matvec: Callable, /, *, reortho: str = "full", use_efficient_adjoint: bool = True) -> Callable:
    def quadform(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)
        scale = jnp.linalg.norm(v0_flat)
        v0_flat /= scale

        @jax.tree_util.Partial
        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, _ = jax.flatten_util.ravel_pytree(av)
            return flat

        algorithm = decompositions.lanczos_tridiag(matvec_flat, krylov_depth, custom_vjp=use_efficient_adjoint, reortho=reortho)
        (_basis, (diag, off_diag)), _remainder = algorithm(v0_flat, *parameters)
        dense_matrix = jnp.diag(diag) + jnp.diag(off_diag, -1) + jnp.diag(off_diag, 1)
        eigvals, eigvecs = jnp.linalg.eigh(dense_matrix)
        fx_eigvals = jax.vmap(matfun)(eigvals)
        return scale**2 * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def cg_fixed_iterations(num_matvecs: int) -> Callable:
    def pcg(A: Callable, b: jax.Array, P: Callable):
        return jax.lax.custom_linear_solve(A, b, lambda a, r: pcg_impl(a, r, P=P), symmetric=True, has_aux=True)

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
    return num / (denom + eps)


def low_rank_preconditioner(cholesky: Callable, /) -> Callable:
    def solve_with_preconditioner(lazy_kernel, /, nrows: int):
        chol, info = cholesky(lazy_kernel, nrows)
        N, n = jnp.shape(chol)
        if n > N:
            raise ValueError(f"Low-rank matrix must be tall (N >= n), got {N} < {n}")

        @jax.custom_vjp
        def solve(v: jax.Array, s: float):
            U = chol / jnp.sqrt(s)
            V = chol.T / jnp.sqrt(s)
            v_scaled = v / s
            eye_n = jnp.eye(n, dtype=chol.dtype)
            chol_cap = jnp.linalg.cholesky(eye_n + V @ U)
            rhs = V @ v_scaled
            sol = lax.linalg.triangular_solve(chol_cap, rhs[:, None], left_side=True, lower=True)
            sol = lax.linalg.triangular_solve(chol_cap.T.conj(), sol, left_side=True, lower=False)
            sol = jnp.squeeze(sol, axis=-1)
            return v_scaled - U @ sol

        def fwd(v, s):
            return solve(v, s), None

        def bwd(_cache, _vjp_incoming):
            raise RuntimeError("Differentiation through preconditioner not supported")

        solve.defvjp(fwd, bwd)
        return solve, info

    return solve_with_preconditioner


def partial_cholesky(*, rank: int) -> Callable:
    def cholesky(lazy_kernel: Callable, n: int, /):
        if rank > n:
            raise ValueError(f"Rank {rank} exceeds matrix size {n}")
        if rank < 1:
            raise ValueError(f"Rank must be positive, got {rank}")
        i, j = 0, 0
        element, aux_args = jax.closure_convert(lazy_kernel, i, j)
        return _cholesky(element, n, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0, 1])
    def _cholesky(lazy_kernel: Callable, n: int, *params):
        step = _cholesky_partial_body(lazy_kernel, n, *params)
        chol = jnp.zeros((n, rank))
        return jax.lax.fori_loop(0, rank, step, chol), {}

    def _fwd(*args):
        return _cholesky(*args), None

    def _bwd(*_args):
        raise RuntimeError("Differentiation through Cholesky not supported")

    _cholesky.defvjp(_fwd, _bwd)
    return cholesky


def _cholesky_partial_body(fn: Callable, n: int, *args):
    idx = jnp.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_column(i):
        fun = jax.vmap(matrix_element, in_axes=(0, None))
        return fun(idx, i)

    def body(i, L):
        element = matrix_element(i, i)
        l_ii = jnp.sqrt(element - jnp.dot(L[i], L[i]))
        column = matrix_column(i)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii
        return L.at[:, i].set(l_ji)

    return body


def partial_cholesky_pivoted(*, rank: int) -> Callable:
    def cholesky(matrix_element: Callable, n: int):
        if rank > n:
            raise ValueError(f"Rank {rank} exceeds matrix size {n}")
        if rank < 1:
            raise ValueError(f"Rank must be positive, got {rank}")
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

    def fwd(*args):
        return call_backend(*args), None

    def bwd(*_args):
        raise RuntimeError("Differentiation through pivoted Cholesky not supported")

    call_backend.defvjp(fwd, bwd)
    return cholesky


def _cholesky_partial_pivot_body(fn: Callable, n: int, *args):
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
        diagonal = matrix_diagonal_p(permute=P_matrix)
        residual_diag = diagonal - jax.vmap(jnp.dot)(L, L)
        k = jnp.argmax(jnp.abs(residual_diag))
        P_matrix = _swap_cols(P_matrix, i, k)
        L = _swap_rows(L, i, k)
        P = _swap_rows(P, i, k)
        element = matrix_element_p(i, i, permute=P_matrix)
        column = matrix_column_p(i, permute=P_matrix)
        l_ii_squared = element - jnp.dot(L[i], L[i])
        l_ii = jnp.sqrt(l_ii_squared)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii
        success = jnp.logical_and(success, l_ii_squared > 0.0)
        L = L.at[:, i].set(l_ji)
        return L, P, P_matrix, success

    return body


def _swap_cols(arr, i, j):
    return arr.at[[i, j]].set(arr[[j, i]])


def _swap_rows(arr, i, j):
    return arr.at[[i, j], ...].set(arr[[j, i], ...])


def _pivot_invert(L, P):
    return L[P.argsort()]
