from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import make_curvature_operator
from .types import CurvatureSpec


def hvp(fun, x, v, *, args=None):
    args = () if args is None else tuple(args)
    grad_fun = jax.grad(lambda z: fun(z, *args))
    return jax.jvp(grad_fun, (x,), (v,))[1]


def batched_hvp(fun, x, V, *, args=None):
    return jax.vmap(lambda v: hvp(fun, x, v, args=args))(V)


def linearize_hvp(fun, x, *, args=None):
    return lambda v: hvp(fun, x, v, args=args)


def make_hvp_operator(
    fun,
    x,
    *,
    args=None,
    symmetric: bool = True,
    psd: bool | None = None,
    dense: bool = False,
    spec: CurvatureSpec | None = None,
):
    x_arr = jnp.asarray(x)
    dim = int(x_arr.size)
    op = linearize_hvp(fun, x_arr, args=args)
    dense_h = None
    if dense:
        eye = jnp.eye(dim, dtype=x_arr.dtype)
        dense_h = jax.vmap(op)(eye)
    return make_curvature_operator(
        shape=(dim, dim),
        dtype=x_arr.dtype,
        matvec=(lambda v: dense_h @ v) if dense_h is not None else op,
        rmatvec=(lambda v: dense_h @ v) if dense_h is not None else (op if symmetric else None),
        to_dense_fn=(lambda: dense_h) if dense_h is not None else None,
        diagonal_fn=(lambda: jnp.diag(dense_h)) if dense_h is not None else None,
        trace_fn=(lambda: jnp.trace(dense_h)) if dense_h is not None else None,
        solve_fn=(lambda b, **kwargs: jnp.linalg.solve(dense_h, jnp.asarray(b, dtype=x_arr.dtype))) if dense_h is not None else None,
        logdet_fn=(lambda **kwargs: jnp.linalg.slogdet(dense_h)[1]) if dense_h is not None else None,
        inverse_diagonal_fn=(lambda **kwargs: jnp.diag(jnp.linalg.inv(dense_h))) if dense_h is not None else None,
        metadata={
            "kind": "hvp",
            "symmetric": symmetric,
            "psd": psd,
        },
        spec=spec,
    )
