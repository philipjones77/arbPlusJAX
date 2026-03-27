from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import make_curvature_operator
from .hvp import hvp
from .types import CurvatureSpec


def hessian_dense(fun, x, *, args=None):
    args = () if args is None else tuple(args)
    return jax.hessian(lambda z: fun(z, *args))(x)


def hessian_blocks(fun, x, block_structure, *, args=None):
    dense = hessian_dense(fun, x, args=args)
    return [dense[idx] for idx in block_structure]


def make_hessian_operator(fun, x, *, args=None, dense: bool = False, spec: CurvatureSpec | None = None):
    x_arr = jnp.asarray(x)
    dim = int(x_arr.size)
    dense_h = hessian_dense(fun, x_arr, args=args) if dense else None
    return make_curvature_operator(
        shape=(dim, dim),
        dtype=x_arr.dtype,
        matvec=(lambda v: dense_h @ v) if dense_h is not None else (lambda v: hvp(fun, x_arr, v, args=args)),
        rmatvec=(lambda v: dense_h @ v) if dense_h is not None else (lambda v: hvp(fun, x_arr, v, args=args)),
        to_dense_fn=(lambda: dense_h) if dense_h is not None else None,
        diagonal_fn=(lambda: jnp.diag(dense_h)) if dense_h is not None else None,
        trace_fn=(lambda: jnp.trace(dense_h)) if dense_h is not None else None,
        metadata={"kind": "hessian", "symmetric": True},
        spec=spec,
    )
