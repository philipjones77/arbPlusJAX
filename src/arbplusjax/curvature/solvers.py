from __future__ import annotations

import jax.numpy as jnp

from .base import CurvatureOperator
from .composition import add_jitter


def solve(curv_op: CurvatureOperator, b, **kwargs):
    return curv_op.solve(b, **kwargs)


def newton_step(grad, curv_op: CurvatureOperator, *, damping: float = 0.0, **solve_kwargs):
    rhs = -jnp.asarray(grad)
    operator = add_jitter(curv_op, damping) if damping != 0.0 else curv_op
    if operator.solve_fn is None and operator.to_dense_fn is not None:
        dense = operator.to_dense()
        return jnp.linalg.solve(dense, rhs)
    return operator.solve(rhs, **solve_kwargs)


def damped_newton_step(grad, curv_op: CurvatureOperator, *, damping: float = 0.0, **solve_kwargs):
    return newton_step(grad, curv_op, damping=damping, **solve_kwargs)
