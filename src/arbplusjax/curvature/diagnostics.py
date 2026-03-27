from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import CurvatureOperator


def estimate_extreme_eigenvalues(curv_op: CurvatureOperator):
    dense = curv_op.to_dense()
    vals = jnp.linalg.eigvalsh(0.5 * (dense + jnp.swapaxes(jnp.conjugate(dense), -1, -2)))
    return jnp.min(vals), jnp.max(vals)


def estimate_condition_number(curv_op: CurvatureOperator):
    lo, hi = estimate_extreme_eigenvalues(curv_op)
    lo_abs = jnp.maximum(jnp.abs(lo), jnp.asarray(1e-30, dtype=jnp.float64))
    return jnp.abs(hi) / lo_abs


def detect_negative_curvature(curv_op: CurvatureOperator):
    lo, _ = estimate_extreme_eigenvalues(curv_op)
    return lo < 0


def dot_test_curvature(curv_op: CurvatureOperator, x, y, *, rtol: float = 1e-8, atol: float = 1e-8):
    lhs = jnp.vdot(curv_op.matvec(x), y)
    rhs = jnp.vdot(x, curv_op.transpose_matvec(y))
    return jnp.allclose(lhs, rhs, rtol=rtol, atol=atol)


def curvature_regime_report(curv_op: CurvatureOperator):
    lo, hi = estimate_extreme_eigenvalues(curv_op)
    return {
        "shape": curv_op.shape,
        "symmetric": curv_op.is_symmetric(),
        "psd": curv_op.is_psd(),
        "lambda_min": lo,
        "lambda_max": hi,
        "condition_number": estimate_condition_number(curv_op),
        "negative_curvature": detect_negative_curvature(curv_op),
    }
