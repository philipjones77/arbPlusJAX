from __future__ import annotations

import jax.numpy as jnp

from .base import CurvatureOperator, make_curvature_operator
from .types import CurvatureSpec


def _metadata_scalar(value):
    try:
        return float(value)
    except Exception:
        return "traced"


def _dense_solve_fn(dense_fn):
    return lambda b, **kwargs: jnp.linalg.solve(dense_fn(), jnp.asarray(b))


def _dense_logdet_fn(dense_fn):
    return lambda **kwargs: jnp.linalg.slogdet(dense_fn())[1]


def _dense_inverse_diagonal_fn(dense_fn):
    return lambda **kwargs: jnp.diag(jnp.linalg.inv(dense_fn()))


def symmetrize_operator(curv_op: CurvatureOperator) -> CurvatureOperator:
    dense_fn = None
    if curv_op.to_dense_fn is not None:
        dense_fn = lambda: 0.5 * (curv_op.to_dense() + jnp.swapaxes(jnp.conjugate(curv_op.to_dense()), -1, -2))
    return make_curvature_operator(
        shape=curv_op.shape,
        dtype=curv_op.dtype,
        matvec=lambda v: 0.5 * (curv_op.matvec(v) + curv_op.transpose_matvec(v)),
        rmatvec=lambda v: 0.5 * (curv_op.matvec(v) + curv_op.transpose_matvec(v)),
        to_dense_fn=dense_fn,
        diagonal_fn=(lambda: jnp.diag(dense_fn())) if dense_fn is not None else None,
        trace_fn=(lambda: jnp.trace(dense_fn())) if dense_fn is not None else None,
        solve_fn=_dense_solve_fn(dense_fn) if dense_fn is not None else None,
        logdet_fn=_dense_logdet_fn(dense_fn) if dense_fn is not None else None,
        inverse_diagonal_fn=_dense_inverse_diagonal_fn(dense_fn) if dense_fn is not None else None,
        metadata={**curv_op.metadata, "symmetric": True},
    )


def add_jitter(curv_op: CurvatureOperator, jitter: float) -> CurvatureOperator:
    value = jnp.asarray(jitter, dtype=curv_op.dtype)
    dense_fn = None
    if curv_op.to_dense_fn is not None:
        dense_fn = lambda: curv_op.to_dense() + value * jnp.eye(curv_op.shape[0], dtype=curv_op.dtype)
    return make_curvature_operator(
        shape=curv_op.shape,
        dtype=curv_op.dtype,
        matvec=lambda v: curv_op.matvec(v) + value * v,
        rmatvec=lambda v: curv_op.transpose_matvec(v) + value * v,
        to_dense_fn=dense_fn,
        diagonal_fn=(lambda: curv_op.diagonal() + value) if curv_op.diagonal_fn is not None or curv_op.to_dense_fn is not None else None,
        trace_fn=(lambda: curv_op.trace() + value * jnp.asarray(curv_op.shape[0], dtype=curv_op.dtype)) if curv_op.trace_fn is not None or curv_op.to_dense_fn is not None else None,
        solve_fn=_dense_solve_fn(dense_fn) if dense_fn is not None else None,
        logdet_fn=_dense_logdet_fn(dense_fn) if dense_fn is not None else None,
        inverse_diagonal_fn=_dense_inverse_diagonal_fn(dense_fn) if dense_fn is not None else None,
        metadata={**curv_op.metadata, "jitter": _metadata_scalar(jitter)},
    )


def ensure_psd(curv_op: CurvatureOperator, *, jitter: float = 0.0) -> CurvatureOperator:
    out = symmetrize_operator(curv_op) if not curv_op.is_symmetric() else curv_op
    if jitter > 0.0:
        out = add_jitter(out, jitter)
    return make_curvature_operator(
        shape=out.shape,
        dtype=out.dtype,
        matvec=out.matvec,
        rmatvec=out.rmatvec,
        to_dense_fn=out.to_dense_fn,
        diagonal_fn=out.diagonal_fn,
        trace_fn=out.trace_fn,
        solve_fn=out.solve_fn,
        logdet_fn=out.logdet_fn,
        inverse_diagonal_fn=out.inverse_diagonal_fn,
        metadata={**out.metadata, "symmetric": True, "psd": True},
    )


def make_posterior_precision_operator(
    prior_precision: CurvatureOperator,
    likelihood_curvature: CurvatureOperator,
    *,
    damping: float = 0.0,
    jitter: float = 0.0,
    spec: CurvatureSpec | None = None,
) -> CurvatureOperator:
    damping_value = jnp.asarray(damping, dtype=prior_precision.dtype)
    jitter_value = jnp.asarray(jitter, dtype=prior_precision.dtype)
    total_shift = damping_value + jitter_value
    dense_fn = None
    if prior_precision.to_dense_fn is not None and likelihood_curvature.to_dense_fn is not None:
        dense_fn = (
            lambda: prior_precision.to_dense()
            + likelihood_curvature.to_dense()
            + total_shift * jnp.eye(prior_precision.shape[0], dtype=prior_precision.dtype)
        )
    return make_curvature_operator(
        shape=prior_precision.shape,
        dtype=prior_precision.dtype,
        matvec=lambda v: prior_precision.matvec(v) + likelihood_curvature.matvec(v) + total_shift * v,
        rmatvec=lambda v: prior_precision.transpose_matvec(v) + likelihood_curvature.transpose_matvec(v) + total_shift * v,
        to_dense_fn=dense_fn,
        diagonal_fn=(lambda: prior_precision.diagonal() + likelihood_curvature.diagonal() + total_shift) if dense_fn is not None else None,
        trace_fn=(lambda: prior_precision.trace() + likelihood_curvature.trace() + total_shift * jnp.asarray(prior_precision.shape[0], dtype=prior_precision.dtype))
        if dense_fn is not None
        else None,
        solve_fn=_dense_solve_fn(dense_fn) if dense_fn is not None else None,
        logdet_fn=_dense_logdet_fn(dense_fn) if dense_fn is not None else None,
        inverse_diagonal_fn=_dense_inverse_diagonal_fn(dense_fn) if dense_fn is not None else None,
        metadata={
            "kind": "posterior_precision",
            "symmetric": bool(prior_precision.is_symmetric() and likelihood_curvature.is_symmetric()),
            "psd": True if prior_precision.is_psd() and likelihood_curvature.is_psd() else None,
            "damping": _metadata_scalar(damping),
            "jitter": _metadata_scalar(jitter),
        },
        spec=spec,
    )
