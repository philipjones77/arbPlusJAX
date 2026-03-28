from __future__ import annotations

import jax.numpy as jnp

from .base import CurvatureOperator, make_curvature_operator
from .composition import add_jitter, ensure_psd, make_posterior_precision_operator, symmetrize_operator
from .diagnostics import (
    curvature_regime_report,
    detect_negative_curvature,
    dot_test_curvature,
    estimate_condition_number,
    estimate_extreme_eigenvalues,
)
from .fisher import fisher_matvec, make_fisher_operator
from .ggn import ggn_matvec, make_ggn_operator
from .hessian import hessian_blocks, hessian_dense, make_hessian_operator
from .hvp import batched_hvp, hvp, linearize_hvp, make_hvp_operator
from .inverse import (
    colored_inverse_diagonal_estimate,
    colored_inverse_diagonal_with_diagnostics,
    covariance_pushforward,
    inverse_diagonal_estimate,
    posterior_marginal_variances,
    selected_inverse,
)
from .solvers import damped_newton_step, newton_step, solve
from .types import CurvatureSpec


def _dense_operator_from_matrix(a):
    arr = jnp.asarray(a)
    return lambda v: arr @ v


def make_dense_curvature_operator(a, *, symmetric: bool | None = None, psd: bool | None = None) -> CurvatureOperator:
    arr = jnp.asarray(a)
    sym = bool(jnp.allclose(arr, jnp.swapaxes(jnp.conjugate(arr), -1, -2))) if symmetric is None else symmetric
    return make_curvature_operator(
        shape=(int(arr.shape[-2]), int(arr.shape[-1])),
        dtype=arr.dtype,
        matvec=_dense_operator_from_matrix(arr),
        rmatvec=_dense_operator_from_matrix(jnp.swapaxes(jnp.conjugate(arr), -1, -2)),
        to_dense_fn=lambda: arr,
        diagonal_fn=lambda: jnp.diag(arr),
        trace_fn=lambda: jnp.trace(arr),
        solve_fn=lambda b, **kwargs: jnp.linalg.solve(arr, jnp.asarray(b)),
        logdet_fn=lambda **kwargs: jnp.linalg.slogdet(arr)[1],
        inverse_diagonal_fn=lambda **kwargs: jnp.diag(jnp.linalg.inv(arr)),
        metadata={"kind": "dense", "symmetric": sym, "psd": psd},
    )


def make_jrb_curvature_operator(
    matvec,
    *,
    shape: tuple[int, int],
    probes=None,
    steps: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    sparse_bcoo=None,
) -> CurvatureOperator:
    from .. import double_interval as di
    from .. import jrb_mat

    def _lift(v):
        v = jnp.asarray(v, dtype=jnp.float64)
        return di.interval(di._below(v), di._above(v))

    def _solve(b, **kwargs):
        out = jrb_mat.jrb_mat_solve_action_point(
            matvec,
            _lift(b),
            symmetric=symmetric,
            preconditioner=preconditioner,
            **kwargs,
        )
        return di.midpoint(out)

    def _logdet(**kwargs):
        if probes is None or steps is None:
            raise ValueError("probes and steps are required for logdet")
        return jrb_mat.jrb_mat_logdet_estimate_point(matvec, probes, steps)

    def _inverse_diagonal(**kwargs):
        if sparse_bcoo is None:
            dense = None
            if "to_dense" in kwargs:
                dense = kwargs["to_dense"]
            raise NotImplementedError("inverse diagonal currently requires sparse_bcoo metadata or dense fallback")
        return di.midpoint(jrb_mat.jrb_mat_bcoo_inverse_diagonal_point(sparse_bcoo, **kwargs))

    return make_curvature_operator(
        shape=shape,
        dtype=jnp.float64,
        matvec=lambda v: di.midpoint(jrb_mat.jrb_mat_operator_apply_point(matvec, _lift(v))),
        rmatvec=lambda v: di.midpoint(jrb_mat.jrb_mat_operator_apply_point(jrb_mat.matrix_free_core.operator_transpose_plan(matvec, conjugate=False), _lift(v))),
        solve_fn=_solve,
        logdet_fn=_logdet if probes is not None and steps is not None else None,
        inverse_diagonal_fn=_inverse_diagonal if sparse_bcoo is not None else None,
        metadata={
            "kind": "matrix_free",
            "algebra": "jrb",
            "symmetric": symmetric,
            "psd": True if symmetric else None,
            "sparse_bcoo": sparse_bcoo,
        },
    )


def make_jcb_curvature_operator(
    matvec,
    *,
    shape: tuple[int, int],
    probes=None,
    steps: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
) -> CurvatureOperator:
    from .. import acb_core
    from .. import double_interval as di
    from .. import jcb_mat

    def _lift(v):
        z = jnp.asarray(v, dtype=jnp.complex128)
        return acb_core.acb_box(
            di.interval(di._below(jnp.real(z)), di._above(jnp.real(z))),
            di.interval(di._below(jnp.imag(z)), di._above(jnp.imag(z))),
        )

    def _solve(b, **kwargs):
        out = jcb_mat.jcb_mat_solve_action_point(
            matvec,
            _lift(b),
            hermitian=hermitian,
            preconditioner=preconditioner,
            **kwargs,
        )
        return acb_core.acb_midpoint(out)

    def _logdet(**kwargs):
        if probes is None or steps is None:
            raise ValueError("probes and steps are required for logdet")
        if hermitian:
            return jcb_mat.jcb_mat_logdet_slq_hermitian_point(matvec, probes, steps)
        return jcb_mat.jcb_mat_logdet_estimate_point(matvec, probes, steps)

    return make_curvature_operator(
        shape=shape,
        dtype=jnp.complex128,
        matvec=lambda v: acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_apply_point(matvec, _lift(v))),
        rmatvec=lambda v: acb_core.acb_midpoint(
            jcb_mat.jcb_mat_operator_apply_point(jcb_mat.matrix_free_core.operator_transpose_plan(matvec, conjugate=True), _lift(v))
        ),
        solve_fn=_solve,
        logdet_fn=_logdet if probes is not None and steps is not None else None,
        metadata={
            "kind": "matrix_free",
            "algebra": "jcb",
            "symmetric": hermitian,
            "psd": True if hermitian else None,
        },
    )


def make_jrb_sparse_curvature_operator(
    sparse_bcoo,
    *,
    probes=None,
    steps: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
) -> CurvatureOperator:
    from .. import jrb_mat
    from .. import sparse_common

    plan = jrb_mat.jrb_mat_bcoo_operator_plan_prepare(sparse_bcoo)
    shape = (int(sparse_bcoo.rows), int(sparse_bcoo.cols))
    dense = sparse_common.sparse_bcoo_to_dense(sparse_bcoo, algebra="jrb", label="curvature.jrb_sparse.to_dense")
    base = make_jrb_curvature_operator(
        plan,
        shape=shape,
        probes=probes,
        steps=steps,
        symmetric=symmetric,
        preconditioner=preconditioner,
        sparse_bcoo=sparse_bcoo,
    )
    return make_curvature_operator(
        shape=base.shape,
        dtype=base.dtype,
        matvec=base.matvec,
        rmatvec=base.rmatvec,
        to_dense_fn=lambda: dense,
        diagonal_fn=lambda: jnp.diag(dense),
        trace_fn=lambda: jnp.trace(dense),
        solve_fn=base.solve_fn,
        logdet_fn=base.logdet_fn,
        inverse_diagonal_fn=lambda **kwargs: jnp.diag(jnp.linalg.inv(dense)),
        metadata=base.metadata,
    )


def make_jcb_sparse_curvature_operator(
    sparse_bcoo,
    *,
    probes=None,
    steps: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
) -> CurvatureOperator:
    from .. import jcb_mat
    from .. import sparse_common

    plan = jcb_mat.jcb_mat_bcoo_operator_plan_prepare(sparse_bcoo)
    shape = (int(sparse_bcoo.rows), int(sparse_bcoo.cols))
    dense = sparse_common.sparse_bcoo_to_dense(sparse_bcoo, algebra="jcb", label="curvature.jcb_sparse.to_dense")
    base = make_jcb_curvature_operator(
        plan,
        shape=shape,
        probes=probes,
        steps=steps,
        hermitian=hermitian,
        preconditioner=preconditioner,
    )
    return make_curvature_operator(
        shape=base.shape,
        dtype=base.dtype,
        matvec=base.matvec,
        rmatvec=base.rmatvec,
        to_dense_fn=lambda: dense,
        diagonal_fn=lambda: jnp.diag(dense),
        trace_fn=lambda: jnp.trace(dense),
        solve_fn=base.solve_fn,
        logdet_fn=base.logdet_fn,
        metadata={**base.metadata, "sparse_bcoo": sparse_bcoo},
    )


__all__ = [
    "CurvatureOperator",
    "CurvatureSpec",
    "make_curvature_operator",
    "make_dense_curvature_operator",
    "make_jrb_curvature_operator",
    "make_jcb_curvature_operator",
    "make_jrb_sparse_curvature_operator",
    "make_jcb_sparse_curvature_operator",
    "make_hvp_operator",
    "make_hessian_operator",
    "make_ggn_operator",
    "make_fisher_operator",
    "make_posterior_precision_operator",
    "hvp",
    "batched_hvp",
    "linearize_hvp",
    "hessian_dense",
    "hessian_blocks",
    "ggn_matvec",
    "fisher_matvec",
    "solve",
    "newton_step",
    "damped_newton_step",
    "symmetrize_operator",
    "ensure_psd",
    "add_jitter",
    "inverse_diagonal_estimate",
    "colored_inverse_diagonal_estimate",
    "colored_inverse_diagonal_with_diagnostics",
    "selected_inverse",
    "posterior_marginal_variances",
    "covariance_pushforward",
    "estimate_extreme_eigenvalues",
    "estimate_condition_number",
    "detect_negative_curvature",
    "dot_test_curvature",
    "curvature_regime_report",
]
