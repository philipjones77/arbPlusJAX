from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp

from .autodiff import ad_rules
from .autodiff import fingerprints
from . import iterative_solvers
from . import matfree_adjoints


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OperatorPlan:
    kind: str
    payload: object
    orientation: str
    algebra: str

    def tree_flatten(self):
        return (self.payload,), {
            "kind": self.kind,
            "orientation": self.orientation,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload,) = children
        return cls(
            kind=aux_data["kind"],
            payload=payload,
            orientation=aux_data["orientation"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ScaledOperator:
    operator: object
    scale: object

    def tree_flatten(self):
        return (self.operator, self.scale), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        operator, scale = children
        return cls(operator=operator, scale=scale)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PreconditionerPlan:
    kind: str
    payload: object
    orientation: str
    algebra: str

    def tree_flatten(self):
        return (self.payload,), {
            "kind": self.kind,
            "orientation": self.orientation,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload,) = children
        return cls(
            kind=aux_data["kind"],
            payload=payload,
            orientation=aux_data["orientation"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ShiftedSolvePlan:
    operator: object
    shifts: object
    preconditioner: object | None
    solver: str
    algebra: str
    structured: str

    def tree_flatten(self):
        children = (self.operator, self.shifts, self.preconditioner)
        aux = {
            "solver": self.solver,
            "algebra": self.algebra,
            "structured": self.structured,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        operator, shifts, preconditioner = children
        return cls(
            operator=operator,
            shifts=shifts,
            preconditioner=preconditioner,
            solver=aux_data["solver"],
            algebra=aux_data["algebra"],
            structured=aux_data["structured"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RecycledKrylovState:
    basis: object
    projected: object
    residual: object
    preconditioner: object | None
    algorithm: str
    algebra: str
    structured: str

    def tree_flatten(self):
        children = (self.basis, self.projected, self.residual, self.preconditioner)
        aux = {
            "algorithm": self.algorithm,
            "algebra": self.algebra,
            "structured": self.structured,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        basis, projected, residual, preconditioner = children
        return cls(
            basis=basis,
            projected=projected,
            residual=residual,
            preconditioner=preconditioner,
            algorithm=aux_data["algorithm"],
            algebra=aux_data["algebra"],
            structured=aux_data["structured"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LogdetSolveAux:
    operator: object
    transpose_operator: object | None
    logdet_diagnostics: object
    solve_diagnostics: object
    preconditioner: object | None
    solver: str
    implicit_adjoint: bool
    structured: str
    algebra: str

    def tree_flatten(self):
        children = (
            self.operator,
            self.transpose_operator,
            self.logdet_diagnostics,
            self.solve_diagnostics,
            self.preconditioner,
        )
        aux = {
            "solver": self.solver,
            "implicit_adjoint": self.implicit_adjoint,
            "structured": self.structured,
            "algebra": self.algebra,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        operator, transpose_operator, logdet_diagnostics, solve_diagnostics, preconditioner = children
        return cls(
            operator=operator,
            transpose_operator=transpose_operator,
            logdet_diagnostics=logdet_diagnostics,
            solve_diagnostics=solve_diagnostics,
            preconditioner=preconditioner,
            solver=aux_data["solver"],
            implicit_adjoint=aux_data["implicit_adjoint"],
            structured=aux_data["structured"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LogdetSolveResult:
    logdet: object
    solve: object
    aux: LogdetSolveAux

    def tree_flatten(self):
        return (self.logdet, self.solve, self.aux), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        logdet, solve, aux = children
        return cls(logdet=logdet, solve=solve, aux=aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ImplicitAdjointSolveMetadata:
    operator: object
    transpose_operator: object | None
    preconditioner: object | None
    transpose_preconditioner: object | None
    solver: str
    structured: str
    algebra: str
    implicit_adjoint: bool

    def tree_flatten(self):
        children = (
            self.operator,
            self.transpose_operator,
            self.preconditioner,
            self.transpose_preconditioner,
        )
        aux = {
            "solver": self.solver,
            "structured": self.structured,
            "algebra": self.algebra,
            "implicit_adjoint": self.implicit_adjoint,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        operator, transpose_operator, preconditioner, transpose_preconditioner = children
        return cls(
            operator=operator,
            transpose_operator=transpose_operator,
            preconditioner=preconditioner,
            transpose_preconditioner=transpose_preconditioner,
            solver=aux_data["solver"],
            structured=aux_data["structured"],
            algebra=aux_data["algebra"],
            implicit_adjoint=aux_data["implicit_adjoint"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ShellCallbackPayload:
    callback: object
    context: object | None

    def tree_flatten(self):
        return ((self.context,), {"callback": self.callback})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (context,) = children
        return cls(callback=aux_data["callback"], context=context)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FiniteDifferenceOperatorPayload:
    function: object
    base_point: object
    base_value: object | None
    context: object | None
    relative_error: float
    umin: float

    def tree_flatten(self):
        return ((self.base_point, self.base_value, self.context), {
            "function": self.function,
            "relative_error": self.relative_error,
            "umin": self.umin,
        })

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        base_point, base_value, context = children
        return cls(
            function=aux_data["function"],
            base_point=base_point,
            base_value=base_value,
            context=context,
            relative_error=aux_data["relative_error"],
            umin=aux_data["umin"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProbeEstimateStatistics:
    mean: object
    variance: object
    stderr: object
    probe_count: object
    recommended_probe_count: object

    def tree_flatten(self):
        return (
            self.mean,
            self.variance,
            self.stderr,
            self.probe_count,
            self.recommended_probe_count,
        ), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        mean, variance, stderr, probe_count, recommended_probe_count = children
        return cls(
            mean=mean,
            variance=variance,
            stderr=stderr,
            probe_count=probe_count,
            recommended_probe_count=recommended_probe_count,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SlqQuadratureMetadata:
    projected: object
    beta0: object
    nodes: object
    weights: object
    statistics: ProbeEstimateStatistics
    steps: object
    hermitian: object

    def tree_flatten(self):
        return (
            self.projected,
            self.beta0,
            self.nodes,
            self.weights,
            self.statistics,
            self.steps,
            self.hermitian,
        ), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        projected, beta0, nodes, weights, statistics, steps, hermitian = children
        return cls(
            projected=projected,
            beta0=beta0,
            nodes=nodes,
            weights=weights,
            statistics=statistics,
            steps=steps,
            hermitian=hermitian,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HutchppTraceMetadata:
    basis: object
    low_rank_trace: object
    residual_trace: object
    statistics: ProbeEstimateStatistics

    def tree_flatten(self):
        return (self.basis, self.low_rank_trace, self.residual_trace, self.statistics), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        basis, low_rank_trace, residual_trace, statistics = children
        return cls(
            basis=basis,
            low_rank_trace=low_rank_trace,
            residual_trace=residual_trace,
            statistics=statistics,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DeflatedOperatorMetadata:
    basis: object
    image: object
    low_rank_trace: object

    def tree_flatten(self):
        return (self.basis, self.image, self.low_rank_trace), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        basis, image, low_rank_trace = children
        return cls(
            basis=basis,
            image=image,
            low_rank_trace=low_rank_trace,
        )


_STRUCTURE_CODE = {
    "general": 0,
    "symmetric": 1,
    "spd": 2,
    "hermitian": 3,
    "hpd": 4,
}

_SOLVER_CODE = {
    "direct": 0,
    "cg": 1,
    "gmres": 2,
    "lanczos": 3,
    "arnoldi": 4,
    "leja": 5,
    "multi_shift_cg": 6,
    "multi_shift_gmres": 7,
    "minres": 8,
}


def structure_code(name: str) -> jax.Array:
    return jnp.asarray(_STRUCTURE_CODE.get(name, -1), dtype=jnp.int32)


def solver_code(name: str) -> jax.Array:
    return jnp.asarray(_SOLVER_CODE.get(name, -1), dtype=jnp.int32)


def _oriented_dense_matrix(mid: jax.Array, *, orientation: str):
    if orientation == "forward":
        return mid
    if orientation == "transpose":
        return jnp.swapaxes(mid, -1, -2)
    if orientation == "adjoint":
        return jnp.swapaxes(jnp.conjugate(mid), -1, -2)
    raise ValueError(f"unsupported operator orientation: {orientation}")


def dense_operator_plan(mid: jax.Array, *, orientation: str, algebra: str) -> OperatorPlan:
    return OperatorPlan(kind="dense", payload=_oriented_dense_matrix(mid, orientation=orientation), orientation=orientation, algebra=algebra)


def parametric_dense_operator_plan(mid: jax.Array, *, orientation: str, algebra: str) -> OperatorPlan:
    return OperatorPlan(
        kind="dense",
        payload=_oriented_dense_matrix(jnp.asarray(mid), orientation=orientation),
        orientation=orientation,
        algebra=algebra,
    )


def shell_operator_plan(callback, *, context=None, orientation: str = "forward", algebra: str) -> OperatorPlan:
    return OperatorPlan(
        kind="shell",
        payload=ShellCallbackPayload(callback=callback, context=context),
        orientation=orientation,
        algebra=algebra,
    )


def generalized_shell_operator_plan(
    apply_callback,
    solve_callback,
    *,
    context=None,
    orientation: str = "forward",
    algebra: str,
) -> OperatorPlan:
    def callback(v, ctx):
        return solve_callback(apply_callback(v, ctx), ctx)

    return shell_operator_plan(callback, context=context, orientation=orientation, algebra=algebra)


def oriented_shell_operator_plan(
    *,
    context,
    algebra: str,
    orientation: str,
    forward_callback,
    transpose_callback=None,
    adjoint_callback=None,
) -> OperatorPlan:
    if orientation == "forward":
        callback = forward_callback
    elif orientation == "transpose":
        callback = transpose_callback if transpose_callback is not None else forward_callback
    elif orientation == "adjoint":
        callback = adjoint_callback if adjoint_callback is not None else transpose_callback
        if callback is None:
            callback = forward_callback
    else:
        raise ValueError(f"unsupported operator orientation: {orientation}")
    return shell_operator_plan(callback, context=context, orientation=orientation, algebra=algebra)


def dense_preconditioner_plan(mid: jax.Array, *, orientation: str = "forward", algebra: str) -> PreconditionerPlan:
    return PreconditionerPlan(
        kind="dense",
        payload=_oriented_dense_matrix(mid, orientation=orientation),
        orientation=orientation,
        algebra=algebra,
    )


def shell_preconditioner_plan(callback, *, context=None, orientation: str = "forward", algebra: str) -> PreconditionerPlan:
    return PreconditionerPlan(
        kind="shell",
        payload=ShellCallbackPayload(callback=callback, context=context),
        orientation=orientation,
        algebra=algebra,
    )


def oriented_shell_preconditioner_plan(
    *,
    context,
    algebra: str,
    orientation: str,
    forward_callback,
    transpose_callback=None,
    adjoint_callback=None,
) -> PreconditionerPlan:
    if orientation == "forward":
        callback = forward_callback
    elif orientation == "transpose":
        callback = transpose_callback if transpose_callback is not None else forward_callback
    elif orientation == "adjoint":
        callback = adjoint_callback if adjoint_callback is not None else transpose_callback
        if callback is None:
            callback = forward_callback
    else:
        raise ValueError(f"unsupported preconditioner orientation: {orientation}")
    return shell_preconditioner_plan(callback, context=context, orientation=orientation, algebra=algebra)


def identity_preconditioner_plan(*, size: int, dtype, algebra: str) -> PreconditionerPlan:
    return PreconditionerPlan(
        kind="identity",
        payload=jnp.eye(size, dtype=dtype),
        orientation="forward",
        algebra=algebra,
    )


def diagonal_preconditioner_plan(diagonal: jax.Array, *, algebra: str) -> PreconditionerPlan:
    return PreconditionerPlan(
        kind="diagonal",
        payload=jnp.asarray(diagonal),
        orientation="forward",
        algebra=algebra,
    )


def dense_jacobi_preconditioner_plan(mid: jax.Array, *, algebra: str, eps: float = 1e-12) -> PreconditionerPlan:
    diag = jnp.diagonal(jnp.asarray(mid), axis1=-2, axis2=-1)
    safe = jnp.where(jnp.abs(diag) > jnp.asarray(eps, dtype=diag.real.dtype), diag, jnp.ones_like(diag))
    inv_diag = 1.0 / safe
    inv_diag = jnp.where(jnp.abs(diag) > jnp.asarray(eps, dtype=diag.real.dtype), inv_diag, jnp.zeros_like(inv_diag))
    return diagonal_preconditioner_plan(inv_diag, algebra=algebra)


def finite_difference_operator_plan(
    function,
    *,
    base_point,
    base_value=None,
    context=None,
    algebra: str,
    relative_error: float = 1e-7,
    umin: float = 1e-6,
) -> OperatorPlan:
    return OperatorPlan(
        kind="finite_difference",
        payload=FiniteDifferenceOperatorPayload(
            function=function,
            base_point=jnp.asarray(base_point),
            base_value=None if base_value is None else jnp.asarray(base_value),
            context=context,
            relative_error=float(relative_error),
            umin=float(umin),
        ),
        orientation="forward",
        algebra=algebra,
    )


def finite_difference_operator_plan_set_base(plan: OperatorPlan, *, base_point, base_value=None) -> OperatorPlan:
    if plan.kind != "finite_difference":
        raise ValueError("finite_difference_operator_plan_set_base requires a finite_difference operator plan.")
    payload = plan.payload
    return OperatorPlan(
        kind=plan.kind,
        payload=FiniteDifferenceOperatorPayload(
            function=payload.function,
            base_point=jnp.asarray(base_point),
            base_value=None if base_value is None else jnp.asarray(base_value),
            context=payload.context,
            relative_error=payload.relative_error,
            umin=payload.umin,
        ),
        orientation=plan.orientation,
        algebra=plan.algebra,
    )


def sparse_bcoo_operator_plan(a, *, as_sparse_bcoo, sparse_bcoo_cls, orientation: str, algebra: str, conjugate_transpose: bool = False) -> OperatorPlan:
    a = as_sparse_bcoo(a, algebra=algebra, label=f"matrix_free_core.{algebra}.operator_plan")
    if orientation == "forward":
        payload = a
    elif orientation == "transpose":
        payload = sparse_bcoo_cls(
            data=a.data,
            indices=a.indices[:, ::-1],
            rows=a.cols,
            cols=a.rows,
            algebra=algebra,
        )
    elif orientation == "adjoint":
        payload = sparse_bcoo_cls(
            data=jnp.conjugate(a.data) if conjugate_transpose else a.data,
            indices=a.indices[:, ::-1],
            rows=a.cols,
            cols=a.rows,
            algebra=algebra,
        )
    else:
        raise ValueError(f"unsupported operator orientation: {orientation}")
    return OperatorPlan(kind="sparse_bcoo", payload=payload, orientation=orientation, algebra=algebra)


def parametric_bcoo_operator_plan(indices: jax.Array, data: jax.Array, *, shape: tuple[int, int], dtype, algebra: str) -> OperatorPlan:
    idx = jnp.asarray(indices, dtype=jnp.int32)
    if idx.ndim != 2 or idx.shape[-1] != 2:
        raise ValueError("parametric_bcoo_operator_plan expects indices with shape (nnz, 2).")
    row_ids = idx[:, 0]
    col_ids = idx[:, 1]
    rows = int(shape[0])

    def callback(v, context):
        vv = jnp.asarray(v, dtype=dtype)
        vals = jnp.asarray(context["data"], dtype=dtype) * vv[context["col_ids"]]
        return jax.ops.segment_sum(vals, context["row_ids"], context["rows"])

    return shell_operator_plan(
        callback,
        context={
            "data": jnp.asarray(data, dtype=dtype),
            "row_ids": row_ids,
            "col_ids": col_ids,
            "rows": rows,
        },
        orientation="forward",
        algebra=algebra,
    )


def sparse_bcoo_preconditioner_plan(
    a,
    *,
    as_sparse_bcoo,
    sparse_bcoo_cls,
    orientation: str = "forward",
    algebra: str,
    conjugate_transpose: bool = False,
) -> PreconditionerPlan:
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation=orientation,
        algebra=algebra,
        conjugate_transpose=conjugate_transpose,
    )
    return PreconditionerPlan(
        kind=plan.kind,
        payload=plan.payload,
        orientation=plan.orientation,
        algebra=plan.algebra,
    )


def sparse_bcoo_jacobi_preconditioner_plan(
    a,
    *,
    as_sparse_bcoo,
    algebra: str,
    eps: float = 1e-12,
) -> PreconditionerPlan:
    a = as_sparse_bcoo(a, algebra=algebra, label=f"matrix_free_core.{algebra}.jacobi_preconditioner")
    rows = jnp.asarray(a.indices[:, 0], dtype=jnp.int32)
    cols = jnp.asarray(a.indices[:, 1], dtype=jnp.int32)
    diag = jax.ops.segment_sum(jnp.where(rows == cols, a.data, jnp.zeros_like(a.data)), rows, a.rows)
    safe = jnp.where(jnp.abs(diag) > jnp.asarray(eps, dtype=diag.real.dtype), diag, jnp.ones_like(diag))
    inv_diag = 1.0 / safe
    inv_diag = jnp.where(jnp.abs(diag) > jnp.asarray(eps, dtype=diag.real.dtype), inv_diag, jnp.zeros_like(inv_diag))
    return diagonal_preconditioner_plan(inv_diag, algebra=algebra)


def _recover_dense_forward_payload(payload: jax.Array, *, orientation: str):
    if orientation == "forward":
        return jnp.asarray(payload)
    if orientation == "transpose":
        return jnp.swapaxes(jnp.asarray(payload), -1, -2)
    if orientation == "adjoint":
        return jnp.conjugate(jnp.swapaxes(jnp.asarray(payload), -1, -2))
    raise ValueError(f"unsupported dense orientation: {orientation}")


def _orient_sparse_bcoo_payload(payload, *, orientation: str, conjugate_transpose: bool):
    sparse_cls = payload.__class__
    if orientation == "forward":
        return sparse_cls(
            data=payload.data,
            indices=payload.indices,
            rows=payload.rows,
            cols=payload.cols,
            algebra=payload.algebra,
        )
    if orientation == "transpose":
        return sparse_cls(
            data=payload.data,
            indices=payload.indices[:, ::-1],
            rows=payload.cols,
            cols=payload.rows,
            algebra=payload.algebra,
        )
    if orientation == "adjoint":
        return sparse_cls(
            data=jnp.conjugate(payload.data) if conjugate_transpose else payload.data,
            indices=payload.indices[:, ::-1],
            rows=payload.cols,
            cols=payload.rows,
            algebra=payload.algebra,
        )
    raise ValueError(f"unsupported sparse orientation: {orientation}")


def _recover_sparse_bcoo_forward_payload(payload, *, orientation: str):
    if orientation == "forward":
        return payload
    if orientation == "transpose":
        return _orient_sparse_bcoo_payload(payload, orientation="transpose", conjugate_transpose=False)
    if orientation == "adjoint":
        return _orient_sparse_bcoo_payload(payload, orientation="adjoint", conjugate_transpose=True)
    raise ValueError(f"unsupported sparse orientation: {orientation}")


def operator_transpose_plan(operator, *, algebra: str | None = None, conjugate: bool = False):
    target_orientation = "adjoint" if conjugate else "transpose"
    if isinstance(operator, ScaledOperator):
        inner = operator_transpose_plan(operator.operator, algebra=algebra, conjugate=conjugate)
        if inner is None:
            return None
        scale = jnp.conjugate(operator.scale) if conjugate else operator.scale
        return ScaledOperator(operator=inner, scale=scale)
    if isinstance(operator, OperatorPlan):
        if operator.kind == "dense":
            forward_mid = _recover_dense_forward_payload(operator.payload, orientation=operator.orientation)
            return dense_operator_plan(forward_mid, orientation=target_orientation, algebra=algebra or operator.algebra)
        if operator.kind == "sparse_bcoo":
            forward_payload = _recover_sparse_bcoo_forward_payload(operator.payload, orientation=operator.orientation)
            payload = _orient_sparse_bcoo_payload(
                forward_payload,
                orientation=target_orientation,
                conjugate_transpose=conjugate,
            )
            return OperatorPlan(
                kind="sparse_bcoo",
                payload=payload,
                orientation=target_orientation,
                algebra=algebra or operator.algebra,
            )
    return None


def preconditioner_transpose_plan(preconditioner, *, algebra: str | None = None, conjugate: bool = False):
    target_orientation = "adjoint" if conjugate else "transpose"
    if preconditioner is None:
        return None
    if isinstance(preconditioner, PreconditionerPlan):
        if preconditioner.kind == "identity":
            return preconditioner
        if preconditioner.kind == "diagonal":
            payload = jnp.conjugate(preconditioner.payload) if conjugate else preconditioner.payload
            return diagonal_preconditioner_plan(payload, algebra=algebra or preconditioner.algebra)
        if preconditioner.kind == "dense":
            forward_mid = _recover_dense_forward_payload(preconditioner.payload, orientation=preconditioner.orientation)
            return dense_preconditioner_plan(forward_mid, orientation=target_orientation, algebra=algebra or preconditioner.algebra)
        if preconditioner.kind == "sparse_bcoo":
            forward_payload = _recover_sparse_bcoo_forward_payload(preconditioner.payload, orientation=preconditioner.orientation)
            payload = _orient_sparse_bcoo_payload(
                forward_payload,
                orientation=target_orientation,
                conjugate_transpose=conjugate,
            )
            return PreconditionerPlan(
                kind="sparse_bcoo",
                payload=payload,
                orientation=target_orientation,
                algebra=algebra or preconditioner.algebra,
            )
        if preconditioner.kind == "shell":
            payload = preconditioner.payload
            ctx = payload.context
            if isinstance(ctx, dict) and "transpose_callback" in ctx:
                return oriented_shell_preconditioner_plan(
                    context=ctx,
                    algebra=algebra or preconditioner.algebra,
                    orientation=target_orientation,
                    forward_callback=ctx.get("forward_callback", payload.callback),
                    transpose_callback=ctx.get("transpose_callback"),
                    adjoint_callback=ctx.get("adjoint_callback"),
                )
            return shell_preconditioner_plan(
                payload.callback,
                context=ctx,
                orientation=target_orientation,
                algebra=algebra or preconditioner.algebra,
            )
    return None


def operator_plan_apply(plan: OperatorPlan, v: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    vv = midpoint_vector(v)
    if plan.kind == "dense":
        return jnp.asarray(jnp.einsum("...ij,...j->...i", plan.payload, vv), dtype=dtype)
    if plan.kind == "shell":
        if plan.payload.context is None:
            return jnp.asarray(plan.payload.callback(vv), dtype=dtype)
        return jnp.asarray(plan.payload.callback(vv, plan.payload.context), dtype=dtype)
    if plan.kind == "sparse_bcoo":
        return jnp.asarray(
            sparse_bcoo_matvec(plan.payload, vv, algebra=plan.algebra, label=f"matrix_free_core.{plan.algebra}.operator_plan_apply"),
            dtype=dtype,
        )
    if plan.kind == "finite_difference":
        payload = plan.payload
        base = jnp.asarray(payload.base_point, dtype=dtype)
        direction = jnp.asarray(vv, dtype=dtype)
        norm_base = jnp.linalg.norm(jnp.ravel(base))
        norm_direction = jnp.linalg.norm(jnp.ravel(direction))
        denom = jnp.maximum(norm_direction, jnp.asarray(1.0, dtype=norm_direction.dtype))
        scale = jnp.maximum(norm_base, jnp.asarray(payload.umin, dtype=norm_base.dtype))
        step = jnp.asarray(payload.relative_error, dtype=scale.dtype) * scale / denom
        delta = jnp.asarray(step, dtype=dtype) * direction
        if payload.context is None:
            f1 = payload.function(base + delta)
            f0 = payload.function(base) if payload.base_value is None else payload.base_value
        else:
            f1 = payload.function(base + delta, payload.context)
            f0 = payload.function(base, payload.context) if payload.base_value is None else payload.base_value
        return jnp.asarray((jnp.asarray(f1, dtype=dtype) - jnp.asarray(f0, dtype=dtype)) / jnp.asarray(step, dtype=dtype), dtype=dtype)
    raise ValueError(f"unsupported operator plan kind: {plan.kind}")


def preconditioner_plan_apply(plan: PreconditionerPlan, v: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    vv = midpoint_vector(v)
    if plan.kind == "identity":
        return jnp.asarray(vv, dtype=dtype)
    if plan.kind == "diagonal":
        return jnp.asarray(jnp.asarray(plan.payload, dtype=dtype) * vv, dtype=dtype)
    if plan.kind == "dense":
        return jnp.asarray(jnp.einsum("...ij,...j->...i", plan.payload, vv), dtype=dtype)
    if plan.kind == "shell":
        if plan.payload.context is None:
            return jnp.asarray(plan.payload.callback(vv), dtype=dtype)
        return jnp.asarray(plan.payload.callback(vv, plan.payload.context), dtype=dtype)
    if plan.kind == "sparse_bcoo":
        return jnp.asarray(
            sparse_bcoo_matvec(
                plan.payload,
                vv,
                algebra=plan.algebra,
                label=f"matrix_free_core.{plan.algebra}.preconditioner_plan_apply",
            ),
            dtype=dtype,
        )
    raise ValueError(f"unsupported preconditioner plan kind: {plan.kind}")


def operator_apply_midpoint(operator, v: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    if isinstance(operator, ScaledOperator):
        applied = operator_apply_midpoint(
            operator.operator,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
        return jnp.asarray(operator.scale, dtype=dtype) * applied
    if isinstance(operator, OperatorPlan):
        return operator_plan_apply(
            operator,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
    return jnp.asarray(operator(v), dtype=dtype)


def preconditioner_apply_midpoint(preconditioner, v: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    if preconditioner is None:
        return jnp.asarray(midpoint_vector(v), dtype=dtype)
    if isinstance(preconditioner, PreconditionerPlan):
        return preconditioner_plan_apply(
            preconditioner,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
    return jnp.asarray(preconditioner(v), dtype=dtype)


def finite_difference_jacobi_preconditioner_plan(
    plan: OperatorPlan,
    *,
    midpoint_vector,
    sparse_bcoo_matvec,
    dtype,
    algebra: str,
    eps: float = 1e-12,
) -> PreconditionerPlan:
    if plan.kind != "finite_difference":
        raise ValueError("finite_difference_jacobi_preconditioner_plan requires a finite_difference operator plan.")
    n = int(jnp.asarray(plan.payload.base_point).shape[-1])
    eye = jnp.eye(n, dtype=dtype)
    cols = jax.vmap(
        lambda e: operator_plan_apply(
            plan,
            e,
            midpoint_vector=lambda x: x,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
    )(eye)
    diag = jnp.diagonal(cols, axis1=0, axis2=1)
    safe = jnp.where(jnp.abs(diag) > jnp.asarray(eps, dtype=diag.real.dtype), diag, jnp.ones_like(diag))
    inv_diag = 1.0 / safe
    inv_diag = jnp.where(jnp.abs(diag) > jnp.asarray(eps, dtype=diag.real.dtype), inv_diag, jnp.zeros_like(inv_diag))
    return diagonal_preconditioner_plan(inv_diag, algebra=algebra)


def scaled_operator(operator, scale) -> ScaledOperator:
    return ScaledOperator(operator=operator, scale=scale)


def canonicalize_sparse_bcoo(x, *, algebra: str, sparse_common, label: str):
    if isinstance(x, sparse_common.SparseCOO):
        return sparse_common.coo_to_bcoo(x)
    if isinstance(x, sparse_common.SparseCSR):
        dense = sparse_common.sparse_to_dense(x, algebra=algebra, label=label)
        return sparse_common.dense_to_sparse_bcoo(dense, algebra=algebra)
    return sparse_common.as_sparse_bcoo(x, algebra=algebra, label=label)


def det_from_logdet(logdet):
    return jnp.exp(logdet)


def matrix_free_fingerprint(
    *,
    regime: str,
    method: str,
    work_units,
    scale=1.0,
    compensated_sum=False,
    adjoint_residual=0.0,
    note: str = "",
):
    return fingerprints.make_fingerprint(
        regime_code_value=fingerprints.regime_code(regime),
        method_code_value=solver_code(method),
        work_units=work_units,
        scale=scale,
        compensated_sum=compensated_sum,
        adjoint_residual=adjoint_residual,
        note=note,
    )


def attach_krylov_metadata(
    diag,
    *,
    regime: str,
    method: str,
    structure: str,
    work_units,
    primal_residual=0.0,
    adjoint_residual=0.0,
    compensated_sum=False,
    note: str = "",
):
    attachment = ad_rules.attach_rule_artifacts(
        matrix_free_fingerprint(
            regime=regime,
            method=method,
            work_units=work_units,
            compensated_sum=compensated_sum,
            adjoint_residual=adjoint_residual,
            note=note,
        ),
        primal_residual=primal_residual,
        adjoint_residual=adjoint_residual,
        note=note,
    )
    return diag._replace(
        primal_residual=jnp.asarray(attachment.residuals.primal_residual, dtype=jnp.float64),
        adjoint_residual=jnp.asarray(attachment.residuals.adjoint_residual, dtype=jnp.float64),
        regime_code=jnp.asarray(attachment.fingerprint.regime_code, dtype=jnp.int32),
        method_code=jnp.asarray(attachment.fingerprint.method_code, dtype=jnp.int32),
        solver_code=solver_code(method),
        structure_code=structure_code(structure),
    )


def make_shifted_solve_plan(
    operator,
    shifts,
    *,
    preconditioner=None,
    solver: str,
    algebra: str,
    structured: str = "general",
) -> ShiftedSolvePlan:
    return ShiftedSolvePlan(
        operator=operator,
        shifts=jnp.asarray(shifts),
        preconditioner=preconditioner,
        solver=solver,
        algebra=algebra,
        structured=structured,
    )


def make_recycled_krylov_state(
    *,
    basis,
    projected,
    residual,
    preconditioner=None,
    algorithm: str,
    algebra: str,
    structured: str = "general",
) -> RecycledKrylovState:
    return RecycledKrylovState(
        basis=basis,
        projected=projected,
        residual=residual,
        preconditioner=preconditioner,
        algorithm=algorithm,
        algebra=algebra,
        structured=structured,
    )


def make_logdet_solve_result(
    *,
    logdet,
    solve,
    operator,
    transpose_operator,
    logdet_diagnostics,
    solve_diagnostics,
    preconditioner=None,
    solver: str = "",
    implicit_adjoint: bool = False,
    structured: str = "general",
    algebra: str,
) -> LogdetSolveResult:
    return LogdetSolveResult(
        logdet=logdet,
        solve=solve,
        aux=LogdetSolveAux(
            operator=operator,
            transpose_operator=transpose_operator,
            logdet_diagnostics=logdet_diagnostics,
            solve_diagnostics=solve_diagnostics,
            preconditioner=preconditioner,
            solver=solver,
            implicit_adjoint=implicit_adjoint,
            structured=structured,
            algebra=algebra,
        ),
    )


def combine_logdet_solve_point(
    *,
    operator,
    transpose_operator,
    rhs,
    probes,
    solve_with_diagnostics,
    logdet_with_diagnostics,
    preconditioner=None,
    solver: str = "",
    implicit_adjoint: bool = False,
    structured: str = "general",
    algebra: str,
) -> LogdetSolveResult:
    solve_value, solve_diag = solve_with_diagnostics(operator, rhs)
    logdet_value, logdet_diag = logdet_with_diagnostics(operator, probes)
    return make_logdet_solve_result(
        logdet=logdet_value,
        solve=solve_value,
        operator=operator,
        transpose_operator=transpose_operator,
        logdet_diagnostics=logdet_diag,
        solve_diagnostics=solve_diag,
        preconditioner=preconditioner,
        solver=solver,
        implicit_adjoint=implicit_adjoint,
        structured=structured,
        algebra=algebra,
    )


def _operator_apply_linear_midpoint(operator, v_mid: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    if isinstance(operator, ScaledOperator):
        applied = _operator_apply_linear_midpoint(
            operator.operator,
            v_mid,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
        return jnp.asarray(operator.scale, dtype=dtype) * applied
    if isinstance(operator, OperatorPlan):
        if operator.kind == "dense":
            return jnp.asarray(jnp.einsum("...ij,...j->...i", operator.payload, jnp.asarray(v_mid, dtype=dtype)), dtype=dtype)
        if operator.kind == "sparse_bcoo":
            return jnp.asarray(
                sparse_bcoo_matvec(
                    operator.payload,
                    jnp.asarray(v_mid, dtype=dtype),
                    algebra=operator.algebra,
                    label=f"matrix_free_core.{operator.algebra}.multi_shift.operator_apply",
                ),
                dtype=dtype,
            )
    return operator_apply_midpoint(
        operator,
        v_mid,
        midpoint_vector=midpoint_vector,
        sparse_bcoo_matvec=sparse_bcoo_matvec,
        dtype=dtype,
    )


def _preconditioner_apply_linear_midpoint(preconditioner, v_mid: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    if preconditioner is None:
        return jnp.asarray(v_mid, dtype=dtype)
    if isinstance(preconditioner, PreconditionerPlan):
        if preconditioner.kind == "identity":
            return jnp.asarray(v_mid, dtype=dtype)
        if preconditioner.kind == "diagonal":
            return jnp.asarray(jnp.asarray(preconditioner.payload, dtype=dtype) * jnp.asarray(v_mid, dtype=dtype), dtype=dtype)
        if preconditioner.kind == "dense":
            return jnp.asarray(jnp.einsum("...ij,...j->...i", preconditioner.payload, jnp.asarray(v_mid, dtype=dtype)), dtype=dtype)
        if preconditioner.kind == "sparse_bcoo":
            return jnp.asarray(
                sparse_bcoo_matvec(
                    preconditioner.payload,
                    jnp.asarray(v_mid, dtype=dtype),
                    algebra=preconditioner.algebra,
                    label=f"matrix_free_core.{preconditioner.algebra}.multi_shift.preconditioner_apply",
                ),
                dtype=dtype,
            )
    return preconditioner_apply_midpoint(
        preconditioner,
        v_mid,
        midpoint_vector=midpoint_vector,
        sparse_bcoo_matvec=sparse_bcoo_matvec,
        dtype=dtype,
    )


def multi_shift_solve_point(
    plan: ShiftedSolvePlan,
    rhs: jax.Array,
    *,
    apply_operator,
    midpoint_vector,
    sparse_bcoo_matvec,
    dtype,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
):
    rhs_mid = midpoint_vector(rhs)

    def solve_one(shift):
        shift_arr = jnp.asarray(shift, dtype=dtype)

        def shifted_matvec(v):
            base = _operator_apply_linear_midpoint(
                plan.operator,
                v,
                midpoint_vector=midpoint_vector,
                sparse_bcoo_matvec=sparse_bcoo_matvec,
                dtype=dtype,
            )
            return base + shift_arr * jnp.asarray(v, dtype=dtype)

        precond = None
        if plan.preconditioner is not None:
            precond = lambda v: _preconditioner_apply_linear_midpoint(
                plan.preconditioner,
                v,
                midpoint_vector=midpoint_vector,
                sparse_bcoo_matvec=sparse_bcoo_matvec,
                dtype=dtype,
            )

        if plan.solver in ("cg", "multi_shift_cg"):
            x_mid, _ = apply_operator.cg(shifted_matvec, rhs_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
        else:
            x_mid, _ = apply_operator.gmres(shifted_matvec, rhs_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
        return x_mid

    return jax.vmap(solve_one)(jnp.asarray(plan.shifts))


def krylov_solve_midpoint(
    operator,
    rhs: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
    solver: str,
    midpoint_vector,
    lift_vector,
    sparse_bcoo_matvec,
    dtype,
):
    rhs_mid = midpoint_vector(rhs)
    x0_mid = None if x0 is None else midpoint_vector(x0)

    def mv(v):
        return operator_apply_midpoint(
            operator,
            lift_vector(v),
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )

    precond = None
    if preconditioner is not None:
        precond = lambda v: preconditioner_apply_midpoint(
            preconditioner,
            lift_vector(v),
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )

    if solver == "cg":
        x_mid, info = iterative_solvers.cg(mv, rhs_mid, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
    elif solver == "gmres":
        x_mid, info = iterative_solvers.gmres(mv, rhs_mid, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
    elif solver == "minres":
        x_mid, info = iterative_solvers.minres(mv, rhs_mid, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
    else:
        raise ValueError(f"unsupported Krylov solver: {solver}")

    residual = jnp.linalg.norm(mv(x_mid) - rhs_mid)
    return x_mid, info, residual, jnp.linalg.norm(rhs_mid)


def implicit_krylov_solve_midpoint(
    operator,
    rhs: jax.Array,
    *,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    preconditioner=None,
    solver: str,
    structured: str,
    midpoint_vector,
    lift_vector,
    sparse_bcoo_matvec,
    dtype,
    transpose_operator=None,
    transpose_preconditioner=None,
    use_implicit_adjoint: bool | None = None,
):
    rhs_mid = midpoint_vector(rhs)
    x0_mid = None if x0 is None else midpoint_vector(x0)
    conjugate = jnp.issubdtype(jnp.asarray(rhs_mid).dtype, jnp.complexfloating)
    if transpose_operator is None:
        transpose_operator = operator_transpose_plan(operator, conjugate=conjugate)
    if transpose_preconditioner is None:
        transpose_preconditioner = preconditioner_transpose_plan(preconditioner, conjugate=conjugate)
    if use_implicit_adjoint is None:
        use_implicit_adjoint = structured in {"symmetric", "spd", "hermitian", "hpd"} or transpose_operator is not None

    def mv(v):
        return operator_apply_midpoint(
            operator,
            lift_vector(v),
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )

    def build_preconditioner(plan):
        if plan is None:
            return None
        return lambda v: preconditioner_apply_midpoint(
            plan,
            lift_vector(v),
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )

    def solve_impl(matvec_fn, rhs_value, *, preconditioner_plan):
        precond = build_preconditioner(preconditioner_plan)
        if solver == "cg":
            return iterative_solvers.cg(matvec_fn, rhs_value, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
        if solver == "gmres":
            return iterative_solvers.gmres(matvec_fn, rhs_value, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
        if solver == "minres":
            return iterative_solvers.minres(matvec_fn, rhs_value, x0=x0_mid, tol=tol, atol=atol, maxiter=maxiter, M=precond)
        raise ValueError(f"unsupported Krylov solver: {solver}")

    transpose_structured = structured in {"symmetric", "spd", "hermitian", "hpd"}

    def transpose_mv(v):
        if transpose_operator is None:
            raise ValueError("transpose solve requested without a transpose/adjoint operator.")
        return operator_apply_midpoint(
            transpose_operator,
            lift_vector(v),
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )

    if use_implicit_adjoint:
        transpose_solve = None if transpose_structured else (
            lambda matvec_fn, rhs_value: solve_impl(transpose_mv, rhs_value, preconditioner_plan=transpose_preconditioner)
        )
        x_mid, info = lax.custom_linear_solve(
            mv,
            rhs_mid,
            lambda matvec_fn, rhs_value: solve_impl(matvec_fn, rhs_value, preconditioner_plan=preconditioner),
            transpose_solve=transpose_solve,
            symmetric=transpose_structured,
            has_aux=True,
        )
    else:
        x_mid, info = solve_impl(mv, rhs_mid, preconditioner_plan=preconditioner)

    residual = jnp.linalg.norm(mv(x_mid) - rhs_mid)
    metadata = ImplicitAdjointSolveMetadata(
        operator=operator,
        transpose_operator=transpose_operator,
        preconditioner=preconditioner,
        transpose_preconditioner=transpose_preconditioner,
        solver=solver,
        structured=structured,
        algebra=getattr(operator, "algebra", "matrix_free"),
        implicit_adjoint=bool(use_implicit_adjoint),
    )
    return x_mid, info, residual, jnp.linalg.norm(rhs_mid), metadata


def krylov_diagnostics(
    diagnostics_type,
    *,
    algorithm_code: int,
    steps: int | jax.Array,
    basis_dim: int | jax.Array,
    beta0,
    tail_norm,
    breakdown,
    used_adjoint: bool | jax.Array = False,
    gradient_supported: bool | jax.Array = True,
    probe_count: int | jax.Array = 1,
    restart_count: int | jax.Array = 0,
    primal_residual=0.0,
    adjoint_residual=0.0,
    regime_code_value=-1,
    method_code_value=-1,
    solver_code_value=-1,
    structure_code_value=-1,
    residual_history=None,
    deflated_count: int | jax.Array = 0,
):
    if residual_history is None:
        residual_history = jnp.asarray([tail_norm], dtype=jnp.float64)
    return diagnostics_type(
        algorithm_code=jnp.asarray(algorithm_code, dtype=jnp.int32),
        steps=jnp.asarray(steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(basis_dim, dtype=jnp.int32),
        restart_count=jnp.asarray(restart_count, dtype=jnp.int32),
        beta0=jnp.asarray(beta0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(breakdown),
        used_adjoint=jnp.asarray(used_adjoint),
        gradient_supported=jnp.asarray(gradient_supported),
        probe_count=jnp.asarray(probe_count, dtype=jnp.int32),
        primal_residual=jnp.asarray(primal_residual, dtype=jnp.float64),
        adjoint_residual=jnp.asarray(adjoint_residual, dtype=jnp.float64),
        regime_code=jnp.asarray(regime_code_value, dtype=jnp.int32),
        method_code=jnp.asarray(method_code_value, dtype=jnp.int32),
        solver_code=jnp.asarray(solver_code_value, dtype=jnp.int32),
        structure_code=jnp.asarray(structure_code_value, dtype=jnp.int32),
        residual_history=jnp.asarray(residual_history, dtype=jnp.float64),
        deflated_count=jnp.asarray(deflated_count, dtype=jnp.int32),
    )


def dense_funm_hermitian_eigh(scalar_fun, *, dtype, conjugate_right: bool):
    def apply(matrix: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eigh(jnp.asarray(matrix, dtype=dtype))
        right = jnp.conjugate(evecs).T if conjugate_right else evecs.T
        return evecs @ jnp.diag(scalar_fun(evals)) @ right

    return apply


def dense_funm_general_eig(scalar_fun, *, dtype):
    def apply(matrix: jax.Array) -> jax.Array:
        vals, vecs = jnp.linalg.eig(jnp.asarray(matrix, dtype=dtype))
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(scalar_fun(vals)) @ inv

    return apply


def krylov_step_bucket(steps: int, max_dim: int) -> int:
    resolved_steps = max(1, int(steps))
    resolved_dim = max(1, int(max_dim))
    bucket = 1
    while bucket < resolved_steps and bucket < resolved_dim:
        bucket <<= 1
    return min(bucket, resolved_dim)


def projected_krylov_action_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
    *,
    krylov_decomp,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    coeff_dtype,
):
    basis, projected, beta0 = krylov_decomp(matvec, x, steps)
    basis_norms = jnp.linalg.norm(basis, axis=-1)
    active = jnp.sum(basis_norms > jnp.asarray(1e-30, dtype=basis_norms.dtype))
    active_mask = (jnp.arange(steps, dtype=jnp.int32) < active).astype(projected.dtype)
    masked_basis = basis * active_mask[:, None]
    masked_projected = projected * (active_mask[:, None] * active_mask[None, :]) + jnp.diag(1.0 - active_mask)
    e1 = jnp.zeros((steps,), dtype=coeff_dtype).at[0].set(jnp.asarray(1, dtype=coeff_dtype))
    y = beta0 * (masked_basis.T @ (dense_funm(masked_projected) @ e1))
    out = point_from_midpoint(y)
    finite = finite_mask_fn(y)
    return jnp.where(finite[..., None], out, full_like(out))


def projected_krylov_integrand_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
    *,
    krylov_decomp,
    coeff_dtype,
    scalar_dtype,
    scalar_postprocess,
):
    basis, projected, beta0 = krylov_decomp(matvec, x, steps)
    basis_norms = jnp.linalg.norm(basis, axis=-1)
    active = jnp.sum(basis_norms > jnp.asarray(1e-30, dtype=basis_norms.dtype))
    active_mask = (jnp.arange(steps, dtype=jnp.int32) < active).astype(projected.dtype)
    masked_projected = projected * (active_mask[:, None] * active_mask[None, :]) + jnp.diag(1.0 - active_mask)
    e1 = jnp.zeros((steps,), dtype=coeff_dtype).at[0].set(jnp.asarray(1, dtype=coeff_dtype))
    value = (beta0**2) * jnp.vdot(e1, dense_funm(masked_projected) @ e1)
    return jnp.asarray(scalar_postprocess(value), dtype=scalar_dtype)


def orthonormalize_columns(block: jax.Array) -> jax.Array:
    q, _ = jnp.linalg.qr(jnp.asarray(block), mode="reduced")
    return q


def select_eigen_indices(evals: jax.Array, k: int, which: str) -> jax.Array:
    code = which.lower()
    if code in {"largest", "la", "lm"}:
        return jnp.arange(evals.shape[0] - k, evals.shape[0], dtype=jnp.int32)
    if code in {"smallest", "sa", "sm"}:
        return jnp.arange(0, k, dtype=jnp.int32)
    raise ValueError("which must be one of {'largest', 'smallest', 'la', 'sa', 'lm', 'sm'}")


def eig_locked_mask_from_residuals(residuals: jax.Array, *, tol: float) -> jax.Array:
    return jnp.linalg.norm(residuals, axis=0) <= jnp.asarray(tol, dtype=jnp.float64)


def eig_restart_lock_tolerance(*, steps: int, restarts: int = 1, floor: float = 1e-10) -> float:
    return float(max(floor, 1e-2 / max(int(steps) * max(int(restarts), 1), 1)))


def eig_restart_column_order(evals: jax.Array, residuals: jax.Array, *, which: str, lock_tol: float) -> jax.Array:
    residual_norms = jnp.linalg.norm(residuals, axis=0)
    locked_mask = eig_locked_mask_from_residuals(residuals, tol=lock_tol)
    code = which.lower()
    spectral_key = jnp.real(jnp.asarray(evals))
    if code in {"largest", "la", "lm"}:
        spectral_key = -spectral_key
    elif code not in {"smallest", "sa", "sm"}:
        raise ValueError("which must be one of {'largest', 'smallest', 'la', 'sa', 'lm', 'sm'}")
    primary = jnp.where(locked_mask, 0.0, 1.0)
    secondary = jnp.where(locked_mask, residual_norms, spectral_key)
    tertiary = jnp.where(locked_mask, 0.0, residual_norms)
    return jnp.lexsort((tertiary, secondary, primary))


def eig_expansion_column_order(evals: jax.Array, residuals: jax.Array, *, which: str, lock_tol: float) -> jax.Array:
    residual_norms = jnp.linalg.norm(residuals, axis=0)
    locked_mask = eig_locked_mask_from_residuals(residuals, tol=lock_tol)
    code = which.lower()
    spectral_key = jnp.real(jnp.asarray(evals))
    if code in {"largest", "la", "lm"}:
        spectral_key = -spectral_key
    elif code not in {"smallest", "sa", "sm"}:
        raise ValueError("which must be one of {'largest', 'smallest', 'la', 'sa', 'lm', 'sm'}")
    primary = locked_mask.astype(jnp.int32)
    secondary = jnp.where(locked_mask, 0.0, -residual_norms)
    tertiary = jnp.where(locked_mask, 0.0, spectral_key)
    return jnp.lexsort((tertiary, secondary, primary))


def eig_restart_basis_from_pairs(
    vecs: jax.Array,
    evals: jax.Array,
    residuals: jax.Array,
    *,
    target_cols: int,
    which: str,
    lock_tol: float,
    refill_basis: jax.Array | None = None,
):
    locked_mask = eig_locked_mask_from_residuals(residuals, tol=lock_tol)
    residual_norms = jnp.linalg.norm(residuals, axis=0)
    code = which.lower()
    spectral_key = jnp.real(jnp.asarray(evals))
    if code in {"largest", "la", "lm"}:
        spectral_key = -spectral_key
    elif code not in {"smallest", "sa", "sm"}:
        raise ValueError("which must be one of {'largest', 'smallest', 'la', 'sa', 'lm', 'sm'}")
    primary = jnp.where(locked_mask, 0.0, 1.0)
    secondary = jnp.where(locked_mask, residual_norms, spectral_key)
    tertiary = jnp.where(locked_mask, 0.0, residual_norms)
    order = jnp.lexsort((tertiary, secondary, primary))
    keep = min(int(target_cols), int(vecs.shape[1]))
    pieces = [vecs[:, order[:keep]]]
    if keep < target_cols and refill_basis is not None:
        pieces.append(refill_basis[:, : target_cols - keep])
    basis = orthonormalize_columns(jnp.concatenate(pieces, axis=1))
    return basis[:, :target_cols]


def eig_convergence_summary(
    residuals: jax.Array,
    *,
    tol: float,
    requested: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    residuals_arr = jnp.asarray(residuals, dtype=jnp.float64)
    if residuals_arr.size == 0:
        zero = jnp.asarray(0, dtype=jnp.int32)
        return zero, zero, zero, jnp.asarray(False)
    converged_mask = residuals_arr <= jnp.asarray(tol, dtype=jnp.float64)
    converged_count = jnp.sum(converged_mask.astype(jnp.int32))
    requested_arr = jnp.asarray(requested, dtype=jnp.int32)
    locked_count = jnp.minimum(converged_count, requested_arr)
    deflated_count = converged_count
    return converged_count, locked_count, deflated_count, jnp.asarray(converged_count >= requested_arr)


def ritz_pairs_from_basis(apply_block, basis: jax.Array, *, k: int, which: str, hermitian: bool = True):
    basis = orthonormalize_columns(basis)
    applied = apply_block(basis)
    projected = jnp.conjugate(basis).T @ applied
    if hermitian:
        evals, coeffs = jnp.linalg.eigh(projected)
    else:
        evals, coeffs = jnp.linalg.eig(projected)
    indices = select_eigen_indices(evals, k, which)
    selected_vals = evals[indices]
    vectors = basis @ coeffs[:, indices]
    norms = jnp.maximum(jnp.linalg.norm(vectors, axis=0), jnp.asarray(1e-30, dtype=jnp.float64))
    return selected_vals, vectors / norms[None, :]


def block_subspace_iteration_point(apply_block, basis: jax.Array, *, subspace_iters: int) -> jax.Array:
    if subspace_iters <= 0:
        raise ValueError("subspace_iters must be > 0")
    basis = orthonormalize_columns(basis)

    def body(q, _):
        return orthonormalize_columns(apply_block(q)), None

    basis, _ = lax.scan(body, basis, xs=None, length=int(subspace_iters))
    return basis


def restarted_subspace_iteration_point(
    apply_block,
    basis: jax.Array,
    *,
    subspace_iters: int,
    restarts: int,
    k: int,
    which: str,
    hermitian: bool = True,
) -> jax.Array:
    if restarts <= 0:
        raise ValueError("restarts must be > 0")
    basis = orthonormalize_columns(basis)

    def restart_body(q, _):
        q_iter = block_subspace_iteration_point(apply_block, q, subspace_iters=subspace_iters)
        _, vecs = ritz_pairs_from_basis(apply_block, q_iter, k=k, which=which, hermitian=hermitian)
        next_q = orthonormalize_columns(vecs)
        target_cols = q.shape[1]
        if next_q.shape[1] < target_cols:
            pad = q[:, : target_cols - next_q.shape[1]]
            next_q = orthonormalize_columns(jnp.concatenate([next_q, pad], axis=1))
        return next_q[:, :target_cols], None

    basis, _ = lax.scan(restart_body, basis, xs=None, length=int(restarts))
    return basis


def contour_quadrature_nodes(center, radius, *, quadrature_order: int):
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be > 0")
    theta = (2.0 * jnp.pi / quadrature_order) * (jnp.arange(quadrature_order, dtype=jnp.float64) + 0.5)
    unit = jnp.exp(1j * theta)
    center_arr = jnp.asarray(center, dtype=jnp.complex128)
    radius_arr = jnp.asarray(radius, dtype=jnp.complex128)
    nodes = center_arr + radius_arr * unit
    weights = (radius_arr * unit) / jnp.asarray(quadrature_order, dtype=jnp.complex128)
    return nodes, weights


def contour_filter_subspace_point(
    solve_shifted_block,
    basis: jax.Array,
    *,
    center,
    radius,
    quadrature_order: int,
) -> jax.Array:
    nodes, weights = contour_quadrature_nodes(center, radius, quadrature_order=quadrature_order)
    init = jnp.zeros_like(jnp.asarray(basis), dtype=jnp.complex128)

    def body(acc, nw):
        node, weight = nw
        return acc + weight * solve_shifted_block(node, basis), None

    filtered, _ = lax.scan(body, init, (nodes, weights))
    return orthonormalize_columns(filtered)


def contour_integral_action_point(
    solve_shifted,
    x: jax.Array,
    *,
    center,
    radius,
    quadrature_order: int,
    node_weight_fn=None,
) -> jax.Array:
    nodes, weights = contour_quadrature_nodes(center, radius, quadrature_order=quadrature_order)
    vector = jnp.asarray(x)
    out_dtype = jnp.result_type(vector.dtype, jnp.complex128)
    init = jnp.zeros_like(vector, dtype=out_dtype)

    def body(acc, nw):
        node, weight = nw
        kernel = jnp.asarray(1.0 if node_weight_fn is None else node_weight_fn(node), dtype=out_dtype)
        value = jnp.asarray(solve_shifted(node, vector), dtype=out_dtype)
        return acc + jnp.asarray(weight, dtype=out_dtype) * kernel * value, None

    value, _ = lax.scan(body, init, (nodes, weights))
    return value


def polynomial_spectral_action_midpoint(
    apply_operator,
    x_mid: jax.Array,
    coefficients: jax.Array,
    *,
    coeff_dtype,
) -> jax.Array:
    coeffs = jnp.asarray(coefficients, dtype=coeff_dtype)
    if coeffs.ndim != 1:
        raise ValueError("coefficients must be rank-1")
    x_arr = jnp.asarray(x_mid)
    out_dtype = jnp.result_type(x_arr.dtype, coeffs.dtype)
    init = (jnp.asarray(x_arr, dtype=out_dtype), jnp.zeros_like(x_arr, dtype=out_dtype))

    def step(carry, coeff):
        term, acc = carry
        next_acc = acc + coeff * term
        next_term = jnp.asarray(apply_operator(term), dtype=out_dtype)
        return (next_term, next_acc), None

    (_, acc), _ = lax.scan(step, init, coeffs)
    return acc


def rational_spectral_action_midpoint(
    apply_operator,
    solve_shifted,
    x_mid: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    coeff_dtype,
) -> jax.Array:
    x_arr = jnp.asarray(x_mid)
    shifts_arr = jnp.asarray(shifts)
    weights_arr = jnp.asarray(weights, dtype=coeff_dtype)
    if shifts_arr.ndim != 1 or weights_arr.ndim != 1 or shifts_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("shifts and weights must be rank-1 with matching length")
    out_dtype = jnp.result_type(x_arr.dtype, weights_arr.dtype, shifts_arr.dtype)
    acc = jnp.zeros_like(x_arr, dtype=out_dtype)
    if polynomial_coefficients is not None:
        acc = acc + jnp.asarray(
            polynomial_spectral_action_midpoint(
                apply_operator,
                x_arr,
                polynomial_coefficients,
                coeff_dtype=coeff_dtype,
            ),
            dtype=out_dtype,
        )

    def body(carry, sw):
        shift, weight = sw
        resolved = jnp.asarray(solve_shifted(shift, x_arr), dtype=out_dtype)
        return carry + jnp.asarray(weight, dtype=out_dtype) * resolved, None

    value, _ = lax.scan(body, acc, (shifts_arr, weights_arr))
    return value


def complexify_real_linear_operator(real_apply, v: jax.Array) -> jax.Array:
    vv = jnp.asarray(v, dtype=jnp.complex128)
    re = jnp.asarray(real_apply(jnp.real(vv)), dtype=jnp.float64)
    im = jnp.asarray(real_apply(jnp.imag(vv)), dtype=jnp.float64)
    return re + 1j * im


def dense_operator(mid: jax.Array, *, midpoint_vector, orientation: str = "forward"):
    plan = dense_operator_plan(mid, orientation=orientation, algebra="matrix_free")

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(
            plan,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=lambda *_args, **_kwargs: None,
            dtype=plan.payload.dtype,
        )

    return matvec


def dense_operator_adjoint(mid: jax.Array, *, midpoint_vector, conjugate: bool):
    return dense_operator(mid, midpoint_vector=midpoint_vector, orientation="adjoint" if conjugate else "transpose")


def dense_operator_rmatvec(mid: jax.Array, *, midpoint_vector):
    return dense_operator(mid, midpoint_vector=midpoint_vector, orientation="transpose")


def sparse_bcoo_operator(a, *, as_sparse_bcoo, sparse_bcoo_cls, sparse_bcoo_matvec, midpoint_vector, dtype, algebra: str, label: str):
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation="forward",
        algebra=algebra,
    )

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(plan, v, midpoint_vector=midpoint_vector, sparse_bcoo_matvec=sparse_bcoo_matvec, dtype=dtype)

    return matvec


def sparse_bcoo_operator_adjoint(
    a,
    *,
    as_sparse_bcoo,
    sparse_bcoo_cls,
    sparse_bcoo_matvec,
    midpoint_vector,
    dtype,
    algebra: str,
    label: str,
    conjugate: bool,
):
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation="adjoint" if conjugate else "transpose",
        algebra=algebra,
        conjugate_transpose=conjugate,
    )

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(plan, v, midpoint_vector=midpoint_vector, sparse_bcoo_matvec=sparse_bcoo_matvec, dtype=dtype)

    return matvec


def sparse_bcoo_operator_rmatvec(
    a,
    *,
    as_sparse_bcoo,
    sparse_bcoo_cls,
    sparse_bcoo_matvec,
    midpoint_vector,
    dtype,
    algebra: str,
    label: str,
):
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation="transpose",
        algebra=algebra,
    )

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(plan, v, midpoint_vector=midpoint_vector, sparse_bcoo_matvec=sparse_bcoo_matvec, dtype=dtype)

    return matvec


def operator_apply_point(
    operator,
    x: jax.Array,
    *,
    midpoint_apply,
    coerce_vector,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    dtype,
) -> jax.Array:
    x = coerce_vector(x)
    y = midpoint_apply(operator, x)
    out = point_from_midpoint(y)
    finite = finite_mask_fn(y)
    return jnp.where(finite[..., None, None], out, full_like(out))


def poly_action_point(
    operator,
    x: jax.Array,
    coefficients: jax.Array,
    *,
    midpoint_apply,
    coerce_vector,
    midpoint_vector,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    coeff_dtype,
):
    x = coerce_vector(x)
    coeffs = jnp.asarray(coefficients, dtype=coeff_dtype)
    x_mid = midpoint_vector(x)

    def step(carry, coeff):
        term, acc = carry
        next_acc = acc + coeff * term
        next_term = midpoint_apply(operator, point_from_midpoint(term))
        return (next_term, next_acc), None

    if coeffs.ndim != 1:
        raise ValueError("coefficients must be rank-1")
    init = (x_mid, jnp.zeros_like(x_mid))
    (_, acc), _ = lax.scan(step, init, coeffs)
    out = point_from_midpoint(acc)
    finite = finite_mask_fn(acc)
    return jnp.where(finite[..., None, None], out, full_like(out))


def expm_action_point(
    operator,
    x: jax.Array,
    *,
    terms: int,
    midpoint_apply,
    coerce_vector,
    midpoint_vector,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    scalar_dtype,
):
    x = coerce_vector(x)
    if terms <= 0:
        raise ValueError("terms must be > 0")
    x_mid = midpoint_vector(x)

    def step(carry, k):
        term, acc = carry
        next_term = midpoint_apply(operator, point_from_midpoint(term)) / jnp.asarray(k, dtype=scalar_dtype)
        next_acc = acc + next_term
        return (next_term, next_acc), None

    init = (x_mid, x_mid)
    (_, acc), _ = lax.scan(step, init, jnp.arange(1, terms, dtype=jnp.int32))
    out = point_from_midpoint(acc)
    finite = finite_mask_fn(acc)
    return jnp.where(finite[..., None, None], out, full_like(out))


def restarted_action_point(apply_once, x: jax.Array, *, restarts: int):
    if restarts <= 0:
        raise ValueError("restarts must be > 0")

    def body(y, _):
        next_y = apply_once(y)
        return next_y, None

    y, _ = lax.scan(body, x, xs=None, length=restarts)
    return y


def block_action_point(apply_one, xs: jax.Array) -> jax.Array:
    return jax.vmap(apply_one)(xs)


def rademacher_probes_real(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    mids = jax.random.rademacher(key, shape=(num, length), dtype=jnp.float64)
    return jax.vmap(point_from_midpoint)(mids)


def normal_probes_real(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    mids = jax.random.normal(key, shape=(num, length), dtype=jnp.float64)
    return jax.vmap(point_from_midpoint)(mids)


def rademacher_probes_complex(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    key_re, key_im = jax.random.split(key)
    re = jax.random.rademacher(key_re, shape=(num, length), dtype=jnp.float64)
    im = jax.random.rademacher(key_im, shape=(num, length), dtype=jnp.float64)
    return jax.vmap(point_from_midpoint)(re + 1j * im)


def normal_probes_complex(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    key_re, key_im = jax.random.split(key)
    re = jax.random.normal(key_re, shape=(num, length), dtype=jnp.float64)
    im = jax.random.normal(key_im, shape=(num, length), dtype=jnp.float64)
    mids = (re + 1j * im) / jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return jax.vmap(point_from_midpoint)(mids)


def orthogonal_rademacher_probe_block_real(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    if num <= 0:
        raise ValueError("num must be > 0")
    if num > length:
        raise ValueError("num must be <= length for orthogonal real probe blocks")
    samples = jax.random.rademacher(key, shape=(length, num), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(samples, mode="reduced")
    mids = q.T
    return jax.vmap(point_from_midpoint)(mids)


def orthogonal_normal_probe_block_real(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    if num <= 0:
        raise ValueError("num must be > 0")
    if num > length:
        raise ValueError("num must be <= length for orthogonal real probe blocks")
    samples = jax.random.normal(key, shape=(length, num), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(samples, mode="reduced")
    mids = q.T
    return jax.vmap(point_from_midpoint)(mids)


def orthogonal_rademacher_probe_block_complex(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    if num <= 0:
        raise ValueError("num must be > 0")
    if num > length:
        raise ValueError("num must be <= length for orthogonal complex probe blocks")
    key_re, key_im = jax.random.split(key)
    re = jax.random.rademacher(key_re, shape=(length, num), dtype=jnp.float64)
    im = jax.random.rademacher(key_im, shape=(length, num), dtype=jnp.float64)
    samples = (re + 1j * im) / jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    q, _ = jnp.linalg.qr(samples, mode="reduced")
    mids = q.T
    return jax.vmap(point_from_midpoint)(mids)


def orthogonal_normal_probe_block_complex(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    if num <= 0:
        raise ValueError("num must be > 0")
    if num > length:
        raise ValueError("num must be <= length for orthogonal complex probe blocks")
    key_re, key_im = jax.random.split(key)
    re = jax.random.normal(key_re, shape=(length, num), dtype=jnp.float64)
    im = jax.random.normal(key_im, shape=(length, num), dtype=jnp.float64)
    samples = (re + 1j * im) / jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    q, _ = jnp.linalg.qr(samples, mode="reduced")
    mids = q.T
    return jax.vmap(point_from_midpoint)(mids)


def probe_sample_statistics(samples: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    values = jnp.asarray(samples)
    if values.ndim == 0:
        raise ValueError("samples must have at least one probe axis")
    count = values.shape[0]
    if count <= 0:
        raise ValueError("samples must contain at least one probe")
    mean = jnp.mean(values, axis=0)
    centered = values - mean
    sq_norm = jnp.real(centered * jnp.conjugate(centered))
    variance = jnp.mean(sq_norm, axis=0) if count == 1 else jnp.sum(sq_norm, axis=0) / jnp.asarray(count - 1, dtype=jnp.float64)
    stderr = jnp.sqrt(variance / jnp.asarray(count, dtype=jnp.float64))
    return mean, variance, stderr


def make_probe_estimate_statistics(
    samples: jax.Array,
    *,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> ProbeEstimateStatistics:
    values = jnp.asarray(samples)
    mean, variance, stderr = probe_sample_statistics(values)
    probe_count = jnp.asarray(values.shape[0], dtype=jnp.int32)
    if target_stderr is None:
        recommended = probe_count
    else:
        recommended = adaptive_probe_count_from_pilot(
            values,
            target_stderr=target_stderr,
            min_probes=min_probes,
            max_probes=max_probes,
            block_size=block_size,
        )
    return ProbeEstimateStatistics(
        mean=mean,
        variance=variance,
        stderr=stderr,
        probe_count=probe_count,
        recommended_probe_count=jnp.asarray(recommended, dtype=jnp.int32),
    )


def adaptive_probe_count_from_pilot(
    pilot_samples: jax.Array,
    *,
    target_stderr: float,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> jax.Array:
    values = jnp.asarray(pilot_samples)
    if values.ndim == 0:
        raise ValueError("pilot_samples must have a probe axis")
    count = int(values.shape[0])
    if count <= 0:
        raise ValueError("pilot_samples must contain at least one probe")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    _, _, stderr = probe_sample_statistics(values)
    stderr_scalar = jnp.max(jnp.asarray(stderr, dtype=jnp.float64))
    target = jnp.maximum(jnp.asarray(target_stderr, dtype=jnp.float64), jnp.asarray(1e-30, dtype=jnp.float64))
    required = jnp.ceil((stderr_scalar / target) ** 2 * jnp.asarray(count, dtype=jnp.float64)).astype(jnp.int32)
    required = jnp.maximum(required, jnp.asarray(count if min_probes is None else min_probes, dtype=jnp.int32))
    if max_probes is not None:
        required = jnp.minimum(required, jnp.asarray(max_probes, dtype=jnp.int32))
    block = jnp.asarray(block_size, dtype=jnp.int32)
    rounded = block * ((required + block - 1) // block)
    if max_probes is not None:
        rounded = jnp.minimum(rounded, jnp.asarray(max_probes, dtype=jnp.int32))
    return rounded


def probe_statistics_target_met(
    statistics: ProbeEstimateStatistics,
    *,
    target_stderr: float,
) -> jax.Array:
    target = jnp.asarray(target_stderr, dtype=jnp.float64)
    return jnp.asarray(jnp.max(jnp.asarray(statistics.stderr, dtype=jnp.float64)) <= target)


def probe_statistics_should_stop(
    statistics: ProbeEstimateStatistics,
    *,
    target_stderr: float,
    max_probes: int | None = None,
) -> jax.Array:
    met = probe_statistics_target_met(statistics, target_stderr=target_stderr)
    if max_probes is None:
        return met
    return jnp.asarray(met | (jnp.asarray(statistics.probe_count, dtype=jnp.int32) >= jnp.asarray(max_probes, dtype=jnp.int32)))


def expand_subspace_with_corrections(
    basis: jax.Array,
    vecs: jax.Array,
    residuals: jax.Array,
    *,
    orthonormalize_columns_fn,
    apply_preconditioner=None,
    vals: jax.Array | None = None,
    target_cols: int,
    which: str = "largest",
    lock_tol: float = 1e-4,
    jacobi_davidson: bool = False,
    conjugate_inner: bool = False,
) -> jax.Array:
    max_new_cols = max(0, min(int(target_cols) - int(basis.shape[1]), int(residuals.shape[1])))
    if vals is not None and max_new_cols > 0:
        order = eig_expansion_column_order(vals, residuals, which=which, lock_tol=lock_tol)
        chosen = order[:max_new_cols]
        vecs = vecs[:, chosen]
        residuals = residuals[:, chosen]
    corrections = residuals
    residual_norms = jnp.linalg.norm(corrections, axis=0, keepdims=True)
    safe_norms = jnp.where(residual_norms > 1e-12, residual_norms, 1.0)
    corrections = corrections / safe_norms
    if apply_preconditioner is not None:
        corrections = jax.vmap(apply_preconditioner, in_axes=1, out_axes=1)(corrections)
    if jacobi_davidson:
        if conjugate_inner:
            coeffs = jnp.sum(jnp.conj(vecs) * corrections, axis=0, keepdims=True)
        else:
            coeffs = jnp.sum(vecs * corrections, axis=0, keepdims=True)
        projected = corrections - vecs * coeffs
        proj_norms = jnp.linalg.norm(projected, axis=0, keepdims=True)
        corrections = jnp.where(
            proj_norms > 1e-12,
            projected / jnp.where(proj_norms > 1e-12, proj_norms, 1.0),
            0.0,
        )
    correction_norms = jnp.linalg.norm(corrections, axis=0, keepdims=True)
    corrections = jnp.where(correction_norms > 1e-10, corrections, 0.0)
    trial = jnp.concatenate([basis, corrections], axis=1)
    basis_next = orthonormalize_columns_fn(trial)
    if basis_next.shape[1] < target_cols:
        pad = basis[:, : target_cols - basis_next.shape[1]]
        basis_next = orthonormalize_columns_fn(jnp.concatenate([basis_next, pad], axis=1))
    return basis_next[:, :target_cols]


def apply_action_over_probe_block_point(
    action_fn,
    probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
) -> jax.Array:
    coerced = coerce_probes(probes)
    outputs = jax.vmap(action_fn)(coerced)
    return midpoint_value(outputs)


def hutchpp_trace_with_metadata_projected_point(
    action_fn,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
    point_from_midpoint,
    basis_dtype,
    trace_inner,
    residual_project,
    quadratic_reduce,
    zero_scalar,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> HutchppTraceMetadata:
    sketch = coerce_probes(sketch_probes)
    residual = coerce_probes(residual_probes)
    n = int(sketch.shape[-2] if sketch.shape[0] > 0 else residual.shape[-2])

    if sketch.shape[0] > 0:
        y_cols = jnp.swapaxes(
            apply_action_over_probe_block_point(
                action_fn,
                sketch,
                coerce_probes=coerce_probes,
                midpoint_value=midpoint_value,
            ),
            0,
            1,
        )
        q, _ = jnp.linalg.qr(y_cols, mode="reduced")
        fq_cols = jnp.swapaxes(
            apply_action_over_probe_block_point(
                action_fn,
                jax.vmap(point_from_midpoint)(q.T),
                coerce_probes=coerce_probes,
                midpoint_value=midpoint_value,
            ),
            0,
            1,
        )
        trace_lr = trace_inner(q, fq_cols)
    else:
        q = jnp.zeros((n, 0), dtype=basis_dtype)
        trace_lr = zero_scalar

    if residual.shape[0] > 0:
        z = midpoint_value(residual)
        z_proj = residual_project(z, q)
        hz = apply_action_over_probe_block_point(
            action_fn,
            jax.vmap(point_from_midpoint)(z_proj),
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        )
        residual_samples = quadratic_reduce(z_proj, hz)
    else:
        residual_samples = jnp.zeros((1,), dtype=jnp.asarray(zero_scalar).dtype)

    return make_hutchpp_trace_metadata(
        basis=q,
        low_rank_trace=trace_lr,
        residual_samples=residual_samples,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def make_deflated_operator_metadata(
    *,
    basis: jax.Array,
    image: jax.Array,
    low_rank_trace,
) -> DeflatedOperatorMetadata:
    return DeflatedOperatorMetadata(
        basis=jnp.asarray(basis),
        image=jnp.asarray(image),
        low_rank_trace=low_rank_trace,
    )


def prepare_deflated_operator_metadata_point(
    action_fn,
    sketch_probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
    point_from_midpoint,
    basis_dtype,
    trace_inner,
) -> DeflatedOperatorMetadata:
    sketch = coerce_probes(sketch_probes)
    n = int(sketch.shape[-2])
    if sketch.shape[0] == 0:
        empty = jnp.zeros((n, 0), dtype=basis_dtype)
        return make_deflated_operator_metadata(
            basis=empty,
            image=empty,
            low_rank_trace=jnp.asarray(0.0, dtype=basis_dtype),
        )
    y_cols = jnp.swapaxes(
        apply_action_over_probe_block_point(
            action_fn,
            sketch,
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        ),
        0,
        1,
    )
    q, _ = jnp.linalg.qr(y_cols, mode="reduced")
    aq = jnp.swapaxes(
        apply_action_over_probe_block_point(
            action_fn,
            jax.vmap(point_from_midpoint)(q.T),
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        ),
        0,
        1,
    )
    return make_deflated_operator_metadata(
        basis=q,
        image=aq,
        low_rank_trace=trace_inner(q, aq),
    )


def deflated_operator_apply_midpoint(
    x_mid: jax.Array,
    *,
    deflation: DeflatedOperatorMetadata,
    apply_operator_midpoint,
    conjugate_inner: bool = False,
) -> jax.Array:
    x_arr = jnp.asarray(x_mid)
    ax = jnp.asarray(apply_operator_midpoint(x_arr))
    basis = jnp.asarray(deflation.basis)
    if basis.shape[1] == 0:
        return ax
    if conjugate_inner:
        coeffs = jnp.conjugate(basis).T @ x_arr
    else:
        coeffs = basis.T @ x_arr
    return ax - jnp.asarray(deflation.image) @ coeffs


def deflated_trace_estimate_from_metadata_point(
    action_fn,
    deflation: DeflatedOperatorMetadata,
    residual_probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
    point_from_midpoint,
    residual_project,
    quadratic_reduce,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> HutchppTraceMetadata:
    residual = coerce_probes(residual_probes)
    if residual.shape[0] > 0:
        z = midpoint_value(residual)
        z_proj = residual_project(z, jnp.asarray(deflation.basis))
        hz = apply_action_over_probe_block_point(
            lambda probe: point_from_midpoint(
                deflated_operator_apply_midpoint(
                    midpoint_value(probe),
                    deflation=deflation,
                    apply_operator_midpoint=lambda v_mid: midpoint_value(action_fn(point_from_midpoint(v_mid))),
                    conjugate_inner=jnp.iscomplexobj(jnp.asarray(deflation.basis)),
                )
            ),
            jax.vmap(point_from_midpoint)(z_proj),
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        )
        residual_samples = quadratic_reduce(z_proj, hz)
    else:
        residual_samples = jnp.zeros((1,), dtype=jnp.asarray(deflation.low_rank_trace).dtype)
    return make_hutchpp_trace_metadata(
        basis=deflation.basis,
        low_rank_trace=deflation.low_rank_trace,
        residual_samples=residual_samples,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def slq_nodes_weights(projected: jax.Array, beta0, *, hermitian: bool) -> tuple[jax.Array, jax.Array]:
    projected_arr = jnp.asarray(projected)
    beta0_arr = jnp.asarray(beta0)
    if hermitian:
        nodes, vecs = jnp.linalg.eigh(projected_arr)
        first_row = vecs[0, :]
        weights = (jnp.real(beta0_arr * jnp.conjugate(beta0_arr)) * jnp.real(first_row * jnp.conjugate(first_row))).astype(jnp.float64)
        return jnp.asarray(nodes), weights
    nodes, vecs = jnp.linalg.eig(projected_arr)
    first_row = vecs[0, :]
    weights = (beta0_arr * jnp.conjugate(beta0_arr)) * (first_row * jnp.conjugate(first_row))
    return jnp.asarray(nodes), jnp.asarray(weights)


def slq_scalar_quadrature(nodes: jax.Array, weights: jax.Array, scalar_fun, *, scalar_postprocess=None):
    values = weights * scalar_fun(nodes)
    total = jnp.sum(values)
    return total if scalar_postprocess is None else scalar_postprocess(total)


def slq_heat_trace(nodes: jax.Array, weights: jax.Array, time):
    time_value = jnp.asarray(time)
    return slq_scalar_quadrature(
        nodes,
        weights,
        lambda vals: jnp.exp(-time_value * vals),
    )


def slq_spectral_density(nodes: jax.Array, weights: jax.Array, bin_edges: jax.Array, *, normalize: bool = False) -> jax.Array:
    edges = jnp.asarray(bin_edges)
    vals = jnp.asarray(nodes)
    coeffs = jnp.real(jnp.asarray(weights, dtype=jnp.complex128))
    left = vals[:, None] >= edges[:-1][None, :]
    right = vals[:, None] < edges[1:][None, :]
    last_bin = vals[:, None] == edges[-1][None, :]
    mask = (left & right) | (last_bin & (jnp.arange(edges.shape[0] - 1) == edges.shape[0] - 2)[None, :])
    hist = jnp.sum(jnp.where(mask, coeffs[:, None], 0.0), axis=0)
    if normalize:
        total = jnp.sum(hist)
        hist = jnp.where(total > 0, hist / total, hist)
    return hist


def slq_functional_statistics_from_metadata(
    metadata: SlqQuadratureMetadata,
    scalar_function,
) -> ProbeEstimateStatistics:
    values = jax.vmap(lambda nodes, weights: scalar_function(nodes, weights))(metadata.nodes, metadata.weights)
    return make_probe_estimate_statistics(values)


def slq_functional_mean_from_metadata(
    metadata: SlqQuadratureMetadata,
    scalar_function,
):
    return slq_functional_statistics_from_metadata(metadata, scalar_function).mean


def slq_heat_trace_from_metadata(metadata: SlqQuadratureMetadata, time):
    return slq_functional_mean_from_metadata(
        metadata,
        lambda nodes, weights: slq_heat_trace(nodes, weights, time),
    )


def slq_spectral_density_from_metadata(
    metadata: SlqQuadratureMetadata,
    bin_edges: jax.Array,
    *,
    normalize: bool = False,
) -> jax.Array:
    return jnp.mean(
        jax.vmap(lambda nodes, weights: slq_spectral_density(nodes, weights, bin_edges, normalize=normalize))(metadata.nodes, metadata.weights),
        axis=0,
    )


def hutchpp_trace_from_metadata(metadata: HutchppTraceMetadata):
    return metadata.low_rank_trace + metadata.residual_trace


def slq_prepare_metadata_point(
    lanczos_tridiag,
    probes: jax.Array,
    steps: int,
    *,
    coerce_probes,
    hermitian: bool,
    scalar_fun,
    scalar_postprocess=None,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> SlqQuadratureMetadata:
    coerced = coerce_probes(probes)
    projected, beta0 = jax.vmap(lambda v: lanczos_tridiag(v, steps)[1:])(coerced)
    sample_values = jax.vmap(
        lambda proj, beta: slq_scalar_quadrature(
            *slq_nodes_weights(proj, beta, hermitian=hermitian),
            scalar_fun,
            scalar_postprocess=scalar_postprocess,
        )
    )(projected, beta0)
    return make_slq_quadrature_metadata(
        projected,
        beta0,
        sample_values,
        hermitian=hermitian,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def make_slq_quadrature_metadata(
    projected: jax.Array,
    beta0,
    sample_values: jax.Array,
    *,
    hermitian: bool,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> SlqQuadratureMetadata:
    nodes, weights = jax.vmap(lambda proj, beta: slq_nodes_weights(proj, beta, hermitian=hermitian))(jnp.asarray(projected), jnp.asarray(beta0))
    stats = make_probe_estimate_statistics(
        sample_values,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )
    return SlqQuadratureMetadata(
        projected=projected,
        beta0=beta0,
        nodes=nodes,
        weights=weights,
        statistics=stats,
        steps=jnp.asarray(jnp.asarray(projected).shape[-1], dtype=jnp.int32),
        hermitian=jnp.asarray(hermitian),
    )


def make_hutchpp_trace_metadata(
    *,
    basis: jax.Array,
    low_rank_trace,
    residual_samples: jax.Array,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> HutchppTraceMetadata:
    stats = make_probe_estimate_statistics(
        residual_samples,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )
    return HutchppTraceMetadata(
        basis=basis,
        low_rank_trace=low_rank_trace,
        residual_trace=stats.mean,
        statistics=stats,
    )


__all__ = [
    "OperatorPlan",
    "ScaledOperator",
    "PreconditionerPlan",
    "ShiftedSolvePlan",
    "RecycledKrylovState",
    "LogdetSolveAux",
    "LogdetSolveResult",
    "ImplicitAdjointSolveMetadata",
    "ProbeEstimateStatistics",
    "SlqQuadratureMetadata",
    "HutchppTraceMetadata",
    "DeflatedOperatorMetadata",
    "ShellCallbackPayload",
    "FiniteDifferenceOperatorPayload",
    "structure_code",
    "solver_code",
    "matfree_adjoints",
    "dense_operator",
    "dense_operator_adjoint",
    "dense_operator_rmatvec",
    "dense_operator_plan",
    "parametric_dense_operator_plan",
    "shell_operator_plan",
    "generalized_shell_operator_plan",
    "dense_preconditioner_plan",
    "shell_preconditioner_plan",
    "oriented_shell_preconditioner_plan",
    "identity_preconditioner_plan",
    "diagonal_preconditioner_plan",
    "dense_jacobi_preconditioner_plan",
    "finite_difference_operator_plan",
    "finite_difference_operator_plan_set_base",
    "sparse_bcoo_operator",
    "sparse_bcoo_operator_adjoint",
    "sparse_bcoo_operator_rmatvec",
    "sparse_bcoo_operator_plan",
    "sparse_bcoo_preconditioner_plan",
    "sparse_bcoo_jacobi_preconditioner_plan",
    "operator_transpose_plan",
    "preconditioner_transpose_plan",
    "operator_plan_apply",
    "preconditioner_plan_apply",
    "operator_apply_midpoint",
    "preconditioner_apply_midpoint",
    "scaled_operator",
    "canonicalize_sparse_bcoo",
    "det_from_logdet",
    "matrix_free_fingerprint",
    "attach_krylov_metadata",
    "make_shifted_solve_plan",
    "make_recycled_krylov_state",
    "make_logdet_solve_result",
    "combine_logdet_solve_point",
    "finite_difference_jacobi_preconditioner_plan",
    "multi_shift_solve_point",
    "implicit_krylov_solve_midpoint",
    "krylov_diagnostics",
    "dense_funm_hermitian_eigh",
    "dense_funm_general_eig",
    "projected_krylov_action_point",
    "projected_krylov_integrand_point",
    "orthonormalize_columns",
    "select_eigen_indices",
    "eig_locked_mask_from_residuals",
    "eig_restart_lock_tolerance",
    "eig_restart_column_order",
    "eig_expansion_column_order",
    "eig_restart_basis_from_pairs",
    "ritz_pairs_from_basis",
    "block_subspace_iteration_point",
    "restarted_subspace_iteration_point",
    "contour_quadrature_nodes",
    "contour_filter_subspace_point",
    "contour_integral_action_point",
    "polynomial_spectral_action_midpoint",
    "rational_spectral_action_midpoint",
    "complexify_real_linear_operator",
    "operator_apply_point",
    "poly_action_point",
    "expm_action_point",
    "restarted_action_point",
    "block_action_point",
    "rademacher_probes_real",
    "normal_probes_real",
    "rademacher_probes_complex",
    "normal_probes_complex",
    "orthogonal_rademacher_probe_block_real",
    "orthogonal_normal_probe_block_real",
    "orthogonal_rademacher_probe_block_complex",
    "orthogonal_normal_probe_block_complex",
    "probe_sample_statistics",
    "make_probe_estimate_statistics",
    "adaptive_probe_count_from_pilot",
    "probe_statistics_target_met",
    "probe_statistics_should_stop",
    "expand_subspace_with_corrections",
    "make_deflated_operator_metadata",
    "prepare_deflated_operator_metadata_point",
    "deflated_operator_apply_midpoint",
    "deflated_trace_estimate_from_metadata_point",
    "apply_action_over_probe_block_point",
    "hutchpp_trace_with_metadata_projected_point",
    "slq_nodes_weights",
    "slq_scalar_quadrature",
    "slq_heat_trace",
    "slq_spectral_density",
    "slq_functional_statistics_from_metadata",
    "slq_functional_mean_from_metadata",
    "slq_heat_trace_from_metadata",
    "slq_spectral_density_from_metadata",
    "slq_prepare_metadata_point",
    "hutchpp_trace_from_metadata",
    "make_slq_quadrature_metadata",
    "make_hutchpp_trace_metadata",
]
