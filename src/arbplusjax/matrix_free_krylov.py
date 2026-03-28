from __future__ import annotations

import jax
import jax.numpy as jnp

from . import iterative_solvers
from .matrix_free_core import (
    LogdetSolveAux,
    LogdetSolveResult,
    OperatorPlan,
    PreconditionerPlan,
    RecycledKrylovState,
    ScaledOperator,
    ShiftedSolvePlan,
    operator_apply_midpoint,
    preconditioner_apply_midpoint,
    solver_code,
    structure_code,
)


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
    from .autodiff import fingerprints
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
    from .autodiff import ad_rules
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
    recycled_state=None,
    solver: str,
    algebra: str,
    structured: str = "general",
) -> ShiftedSolvePlan:
    return ShiftedSolvePlan(
        operator=operator,
        shifts=jnp.asarray(shifts),
        preconditioner=preconditioner,
        recycled_state=recycled_state,
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
        if operator.kind == "shell":
            if operator.payload.context is None:
                return jnp.asarray(operator.payload.callback(jnp.asarray(v_mid, dtype=dtype)), dtype=dtype)
            return jnp.asarray(operator.payload.callback(jnp.asarray(v_mid, dtype=dtype), operator.payload.context), dtype=dtype)
        if operator.kind == "sparse_bcoo":
            return jnp.asarray(
                sparse_bcoo_matvec(
                    operator.payload,
                    jnp.asarray(v_mid, dtype=dtype),
                    algebra=operator.algebra,
                    label=f"matrix_free_krylov.{operator.algebra}.multi_shift.operator_apply",
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
        if preconditioner.kind == "shell":
            if preconditioner.payload.context is None:
                return jnp.asarray(preconditioner.payload.callback(jnp.asarray(v_mid, dtype=dtype)), dtype=dtype)
            return jnp.asarray(preconditioner.payload.callback(jnp.asarray(v_mid, dtype=dtype), preconditioner.payload.context), dtype=dtype)
        if preconditioner.kind == "sparse_bcoo":
            return jnp.asarray(
                sparse_bcoo_matvec(
                    preconditioner.payload,
                    jnp.asarray(v_mid, dtype=dtype),
                    algebra=preconditioner.algebra,
                    label=f"matrix_free_krylov.{preconditioner.algebra}.multi_shift.preconditioner_apply",
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

    if plan.recycled_state is not None:
        basis = jnp.asarray(plan.recycled_state.basis, dtype=dtype)
        projected = jnp.asarray(plan.recycled_state.projected, dtype=dtype)
        beta0 = jnp.linalg.norm(rhs_mid)
        e1 = jnp.zeros((projected.shape[0],), dtype=dtype).at[0].set(jnp.asarray(beta0, dtype=dtype))
        eye = jnp.eye(projected.shape[0], dtype=dtype)

        def projected_solve_one(shift):
            shift_arr = jnp.asarray(shift, dtype=dtype)
            coeffs = jnp.linalg.solve(projected + shift_arr * eye, e1)
            return coeffs @ basis

        return jax.vmap(projected_solve_one)(jnp.asarray(plan.shifts))

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
        converged=jnp.asarray(jnp.asarray(tail_norm, dtype=jnp.float64) <= jnp.asarray(primal_residual, dtype=jnp.float64)),
        locked_count=jnp.asarray(0, dtype=jnp.int32),
        convergence_metric=jnp.asarray(tail_norm, dtype=jnp.float64),
        residual_history=jnp.asarray(residual_history, dtype=jnp.float64),
        deflated_count=jnp.asarray(deflated_count, dtype=jnp.int32),
    )


__all__ = [
    "LogdetSolveAux",
    "LogdetSolveResult",
    "RecycledKrylovState",
    "ShiftedSolvePlan",
    "attach_krylov_metadata",
    "combine_logdet_solve_point",
    "krylov_diagnostics",
    "krylov_solve_midpoint",
    "make_logdet_solve_result",
    "make_recycled_krylov_state",
    "make_shifted_solve_plan",
    "matrix_free_fingerprint",
    "multi_shift_solve_point",
]
