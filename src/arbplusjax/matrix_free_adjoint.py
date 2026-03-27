from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from . import iterative_solvers
from .matrix_free_core import (
    ImplicitAdjointSolveMetadata,
    operator_apply_midpoint,
    operator_transpose_plan,
    preconditioner_apply_midpoint,
    preconditioner_transpose_plan,
)


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
    transpose_supported = transpose_structured or transpose_operator is not None

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

    if use_implicit_adjoint and transpose_supported:
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
        implicit_adjoint=bool(use_implicit_adjoint and transpose_supported),
    )
    return x_mid, info, residual, jnp.linalg.norm(rhs_mid), metadata


__all__ = [
    "ImplicitAdjointSolveMetadata",
    "implicit_krylov_solve_midpoint",
]
