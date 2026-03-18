from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .lowering import to_petsc_mat, to_petsc_vec
from .runtime import get_petsc_module


@dataclass(frozen=True)
class LinearSolveConfig:
    ksp_type: str | None = None
    pc_type: str | None = None
    rtol: float | None = None
    atol: float | None = None
    max_it: int | None = None


def create_ksp(
    operator,
    *,
    preconditioner=None,
    config: LinearSolveConfig = LinearSolveConfig(),
    petsc=None,
    operator_shape: tuple[int, int] | None = None,
    preconditioner_shape: tuple[int, int] | None = None,
):
    module = get_petsc_module() if petsc is None else petsc
    native_operator = to_petsc_mat(operator, shape=operator_shape, petsc=module)
    native_preconditioner = native_operator if preconditioner is None else to_petsc_mat(
        preconditioner,
        shape=preconditioner_shape,
        petsc=module,
    )
    ksp = module.KSP().create()
    if config.ksp_type is not None:
        ksp.setType(config.ksp_type)
    ksp.setOperators(native_operator, native_preconditioner)
    if config.pc_type is not None and hasattr(ksp, "getPC"):
        ksp.getPC().setType(config.pc_type)
    tolerance_kwargs = {}
    if config.rtol is not None:
        tolerance_kwargs["rtol"] = float(config.rtol)
    if config.atol is not None:
        tolerance_kwargs["atol"] = float(config.atol)
    if config.max_it is not None:
        tolerance_kwargs["max_it"] = int(config.max_it)
    if tolerance_kwargs and hasattr(ksp, "setTolerances"):
        ksp.setTolerances(**tolerance_kwargs)
    return ksp


def solve_linear_system(
    operator,
    rhs,
    *,
    preconditioner=None,
    config: LinearSolveConfig = LinearSolveConfig(),
    petsc=None,
    operator_shape: tuple[int, int] | None = None,
    preconditioner_shape: tuple[int, int] | None = None,
) -> jnp.ndarray:
    module = get_petsc_module() if petsc is None else petsc
    native_operator = to_petsc_mat(operator, shape=operator_shape, petsc=module)
    native_rhs = to_petsc_vec(rhs, petsc=module)
    ksp = create_ksp(
        native_operator,
        preconditioner=preconditioner,
        config=config,
        petsc=module,
        operator_shape=operator_shape,
        preconditioner_shape=preconditioner_shape,
    )
    if hasattr(native_operator, "createVecRight"):
        native_solution = native_operator.createVecRight()
    else:
        native_solution = module.Vec().createSeq(int(native_operator.getSize()[1]))
    ksp.solve(native_rhs, native_solution)
    try:
        solution = native_solution.getArray(readonly=True)
    except TypeError:
        solution = native_solution.getArray()
    return jnp.asarray(np.asarray(solution))
