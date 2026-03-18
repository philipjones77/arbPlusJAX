from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..petsc.lowering import to_petsc_mat
from ..petsc.runtime import get_petsc_module
from ..petsc.solve import LinearSolveConfig, create_ksp
from .runtime import get_slepc_module


@dataclass(frozen=True)
class EigensolveConfig:
    nev: int
    which: str = "SMALLEST_MAGNITUDE"
    problem_type: str | None = None
    eps_type: str | None = None
    st_type: str | None = None
    shift: float | None = None
    target: float | complex | None = None
    st_ksp: LinearSolveConfig = LinearSolveConfig()


@dataclass(frozen=True)
class EigensolveResult:
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    converged: int


def create_eps(
    operator,
    *,
    mass=None,
    config: EigensolveConfig,
    petsc=None,
    slepc=None,
    operator_shape: tuple[int, int] | None = None,
    mass_shape: tuple[int, int] | None = None,
):
    petsc_module = get_petsc_module() if petsc is None else petsc
    slepc_module = get_slepc_module() if slepc is None else slepc
    native_operator = to_petsc_mat(operator, shape=operator_shape, petsc=petsc_module)
    native_mass = None if mass is None else to_petsc_mat(mass, shape=mass_shape, petsc=petsc_module)
    eps = slepc_module.EPS().create()
    eps.setOperators(native_operator, native_mass)
    if config.problem_type is not None:
        eps.setProblemType(_enum_or_raw(getattr(slepc_module.EPS, "ProblemType", None), config.problem_type))
    if config.eps_type is not None:
        eps.setType(_enum_or_raw(getattr(slepc_module.EPS, "Type", None), config.eps_type))
    eps.setDimensions(int(config.nev))
    eps.setWhichEigenpairs(_enum_or_raw(getattr(slepc_module.EPS, "Which", None), config.which))
    if config.target is not None and hasattr(eps, "setTarget"):
        eps.setTarget(config.target)
    if config.st_type is not None or config.shift is not None or _needs_st_ksp(config.st_ksp):
        st = slepc_module.ST().create()
        if config.st_type is not None:
            st.setType(_enum_or_raw(getattr(slepc_module.ST, "Type", None), config.st_type))
        if config.shift is not None:
            st.setShift(float(config.shift))
        if _needs_st_ksp(config.st_ksp):
            native_ksp = create_ksp(native_operator, config=config.st_ksp, petsc=petsc_module)
            st.setKSP(native_ksp)
        eps.setST(st)
    return eps


def solve_eigenproblem(
    operator,
    *,
    mass=None,
    config: EigensolveConfig,
    petsc=None,
    slepc=None,
    operator_shape: tuple[int, int] | None = None,
    mass_shape: tuple[int, int] | None = None,
) -> EigensolveResult:
    petsc_module = get_petsc_module() if petsc is None else petsc
    eps = create_eps(
        operator,
        mass=mass,
        config=config,
        petsc=petsc_module,
        slepc=slepc,
        operator_shape=operator_shape,
        mass_shape=mass_shape,
    )
    eps.solve()
    converged = min(int(eps.getConverged()), int(config.nev))
    size = int(eps.getOperators()[0].getSize()[0]) if hasattr(eps, "getOperators") else _operator_rows(operator, operator_shape)
    if converged == 0:
        return EigensolveResult(
            eigenvalues=jnp.asarray([]),
            eigenvectors=jnp.zeros((size, 0), dtype=jnp.float64),
            converged=0,
        )
    values: list[complex] = []
    vectors: list[np.ndarray] = []
    for index in range(converged):
        real_vector = petsc_module.Vec().createSeq(size)
        try:
            imag_vector = petsc_module.Vec().createSeq(size)
            eigenvalue = eps.getEigenpair(index, real_vector, imag_vector)
        except TypeError:
            eigenvalue = eps.getEigenpair(index, real_vector)
        try:
            array = real_vector.getArray(readonly=True)
        except TypeError:
            array = real_vector.getArray()
        values.append(eigenvalue)
        vectors.append(np.asarray(array))
    return EigensolveResult(
        eigenvalues=jnp.asarray(np.asarray(values)),
        eigenvectors=jnp.asarray(np.stack(vectors, axis=1)),
        converged=converged,
    )


def _enum_or_raw(container, name: str):
    if container is None:
        return name
    return getattr(container, name, name)


def _needs_st_ksp(config: LinearSolveConfig) -> bool:
    return any(
        value is not None
        for value in (config.ksp_type, config.pc_type, config.rtol, config.atol, config.max_it)
    )


def _operator_rows(operator, explicit_shape: tuple[int, int] | None) -> int:
    if explicit_shape is not None:
        return int(explicit_shape[0])
    if hasattr(operator, "shape"):
        return int(operator.shape[0])
    if hasattr(operator, "rows"):
        return int(operator.rows)
    return int(jnp.asarray(operator).shape[0])
