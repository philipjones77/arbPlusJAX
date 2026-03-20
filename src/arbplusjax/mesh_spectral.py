from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from . import mat_common
from . import matfree_adjoints
from . import scb_block_mat
from . import scb_mat
from . import scb_vblock_mat
from . import sparse_common
from . import srb_block_mat
from . import srb_mat
from . import srb_vblock_mat
from .backends.petsc.solve import LinearSolveConfig
from .backends.slepc import eigs as slepc_eigs


@dataclass(frozen=True)
class MeshEigenConfig:
    k: int
    backend: str = "jax"
    which: str = "smallest_magnitude"
    krylov_dim: int | None = None
    reortho: str = "full"
    tol: float = 1e-8
    problem_type: str | None = None
    eps_type: str | None = None
    st_type: str | None = None
    shift: float | None = None
    target: float | complex | None = None
    st_ksp: LinearSolveConfig = LinearSolveConfig()


@dataclass(frozen=True)
class MeshEigenResult:
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    converged: jnp.ndarray
    residual_norms: jnp.ndarray
    backend: str


def graph_laplacian(adjacency: jnp.ndarray, *, normalized: bool = False) -> jnp.ndarray:
    adjacency = jnp.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph_laplacian expects a square dense adjacency matrix")
    degrees = jnp.sum(adjacency, axis=-1)
    if not normalized:
        return jnp.diag(degrees) - adjacency
    inv_sqrt = jnp.where(degrees > 0, jnp.reciprocal(jnp.sqrt(degrees)), 0.0)
    identity = jnp.eye(adjacency.shape[0], dtype=adjacency.dtype)
    scaled = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return identity - scaled


def solve_mesh_eigenproblem(
    operator,
    *,
    mass=None,
    config: MeshEigenConfig,
    shape: tuple[int, int] | None = None,
    mass_shape: tuple[int, int] | None = None,
    initial_vector: jnp.ndarray | None = None,
) -> MeshEigenResult:
    _validate_config(config)
    backend = _canonical_backend(config.backend)
    resolved_shape = _resolve_shape(operator, shape)
    if resolved_shape[0] != resolved_shape[1]:
        raise ValueError("solve_mesh_eigenproblem expects a square operator")
    if config.k > resolved_shape[0]:
        raise ValueError("MeshEigenConfig.k cannot exceed the operator dimension")
    if backend == "jax":
        return _solve_mesh_eigenproblem_jax(
            operator,
            mass=mass,
            config=config,
            shape=resolved_shape,
            mass_shape=mass_shape,
            initial_vector=initial_vector,
        )
    return _solve_mesh_eigenproblem_slepc(
        operator,
        mass=mass,
        config=config,
        shape=resolved_shape,
        mass_shape=mass_shape,
    )


def _solve_mesh_eigenproblem_jax(
    operator,
    *,
    mass,
    config: MeshEigenConfig,
    shape: tuple[int, int],
    mass_shape: tuple[int, int] | None,
    initial_vector: jnp.ndarray | None,
) -> MeshEigenResult:
    if mass is not None:
        dense_operator = _materialize_dense_operator(operator, shape=shape)
        dense_mass = _materialize_dense_operator(mass, shape=_resolve_shape(mass, mass_shape))
        return _solve_dense_generalized_eigenproblem(dense_operator, dense_mass, config=config)
    if _is_dense_explicit(operator):
        dense_operator = _materialize_dense_operator(operator, shape=shape)
        return _solve_dense_standard_eigenproblem(dense_operator, config=config)
    return _solve_lanczos_standard_eigenproblem(
        operator,
        config=config,
        shape=shape,
        initial_vector=initial_vector,
    )


def _solve_dense_standard_eigenproblem(operator, *, config: MeshEigenConfig) -> MeshEigenResult:
    dense = jnp.asarray(_materialize_dense_operator(operator), dtype=jnp.float64)
    eigenvalues, eigenvectors = jnp.linalg.eigh(dense)
    selected_values, selected_vectors = _select_eigenpairs(eigenvalues, eigenvectors, config)
    residual_norms = _standard_residual_norms(dense, selected_values, selected_vectors)
    converged = _count_converged(selected_values, residual_norms, config.tol)
    return MeshEigenResult(
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
        converged=converged,
        residual_norms=residual_norms,
        backend="jax",
    )


def _solve_dense_generalized_eigenproblem(operator, mass, *, config: MeshEigenConfig) -> MeshEigenResult:
    dense_operator = jnp.asarray(_materialize_dense_operator(operator), dtype=jnp.float64)
    dense_mass = jnp.asarray(_materialize_dense_operator(mass), dtype=jnp.float64)
    chol = jnp.linalg.cholesky(dense_mass)
    reduced_left = lax.linalg.triangular_solve(chol, dense_operator, left_side=True, lower=True)
    reduced = lax.linalg.triangular_solve(chol, reduced_left.T, left_side=True, lower=True).T
    eigenvalues, reduced_vectors = jnp.linalg.eigh(reduced)
    eigenvectors = lax.linalg.triangular_solve(chol, reduced_vectors, left_side=True, lower=True, transpose_a=True)
    mass_norms = jnp.sqrt(jnp.sum(eigenvectors * (dense_mass @ eigenvectors), axis=0))
    eigenvectors = eigenvectors / jnp.maximum(mass_norms, jnp.asarray(1e-30, dtype=eigenvectors.dtype))
    selected_values, selected_vectors = _select_eigenpairs(eigenvalues, eigenvectors, config)
    residual_norms = _generalized_residual_norms(dense_operator, dense_mass, selected_values, selected_vectors)
    converged = _count_converged(selected_values, residual_norms, config.tol)
    return MeshEigenResult(
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
        converged=converged,
        residual_norms=residual_norms,
        backend="jax",
    )


def _solve_lanczos_standard_eigenproblem(
    operator,
    *,
    config: MeshEigenConfig,
    shape: tuple[int, int],
    initial_vector: jnp.ndarray | None,
) -> MeshEigenResult:
    matvec = _operator_matvec(operator, shape=shape)
    dimension = int(shape[0])
    krylov_dim = int(config.krylov_dim or min(dimension, max(config.k + 8, 2 * config.k + 2)))
    if krylov_dim < config.k or krylov_dim > dimension:
        raise ValueError("MeshEigenConfig.krylov_dim must satisfy k <= krylov_dim <= matrix dimension")
    v0 = _initial_lanczos_vector(dimension, initial_vector)
    sample = jnp.asarray(matvec(v0))
    if jnp.iscomplexobj(sample):
        raise ValueError("JAX Lanczos mesh eigensolve currently expects a real symmetric operator; use backend='slepc' for complex problems")
    lanczos = matfree_adjoints.lanczos_tridiag(
        lambda vector: jnp.asarray(matvec(vector), dtype=jnp.float64),
        krylov_dim,
        reortho=config.reortho,
        custom_vjp=False,
    )
    (basis, (diagonals, offdiagonals)), _ = lanczos(v0)
    tridiagonal = jnp.diag(diagonals) + jnp.diag(offdiagonals, 1) + jnp.diag(offdiagonals, -1)
    eigenvalues, krylov_vectors = jnp.linalg.eigh(tridiagonal)
    eigenvectors = basis.T @ krylov_vectors
    norms = jnp.linalg.norm(eigenvectors, axis=0)
    eigenvectors = eigenvectors / jnp.maximum(norms, jnp.asarray(1e-30, dtype=eigenvectors.dtype))
    selected_values, selected_vectors = _select_eigenpairs(eigenvalues, eigenvectors, config)
    residual_norms = _operator_standard_residual_norms(matvec, selected_values, selected_vectors)
    converged = _count_converged(selected_values, residual_norms, config.tol)
    return MeshEigenResult(
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
        converged=converged,
        residual_norms=residual_norms,
        backend="jax",
    )


def _solve_mesh_eigenproblem_slepc(
    operator,
    *,
    mass,
    config: MeshEigenConfig,
    shape: tuple[int, int],
    mass_shape: tuple[int, int] | None,
) -> MeshEigenResult:
    slepc_result = slepc_eigs.solve_eigenproblem(
        operator,
        mass=mass,
        config=slepc_eigs.EigensolveConfig(
            nev=config.k,
            which=_slepc_which_name(config.which),
            problem_type=config.problem_type,
            eps_type=config.eps_type,
            st_type=config.st_type,
            shift=config.shift,
            target=config.target,
            st_ksp=config.st_ksp,
        ),
        operator_shape=shape,
        mass_shape=mass_shape,
    )
    residual_norms = _residual_norms_for_result(
        operator,
        eigenvalues=slepc_result.eigenvalues,
        eigenvectors=slepc_result.eigenvectors,
        mass=mass,
        shape=shape,
        mass_shape=mass_shape,
    )
    return MeshEigenResult(
        eigenvalues=slepc_result.eigenvalues,
        eigenvectors=slepc_result.eigenvectors,
        converged=jnp.asarray(int(slepc_result.converged), dtype=jnp.int32),
        residual_norms=residual_norms,
        backend="slepc",
    )


def _validate_config(config: MeshEigenConfig) -> None:
    if config.k <= 0:
        raise ValueError("MeshEigenConfig.k must be positive")
    _canonical_backend(config.backend)
    _canonical_which(config.which)
    if config.reortho not in {"none", "full"}:
        raise ValueError("MeshEigenConfig.reortho must be 'none' or 'full'")
    if config.tol < 0:
        raise ValueError("MeshEigenConfig.tol must be non-negative")


def _canonical_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in {"jax", "slepc"}:
        raise ValueError(f"Unsupported eigensolver backend: {backend}")
    return normalized


def _canonical_which(which: str) -> str:
    mapping = {
        "smallest_magnitude": "smallest_magnitude",
        "largest_magnitude": "largest_magnitude",
        "smallest_real": "smallest_real",
        "largest_real": "largest_real",
        "SMALLEST_MAGNITUDE": "smallest_magnitude",
        "LARGEST_MAGNITUDE": "largest_magnitude",
        "SMALLEST_REAL": "smallest_real",
        "LARGEST_REAL": "largest_real",
    }
    try:
        return mapping[str(which)]
    except KeyError as error:
        raise ValueError(f"Unsupported eigenpair selection: {which}") from error


def _slepc_which_name(which: str) -> str:
    canonical = _canonical_which(which)
    return canonical.upper()


def _select_eigenpairs(eigenvalues: jnp.ndarray, eigenvectors: jnp.ndarray, config: MeshEigenConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    indices = _order_indices(eigenvalues, config.which)[: config.k]
    return eigenvalues[indices], eigenvectors[:, indices]


def _order_indices(eigenvalues: jnp.ndarray, which: str) -> jnp.ndarray:
    canonical = _canonical_which(which)
    if canonical == "smallest_magnitude":
        return jnp.argsort(jnp.abs(eigenvalues))
    if canonical == "largest_magnitude":
        return jnp.argsort(-jnp.abs(eigenvalues))
    if canonical == "smallest_real":
        return jnp.argsort(jnp.real(eigenvalues))
    return jnp.argsort(-jnp.real(eigenvalues))


def _resolve_shape(operator, explicit_shape: tuple[int, int] | None) -> tuple[int, int]:
    if explicit_shape is not None:
        return int(explicit_shape[0]), int(explicit_shape[1])
    operator = getattr(operator, "native", operator)
    if hasattr(operator, "shape"):
        shape = tuple(int(axis) for axis in operator.shape)
        if len(shape) == 2:
            return shape
    if hasattr(operator, "rows") and hasattr(operator, "cols"):
        return int(operator.rows), int(operator.cols)
    if hasattr(operator, "getSize"):
        rows, cols = operator.getSize()
        return int(rows), int(cols)
    array = jnp.asarray(operator)
    if array.ndim != 2:
        raise ValueError("Unable to infer operator shape")
    return int(array.shape[0]), int(array.shape[1])


def _is_dense_explicit(operator) -> bool:
    operator = getattr(operator, "native", operator)
    if isinstance(operator, mat_common.DenseMatvecPlan):
        return True
    if sparse.issparse(operator):
        return False
    if isinstance(
        operator,
        (
            sparse_common.SparseCOO,
            sparse_common.SparseCSR,
            sparse_common.SparseBCOO,
            sparse_common.SparseMatvecPlan,
            sparse_common.BlockSparseCOO,
            sparse_common.BlockSparseCSR,
            sparse_common.BlockSparseMatvecPlan,
            sparse_common.VariableBlockSparseCOO,
            sparse_common.VariableBlockSparseCSR,
            sparse_common.VariableBlockSparseMatvecPlan,
        ),
    ):
        return False
    if callable(operator) or hasattr(operator, "matvec"):
        return False
    array = jnp.asarray(operator)
    return array.ndim == 2


def _materialize_dense_operator(operator, *, shape: tuple[int, int] | None = None) -> jnp.ndarray:
    operator = getattr(operator, "native", operator)
    if sparse.issparse(operator):
        return jnp.asarray(operator.toarray())
    if isinstance(operator, mat_common.DenseMatvecPlan):
        return jnp.asarray(operator.matrix)
    if isinstance(operator, sparse_common.SparseMatvecPlan):
        return _materialize_dense_from_matvec(_operator_matvec(operator, shape=(operator.rows, operator.cols)), (operator.rows, operator.cols))
    if isinstance(operator, (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)):
        return srb_mat.srb_mat_to_dense(operator) if operator.algebra == "srb" else scb_mat.scb_mat_to_dense(operator)
    if isinstance(operator, (sparse_common.BlockSparseCOO, sparse_common.BlockSparseCSR)):
        return srb_block_mat.srb_block_mat_to_dense(operator) if operator.algebra == "srb" else scb_block_mat.scb_block_mat_to_dense(operator)
    if isinstance(operator, (sparse_common.VariableBlockSparseCOO, sparse_common.VariableBlockSparseCSR)):
        return srb_vblock_mat.srb_vblock_mat_to_dense(operator) if operator.algebra == "srb" else scb_vblock_mat.scb_vblock_mat_to_dense(operator)
    if isinstance(
        operator,
        (
            sparse_common.BlockSparseMatvecPlan,
            sparse_common.VariableBlockSparseMatvecPlan,
        ),
    ):
        resolved_shape = _resolve_shape(operator, shape)
        return _materialize_dense_from_matvec(_operator_matvec(operator, shape=resolved_shape), resolved_shape)
    if callable(operator) or hasattr(operator, "matvec"):
        resolved_shape = _resolve_shape(operator, shape)
        return _materialize_dense_from_matvec(_operator_matvec(operator, shape=resolved_shape), resolved_shape)
    array = jnp.asarray(operator)
    if array.ndim != 2:
        raise ValueError("Dense materialization expects a 2D operator")
    return array


def _materialize_dense_from_matvec(matvec: Callable[[jnp.ndarray], jnp.ndarray], shape: tuple[int, int]) -> jnp.ndarray:
    eye = jnp.eye(shape[1], dtype=jnp.float64)
    return jax.vmap(matvec, in_axes=1, out_axes=1)(eye)


def _operator_matvec(operator, *, shape: tuple[int, int] | None = None) -> Callable[[jnp.ndarray], jnp.ndarray]:
    operator = getattr(operator, "native", operator)
    if isinstance(operator, mat_common.DenseMatvecPlan):
        return lambda vector: jnp.asarray(operator.matrix) @ jnp.asarray(vector)
    if isinstance(operator, sparse_common.SparseMatvecPlan):
        if operator.algebra == "srb":
            return lambda vector: srb_mat.srb_mat_matvec_cached_apply(operator, jnp.asarray(vector))
        return lambda vector: scb_mat.scb_mat_matvec_cached_apply(operator, jnp.asarray(vector))
    if isinstance(operator, (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)):
        if operator.algebra == "srb":
            return lambda vector: srb_mat.srb_mat_matvec(operator, jnp.asarray(vector))
        return lambda vector: scb_mat.scb_mat_matvec(operator, jnp.asarray(vector))
    if isinstance(operator, (sparse_common.BlockSparseCOO, sparse_common.BlockSparseCSR)):
        if operator.algebra == "srb":
            return lambda vector: srb_block_mat.srb_block_mat_matvec(operator, jnp.asarray(vector))
        return lambda vector: scb_block_mat.scb_block_mat_matvec(operator, jnp.asarray(vector))
    if isinstance(operator, sparse_common.BlockSparseMatvecPlan):
        if operator.algebra == "srb":
            return lambda vector: srb_block_mat.srb_block_mat_matvec_cached_apply(operator, jnp.asarray(vector))
        return lambda vector: scb_block_mat.scb_block_mat_matvec_cached_apply(operator, jnp.asarray(vector))
    if isinstance(operator, (sparse_common.VariableBlockSparseCOO, sparse_common.VariableBlockSparseCSR)):
        if operator.algebra == "srb":
            return lambda vector: srb_vblock_mat.srb_vblock_mat_matvec(operator, jnp.asarray(vector))
        return lambda vector: scb_vblock_mat.scb_vblock_mat_matvec(operator, jnp.asarray(vector))
    if isinstance(operator, sparse_common.VariableBlockSparseMatvecPlan):
        if operator.algebra == "srb":
            return lambda vector: srb_vblock_mat.srb_vblock_mat_matvec_cached_apply(operator, jnp.asarray(vector))
        return lambda vector: scb_vblock_mat.scb_vblock_mat_matvec_cached_apply(operator, jnp.asarray(vector))
    if sparse.issparse(operator):
        return lambda vector: jnp.asarray(operator @ np.asarray(vector))
    if callable(operator):
        return lambda vector: jnp.asarray(operator(jnp.asarray(vector)))
    if hasattr(operator, "matvec"):
        return lambda vector: jnp.asarray(operator.matvec(jnp.asarray(vector)))
    dense = _materialize_dense_operator(operator, shape=shape)
    return lambda vector: dense @ jnp.asarray(vector)


def _initial_lanczos_vector(dimension: int, initial_vector: jnp.ndarray | None) -> jnp.ndarray:
    if initial_vector is not None:
        vector = jnp.asarray(initial_vector, dtype=jnp.float64)
        if vector.ndim != 1 or vector.shape[0] != dimension:
            raise ValueError("initial_vector must be one-dimensional and match the operator dimension")
    else:
        grid = jnp.arange(dimension, dtype=jnp.float64)
        vector = jnp.cos(grid + 1.0) + jnp.sin(0.5 * (grid + 1.0))
    norm = jnp.linalg.norm(vector)
    return vector / jnp.maximum(norm, jnp.asarray(1e-30, dtype=vector.dtype))


def _standard_residual_norms(operator: jnp.ndarray, eigenvalues: jnp.ndarray, eigenvectors: jnp.ndarray) -> jnp.ndarray:
    return _operator_standard_residual_norms(lambda vector: operator @ vector, eigenvalues, eigenvectors)


def _operator_standard_residual_norms(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
) -> jnp.ndarray:
    def residual(eigenvalue, eigenvector):
        return jnp.linalg.norm(jnp.asarray(matvec(eigenvector)) - eigenvalue * eigenvector)

    return jax.vmap(residual, in_axes=(0, 1))(eigenvalues, eigenvectors)


def _generalized_residual_norms(
    operator: jnp.ndarray,
    mass: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
) -> jnp.ndarray:
    def residual(eigenvalue, eigenvector):
        return jnp.linalg.norm((operator @ eigenvector) - eigenvalue * (mass @ eigenvector))

    return jax.vmap(residual, in_axes=(0, 1))(eigenvalues, eigenvectors)


def _residual_norms_for_result(
    operator,
    *,
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
    mass=None,
    shape: tuple[int, int] | None = None,
    mass_shape: tuple[int, int] | None = None,
) -> jnp.ndarray:
    operator_matvec = _operator_matvec(operator, shape=shape)
    if mass is None:
        return _operator_standard_residual_norms(operator_matvec, eigenvalues, eigenvectors)
    mass_matvec = _operator_matvec(mass, shape=_resolve_shape(mass, mass_shape))

    def residual(eigenvalue, eigenvector):
        return jnp.linalg.norm(operator_matvec(eigenvector) - eigenvalue * mass_matvec(eigenvector))

    return jax.vmap(residual, in_axes=(0, 1))(eigenvalues, eigenvectors)


def _count_converged(eigenvalues: jnp.ndarray, residual_norms: jnp.ndarray, tol: float) -> jnp.ndarray:
    threshold = tol * jnp.maximum(jnp.abs(eigenvalues), jnp.asarray(1.0, dtype=eigenvalues.dtype))
    return jnp.sum(residual_norms <= threshold).astype(jnp.int32)


__all__ = [
    "MeshEigenConfig",
    "MeshEigenResult",
    "graph_laplacian",
    "solve_mesh_eigenproblem",
]
