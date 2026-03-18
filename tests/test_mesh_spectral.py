from __future__ import annotations

import jax
import jax.numpy as jnp

from arbplusjax import mesh_spectral
from arbplusjax import srb_mat


def _path_graph_adjacency(n: int, weight: float = 1.0) -> jnp.ndarray:
    adjacency = jnp.zeros((n, n), dtype=jnp.float64)
    edge_weight = jnp.asarray(weight, dtype=jnp.float64)
    for index in range(n - 1):
        adjacency = adjacency.at[index, index + 1].set(edge_weight)
        adjacency = adjacency.at[index + 1, index].set(edge_weight)
    return adjacency


def test_dense_mesh_eigensolve_matches_exact_and_is_differentiable() -> None:
    base = _path_graph_adjacency(3, weight=1.0)
    config = mesh_spectral.MeshEigenConfig(k=2, backend="jax", which="smallest_real")

    def loss(weight):
        laplacian = mesh_spectral.graph_laplacian(weight * base)
        result = mesh_spectral.solve_mesh_eigenproblem(laplacian, config=config)
        return jnp.sum(result.eigenvalues)

    weight = jnp.asarray(2.0, dtype=jnp.float64)
    result = mesh_spectral.solve_mesh_eigenproblem(mesh_spectral.graph_laplacian(weight * base), config=config)
    exact = jnp.linalg.eigvalsh(mesh_spectral.graph_laplacian(weight * base))[:2]
    gradient = jax.grad(loss)(weight)

    assert jnp.allclose(result.eigenvalues, exact, atol=1e-8)
    assert jnp.allclose(gradient, jnp.asarray(1.0, dtype=jnp.float64), atol=1e-8)
    assert int(result.converged) == 2


def test_sparse_mesh_eigensolve_uses_lanczos_path() -> None:
    laplacian = mesh_spectral.graph_laplacian(_path_graph_adjacency(6))
    sparse_laplacian = srb_mat.srb_mat_from_dense_csr(laplacian)

    result = mesh_spectral.solve_mesh_eigenproblem(
        sparse_laplacian,
        config=mesh_spectral.MeshEigenConfig(k=3, backend="jax", which="smallest_real", krylov_dim=6),
    )

    exact = jnp.linalg.eigvalsh(laplacian)[:3]

    assert result.backend == "jax"
    assert result.eigenvectors.shape == (6, 3)
    assert jnp.allclose(result.eigenvalues, exact, atol=1e-6)


def test_generalized_dense_mesh_eigensolve_matches_cholesky_reduction() -> None:
    stiffness = jnp.diag(jnp.asarray([2.0, 5.0, 11.0], dtype=jnp.float64))
    mass = jnp.diag(jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64))

    result = mesh_spectral.solve_mesh_eigenproblem(
        stiffness,
        mass=mass,
        config=mesh_spectral.MeshEigenConfig(k=2, backend="jax", which="smallest_real"),
    )

    assert jnp.allclose(result.eigenvalues, jnp.asarray([2.0, 2.5], dtype=jnp.float64), atol=1e-8)
    assert int(result.converged) == 2


def test_slepc_backend_routing_uses_public_config(monkeypatch) -> None:
    captured = {}

    def fake_solve(operator, *, mass, config, shape, mass_shape):
        del operator, mass
        captured["config"] = config
        captured["shape"] = shape
        captured["mass_shape"] = mass_shape
        return mesh_spectral.MeshEigenResult(
            eigenvalues=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            eigenvectors=jnp.eye(3, 2, dtype=jnp.float64),
            converged=jnp.asarray(2, dtype=jnp.int32),
            residual_norms=jnp.zeros((2,), dtype=jnp.float64),
            backend="slepc",
        )

    monkeypatch.setattr(mesh_spectral, "_solve_mesh_eigenproblem_slepc", fake_solve)
    laplacian = mesh_spectral.graph_laplacian(_path_graph_adjacency(3))
    mass = jnp.eye(3, dtype=jnp.float64)

    result = mesh_spectral.solve_mesh_eigenproblem(
        laplacian,
        mass=mass,
        config=mesh_spectral.MeshEigenConfig(
            k=2,
            backend="slepc",
            which="smallest_real",
            problem_type="GHEP",
            st_type="SINVERT",
            shift=0.0,
        ),
    )

    assert result.backend == "slepc"
    assert captured["shape"] == (3, 3)
    assert captured["mass_shape"] is None
    assert captured["config"].problem_type == "GHEP"
    assert captured["config"].st_type == "SINVERT"
