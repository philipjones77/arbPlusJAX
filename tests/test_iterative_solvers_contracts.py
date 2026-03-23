import jax.numpy as jnp

from arbplusjax import iterative_solvers as its


def test_iterative_solver_cg_matches_dense_spd_reference():
    a = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    exact = jnp.linalg.solve(a, b)

    x, info = its.cg(lambda v: a @ v, b, tol=1e-10, atol=1e-10, maxiter=8)

    assert jnp.allclose(x, exact, rtol=1e-6, atol=1e-6)
    assert bool(info["converged"])
    assert info["residuals"].shape == (8,)


def test_iterative_solver_bicgstab_and_gmres_solve_general_systems():
    a = jnp.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    exact = jnp.linalg.solve(a, b)

    x_bi, info_bi = its.bicgstab(lambda v: a @ v, b, tol=1e-10, atol=1e-10, maxiter=12)
    x_gm, info_gm = its.gmres(lambda v: a @ v, b, tol=1e-10, atol=1e-10, restart=2, maxiter=4)

    assert jnp.allclose(x_bi, exact, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(x_gm, exact, rtol=1e-5, atol=1e-5)
    assert bool(info_bi["converged"])
    assert info_gm["residuals"].shape == (4,)


def test_iterative_solver_minres_handles_symmetric_indefinite_system():
    a = jnp.asarray([[2.0, 0.0], [0.0, -1.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    exact = jnp.linalg.solve(a, b)

    x, info = its.minres(lambda v: a @ v, b, tol=1e-10, atol=1e-10, maxiter=4)

    assert jnp.allclose(x, exact, rtol=1e-6, atol=1e-6)
    assert info["residuals"].shape == (4,)
