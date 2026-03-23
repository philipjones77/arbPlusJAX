import jax.numpy as jnp

from arbplusjax import krylov_solvers as ks


def test_cg_solves_spd_system_and_honors_left_preconditioner():
    a = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    exact = jnp.linalg.solve(a, b)

    matvec = lambda x: a @ x
    precond = lambda x: x / jnp.asarray([4.0, 3.0], dtype=jnp.float64)

    x_plain, info_plain = ks.cg(matvec, b, tol=1e-10, atol=1e-10, maxiter=8)
    x_prec, info_prec = ks.cg(matvec, b, tol=1e-10, atol=1e-10, maxiter=8, M=precond)

    assert jnp.allclose(x_plain, exact, rtol=1e-8, atol=1e-8)
    assert jnp.allclose(x_prec, exact, rtol=1e-8, atol=1e-8)
    assert jnp.isfinite(info_plain["residual"])
    assert jnp.isfinite(info_prec["residual"])


def test_gmres_solves_nonsymmetric_system_with_restart_contract():
    a = jnp.asarray([[3.0, 1.0], [0.0, 2.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    exact = jnp.linalg.solve(a, b)

    x, info = ks.gmres(lambda v: a @ v, b, tol=1e-10, atol=1e-10, maxiter=6, restart=2)

    assert jnp.allclose(x, exact, rtol=1e-8, atol=1e-8)
    assert jnp.isfinite(info["residual"])
    assert int(info["iterations"]) == 6


def test_bicgstab_solves_general_system():
    a = jnp.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    exact = jnp.linalg.solve(a, b)

    x, info = ks.bicgstab(lambda v: a @ v, b, tol=1e-10, atol=1e-10, maxiter=12)

    assert jnp.allclose(x, exact, rtol=1e-7, atol=1e-7)
    assert jnp.isfinite(info["residual"])
    assert int(info["iterations"]) == 12


def test_krylov_safe_divide_and_norm_are_finite_on_edge_cases():
    zero = jnp.asarray([0.0, 0.0], dtype=jnp.float64)

    assert jnp.isclose(ks._norm(zero), 0.0)
    assert jnp.isclose(ks._safe_divide(jnp.asarray(1.0), jnp.asarray(0.0)), 0.0)
