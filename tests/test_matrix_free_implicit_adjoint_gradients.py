import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat


def _exact_interval_vector(values):
    arr = jnp.asarray(values, dtype=jnp.float64)
    return di.interval(arr, arr)


def _dense_plan(theta):
    dense = jnp.asarray(
        [
            [2.0 + theta, 0.0],
            [0.0, 4.0 + 0.5 * theta],
        ],
        dtype=jnp.float64,
    )
    return jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))


def _complex_box_vector(values):
    arr = jnp.asarray(values, dtype=jnp.complex128)
    return acb_core.acb_box(
        di.interval(jnp.real(arr), jnp.real(arr)),
        di.interval(jnp.imag(arr), jnp.imag(arr)),
    )


def _complex_dense_plan(theta):
    dense_mid = jnp.asarray(
        [
            [2.0 + theta + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 4.0 + 0.5 * theta + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    return jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)


def test_fp64_dot_test_for_implicit_solve_gradient():
    rhs = _exact_interval_vector([1.0, -2.0])
    theta = jnp.asarray(0.3, dtype=jnp.float64)
    tangent = jnp.asarray(0.7, dtype=jnp.float64)
    cotangent = jnp.asarray([0.25, -0.4], dtype=jnp.float64)

    def solve_mid(theta_value):
        solved, diag = jrb_mat.jrb_mat_solve_action_with_diagnostics_point(
            _dense_plan(theta_value),
            rhs,
            symmetric=True,
        )
        return di.midpoint(solved)

    _, diag = jrb_mat.jrb_mat_solve_action_with_diagnostics_point(_dense_plan(theta), rhs, symmetric=True)
    assert bool(diag.used_adjoint)
    assert bool(diag.gradient_supported)

    primal, tangent_out = jax.jvp(solve_mid, (theta,), (tangent,))
    del primal
    _, vjp = jax.vjp(solve_mid, theta)
    (cotangent_out,) = vjp(cotangent)

    lhs = jnp.vdot(cotangent, tangent_out)
    rhs_dot = tangent * cotangent_out
    assert jnp.allclose(lhs, rhs_dot, rtol=1e-6, atol=1e-6)


def test_fp64_logdet_solve_gradient_matches_finite_difference():
    rhs = _exact_interval_vector([1.0, -2.0])
    probes = jnp.stack(
        [
            _exact_interval_vector([1.0, 0.0]),
            _exact_interval_vector([0.0, 1.0]),
        ],
        axis=0,
    )

    def objective(theta_value):
        bundle = jrb_mat.jrb_mat_logdet_solve_point(
            _dense_plan(theta_value),
            rhs,
            probes,
            2,
            symmetric=True,
        )
        return jnp.real(bundle.logdet) + 0.1 * jnp.sum(di.midpoint(bundle.solve))

    theta = jnp.asarray(0.15, dtype=jnp.float64)
    bundle = jrb_mat.jrb_mat_logdet_solve_point(_dense_plan(theta), rhs, probes, 2, symmetric=True)
    assert bundle.aux.implicit_adjoint
    assert bundle.aux.transpose_operator is not None
    grad_value = jax.grad(objective)(theta)
    step = jnp.asarray(1e-5, dtype=jnp.float64)
    finite_diff = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert jnp.allclose(grad_value, finite_diff, rtol=5e-4, atol=5e-4)


def test_fp64_inverse_action_gradient_matches_finite_difference():
    rhs = _exact_interval_vector([1.0, -2.0])

    def objective(theta_value):
        solved = jrb_mat.jrb_mat_inverse_action_point(
            _dense_plan(theta_value),
            rhs,
            symmetric=True,
        )
        return jnp.sum(di.midpoint(solved))

    theta = jnp.asarray(0.2, dtype=jnp.float64)
    grad_value = jax.grad(objective)(theta)
    step = jnp.asarray(1e-5, dtype=jnp.float64)
    finite_diff = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert jnp.allclose(grad_value, finite_diff, rtol=5e-4, atol=5e-4)


def test_fp64_log_action_gradient_matches_finite_difference():
    vec = _exact_interval_vector([1.0, -1.5])

    def objective(theta_value):
        value = jrb_mat.jrb_mat_log_action_lanczos_point(
            _dense_plan(theta_value),
            vec,
            2,
        )
        return jnp.sum(di.midpoint(value))

    theta = jnp.asarray(0.1, dtype=jnp.float64)
    grad_value = jax.grad(objective)(theta)
    step = jnp.asarray(1e-5, dtype=jnp.float64)
    finite_diff = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert jnp.allclose(grad_value, finite_diff, rtol=5e-4, atol=5e-4)


def test_complex_hermitian_logdet_solve_gradient_matches_finite_difference():
    rhs = _complex_box_vector([1.0 + 0.0j, -2.0 + 0.0j])
    probes = jnp.stack(
        [
            _complex_box_vector([1.0 + 0.0j, 0.0 + 0.0j]),
            _complex_box_vector([0.0 + 0.0j, 1.0 + 0.0j]),
        ],
        axis=0,
    )

    def objective(theta_value):
        bundle = jcb_mat.jcb_mat_logdet_solve_point(
            _complex_dense_plan(theta_value),
            rhs,
            probes,
            2,
            hermitian=True,
        )
        return jnp.real(bundle.logdet) + 0.1 * jnp.sum(jnp.real(acb_core.acb_midpoint(bundle.solve)))

    theta = jnp.asarray(0.15, dtype=jnp.float64)
    grad_value = jax.grad(objective)(theta)
    step = jnp.asarray(1e-5, dtype=jnp.float64)
    finite_diff = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert jnp.allclose(grad_value, finite_diff, rtol=5e-4, atol=5e-4)
