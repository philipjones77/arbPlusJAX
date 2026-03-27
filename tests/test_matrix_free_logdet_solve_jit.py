import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat


def _real_vec(values):
    arr = jnp.asarray(values, dtype=jnp.float64)
    return di.interval(arr, arr)


def _complex_box(values):
    arr = jnp.asarray(values, dtype=jnp.complex128)
    return acb_core.acb_box(
        di.interval(jnp.real(arr), jnp.real(arr)),
        di.interval(jnp.imag(arr), jnp.imag(arr)),
    )


def test_jrb_logdet_solve_point_jit_matches_plan_surface():
    dense = jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64))
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))
    rhs = _real_vec([2.0, 8.0])
    probes = jnp.stack([rhs, _real_vec([1.0, 1.0])], axis=0)

    eager = jrb_mat.jrb_mat_logdet_solve_point(plan, rhs, probes, steps=2, symmetric=True)
    compiled = jrb_mat.jrb_mat_logdet_solve_point_jit(plan, rhs, probes, steps=2, symmetric=True)

    assert jnp.allclose(compiled.logdet, eager.logdet, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(di.midpoint(compiled.solve), di.midpoint(eager.solve), rtol=1e-6, atol=1e-6)
    assert bool(compiled.aux.implicit_adjoint)


def test_jcb_logdet_solve_point_jit_matches_plan_surface():
    dense_mid = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)
    rhs = _complex_box([2.0 + 0.0j, 15.0 + 0.0j])
    probes = jnp.stack([rhs, _complex_box([1.0 + 0.0j, 1.0 + 0.0j])], axis=0)

    eager = jcb_mat.jcb_mat_logdet_solve_point(plan, rhs, probes, steps=2, hermitian=True)
    compiled = jcb_mat.jcb_mat_logdet_solve_point_jit(plan, rhs, probes, steps=2, hermitian=True)

    assert jnp.allclose(compiled.logdet, eager.logdet, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(acb_core.acb_midpoint(compiled.solve), acb_core.acb_midpoint(eager.solve), rtol=1e-6, atol=1e-6)
    assert bool(compiled.aux.implicit_adjoint)


def test_jrb_multi_shift_solve_point_jit_matches_plan_surface():
    dense = jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64))
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))
    rhs = _real_vec([2.0, 8.0])
    shifts = jnp.asarray([0.0, 1.0], dtype=jnp.float64)

    eager = jrb_mat.jrb_mat_multi_shift_solve_point(plan, rhs, shifts, symmetric=True)
    compiled = jrb_mat.jrb_mat_multi_shift_solve_point_jit(plan, rhs, shifts, symmetric=True)

    assert jnp.allclose(di.midpoint(compiled), di.midpoint(eager), rtol=1e-6, atol=1e-6)


def test_jcb_multi_shift_solve_point_jit_matches_plan_surface():
    dense_mid = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)
    rhs = _complex_box([2.0 + 0.0j, 15.0 + 0.0j])
    shifts = jnp.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128)

    eager = jcb_mat.jcb_mat_multi_shift_solve_point(plan, rhs, shifts, hermitian=True)
    compiled = jcb_mat.jcb_mat_multi_shift_solve_point_jit(plan, rhs, shifts, hermitian=True)

    assert jnp.allclose(acb_core.acb_midpoint(compiled), acb_core.acb_midpoint(eager), rtol=1e-6, atol=1e-6)
