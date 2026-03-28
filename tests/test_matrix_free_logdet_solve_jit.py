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


def test_jrb_multi_shift_solve_point_jit_matches_recycled_plan_surface():
    dense = jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64))
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))
    rhs = _real_vec([2.0, 8.0])
    shifts = jnp.asarray([0.0, 1.0], dtype=jnp.float64)

    eager = jrb_mat.jrb_mat_multi_shift_solve_point(plan, rhs, shifts, symmetric=True, recycled_steps=2)
    compiled = jrb_mat.jrb_mat_multi_shift_solve_point_jit(plan, rhs, shifts, symmetric=True, recycled_steps=2)

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


def test_jcb_multi_shift_solve_point_jit_matches_recycled_plan_surface():
    dense_mid = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)
    rhs = _complex_box([2.0 + 0.0j, 15.0 + 0.0j])
    shifts = jnp.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128)

    eager = jcb_mat.jcb_mat_multi_shift_solve_point(plan, rhs, shifts, hermitian=True, recycled_steps=2)
    compiled = jcb_mat.jcb_mat_multi_shift_solve_point_jit(plan, rhs, shifts, hermitian=True, recycled_steps=2)

    assert jnp.allclose(acb_core.acb_midpoint(compiled), acb_core.acb_midpoint(eager), rtol=1e-6, atol=1e-6)


def test_jrb_logdet_rational_hutchpp_point_jit_matches_plan_surface():
    dense = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64))
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))
    sketch = jnp.stack([_real_vec([1.0, 0.0])], axis=0)
    residual = jnp.stack([_real_vec([0.0, 1.0])], axis=0)
    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    coeffs = jnp.asarray([log2 - 2.0 * slope, slope], dtype=jnp.float64)

    eager = jrb_mat.jrb_mat_logdet_rational_hutchpp_point(
        plan,
        sketch,
        residual,
        shifts=jnp.zeros((0,), dtype=jnp.float64),
        weights=jnp.zeros((0,), dtype=jnp.float64),
        polynomial_coefficients=coeffs,
    )
    compiled = jrb_mat.jrb_mat_logdet_rational_hutchpp_point_jit(
        plan,
        sketch,
        residual,
        shifts=jnp.zeros((0,), dtype=jnp.float64),
        weights=jnp.zeros((0,), dtype=jnp.float64),
        polynomial_coefficients=coeffs,
    )

    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)


def test_jcb_logdet_rational_hutchpp_point_jit_matches_plan_surface():
    dense_mid = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)
    sketch = jnp.stack([_complex_box([1.0 + 0.0j, 0.0 + 0.0j])], axis=0)
    residual = jnp.stack([_complex_box([0.0 + 0.0j, 1.0 + 0.0j])], axis=0)
    log2 = jnp.log(jnp.asarray(2.0 + 0.0j, dtype=jnp.complex128))
    slope = jnp.log(jnp.asarray(3.0 / 2.0 + 0.0j, dtype=jnp.complex128))
    coeffs = jnp.asarray([log2 - 2.0 * slope, slope], dtype=jnp.complex128)

    eager = jcb_mat.jcb_mat_logdet_rational_hutchpp_point(
        plan,
        sketch,
        residual,
        shifts=jnp.zeros((0,), dtype=jnp.complex128),
        weights=jnp.zeros((0,), dtype=jnp.complex128),
        polynomial_coefficients=coeffs,
        hermitian=True,
    )
    compiled = jcb_mat.jcb_mat_logdet_rational_hutchpp_point_jit(
        plan,
        sketch,
        residual,
        shifts=jnp.zeros((0,), dtype=jnp.complex128),
        weights=jnp.zeros((0,), dtype=jnp.complex128),
        polynomial_coefficients=coeffs,
        hermitian=True,
    )

    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)


def test_jrb_leja_hutchpp_point_jit_matches_plan_surface():
    dense = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64))
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))
    probes = jnp.stack([_real_vec([1.0, 1.0]), _real_vec([1.0, -1.0])], axis=0)
    residual = di.interval(jnp.zeros((0, 2), dtype=jnp.float64), jnp.zeros((0, 2), dtype=jnp.float64))

    eager_logdet = jrb_mat.jrb_mat_logdet_leja_hutchpp_point(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )
    compiled_logdet = jrb_mat.jrb_mat_logdet_leja_hutchpp_point_jit(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )
    eager_det = jrb_mat.jrb_mat_det_leja_hutchpp_point(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )
    compiled_det = jrb_mat.jrb_mat_det_leja_hutchpp_point_jit(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )

    assert jnp.allclose(compiled_logdet, eager_logdet, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(compiled_det, eager_det, rtol=1e-6, atol=1e-6)


def test_jcb_leja_hutchpp_point_jit_matches_plan_surface():
    dense_mid = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)
    probes = jnp.stack(
        [_complex_box([1.0 + 0.0j, 1.0 + 0.0j]), _complex_box([1.0 + 0.0j, -1.0 + 0.0j])],
        axis=0,
    )
    residual = jnp.zeros((0, 2, 4), dtype=jnp.float64)

    eager_logdet = jcb_mat.jcb_mat_logdet_leja_hutchpp_point(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )
    compiled_logdet = jcb_mat.jcb_mat_logdet_leja_hutchpp_point_jit(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )
    eager_det = jcb_mat.jcb_mat_det_leja_hutchpp_point(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )
    compiled_det = jcb_mat.jcb_mat_det_leja_hutchpp_point_jit(
        plan, probes, residual, degree=6, spectral_bounds=(2.0, 3.0)
    )

    assert jnp.allclose(compiled_logdet, eager_logdet, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(compiled_det, eager_det, rtol=1e-5, atol=1e-5)


def test_jrb_heat_trace_and_spectral_density_slq_point_jit_match_plan_surface():
    dense = jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64))
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))
    probes = jnp.stack([_real_vec([1.0, 1.0]), _real_vec([1.0, -1.0])], axis=0)
    bin_edges = jnp.asarray([1.5, 2.5, 4.5], dtype=jnp.float64)

    eager_heat = jrb_mat.jrb_mat_heat_trace_slq_point(plan, probes, 2, time=jnp.asarray(0.5, dtype=jnp.float64))
    compiled_heat = jrb_mat.jrb_mat_heat_trace_slq_point_jit(
        plan,
        probes,
        2,
        time=jnp.asarray(0.5, dtype=jnp.float64),
    )
    eager_density = jrb_mat.jrb_mat_spectral_density_slq_point(plan, probes, 2, bin_edges=bin_edges, normalize=True)
    compiled_density = jrb_mat.jrb_mat_spectral_density_slq_point_jit(
        plan,
        probes,
        2,
        bin_edges=bin_edges,
        normalize=True,
    )

    assert jnp.allclose(compiled_heat, eager_heat, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(compiled_density, eager_density, rtol=1e-6, atol=1e-6)


def test_jcb_heat_trace_and_spectral_density_slq_point_jit_match_plan_surface():
    dense_mid = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense = acb_core.acb_box(
        di.interval(jnp.real(dense_mid), jnp.real(dense_mid)),
        di.interval(jnp.imag(dense_mid), jnp.imag(dense_mid)),
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense)
    probes = jnp.stack([_complex_box([1.0 + 0.0j, 1.0 + 0.0j]), _complex_box([1.0 + 0.0j, -1.0 + 0.0j])], axis=0)
    bin_edges = jnp.asarray([1.5, 3.5, 5.5], dtype=jnp.float64)

    eager_heat = jcb_mat.jcb_mat_heat_trace_slq_hermitian_point(plan, probes, 2, time=jnp.asarray(0.5, dtype=jnp.float64))
    compiled_heat = jcb_mat.jcb_mat_heat_trace_slq_hermitian_point_jit(
        plan,
        probes,
        2,
        time=jnp.asarray(0.5, dtype=jnp.float64),
    )
    eager_density = jcb_mat.jcb_mat_spectral_density_slq_hermitian_point(
        plan,
        probes,
        2,
        bin_edges=bin_edges,
        normalize=True,
    )
    compiled_density = jcb_mat.jcb_mat_spectral_density_slq_hermitian_point_jit(
        plan,
        probes,
        2,
        bin_edges=bin_edges,
        normalize=True,
    )

    assert jnp.allclose(compiled_heat, eager_heat, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(compiled_density, eager_density, rtol=1e-6, atol=1e-6)
