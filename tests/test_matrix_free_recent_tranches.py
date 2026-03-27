import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import matrix_free_core


def _mat2(a11, a12, a21, a22):
    dense = jnp.asarray([[a11, a12], [a21, a22]], dtype=jnp.float64)
    return jax.vmap(jax.vmap(lambda v: di.interval(v, v)))(dense)


def _box(re, im):
    interval_re = di.interval(jnp.asarray(re, dtype=jnp.float64), jnp.asarray(re, dtype=jnp.float64))
    interval_im = di.interval(jnp.asarray(im, dtype=jnp.float64), jnp.asarray(im, dtype=jnp.float64))
    return acb_core.acb_box(interval_re, interval_im)


def _cmat2(a11, a12, a21, a22):
    return jnp.stack(
        [
            jnp.stack([_box(jnp.real(a11), jnp.imag(a11)), _box(jnp.real(a12), jnp.imag(a12))], axis=0),
            jnp.stack([_box(jnp.real(a21), jnp.imag(a21)), _box(jnp.real(a22), jnp.imag(a22))], axis=0),
        ],
        axis=0,
    )


def _vec2(x0, x1):
    return jnp.asarray([x0, x1], dtype=jnp.float64)


def _cvec2(x0, x1):
    return jnp.asarray([x0, x1], dtype=jnp.complex128)


def _ivec2(x0, x1):
    return jnp.stack(
        [
            di.interval(jnp.asarray(x0, dtype=jnp.float64), jnp.asarray(x0, dtype=jnp.float64)),
            di.interval(jnp.asarray(x1, dtype=jnp.float64), jnp.asarray(x1, dtype=jnp.float64)),
        ],
        axis=0,
    )


def _bvec2(x0, x1):
    return jnp.stack([_box(jnp.real(x0), jnp.imag(x0)), _box(jnp.real(x1), jnp.imag(x1))], axis=0)


def test_recent_real_matrix_free_contour_and_cached_rational_surfaces():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    x = _ivec2(1.0, -1.0)
    center = jnp.asarray(2.5, dtype=jnp.float64)
    radius = jnp.asarray(0.75, dtype=jnp.float64)

    sinh_val = di.midpoint(jrb_mat.jrb_mat_sinh_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    cosh_val = di.midpoint(jrb_mat.jrb_mat_cosh_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    tanh_val = di.midpoint(jrb_mat.jrb_mat_tanh_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    exp_val = di.midpoint(jrb_mat.jrb_mat_exp_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    tan_val = di.midpoint(jrb_mat.jrb_mat_tan_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))

    assert sinh_val.shape == (2,)
    assert cosh_val.shape == (2,)
    assert tanh_val.shape == (2,)
    assert exp_val.shape == (2,)
    assert tan_val.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(sinh_val)))
    assert bool(jnp.all(jnp.isfinite(cosh_val)))
    assert bool(jnp.all(jnp.isfinite(tanh_val)))
    assert bool(jnp.all(jnp.isfinite(exp_val)))
    assert bool(jnp.all(jnp.isfinite(tan_val)))

    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    intercept = log2 - 2.0 * slope
    coeffs = jnp.asarray([intercept, slope], dtype=jnp.float64)
    sketch = jnp.stack([_ivec2(1.0, 0.0)], axis=0)
    residual = jnp.stack([_ivec2(0.0, 1.0)], axis=0)
    metadata = jrb_mat.jrb_mat_logdet_rational_hutchpp_prepare_point(
        op,
        sketch,
        shifts=jnp.zeros((0,), dtype=jnp.float64),
        weights=jnp.zeros((0,), dtype=jnp.float64),
        polynomial_coefficients=coeffs,
    )
    cached = jrb_mat.jrb_mat_logdet_rational_hutchpp_from_metadata_point(metadata, residual)
    direct = jrb_mat.jrb_mat_logdet_rational_hutchpp_point(
        op,
        sketch,
        residual,
        shifts=jnp.zeros((0,), dtype=jnp.float64),
        weights=jnp.zeros((0,), dtype=jnp.float64),
        polynomial_coefficients=coeffs,
    )
    exact_logdet = jnp.log(2.0) + jnp.log(3.0)
    assert bool(metadata.cached_adjoint_supported)
    assert jnp.allclose(matrix_free_core.hutchpp_trace_from_metadata(cached), exact_logdet, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(direct, exact_logdet, rtol=1e-6, atol=1e-6)
    assert int(matrix_free_core.rational_hutchpp_next_probe_count(metadata, cached)) >= int(cached.statistics.probe_count)
    assert bool(matrix_free_core.rational_hutchpp_should_stop(metadata, cached))


def test_recent_complex_matrix_free_contour_and_cached_rational_surfaces():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    x = _bvec2(1.0 + 0.0j, -1.0 + 0.0j)
    center = jnp.asarray(2.5 + 0.0j, dtype=jnp.complex128)
    radius = jnp.asarray(0.75, dtype=jnp.float64)

    sinh_val = acb_core.acb_midpoint(jcb_mat.jcb_mat_sinh_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    cosh_val = acb_core.acb_midpoint(jcb_mat.jcb_mat_cosh_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    tanh_val = acb_core.acb_midpoint(jcb_mat.jcb_mat_tanh_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    exp_val = acb_core.acb_midpoint(jcb_mat.jcb_mat_exp_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))
    tan_val = acb_core.acb_midpoint(jcb_mat.jcb_mat_tan_action_contour_point(op, x, center=center, radius=radius, quadrature_order=32))

    assert sinh_val.shape == (2,)
    assert cosh_val.shape == (2,)
    assert tanh_val.shape == (2,)
    assert exp_val.shape == (2,)
    assert tan_val.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(jnp.real(sinh_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(sinh_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.real(cosh_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(cosh_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.real(tanh_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(tanh_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.real(exp_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(exp_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.real(tan_val))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(tan_val))))

    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    intercept = log2 - 2.0 * slope
    coeffs = jnp.asarray([intercept, slope], dtype=jnp.float64)
    sketch = jnp.stack([_bvec2(1.0 + 0.0j, 0.0 + 0.0j)], axis=0)
    residual = jnp.stack([_bvec2(0.0 + 0.0j, 1.0 + 0.0j)], axis=0)
    metadata = jcb_mat.jcb_mat_logdet_rational_hutchpp_prepare_point(
        op,
        sketch,
        shifts=jnp.zeros((0,), dtype=jnp.complex128),
        weights=jnp.zeros((0,), dtype=jnp.complex128),
        polynomial_coefficients=coeffs,
        hermitian=True,
    )
    cached = jcb_mat.jcb_mat_logdet_rational_hutchpp_from_metadata_point(metadata, residual)
    direct = jcb_mat.jcb_mat_logdet_rational_hutchpp_point(
        op,
        sketch,
        residual,
        shifts=jnp.zeros((0,), dtype=jnp.complex128),
        weights=jnp.zeros((0,), dtype=jnp.complex128),
        polynomial_coefficients=coeffs,
        hermitian=True,
    )
    exact_logdet = jnp.log(2.0 + 0.0j) + jnp.log(3.0 + 0.0j)
    assert bool(metadata.cached_adjoint_supported)
    assert jnp.allclose(matrix_free_core.hutchpp_trace_from_metadata(cached), exact_logdet, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(direct, exact_logdet, rtol=1e-6, atol=1e-6)
    assert int(matrix_free_core.rational_hutchpp_next_probe_count(metadata, cached)) >= int(cached.statistics.probe_count)
    assert bool(matrix_free_core.rational_hutchpp_should_stop(metadata, cached))


def test_recent_probe_statistics_and_restart_policy_helpers():
    stats = matrix_free_core.make_probe_estimate_statistics(
        jnp.asarray([1.0, 1.5, 0.5, 1.0], dtype=jnp.float64),
        target_stderr=0.6,
        min_probes=2,
        max_probes=8,
        block_size=2,
    )
    assert int(stats.recommended_probe_count) % 2 == 0
    assert bool(matrix_free_core.probe_statistics_should_stop(stats, target_stderr=1.0, max_probes=8))
    assert int(matrix_free_core.probe_statistics_probe_deficit(stats)) >= 0
    assert int(matrix_free_core.probe_statistics_next_probe_count(stats, block_size=2, max_probes=8)) >= int(stats.probe_count)

    evals = jnp.asarray([5.0, 4.0, 3.0], dtype=jnp.float64)
    residuals = jnp.asarray([[1e-12, 1e-4, 1e-2]], dtype=jnp.float64)
    order = matrix_free_core.eig_restart_column_order(evals, residuals, which="largest", lock_tol=1e-6)
    filtered = matrix_free_core.eig_filter_residual_corrections(residuals, lock_tol=1e-6)
    converged_count, locked_count, deflated_count, converged = matrix_free_core.eig_convergence_summary(
        residuals,
        requested=1,
        tol=1e-6,
    )
    assert order.shape == (3,)
    assert float(filtered[0, 0]) == 0.0
    assert int(matrix_free_core.eig_target_subspace_cols(size=6, seed_cols=2, residual_cols=3, block_size=2)) == 4
    assert int(converged_count) == 1
    assert int(locked_count) == 1
    assert int(deflated_count) == 1
    assert bool(converged)
