import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import matrix_free_basic

from tests._test_checks import _check


def _rinterval(x: float) -> jnp.ndarray:
    return di.interval(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64))


def _rvec2(x0: float, x1: float) -> jnp.ndarray:
    return jnp.stack([_rinterval(x0), _rinterval(x1)], axis=0)


def _rmat2(a00: float, a01: float, a10: float, a11: float) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([_rinterval(a00), _rinterval(a01)], axis=0),
            jnp.stack([_rinterval(a10), _rinterval(a11)], axis=0),
        ],
        axis=0,
    )


def _cbox(z: complex) -> jnp.ndarray:
    return acb_core.acb_box(_rinterval(z.real), _rinterval(z.imag))


def _cvec2(x0: complex, x1: complex) -> jnp.ndarray:
    return jnp.stack([_cbox(x0), _cbox(x1)], axis=0)


def _cmat2(a00: complex, a01: complex, a10: complex, a11: complex) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([_cbox(a00), _cbox(a01)], axis=0),
            jnp.stack([_cbox(a10), _cbox(a11)], axis=0),
        ],
        axis=0,
    )


def test_real_matrix_free_basic_contracts():
    a = _rmat2(2.0, 0.0, 0.0, 3.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    probes = jnp.stack([x, _rvec2(1.0, 1.0)], axis=0)

    apply_point = jrb_mat.jrb_mat_operator_apply_point(plan, x)
    apply_basic = jrb_mat.jrb_mat_operator_apply_basic(plan, x)
    _check(bool(jnp.all(di.contains(apply_basic, apply_point))))

    coeffs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    poly_point = jrb_mat.jrb_mat_poly_action_point(plan, x, coeffs)
    poly_basic = jrb_mat.jrb_mat_poly_action_basic(plan, x, coeffs)
    _check(bool(jnp.all(di.contains(poly_basic, poly_point))))

    expm_point = jrb_mat.jrb_mat_expm_action_point(plan, x, terms=8)
    expm_basic = jrb_mat.jrb_mat_expm_action_basic(plan, x, terms=8)
    _check(bool(jnp.all(di.contains(expm_basic, expm_point))))

    log_point = jrb_mat.jrb_mat_log_action_lanczos_point(plan, x, steps=2)
    log_basic = jrb_mat.jrb_mat_log_action_lanczos_basic(plan, x, steps=2)
    _check(bool(jnp.all(di.contains(log_basic, log_point))))

    solve_point = jrb_mat.jrb_mat_solve_action_point(plan, x, symmetric=True, maxiter=8)
    solve_basic = jrb_mat.jrb_mat_solve_action_basic(plan, x, symmetric=True, maxiter=8)
    solve_basic_diag, solve_diag = jrb_mat.jrb_mat_solve_action_with_diagnostics_basic(plan, x, symmetric=True, maxiter=8)
    _check(bool(jnp.all(di.contains(solve_basic, solve_point))))
    _check(bool(jnp.all(di.contains(solve_basic_diag, solve_point))))
    _check(solve_diag is not None)

    inv_point = jrb_mat.jrb_mat_inverse_action_point(plan, x, symmetric=True, maxiter=8)
    inv_basic = jrb_mat.jrb_mat_inverse_action_basic(plan, x, symmetric=True, maxiter=8)
    _check(bool(jnp.all(di.contains(inv_basic, inv_point))))

    minres_point = jrb_mat.jrb_mat_minres_solve_action_point(plan, x, maxiter=8)
    minres_basic = jrb_mat.jrb_mat_minres_solve_action_basic(plan, x, maxiter=8)
    minres_basic_diag, minres_diag = jrb_mat.jrb_mat_minres_solve_action_with_diagnostics_basic(plan, x, maxiter=8)
    minres_inv_basic = jrb_mat.jrb_mat_minres_inverse_action_basic(plan, x, maxiter=8)
    _check(bool(jnp.all(di.contains(minres_basic, minres_point))))
    _check(bool(jnp.all(di.contains(minres_basic_diag, minres_point))))
    _check(bool(jnp.all(di.contains(minres_inv_basic, minres_point))))
    _check(bool(jnp.isfinite(minres_diag.primal_residual)))

    logdet_point = jrb_mat.jrb_mat_logdet_slq_point(plan, probes, 2)
    logdet_est_basic = jrb_mat.jrb_mat_logdet_estimate_basic(plan, probes, 2)
    logdet_basic = jrb_mat.jrb_mat_logdet_slq_basic(plan, probes, 2)
    logdet_basic_diag, logdet_diag = jrb_mat.jrb_mat_logdet_slq_with_diagnostics_basic(plan, probes, 2)
    det_point = jrb_mat.jrb_mat_det_slq_point(plan, probes, 2)
    det_basic = jrb_mat.jrb_mat_det_slq_basic(plan, probes, 2)
    det_basic_diag, det_diag = jrb_mat.jrb_mat_det_slq_with_diagnostics_basic(plan, probes, 2)
    heat_point = jrb_mat.jrb_mat_heat_trace_slq_point(plan, probes, 2, time=0.5)
    heat_basic = jrb_mat.jrb_mat_heat_trace_slq_basic(plan, probes, 2, time=0.5)
    hutch_point = jrb_mat.jrb_mat_hutchpp_trace_estimate_point(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )
    hutch_basic = jrb_mat.jrb_mat_hutchpp_trace_estimate_basic(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )
    _check(logdet_basic.shape == (2,))
    _check(det_basic.shape == (2,))
    _check(heat_basic.shape == (2,))
    _check(hutch_basic.shape == (2,))
    _check(bool(di.contains(logdet_est_basic, di.interval(logdet_point, logdet_point))))
    _check(bool(di.contains(logdet_basic, di.interval(logdet_point, logdet_point))))
    _check(bool(di.contains(det_basic, di.interval(det_point, det_point))))
    _check(bool(di.contains(logdet_basic_diag, di.interval(logdet_point, logdet_point))))
    _check(bool(di.contains(det_basic_diag, di.interval(det_point, det_point))))
    _check(bool(di.contains(heat_basic, di.interval(heat_point, heat_point))))
    _check(bool(di.contains(hutch_basic, di.interval(hutch_point, hutch_point))))
    _check(bool(jnp.isfinite(logdet_diag.primal_residual)))
    _check(bool(jnp.isfinite(det_diag.primal_residual)))


def test_complex_matrix_free_basic_contracts():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 1.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj_plan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)
    probes = jnp.stack([x, _cvec2(1.0 + 0.0j, 1.0 + 0.0j)], axis=0)

    apply_point = jcb_mat.jcb_mat_operator_apply_point(plan, x)
    apply_basic = jcb_mat.jcb_mat_operator_apply_basic(plan, x)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(apply_basic), acb_core.acb_real(apply_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(apply_basic), acb_core.acb_imag(apply_point)))))

    coeffs = jnp.asarray([1.0 + 0.0j, 2.0 - 1.0j], dtype=jnp.complex128)
    poly_point = jcb_mat.jcb_mat_poly_action_point(plan, x, coeffs)
    poly_basic = jcb_mat.jcb_mat_poly_action_basic(plan, x, coeffs)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(poly_basic), acb_core.acb_real(poly_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(poly_basic), acb_core.acb_imag(poly_point)))))

    expm_point = jcb_mat.jcb_mat_expm_action_point(plan, x, terms=8)
    expm_basic = jcb_mat.jcb_mat_expm_action_basic(plan, x, terms=8)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(expm_basic), acb_core.acb_real(expm_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(expm_basic), acb_core.acb_imag(expm_point)))))

    log_point = jcb_mat.jcb_mat_log_action_arnoldi_point(plan, x, steps=2, adjoint_matvec=adj_plan)
    log_basic = jcb_mat.jcb_mat_log_action_arnoldi_basic(plan, x, steps=2, adjoint_matvec=adj_plan)
    herm_point = jcb_mat.jcb_mat_log_action_hermitian_point(plan, x, steps=2, adjoint_matvec=adj_plan)
    herm_basic = jcb_mat.jcb_mat_log_action_hermitian_basic(plan, x, steps=2, adjoint_matvec=adj_plan)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(log_basic), acb_core.acb_real(log_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(log_basic), acb_core.acb_imag(log_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(herm_basic), acb_core.acb_real(herm_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(herm_basic), acb_core.acb_imag(herm_point)))))

    solve_point = jcb_mat.jcb_mat_solve_action_point(plan, x, hermitian=False, maxiter=8)
    solve_basic = jcb_mat.jcb_mat_solve_action_basic(plan, x, hermitian=False, maxiter=8)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(solve_basic), acb_core.acb_real(solve_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(solve_basic), acb_core.acb_imag(solve_point)))))

    inv_point = jcb_mat.jcb_mat_inverse_action_point(plan, x, hermitian=False, maxiter=8)
    inv_basic = jcb_mat.jcb_mat_inverse_action_basic(plan, x, hermitian=False, maxiter=8)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(inv_basic), acb_core.acb_real(inv_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(inv_basic), acb_core.acb_imag(inv_point)))))

    minres_point = jcb_mat.jcb_mat_minres_solve_action_point(plan, x, maxiter=8)
    minres_basic = jcb_mat.jcb_mat_minres_solve_action_basic(plan, x, maxiter=8)
    minres_basic_diag, minres_diag = jcb_mat.jcb_mat_minres_solve_action_with_diagnostics_basic(plan, x, maxiter=8)
    minres_inv_basic = jcb_mat.jcb_mat_minres_inverse_action_basic(plan, x, maxiter=8)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(minres_basic), acb_core.acb_real(minres_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(minres_basic), acb_core.acb_imag(minres_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(minres_basic_diag), acb_core.acb_real(minres_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(minres_basic_diag), acb_core.acb_imag(minres_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(minres_inv_basic), acb_core.acb_real(minres_point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(minres_inv_basic), acb_core.acb_imag(minres_point)))))
    _check(bool(jnp.isfinite(minres_diag.primal_residual)))

    logdet_point = jcb_mat.jcb_mat_logdet_slq_point(plan, probes, 2, adj_plan)
    trace_point = jcb_mat.jcb_mat_trace_estimate_point(plan, probes, adj_plan)
    trace_basic = jcb_mat.jcb_mat_trace_estimate_basic(plan, probes, adj_plan)
    logdet_est_basic = jcb_mat.jcb_mat_logdet_estimate_basic(plan, probes, 2, adj_plan)
    logdet_basic = jcb_mat.jcb_mat_logdet_slq_basic(plan, probes, 2, adj_plan)
    logdet_basic_diag, logdet_diag = jcb_mat.jcb_mat_logdet_slq_with_diagnostics_basic(plan, probes, 2, adj_plan)
    det_point = jcb_mat.jcb_mat_det_slq_point(plan, probes, 2, adj_plan)
    det_basic = jcb_mat.jcb_mat_det_slq_basic(plan, probes, 2, adj_plan)
    det_basic_diag, det_diag = jcb_mat.jcb_mat_det_slq_with_diagnostics_basic(plan, probes, 2, adj_plan)
    heat_point = jcb_mat.jcb_mat_heat_trace_slq_hermitian_point(plan, probes, 2, time=0.5)
    heat_basic = jcb_mat.jcb_mat_heat_trace_slq_hermitian_basic(plan, probes, 2, time=0.5)
    hutch_point = jcb_mat.jcb_mat_hutchpp_trace_estimate_point(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )
    hutch_basic = jcb_mat.jcb_mat_hutchpp_trace_estimate_basic(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )
    _check(logdet_basic.shape == (4,))
    _check(det_basic.shape == (4,))
    _check(trace_basic.shape == (4,))
    _check(logdet_est_basic.shape == (4,))
    _check(heat_basic.shape == (4,))
    _check(hutch_basic.shape == (4,))
    point_trace_box = _cbox(trace_point)
    point_logdet_box = _cbox(logdet_point)
    point_det_box = _cbox(det_point)
    point_heat_box = _cbox(heat_point)
    point_hutch_box = _cbox(hutch_point)
    _check(bool(di.contains(acb_core.acb_real(trace_basic), acb_core.acb_real(point_trace_box))))
    _check(bool(di.contains(acb_core.acb_imag(trace_basic), acb_core.acb_imag(point_trace_box))))
    _check(bool(di.contains(acb_core.acb_real(logdet_est_basic), acb_core.acb_real(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_imag(logdet_est_basic), acb_core.acb_imag(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_real(logdet_basic), acb_core.acb_real(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_imag(logdet_basic), acb_core.acb_imag(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_real(det_basic), acb_core.acb_real(point_det_box))))
    _check(bool(di.contains(acb_core.acb_imag(det_basic), acb_core.acb_imag(point_det_box))))
    _check(bool(di.contains(acb_core.acb_real(logdet_basic_diag), acb_core.acb_real(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_imag(logdet_basic_diag), acb_core.acb_imag(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_real(det_basic_diag), acb_core.acb_real(point_det_box))))
    _check(bool(di.contains(acb_core.acb_imag(det_basic_diag), acb_core.acb_imag(point_det_box))))
    _check(bool(di.contains(acb_core.acb_real(heat_basic), acb_core.acb_real(point_heat_box))))
    _check(bool(di.contains(acb_core.acb_imag(heat_basic), acb_core.acb_imag(point_heat_box))))
    _check(bool(di.contains(acb_core.acb_real(hutch_basic), acb_core.acb_real(point_hutch_box))))
    _check(bool(di.contains(acb_core.acb_imag(hutch_basic), acb_core.acb_imag(point_hutch_box))))
    _check(bool(jnp.isfinite(logdet_diag.primal_residual)))
    _check(bool(jnp.isfinite(det_diag.primal_residual)))


def test_real_basic_solve_invalidates_when_krylov_residual_is_too_large():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    value, diag = jrb_mat.jrb_mat_solve_action_with_diagnostics_basic(
        plan,
        x,
        symmetric=True,
        maxiter=1,
        tol=1e-14,
    )

    _check(bool(diag.primal_residual > jnp.maximum(1e-14 * jnp.abs(diag.beta0), 0.0)))
    _check(bool(jnp.all(jnp.isneginf(value[..., 0]))))
    _check(bool(jnp.all(jnp.isposinf(value[..., 1]))))


def test_real_basic_log_action_invalidates_when_diagnostics_are_non_converged():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    value, diag = jrb_mat.jrb_mat_log_action_lanczos_with_diagnostics_basic(
        plan,
        x,
        steps=2,
        prec_bits=di.DEFAULT_PREC_BITS,
    )

    _check(bool(diag.converged))


def test_real_basic_log_action_inflates_by_reported_krylov_uncertainty():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    value, diag = jrb_mat.jrb_mat_log_action_lanczos_with_diagnostics_basic(
        plan,
        x,
        steps=2,
        prec_bits=di.DEFAULT_PREC_BITS,
    )

    rad = matrix_free_basic.scalar_uncertainty_radius(diag)
    _check(bool(jnp.all(di.ubound_radius(value) >= rad)))


def test_real_basic_sqrt_action_inflates_by_reported_krylov_uncertainty():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    value = jrb_mat.jrb_mat_sqrt_action_lanczos_basic(
        plan,
        x,
        steps=2,
        prec_bits=di.DEFAULT_PREC_BITS,
    )
    _, diag = jrb_mat.jrb_mat_sqrt_action_lanczos_with_diagnostics_point(
        plan,
        x,
        steps=2,
    )

    rad = matrix_free_basic.scalar_uncertainty_radius(diag)
    _check(bool(jnp.all(di.ubound_radius(value) >= rad)))


def test_real_basic_poly_and_expm_actions_inflate_by_reported_uncertainty():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    coeffs = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float64)

    poly_value, poly_diag = jrb_mat.jrb_mat_poly_action_with_diagnostics_basic(
        plan,
        x,
        coeffs,
        prec_bits=di.DEFAULT_PREC_BITS,
    )
    expm_value, expm_diag = jrb_mat.jrb_mat_expm_action_with_diagnostics_basic(
        plan,
        x,
        terms=8,
        prec_bits=di.DEFAULT_PREC_BITS,
    )

    _check(bool(jnp.all(di.ubound_radius(poly_value) >= matrix_free_basic.scalar_uncertainty_radius(poly_diag))))
    _check(bool(jnp.all(di.ubound_radius(expm_value) >= matrix_free_basic.scalar_uncertainty_radius(expm_diag))))


def test_real_basic_logdet_surfaces_inflate_by_reported_scalar_uncertainty():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    probes = jnp.stack([x, _rvec2(1.0, 1.0)], axis=0)

    logdet_basic, logdet_diag = jrb_mat.jrb_mat_logdet_slq_with_diagnostics_basic(plan, probes, 2)
    bundle = jrb_mat.jrb_mat_logdet_solve_basic(plan, x, probes, 2, symmetric=True, maxiter=8)
    rad = matrix_free_basic.scalar_uncertainty_radius(logdet_diag)
    bundle_rad = matrix_free_basic.scalar_uncertainty_radius(bundle.aux.logdet_diagnostics)

    _check(bool(di.ubound_radius(logdet_basic) >= rad))
    _check(bool(di.ubound_radius(bundle.logdet) >= bundle_rad))


def test_real_basic_trace_and_hutchpp_surfaces_inflate_by_estimator_uncertainty():
    a = _rmat2(2.0, 0.0, 0.0, 5.0)
    x = _rvec2(1.0, -2.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    probes = jnp.stack([x, _rvec2(1.0, 1.0)], axis=0)

    trace_basic = jrb_mat.jrb_mat_trace_estimate_basic(plan, probes)
    _, trace_diag = jrb_mat.jrb_mat_trace_estimator_with_diagnostics_point(plan, probes)
    hutch_basic = jrb_mat.jrb_mat_hutchpp_trace_estimate_basic(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )
    hutch_meta = jrb_mat.jrb_mat_hutchpp_trace_with_metadata_point(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )

    _check(bool(di.ubound_radius(trace_basic) >= matrix_free_basic.scalar_uncertainty_radius(trace_diag)))
    _check(bool(di.ubound_radius(hutch_basic) >= matrix_free_basic.scalar_uncertainty_radius(hutch_meta.statistics)))


def test_complex_basic_logdet_surfaces_inflate_by_reported_scalar_uncertainty():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 1.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj_plan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)
    probes = jnp.stack([x, _cvec2(1.0 + 0.0j, 1.0 + 0.0j)], axis=0)

    logdet_basic, logdet_diag = jcb_mat.jcb_mat_logdet_slq_with_diagnostics_basic(plan, probes, 2, adj_plan)
    bundle = jcb_mat.jcb_mat_logdet_solve_basic(plan, x, probes, 2, adj_plan, hermitian=False, maxiter=8)
    rad = matrix_free_basic.scalar_uncertainty_radius(logdet_diag)
    bundle_rad = matrix_free_basic.scalar_uncertainty_radius(bundle.aux.logdet_diagnostics)

    _check(bool(di.ubound_radius(acb_core.acb_real(logdet_basic)) >= rad))
    _check(bool(di.ubound_radius(acb_core.acb_imag(logdet_basic)) >= rad))
    _check(bool(di.ubound_radius(acb_core.acb_real(bundle.logdet)) >= bundle_rad))
    _check(bool(di.ubound_radius(acb_core.acb_imag(bundle.logdet)) >= bundle_rad))


def test_complex_basic_trace_and_hutchpp_surfaces_inflate_by_estimator_uncertainty():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 1.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj_plan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)
    probes = jnp.stack([x, _cvec2(1.0 + 0.0j, 1.0 + 0.0j)], axis=0)

    trace_basic = jcb_mat.jcb_mat_trace_estimate_basic(plan, probes, adj_plan)
    _, trace_diag = jcb_mat.jcb_mat_trace_estimator_with_diagnostics_point(plan, probes, adj_plan)
    hutch_basic = jcb_mat.jcb_mat_hutchpp_trace_estimate_basic(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )
    hutch_meta = jcb_mat.jcb_mat_hutchpp_trace_with_metadata_point(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(plan, v, 2),
        probes[:1],
        probes[1:],
    )

    trace_rad = matrix_free_basic.scalar_uncertainty_radius(trace_diag)
    hutch_rad = matrix_free_basic.scalar_uncertainty_radius(hutch_meta.statistics)
    _check(bool(di.ubound_radius(acb_core.acb_real(trace_basic)) >= trace_rad))
    _check(bool(di.ubound_radius(acb_core.acb_imag(trace_basic)) >= trace_rad))
    _check(bool(di.ubound_radius(acb_core.acb_real(hutch_basic)) >= hutch_rad))
    _check(bool(di.ubound_radius(acb_core.acb_imag(hutch_basic)) >= hutch_rad))


def test_complex_basic_solve_invalidates_when_krylov_residual_is_too_large():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 0.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)

    value, diag = jcb_mat.jcb_mat_solve_action_with_diagnostics_basic(
        plan,
        x,
        hermitian=True,
        maxiter=1,
        tol=1e-14,
    )

    _check(bool(diag.primal_residual > jnp.maximum(1e-14 * jnp.abs(diag.beta0), 0.0)))
    _check(bool(jnp.all(jnp.isneginf(acb_core.acb_real(value)[..., 0]))))
    _check(bool(jnp.all(jnp.isposinf(acb_core.acb_real(value)[..., 1]))))
    _check(bool(jnp.all(jnp.isneginf(acb_core.acb_imag(value)[..., 0]))))
    _check(bool(jnp.all(jnp.isposinf(acb_core.acb_imag(value)[..., 1]))))


def test_complex_basic_log_action_hermitian_with_diagnostics_round_trips():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 0.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)

    value, diag = jcb_mat.jcb_mat_log_action_hermitian_with_diagnostics_basic(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
        prec_bits=di.DEFAULT_PREC_BITS,
    )

    point = jcb_mat.jcb_mat_log_action_hermitian_point(plan, x, steps=2, adjoint_matvec=adj)
    _check(bool(diag.converged))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(value), acb_core.acb_real(point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(value), acb_core.acb_imag(point)))))


def test_complex_basic_log_actions_inflate_by_reported_krylov_uncertainty():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 0.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)

    arnoldi_value, arnoldi_diag = jcb_mat.jcb_mat_log_action_arnoldi_with_diagnostics_basic(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
        prec_bits=di.DEFAULT_PREC_BITS,
    )
    herm_value, herm_diag = jcb_mat.jcb_mat_log_action_hermitian_with_diagnostics_basic(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
        prec_bits=di.DEFAULT_PREC_BITS,
    )

    arnoldi_rad = matrix_free_basic.scalar_uncertainty_radius(arnoldi_diag)
    herm_rad = matrix_free_basic.scalar_uncertainty_radius(herm_diag)
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_real(arnoldi_value)) >= arnoldi_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_imag(arnoldi_value)) >= arnoldi_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_real(herm_value)) >= herm_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_imag(herm_value)) >= herm_rad)))


def test_complex_basic_sqrt_actions_inflate_by_reported_krylov_uncertainty():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 0.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)

    arnoldi_value = jcb_mat.jcb_mat_sqrt_action_arnoldi_basic(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
        prec_bits=di.DEFAULT_PREC_BITS,
    )
    _, arnoldi_diag = jcb_mat.jcb_mat_sqrt_action_arnoldi_with_diagnostics_point(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
    )
    herm_value = jcb_mat.jcb_mat_sqrt_action_hermitian_basic(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
        prec_bits=di.DEFAULT_PREC_BITS,
    )
    _, herm_diag = jcb_mat.jcb_mat_sqrt_action_hermitian_with_diagnostics_point(
        plan,
        x,
        steps=2,
        adjoint_matvec=adj,
    )

    arnoldi_rad = matrix_free_basic.scalar_uncertainty_radius(arnoldi_diag)
    herm_rad = matrix_free_basic.scalar_uncertainty_radius(herm_diag)
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_real(arnoldi_value)) >= arnoldi_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_imag(arnoldi_value)) >= arnoldi_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_real(herm_value)) >= herm_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_imag(herm_value)) >= herm_rad)))


def test_complex_basic_poly_and_expm_actions_inflate_by_reported_uncertainty():
    a = _cmat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 0.0j)
    x = _cvec2(1.0 + 1.0j, -2.0 + 0.5j)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    coeffs = jnp.asarray([1.0 + 0.0j, -0.5 + 0.25j, 0.25 - 0.1j], dtype=jnp.complex128)

    poly_value, poly_diag = jcb_mat.jcb_mat_poly_action_with_diagnostics_basic(
        plan,
        x,
        coeffs,
        prec_bits=di.DEFAULT_PREC_BITS,
    )
    expm_value, expm_diag = jcb_mat.jcb_mat_expm_action_with_diagnostics_basic(
        plan,
        x,
        terms=8,
        prec_bits=di.DEFAULT_PREC_BITS,
    )

    poly_rad = matrix_free_basic.scalar_uncertainty_radius(poly_diag)
    expm_rad = matrix_free_basic.scalar_uncertainty_radius(expm_diag)
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_real(poly_value)) >= poly_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_imag(poly_value)) >= poly_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_real(expm_value)) >= expm_rad)))
    _check(bool(jnp.all(di.ubound_radius(acb_core.acb_imag(expm_value)) >= expm_rad)))
