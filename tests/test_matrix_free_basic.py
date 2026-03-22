import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat

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
    logdet_basic = jrb_mat.jrb_mat_logdet_slq_basic(plan, probes, 2)
    logdet_basic_diag, logdet_diag = jrb_mat.jrb_mat_logdet_slq_with_diagnostics_basic(plan, probes, 2)
    det_point = jrb_mat.jrb_mat_det_slq_point(plan, probes, 2)
    det_basic = jrb_mat.jrb_mat_det_slq_basic(plan, probes, 2)
    det_basic_diag, det_diag = jrb_mat.jrb_mat_det_slq_with_diagnostics_basic(plan, probes, 2)
    _check(logdet_basic.shape == (2,))
    _check(det_basic.shape == (2,))
    _check(bool(di.contains(logdet_basic, di.interval(logdet_point, logdet_point))))
    _check(bool(di.contains(det_basic, di.interval(det_point, det_point))))
    _check(bool(di.contains(logdet_basic_diag, di.interval(logdet_point, logdet_point))))
    _check(bool(di.contains(det_basic_diag, di.interval(det_point, det_point))))
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
    logdet_basic = jcb_mat.jcb_mat_logdet_slq_basic(plan, probes, 2, adj_plan)
    logdet_basic_diag, logdet_diag = jcb_mat.jcb_mat_logdet_slq_with_diagnostics_basic(plan, probes, 2, adj_plan)
    det_point = jcb_mat.jcb_mat_det_slq_point(plan, probes, 2, adj_plan)
    det_basic = jcb_mat.jcb_mat_det_slq_basic(plan, probes, 2, adj_plan)
    det_basic_diag, det_diag = jcb_mat.jcb_mat_det_slq_with_diagnostics_basic(plan, probes, 2, adj_plan)
    _check(logdet_basic.shape == (4,))
    _check(det_basic.shape == (4,))
    point_logdet_box = _cbox(logdet_point)
    point_det_box = _cbox(det_point)
    _check(bool(di.contains(acb_core.acb_real(logdet_basic), acb_core.acb_real(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_imag(logdet_basic), acb_core.acb_imag(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_real(det_basic), acb_core.acb_real(point_det_box))))
    _check(bool(di.contains(acb_core.acb_imag(det_basic), acb_core.acb_imag(point_det_box))))
    _check(bool(di.contains(acb_core.acb_real(logdet_basic_diag), acb_core.acb_real(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_imag(logdet_basic_diag), acb_core.acb_imag(point_logdet_box))))
    _check(bool(di.contains(acb_core.acb_real(det_basic_diag), acb_core.acb_real(point_det_box))))
    _check(bool(di.contains(acb_core.acb_imag(det_basic_diag), acb_core.acb_imag(point_det_box))))
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
    _check(bool(jnp.all(di.contains(value, jrb_mat.jrb_mat_log_action_lanczos_point(plan, x, steps=2)))))


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
