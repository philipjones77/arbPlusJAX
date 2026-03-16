import jax
import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat

from tests._test_checks import _check


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.asarray(lo, dtype=jnp.float64), jnp.asarray(hi, dtype=jnp.float64))


def _box(re: float, im: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re, re), _interval(im, im))


def _mat2(a00: complex, a01: complex, a10: complex, a11: complex) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([_box(a00.real, a00.imag), _box(a01.real, a01.imag)], axis=0),
            jnp.stack([_box(a10.real, a10.imag), _box(a11.real, a11.imag)], axis=0),
        ],
        axis=0,
    )


def _vec2(x0: complex, x1: complex) -> jnp.ndarray:
    return jnp.stack([_box(x0.real, x0.imag), _box(x1.real, x1.imag)], axis=0)


def test_layout_contracts_enforced():
    with pytest.raises(ValueError):
        jcb_mat.jcb_mat_as_box_matrix(jnp.zeros((2, 3, 4), dtype=jnp.float64))
    with pytest.raises(ValueError):
        jcb_mat.jcb_mat_as_box_vector(jnp.zeros((2, 3), dtype=jnp.float64))


def test_matmul_point_and_basic_exact_inputs():
    a = _mat2(1.0 + 1.0j, 2.0 + 0.0j, 0.0 + 1.0j, 3.0 - 1.0j)
    b = _mat2(2.0 + 0.0j, 0.0 + 1.0j, 1.0 - 1.0j, 2.0 + 0.0j)
    expected = jnp.asarray(
        [[(1.0 + 1.0j) * (2.0 + 0.0j) + (2.0 + 0.0j) * (1.0 - 1.0j), (1.0 + 1.0j) * 1.0j + 4.0],
         [(0.0 + 1.0j) * (2.0 + 0.0j) + (3.0 - 1.0j) * (1.0 - 1.0j), (0.0 + 1.0j) * 1.0j + (3.0 - 1.0j) * 2.0]],
        dtype=jnp.complex128,
    )

    point = jcb_mat.jcb_mat_matmul_point(a, b)
    basic = jcb_mat.jcb_mat_matmul_basic(a, b)

    _check(point.shape == (2, 2, 4))
    _check(basic.shape == (2, 2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(point), expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(basic), expected)))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(basic), acb_core.acb_real(point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(basic), acb_core.acb_imag(point)))))


def test_matvec_solve_jit_grad_and_precision():
    a = _mat2(2.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j, 3.0 - 1.0j)
    x = _vec2(1.0 + 2.0j, -1.0 + 1.0j)
    rhs_expected = jnp.asarray(
        [
            (2.0 + 0.0j) * (1.0 + 2.0j) + (1.0 + 1.0j) * (-1.0 + 1.0j),
            (3.0 - 1.0j) * (-1.0 + 1.0j),
        ],
        dtype=jnp.complex128,
    )

    mv = jcb_mat.jcb_mat_matvec_basic_jit(a, x)
    _check(mv.shape == (2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(mv), rhs_expected)))

    rhs = _vec2(rhs_expected[0], rhs_expected[1])
    sol = jcb_mat.jcb_mat_solve_basic_jit(a, rhs)
    _check(sol.shape == (2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), jnp.asarray([1.0 + 2.0j, -1.0 + 1.0j]))))

    hi = jcb_mat.jcb_mat_matvec_basic_prec(a, x, prec_bits=53)
    lo = jcb_mat.jcb_mat_matvec_basic_prec(a, x, prec_bits=20)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi)))))

    def loss(t):
        tt = _box(t, 0.0)
        mat = jnp.stack(
            [
                jnp.stack([tt, _box(1.0, 0.0)], axis=0),
                jnp.stack([_box(0.0, 0.0), _box(2.0, 0.0)], axis=0),
            ],
            axis=0,
        )
        out = jcb_mat.jcb_mat_matvec_point(mat, _vec2(1.0 + 0.0j, 2.0 + 0.0j))
        return jnp.real(jnp.sum(acb_core.acb_midpoint(out)))

    g = jax.grad(loss)(jnp.asarray(3.0, dtype=jnp.float64))
    _check(bool(jnp.isfinite(g)))


def test_triangular_solve_and_lu_substrate():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j, 3.0 + 0.0j)
    rhs = _vec2(4.0 + 0.0j, 10.0 + 2.0j)
    sol = jcb_mat.jcb_mat_triangular_solve_basic_jit(a, rhs, lower=True)
    _check(sol.shape == (2, 4))
    expected = jnp.asarray([2.0 + 0.0j, (8.0 + 0.0j) / 3.0], dtype=jnp.complex128)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), expected)))

    full = _mat2(2.0 + 0.0j, 1.0 + 0.0j, 4.0 + 0.0j, 3.0 + 1.0j)
    p, l, u = jcb_mat.jcb_mat_lu_basic_jit(full)
    p_mid = acb_core.acb_midpoint(p)
    l_mid = acb_core.acb_midpoint(l)
    u_mid = acb_core.acb_midpoint(u)
    a_mid = acb_core.acb_midpoint(full)
    _check(p.shape == (2, 2, 4))
    _check(l.shape == (2, 2, 4))
    _check(u.shape == (2, 2, 4))
    _check(bool(jnp.allclose(p_mid @ a_mid, l_mid @ u_mid)))


def test_matrix_free_operator_apply_poly_and_expm_action():
    a = _mat2(1.0 + 0.0j, 2.0 + 1.0j, 0.0 + 0.0j, 3.0 - 1.0j)
    x = _vec2(1.0 + 1.0j, -1.0 + 0.5j)
    op = jcb_mat.jcb_mat_dense_operator(a)

    applied = jcb_mat.jcb_mat_operator_apply_point(op, x)
    expected_apply = acb_core.acb_midpoint(jcb_mat.jcb_mat_matvec_point(a, x))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(applied), expected_apply)))

    coeffs = jnp.asarray([1.0 + 0.0j, 2.0 - 1.0j], dtype=jnp.complex128)
    poly = jcb_mat.jcb_mat_poly_action_point(op, x, coeffs)
    a_mid = acb_core.acb_midpoint(a)
    x_mid = acb_core.acb_midpoint(x)
    expected_poly = x_mid + (2.0 - 1.0j) * (a_mid @ x_mid)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(poly), expected_poly)))

    zero_op = lambda v: jnp.zeros_like(acb_core.acb_midpoint(v))
    expm = jcb_mat.jcb_mat_expm_action_point(zero_op, x, terms=8)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(expm), x_mid)))
    zero_applied = jcb_mat.jcb_mat_expm_action_basic_jit(zero_op, x, terms=8)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(zero_applied), x_mid)))


def test_arnoldi_funm_action_matches_exact_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    x = _vec2(1.0 + 0.5j, -2.0 + 1.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)

    basis, H, beta0 = jcb_mat.jcb_mat_arnoldi_hessenberg_point(op, x, steps=2)
    _check(basis.shape == (2, 2))
    _check(H.shape == (2, 2))
    _check(bool(jnp.isfinite(beta0)))

    def dense_exp(m):
        vals, vecs = jnp.linalg.eig(m)
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(jnp.exp(vals)) @ inv

    action = jcb_mat.jcb_mat_funm_action_arnoldi_point(op, x, dense_exp, steps=2)
    exact = jnp.exp(jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex128)) * acb_core.acb_midpoint(x)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(action), exact, rtol=1e-6, atol=1e-6)))

    quad = jcb_mat.jcb_mat_funm_integrand_arnoldi_point(op, x, dense_exp, steps=2)
    _check(bool(jnp.isfinite(jnp.real(quad)) and jnp.isfinite(jnp.imag(quad))))


def test_arnoldi_diagnostics_and_with_diagnostics_wrappers():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    x = _vec2(1.0 + 0.5j, -2.0 + 1.0j)
    probes = jnp.stack([x, _vec2(1.0 + 0.0j, 1.0 + 0.0j)], axis=0)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)

    def dense_exp(m):
        vals, vecs = jnp.linalg.eig(m)
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(jnp.exp(vals)) @ inv

    action, action_diag = jcb_mat.jcb_mat_funm_action_arnoldi_with_diagnostics_point(op, x, dense_exp, 2, adj)
    trace_value, trace_diag = jcb_mat.jcb_mat_trace_estimator_with_diagnostics_point(op, probes, adj)
    logdet_value, logdet_diag = jcb_mat.jcb_mat_logdet_slq_with_diagnostics_point(op, probes, 2, adj)

    exact = jnp.exp(jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex128)) * acb_core.acb_midpoint(x)
    _check(action.shape == (2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(action), exact, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(trace_value, jcb_mat.jcb_mat_trace_estimator_point(op, probes, adj), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet_value, jcb_mat.jcb_mat_logdet_slq_point(op, probes, 2, adj), rtol=1e-6, atol=1e-6)))
    _check(int(action_diag.algorithm_code) == 0)
    _check(int(action_diag.steps) == 2)
    _check(bool(action_diag.used_adjoint))
    _check(bool(action_diag.gradient_supported))
    _check(int(trace_diag.algorithm_code) == 1)
    _check(int(trace_diag.probe_count) == 2)
    _check(int(logdet_diag.algorithm_code) == 2)
    _check(int(logdet_diag.probe_count) == 2)


def test_restarted_and_block_arnoldi_expm_actions_match_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    x = _vec2(1.0 + 0.5j, -2.0 + 1.0j)
    xs = jnp.stack([x, _vec2(0.5 + 0.0j, 1.0 - 0.5j)], axis=0)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)

    restarted = jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(op, x, steps=2, restarts=2, adjoint_matvec=adj)
    block = jcb_mat.jcb_mat_expm_action_arnoldi_block_point(op, xs, steps=2, restarts=2, adjoint_matvec=adj)
    restarted_value, restarted_diag = jcb_mat.jcb_mat_expm_action_arnoldi_restarted_with_diagnostics_point(
        op,
        x,
        steps=2,
        restarts=2,
        adjoint_matvec=adj,
    )

    exact_single = jnp.exp(jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex128)) * acb_core.acb_midpoint(x)
    exact_block = jnp.stack(
        [
            exact_single,
            jnp.exp(jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex128)) * acb_core.acb_midpoint(xs[1]),
        ],
        axis=0,
    )

    _check(bool(jnp.allclose(acb_core.acb_midpoint(restarted), exact_single, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(block), exact_block, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(restarted_value), exact_single, rtol=1e-6, atol=1e-6)))
    _check(int(restarted_diag.restart_count) == 2)
    _check(bool(restarted_diag.used_adjoint))


def test_dense_complex_matrix_functions_and_dense_parameter_gradients():
    a = _mat2(4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    x = _vec2(1.0 + 0.0j, -2.0 + 1.0j)

    logm = jcb_mat.jcb_mat_logm(a)
    sqrtm = jcb_mat.jcb_mat_sqrtm(a)
    rootm = jcb_mat.jcb_mat_rootm(a, degree=2)
    signm = jcb_mat.jcb_mat_signm(a)

    expected_log = jnp.diag(jnp.log(jnp.asarray([4.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex128)))
    expected_sqrt = jnp.diag(jnp.sqrt(jnp.asarray([4.0 + 0.0j, 1.0 + 1.0j], dtype=jnp.complex128)))
    expected_sign = jnp.diag(
        jnp.asarray([1.0 + 0.0j, (1.0 + 1.0j) / jnp.sqrt((1.0 + 1.0j) * (1.0 + 1.0j))], dtype=jnp.complex128)
    )
    _check(bool(jnp.allclose(acb_core.acb_midpoint(logm), expected_log, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sqrtm), expected_sqrt, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(rootm), expected_sqrt, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(signm), expected_sign, rtol=1e-6, atol=1e-6)))

    def dense_exp(m):
        vals, vecs = jnp.linalg.eig(m)
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(jnp.exp(vals)) @ inv

    def loss(t):
        mat = _mat2(t + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
        y = jcb_mat.jcb_mat_funm_action_arnoldi_dense_point(mat, x, dense_exp, 2)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    g = jax.grad(loss)(jnp.asarray(2.0, dtype=jnp.float64))
    expected = jnp.exp(jnp.asarray(2.0, dtype=jnp.float64))
    _check(bool(jnp.allclose(g, expected, rtol=1e-6, atol=1e-6)))


def test_matrix_free_complex_trace_and_logdet_estimators_on_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    p1 = _vec2(1.0 + 0.0j, 1.0 + 0.0j)
    p2 = _vec2(1.0 + 0.0j, -1.0 + 0.0j)
    probes = jnp.stack([p1, p2], axis=0)

    trace_est = jcb_mat.jcb_mat_trace_estimator_point(op, probes)
    logdet_est = jcb_mat.jcb_mat_logdet_slq_point(op, probes, steps=2)

    _check(bool(jnp.allclose(trace_est, (2.0 + 0.0j) + (1.0 + 1.0j), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet_est, jnp.log(2.0 + 0.0j) + jnp.log(1.0 + 1.0j), rtol=1e-6, atol=1e-6)))

    sampled = jcb_mat.jcb_mat_normal_probes_like(p1, key=jax.random.PRNGKey(0), num=4)
    _check(sampled.shape == (4, 2, 4))


def test_arnoldi_funm_action_has_custom_vjp_wrt_input_vector():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)

    def dense_exp(m):
        vals, vecs = jnp.linalg.eig(m)
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(jnp.exp(vals)) @ inv

    def loss(t):
        x = _vec2(t + 0.0j, -2.0 + 1.0j)
        y = jcb_mat.jcb_mat_funm_action_arnoldi_point(op, x, dense_exp, 2, adj)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    g = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))
    expected = jnp.exp(jnp.asarray(2.0, dtype=jnp.float64))
    _check(bool(jnp.allclose(g, expected, rtol=1e-6, atol=1e-6)))


def test_trace_and_logdet_estimators_have_probe_gradients():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)
    p2 = _vec2(1.0 + 0.0j, -1.0 + 0.0j)

    def trace_loss(t):
        p1 = _vec2(t + 0.0j, 1.0 + 0.0j)
        probes = jnp.stack([p1, p2], axis=0)
        value = jcb_mat.jcb_mat_trace_estimator_point(op, probes, adj)
        return jnp.real(value)

    def logdet_loss(t):
        p1 = _vec2(t + 0.0j, 1.0 + 0.0j)
        probes = jnp.stack([p1, p2], axis=0)
        value = jcb_mat.jcb_mat_logdet_slq_point(op, probes, steps=2, adjoint_matvec=adj)
        return jnp.real(value)

    trace_grad = jax.grad(trace_loss)(jnp.asarray(1.0, dtype=jnp.float64))
    logdet_grad = jax.grad(logdet_loss)(jnp.asarray(1.0, dtype=jnp.float64))

    _check(bool(jnp.allclose(trace_grad, 4.0, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet_grad, 2.0 * jnp.log(2.0), rtol=1e-6, atol=1e-6)))
