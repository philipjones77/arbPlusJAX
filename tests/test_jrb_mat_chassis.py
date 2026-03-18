import jax
import jax.numpy as jnp
import pytest

from arbplusjax import double_interval as di
from arbplusjax import jrb_mat
from arbplusjax import sparse_common

from tests._test_checks import _check


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.asarray(lo, dtype=jnp.float64), jnp.asarray(hi, dtype=jnp.float64))


def _exact_interval(x: float) -> jnp.ndarray:
    return _interval(x, x)


def _mat2(a00: float, a01: float, a10: float, a11: float) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([_exact_interval(a00), _exact_interval(a01)], axis=0),
            jnp.stack([_exact_interval(a10), _exact_interval(a11)], axis=0),
        ],
        axis=0,
    )


def _vec2(x0: float, x1: float) -> jnp.ndarray:
    return jnp.stack([_exact_interval(x0), _exact_interval(x1)], axis=0)


def _vec3(x0: float, x1: float, x2: float) -> jnp.ndarray:
    return jnp.stack([_exact_interval(x0), _exact_interval(x1), _exact_interval(x2)], axis=0)


def test_layout_contracts_enforced():
    with pytest.raises(ValueError):
        jrb_mat.jrb_mat_as_interval_matrix(jnp.zeros((2, 3, 2), dtype=jnp.float64))
    with pytest.raises(ValueError):
        jrb_mat.jrb_mat_as_interval_vector(jnp.zeros((2, 3), dtype=jnp.float64))


def test_matmul_point_and_basic_exact_inputs():
    a = _mat2(1.0, 2.0, 3.0, 4.0)
    b = _mat2(2.0, 0.0, 1.0, 2.0)
    expected = jnp.asarray([[4.0, 4.0], [10.0, 8.0]], dtype=jnp.float64)

    point = jrb_mat.jrb_mat_matmul_point(a, b)
    basic = jrb_mat.jrb_mat_matmul_basic(a, b)

    _check(point.shape == (2, 2, 2))
    _check(basic.shape == (2, 2, 2))
    _check(bool(jnp.allclose(di.midpoint(point), expected)))
    _check(bool(jnp.allclose(di.midpoint(basic), expected)))
    _check(bool(jnp.all(di.contains(basic, point))))


def test_matvec_solve_jit_grad_and_precision():
    a = _mat2(3.0, 1.0, 0.0, 2.0)
    x = _vec2(5.0, 7.0)
    rhs_expected = jnp.asarray([22.0, 14.0], dtype=jnp.float64)

    mv = jrb_mat.jrb_mat_matvec_basic_jit(a, x)
    _check(mv.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(mv), rhs_expected)))

    rhs = _vec2(22.0, 14.0)
    sol = jrb_mat.jrb_mat_solve_basic_jit(a, rhs)
    _check(sol.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(sol), jnp.asarray([5.0, 7.0], dtype=jnp.float64))))

    hi = jrb_mat.jrb_mat_matvec_basic_prec(a, x, prec_bits=53)
    lo = jrb_mat.jrb_mat_matvec_basic_prec(a, x, prec_bits=20)
    _check(bool(jnp.all(di.contains(lo, hi))))

    def loss(t):
        tt = _exact_interval(t)
        mat = jnp.stack(
            [
                jnp.stack([tt, _exact_interval(1.0)], axis=0),
                jnp.stack([_exact_interval(0.0), _exact_interval(2.0)], axis=0),
            ],
            axis=0,
        )
        out = jrb_mat.jrb_mat_matvec_point(mat, _vec2(2.0, 1.0))
        return jnp.sum(di.midpoint(out))

    g = jax.grad(loss)(jnp.asarray(3.0, dtype=jnp.float64))
    _check(bool(jnp.isfinite(g)))


def test_triangular_solve_and_lu_substrate():
    a = _mat2(2.0, 0.0, 1.0, 3.0)
    rhs = _vec2(4.0, 10.0)
    sol = jrb_mat.jrb_mat_triangular_solve_basic_jit(a, rhs, lower=True)
    _check(sol.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(sol), jnp.asarray([2.0, 8.0 / 3.0], dtype=jnp.float64))))

    full = _mat2(2.0, 1.0, 4.0, 3.0)
    p, l, u = jrb_mat.jrb_mat_lu_basic_jit(full)
    p_mid = di.midpoint(p)
    l_mid = di.midpoint(l)
    u_mid = di.midpoint(u)
    a_mid = di.midpoint(full)
    _check(p.shape == (2, 2, 2))
    _check(l.shape == (2, 2, 2))
    _check(u.shape == (2, 2, 2))
    _check(bool(jnp.allclose(p_mid @ a_mid, l_mid @ u_mid)))


def test_matrix_free_operator_apply_poly_and_expm_action():
    a = _mat2(1.0, 2.0, 0.0, 3.0)
    x = _vec2(1.0, -1.0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    applied = jrb_mat.jrb_mat_operator_apply_point(op, x)
    expected_apply = di.midpoint(jrb_mat.jrb_mat_matvec_point(a, x))
    _check(bool(jnp.allclose(di.midpoint(applied), expected_apply)))

    coeffs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    poly = jrb_mat.jrb_mat_poly_action_point(op, x, coeffs)
    a_mid = di.midpoint(a)
    x_mid = di.midpoint(x)
    expected_poly = x_mid + 2.0 * (a_mid @ x_mid)
    _check(bool(jnp.allclose(di.midpoint(poly), expected_poly)))

    zero = jnp.stack([_exact_interval(0.0), _exact_interval(0.0)], axis=0)
    zero_op = lambda v: jnp.zeros_like(di.midpoint(v))
    expm = jrb_mat.jrb_mat_expm_action_point(zero_op, x, terms=8)
    _check(bool(jnp.allclose(di.midpoint(expm), x_mid)))
    zero_applied = jrb_mat.jrb_mat_expm_action_basic_jit(zero_op, x, terms=8)
    _check(bool(jnp.allclose(di.midpoint(zero_applied), x_mid)))


def test_lanczos_funm_action_matches_exact_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    x = _vec2(1.0, -2.0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    basis, T, beta0 = jrb_mat.jrb_mat_lanczos_tridiag_point(op, x, steps=2)
    _check(basis.shape == (2, 2))
    _check(T.shape == (2, 2))
    _check(bool(jnp.isfinite(beta0)))

    action = jrb_mat.jrb_mat_funm_action_lanczos_point(op, x, lambda m: jnp.linalg.eigh(m)[1] @ jnp.diag(jnp.exp(jnp.linalg.eigh(m)[0])) @ jnp.linalg.eigh(m)[1].T, steps=2)
    exact = jnp.exp(jnp.asarray([2.0, 3.0], dtype=jnp.float64)) * di.midpoint(x)
    _check(bool(jnp.allclose(di.midpoint(action), exact, rtol=1e-6, atol=1e-6)))

    quad = jrb_mat.jrb_mat_funm_integrand_lanczos_point(op, x, lambda m: jnp.linalg.eigh(m)[1] @ jnp.diag(jnp.exp(jnp.linalg.eigh(m)[0])) @ jnp.linalg.eigh(m)[1].T, steps=2)
    _check(bool(jnp.isfinite(quad)))


def test_lanczos_diagnostics_and_with_diagnostics_wrappers():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    x = _vec2(1.0, -2.0)
    probes = jnp.stack([x, _vec2(1.0, 1.0)], axis=0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    def dense_exp(m):
        vals, vecs = jnp.linalg.eigh(m)
        return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T

    action, action_diag = jrb_mat.jrb_mat_funm_action_lanczos_with_diagnostics_point(op, x, dense_exp, 2)
    trace_value, trace_diag = jrb_mat.jrb_mat_trace_estimator_with_diagnostics_point(op, probes)
    logdet_value, logdet_diag = jrb_mat.jrb_mat_logdet_slq_with_diagnostics_point(op, probes, 2)

    _check(action.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(action), jnp.exp(jnp.asarray([2.0, 3.0])) * di.midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(trace_value, jrb_mat.jrb_mat_trace_estimator_point(op, probes), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet_value, jrb_mat.jrb_mat_logdet_slq_point(op, probes, 2), rtol=1e-6, atol=1e-6)))
    _check(int(action_diag.algorithm_code) == 0)
    _check(int(action_diag.steps) == 2)
    _check(int(action_diag.basis_dim) == 2)
    _check(bool(action_diag.gradient_supported))
    _check(not bool(action_diag.used_adjoint))
    _check(int(trace_diag.algorithm_code) == 1)
    _check(int(trace_diag.probe_count) == 2)
    _check(int(logdet_diag.algorithm_code) == 2)
    _check(int(logdet_diag.probe_count) == 2)


def test_restarted_and_block_lanczos_expm_actions_match_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    x = _vec2(1.0, -2.0)
    xs = jnp.stack([x, _vec2(0.5, 1.0)], axis=0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    restarted = jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(op, x, steps=2, restarts=2)
    block = jrb_mat.jrb_mat_expm_action_lanczos_block_point(op, xs, steps=2, restarts=2)
    restarted_value, restarted_diag = jrb_mat.jrb_mat_expm_action_lanczos_restarted_with_diagnostics_point(
        op,
        x,
        steps=2,
        restarts=2,
    )

    exact_single = jnp.exp(jnp.asarray([2.0, 3.0], dtype=jnp.float64)) * di.midpoint(x)
    exact_block = jnp.stack(
        [
            exact_single,
            jnp.exp(jnp.asarray([2.0, 3.0], dtype=jnp.float64)) * di.midpoint(xs[1]),
        ],
        axis=0,
    )

    _check(bool(jnp.allclose(di.midpoint(restarted), exact_single, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(block), exact_block, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(restarted_value), exact_single, rtol=1e-6, atol=1e-6)))
    _check(int(restarted_diag.restart_count) == 2)


def test_dense_real_matrix_functions_and_dense_parameter_gradients():
    a = _mat2(4.0, 0.0, 0.0, 9.0)
    x = _vec2(1.0, -2.0)

    logm = jrb_mat.jrb_mat_logm(a)
    sqrtm = jrb_mat.jrb_mat_sqrtm(a)
    rootm = jrb_mat.jrb_mat_rootm(a, degree=2)
    signm = jrb_mat.jrb_mat_signm(a)

    _check(bool(jnp.allclose(di.midpoint(logm), jnp.diag(jnp.log(jnp.asarray([4.0, 9.0]))), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(sqrtm), jnp.diag(jnp.sqrt(jnp.asarray([4.0, 9.0]))), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(rootm), di.midpoint(sqrtm), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(signm), jnp.eye(2, dtype=jnp.float64), rtol=1e-6, atol=1e-6)))

    def dense_exp(m):
        vals, vecs = jnp.linalg.eigh(m)
        return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T

    def loss(t):
        mat = _mat2(t, 0.0, 0.0, 3.0)
        y = jrb_mat.jrb_mat_funm_action_lanczos_dense_point(mat, x, dense_exp, 2)
        return jnp.sum(di.midpoint(y))

    g = jax.grad(loss)(jnp.asarray(2.0, dtype=jnp.float64))
    expected = jnp.exp(jnp.asarray(2.0, dtype=jnp.float64))
    _check(bool(jnp.allclose(g, expected, rtol=1e-6, atol=1e-6)))


def test_matrix_free_trace_and_logdet_estimators_on_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    p1 = _vec2(1.0, 1.0)
    p2 = _vec2(1.0, -1.0)
    probes = jnp.stack([p1, p2], axis=0)

    trace_est = jrb_mat.jrb_mat_trace_estimator_point(op, probes)
    logdet_est = jrb_mat.jrb_mat_logdet_slq_point(op, probes, steps=2)

    _check(bool(jnp.allclose(trace_est, 5.0, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet_est, jnp.log(2.0) + jnp.log(3.0), rtol=1e-6, atol=1e-6)))

    sampled = jrb_mat.jrb_mat_rademacher_probes_like(p1, key=jax.random.PRNGKey(0), num=4)
    _check(sampled.shape == (4, 2, 2))


def test_lanczos_funm_action_has_custom_vjp_wrt_input_vector():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    def dense_exp(m):
        vals, vecs = jnp.linalg.eigh(m)
        return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T

    def loss(t):
        x = _vec2(t, -2.0)
        y = jrb_mat.jrb_mat_funm_action_lanczos_point(op, x, dense_exp, 2)
        return jnp.sum(di.midpoint(y))

    g = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))
    expected = jnp.exp(jnp.asarray(2.0, dtype=jnp.float64))
    _check(bool(jnp.allclose(g, expected, rtol=1e-6, atol=1e-6)))


def test_lanczos_funm_action_custom_vjp_matches_under_jit_grad():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    def dense_exp(m):
        vals, vecs = jnp.linalg.eigh(m)
        return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T

    def loss(t):
        x = _vec2(t, -2.0)
        y = jrb_mat.jrb_mat_funm_action_lanczos_point(op, x, dense_exp, 2)
        return jnp.sum(di.midpoint(y))

    arg = jnp.asarray(1.0, dtype=jnp.float64)
    eager = jax.grad(loss)(arg)
    jitted = jax.jit(jax.grad(loss))(arg)

    _check(bool(jnp.isfinite(jitted)))
    _check(bool(jnp.allclose(eager, jitted, rtol=1e-12, atol=1e-12)))


def test_trace_and_logdet_estimators_have_probe_gradients():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    p2 = _vec2(1.0, -1.0)

    def trace_loss(t):
        p1 = _vec2(t, 1.0)
        probes = jnp.stack([p1, p2], axis=0)
        return jrb_mat.jrb_mat_trace_estimator_point(op, probes)

    def logdet_loss(t):
        p1 = _vec2(t, 1.0)
        probes = jnp.stack([p1, p2], axis=0)
        return jrb_mat.jrb_mat_logdet_slq_point(op, probes, steps=2)

    trace_grad = jax.grad(trace_loss)(jnp.asarray(1.0, dtype=jnp.float64))
    logdet_grad = jax.grad(logdet_loss)(jnp.asarray(1.0, dtype=jnp.float64))

    _check(bool(jnp.allclose(trace_grad, 4.0, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet_grad, 2.0 * jnp.log(2.0), rtol=1e-6, atol=1e-6)))


def test_sparse_bcoo_logdet_matches_exact_value_and_probe_gradient_is_finite():
    sqrt_n = jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64))
    probes = jnp.stack([
        _vec3(float(sqrt_n), 0.0, 0.0),
        _vec3(0.0, float(sqrt_n), 0.0),
        _vec3(0.0, 0.0, float(sqrt_n)),
    ], axis=0)

    indices = jnp.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
        ],
        dtype=jnp.int32,
    )
    base_data = jnp.asarray([2.0, -0.5, -0.5, 3.0, -0.25, -0.25, 4.0], dtype=jnp.float64)
    bcoo = sparse_common.SparseBCOO(data=base_data, indices=indices, rows=3, cols=3, algebra="jrb")
    matvec = jrb_mat.jrb_mat_bcoo_operator(bcoo)

    def loss(t):
        varied = probes.at[0, 0, :].set(jnp.asarray([t, t], dtype=jnp.float64))
        return jrb_mat.jrb_mat_logdet_slq_point(matvec, varied, 3)

    grad = jax.grad(loss)(jnp.asarray(float(sqrt_n), dtype=jnp.float64))
    _check(bool(jnp.isfinite(grad)))

    dense = jnp.asarray(
        [
            [2.0, -0.5, 0.0],
            [-0.5, 3.0, -0.25],
            [0.0, -0.25, 4.0],
        ],
        dtype=jnp.float64,
    )
    exact = jnp.linalg.slogdet(dense)[1]
    est = jrb_mat.jrb_mat_logdet_slq_point(matvec, probes, 3)
    _check(bool(jnp.allclose(est, exact, rtol=1e-6, atol=1e-6)))


def test_sparse_bcoo_leja_hutchpp_logdet_matches_exact_value():
    indices = jnp.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
        ],
        dtype=jnp.int32,
    )
    data = jnp.asarray([2.0, -0.25, -0.25, 3.0, -0.5, -0.5, 4.0], dtype=jnp.float64)
    bcoo = sparse_common.SparseBCOO(data=data, indices=indices, rows=3, cols=3, algebra="jrb")
    matvec = jrb_mat.jrb_mat_bcoo_operator(bcoo)
    bounds = jrb_mat.jrb_mat_bcoo_gershgorin_bounds(bcoo)
    sketch = jnp.stack([
        _vec3(1.0, 0.0, 0.0),
        _vec3(0.0, 1.0, 0.0),
        _vec3(0.0, 0.0, 1.0),
    ], axis=0)
    residual = jnp.zeros((0, 3, 2), dtype=jnp.float64)

    est = jrb_mat.jrb_mat_logdet_leja_hutchpp_point(
        matvec,
        sketch,
        residual,
        degree=12,
        spectral_bounds=bounds,
        candidate_count=96,
    )
    exact = jnp.linalg.slogdet(
        jnp.asarray(sparse_common.sparse_bcoo_to_dense(bcoo, algebra="jrb", label="test_jrb_mat_chassis.exact"), dtype=jnp.float64)
    )[1]
    _check(bool(jnp.allclose(est, exact, rtol=1e-5, atol=1e-5)))

    value, diag = jrb_mat.jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        matvec,
        sketch,
        residual,
        degree=12,
        spectral_bounds=bounds,
        candidate_count=96,
    )
    _check(bool(jnp.allclose(value, est, rtol=1e-12, atol=1e-12)))
    _check(int(diag.algorithm_code) == 3)
    _check(int(diag.probe_count) == 3)


def test_sparse_bcoo_adaptive_bounds_and_auto_leja_logdet_match_exact_value():
    dense = jnp.asarray(
        [
            [2.5, -0.25, 0.0],
            [-0.25, 3.0, -0.5],
            [0.0, -0.5, 4.5],
        ],
        dtype=jnp.float64,
    )
    bcoo = sparse_common.dense_to_sparse_bcoo(dense, algebra="jrb")
    exact_eigs = jnp.linalg.eigvalsh(dense)
    g_lower, g_upper = jrb_mat.jrb_mat_bcoo_gershgorin_bounds(bcoo)
    a_lower, a_upper = jrb_mat.jrb_mat_bcoo_spectral_bounds_adaptive(bcoo, steps=3)
    _check(bool(a_lower <= exact_eigs[0]))
    _check(bool(a_upper >= exact_eigs[-1]))
    _check(bool(a_upper <= g_upper + 1e-12))

    sketch = jnp.stack([
        _vec3(1.0, 0.0, 0.0),
        _vec3(0.0, 1.0, 0.0),
        _vec3(0.0, 0.0, 1.0),
    ], axis=0)
    residual = jnp.zeros((0, 3, 2), dtype=jnp.float64)
    est, diag = jrb_mat.jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(
        bcoo,
        sketch,
        residual,
        degree=8,
        max_degree=20,
        min_degree=4,
        candidate_count=128,
        bounds_steps=3,
    )
    exact = jnp.linalg.slogdet(dense)[1]
    _check(bool(jnp.allclose(est, exact, rtol=1e-5, atol=1e-5)))
    _check(int(diag.algorithm_code) == 3)
    _check(int(diag.steps) >= 4)
    _check(int(diag.steps) <= 20)


def test_scipy_csr_operator_matches_bcoo_operator_when_available():
    scipy = pytest.importorskip("scipy.sparse")
    dense = jnp.asarray(
        [
            [3.0, 0.0, -1.0],
            [0.0, 2.5, 0.5],
            [-1.0, 0.5, 4.0],
        ],
        dtype=jnp.float64,
    )
    csr = scipy.csr_matrix(dense)
    bcoo = sparse_common.scipy_csr_to_sparse_bcoo(csr, algebra="jrb", dtype=jnp.float64)
    x = jnp.stack([_exact_interval(1.0), _exact_interval(-2.0), _exact_interval(0.5)], axis=0)

    op_csr = jrb_mat.jrb_mat_scipy_csr_operator(csr)
    op_bcoo = jrb_mat.jrb_mat_bcoo_operator(bcoo)
    y_csr = jrb_mat.jrb_mat_operator_apply_point(op_csr, x)
    y_bcoo = jrb_mat.jrb_mat_operator_apply_point(op_bcoo, x)
    _check(bool(jnp.allclose(di.midpoint(y_csr), di.midpoint(y_bcoo), rtol=1e-10, atol=1e-10)))
