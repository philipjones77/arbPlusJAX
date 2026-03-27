import jax
import jax.numpy as jnp
import pytest

from arbplusjax import double_interval as di
from arbplusjax import jrb_mat
from arbplusjax import matrix_free_core
from arbplusjax import sparse_common
from arbplusjax import srb_block_mat
from arbplusjax import srb_vblock_mat

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


def test_shift_invert_operator_plan_exposes_transpose_surface():
    dense = _mat2(4.0, 1.0, 0.0, 2.0)
    operator = jrb_mat.jrb_mat_dense_operator_plan_prepare(dense)
    rhs = _vec2(1.0, 2.0)
    plan = jrb_mat.jrb_mat_shift_invert_operator_plan_prepare(operator, shift=1.0)
    tplan = jrb_mat.matrix_free_core.operator_transpose_plan(plan, conjugate=False)

    _check(tplan is not None)
    out = jrb_mat.jrb_mat_operator_plan_apply(plan, rhs)
    tout = jrb_mat.jrb_mat_operator_plan_apply(tplan, rhs)

    dense_mid = di.midpoint(dense)
    rhs_mid = di.midpoint(rhs)
    expected = jnp.linalg.solve(dense_mid - jnp.eye(2, dtype=jnp.float64), rhs_mid)
    expected_t = jnp.linalg.solve((dense_mid - jnp.eye(2, dtype=jnp.float64)).T, rhs_mid)

    _check(bool(jnp.allclose(di.midpoint(out), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(tout), expected_t, rtol=1e-8, atol=1e-8)))


def test_operator_plans_and_rmatvec_surface():
    a = _mat2(1.0, 2.0, 0.0, 3.0)
    x = _vec2(1.0, -1.0)
    dense = di.midpoint(a)
    x_mid = di.midpoint(x)

    rmat = jrb_mat.jrb_mat_rmatvec_point(a, x)
    _check(bool(jnp.allclose(di.midpoint(rmat), dense.T @ x_mid)))

    dense_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    dense_rplan = jrb_mat.jrb_mat_dense_operator_rmatvec_plan_prepare(a)
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(dense_plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(dense_rplan, x)), dense.T @ x_mid)))

    bdata = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    bindices = jnp.asarray([[0, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    bcoo = sparse_common.SparseBCOO(data=bdata, indices=bindices, rows=2, cols=2, algebra="jrb")
    bplan = jrb_mat.jrb_mat_bcoo_operator_plan_prepare(bcoo)
    brplan = jrb_mat.jrb_mat_bcoo_operator_rmatvec_plan_prepare(bcoo)
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(bplan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(brplan, x)), dense.T @ x_mid)))

    coeffs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    poly_from_plan = jrb_mat.jrb_mat_poly_action_point(dense_plan, x, coeffs)
    _check(bool(jnp.allclose(di.midpoint(poly_from_plan), x_mid + 2.0 * (dense @ x_mid))))

    zero_dense = _mat2(0.0, 0.0, 0.0, 0.0)
    zero_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(zero_dense)
    expm_from_plan = jrb_mat.jrb_mat_expm_action_point(zero_plan, x, terms=8)
    _check(bool(jnp.allclose(di.midpoint(expm_from_plan), x_mid)))
    expm_from_plan_jit = jrb_mat.jrb_mat_expm_action_basic_jit(zero_plan, x, terms=8)
    _check(bool(jnp.allclose(di.midpoint(expm_from_plan_jit), x_mid)))

    spd_dense = _mat2(2.0, 1.0, 1.0, 3.0)
    spd_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(spd_dense)
    basis, T, beta0 = jrb_mat.jrb_mat_lanczos_tridiag_point(spd_plan, x, steps=2)
    _check(basis.shape == (2, 2))
    _check(T.shape == (2, 2))
    _check(bool(jnp.isfinite(beta0)))

    dense_exp = jrb_mat.jrb_mat_dense_funm_sym_eigh_point(jnp.exp)
    funm_plan = jrb_mat.jrb_mat_funm_action_lanczos_point(spd_plan, x, dense_exp, 2)
    diag_funm, diag_info = jrb_mat.jrb_mat_funm_action_lanczos_with_diagnostics_point(spd_plan, x, dense_exp, 2)
    _check(funm_plan.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(funm_plan), di.midpoint(diag_funm), rtol=1e-6, atol=1e-6)))
    _check(int(diag_info.steps) == 2)

    probes = jnp.stack([x, _vec2(1.0, 1.0)], axis=0)
    trace_value = jrb_mat.jrb_mat_trace_estimator_point(spd_plan, probes)
    logdet_value = jrb_mat.jrb_mat_logdet_slq_point(spd_plan, probes, 2)
    logdet_value_jit = jrb_mat.jrb_mat_logdet_slq_point_jit(spd_plan, probes, 2)
    det_value_jit = jrb_mat.jrb_mat_det_slq_point_jit(spd_plan, probes, 2)
    restarted = jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(spd_plan, x, steps=2, restarts=2)
    _check(bool(jnp.isfinite(trace_value)))
    _check(bool(jnp.isfinite(logdet_value)))
    _check(bool(jnp.isfinite(logdet_value_jit)))
    _check(bool(jnp.isfinite(det_value_jit)))
    _check(bool(jnp.allclose(logdet_value_jit, logdet_value, rtol=1e-6, atol=1e-6)))
    _check(restarted.shape == (2, 2))

    spd_data = jnp.asarray([2.0, 1.0, 1.0, 3.0], dtype=jnp.float64)
    spd_indices = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
    spd_bcoo = sparse_common.SparseBCOO(data=spd_data, indices=spd_indices, rows=2, cols=2, algebra="jrb")
    spd_bplan = jrb_mat.jrb_mat_bcoo_operator_plan_prepare(spd_bcoo)
    sparse_trace = jrb_mat.jrb_mat_trace_estimator_point(spd_bplan, probes)
    sparse_logdet = jrb_mat.jrb_mat_logdet_slq_point(spd_bplan, probes, 2)
    sparse_logdet_jit = jrb_mat.jrb_mat_logdet_slq_point_jit(spd_bplan, probes, 2)
    sparse_det_jit = jrb_mat.jrb_mat_det_slq_point_jit(spd_bplan, probes, 2)
    sparse_restarted = jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(spd_bplan, x, steps=2, restarts=2)
    _check(bool(jnp.isfinite(sparse_trace)))
    _check(bool(jnp.isfinite(sparse_logdet)))
    _check(bool(jnp.isfinite(sparse_logdet_jit)))
    _check(bool(jnp.isfinite(sparse_det_jit)))
    _check(bool(jnp.allclose(sparse_logdet_jit, sparse_logdet, rtol=1e-6, atol=1e-6)))
    _check(sparse_restarted.shape == (2, 2))


def test_logdet_slq_plan_jit_matches_point_across_step_buckets():
    a = jnp.stack(
        [
            jnp.stack([_exact_interval(4.0), _exact_interval(1.0), _exact_interval(0.0), _exact_interval(0.0)], axis=0),
            jnp.stack([_exact_interval(1.0), _exact_interval(5.0), _exact_interval(1.0), _exact_interval(0.0)], axis=0),
            jnp.stack([_exact_interval(0.0), _exact_interval(1.0), _exact_interval(6.0), _exact_interval(1.0)], axis=0),
            jnp.stack([_exact_interval(0.0), _exact_interval(0.0), _exact_interval(1.0), _exact_interval(7.0)], axis=0),
        ],
        axis=0,
    )
    x0 = jnp.stack([_exact_interval(1.0), _exact_interval(0.0), _exact_interval(0.0), _exact_interval(0.0)], axis=0)
    x1 = jnp.stack([_exact_interval(0.0), _exact_interval(1.0), _exact_interval(0.0), _exact_interval(0.0)], axis=0)
    probes = jnp.stack([x0, x1], axis=0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    for steps in (3, 4):
        point = jrb_mat.jrb_mat_logdet_slq_point(plan, probes, steps)
        compiled = jrb_mat.jrb_mat_logdet_slq_point_jit(plan, probes, steps)
        _check(bool(jnp.allclose(compiled, point, rtol=1e-6, atol=1e-6)))


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
    _check(float(logdet_diag.primal_residual) >= 0.0)
    _check(int(logdet_diag.regime_code) >= 0)
    _check(int(logdet_diag.solver_code) == int(jrb_mat.matrix_free_core.solver_code("lanczos")))


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
    orth = jrb_mat.jrb_mat_orthogonal_rademacher_probes_like(p1, key=jax.random.PRNGKey(0), num=2)
    orth_mid = di.midpoint(orth)
    _check(orth.shape == (2, 2, 2))
    _check(bool(jnp.allclose(orth_mid @ orth_mid.T, jnp.eye(2, dtype=jnp.float64), atol=1e-6)))
    mean, variance, stderr = jrb_mat.jrb_mat_trace_estimator_probe_statistics_point(op, probes)
    recommended = jrb_mat.jrb_mat_trace_estimator_adaptive_probe_count(
        op,
        probes,
        target_stderr=1.0,
        min_probes=2,
        max_probes=8,
        block_size=2,
    )
    _check(bool(jnp.allclose(mean, trace_est, rtol=1e-6, atol=1e-6)))
    _check(bool(variance >= 0.0))
    _check(bool(stderr >= 0.0))
    _check(int(recommended) >= 2)
    _check(int(recommended) <= 8)
    _check(int(recommended) % 2 == 0)


def test_slq_preparation_heat_trace_spectral_density_and_hutchpp_metadata_on_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    p1 = _vec2(1.0, 1.0)
    p2 = _vec2(1.0, -1.0)
    probes = jnp.stack([p1, p2], axis=0)

    metadata = jrb_mat.jrb_mat_slq_prepare_point(op, probes, 2, target_stderr=1e-4, min_probes=2, max_probes=8, block_size=2)
    logdet = jrb_mat.jrb_mat_logdet_estimate_point(op, probes, 2)
    heat = jrb_mat.jrb_mat_heat_trace_slq_from_metadata_point(metadata, 0.5)
    hist = jrb_mat.jrb_mat_spectral_density_slq_from_metadata_point(
        metadata,
        jnp.asarray([1.0, 2.5, 4.0], dtype=jnp.float64),
        normalize=True,
    )

    _check(bool(jnp.allclose(metadata.statistics.mean, logdet, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(heat, jnp.exp(-1.0) + jnp.exp(-1.5), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(jnp.sum(hist), 1.0, atol=1e-6)))
    _check(int(metadata.statistics.recommended_probe_count) % 2 == 0)

    hutch = jrb_mat.jrb_mat_hutchpp_trace_with_metadata_point(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(op, v, 2),
        probes[:1],
        probes[1:],
        target_stderr=1e-4,
        min_probes=1,
        max_probes=4,
        block_size=1,
    )
    hutch_value = jrb_mat.jrb_mat_hutchpp_trace_estimate_point(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(op, v, 2),
        probes[:1],
        probes[1:],
    )
    _check(bool(jnp.allclose(hutch.low_rank_trace + hutch.residual_trace, hutch_value, rtol=1e-6, atol=1e-6)))

    deflation = jrb_mat.jrb_mat_deflated_operator_prepare_point(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(op, v, 2),
        probes[:1],
    )
    deflated = jrb_mat.jrb_mat_trace_estimate_deflated_point(
        lambda v: jrb_mat.jrb_mat_log_action_lanczos_point(op, v, 2),
        deflation,
        probes[1:],
        target_stderr=1e-4,
        min_probes=1,
        max_probes=4,
        block_size=1,
    )
    _check(bool(jnp.allclose(deflated.low_rank_trace + deflated.residual_trace, hutch_value, rtol=1e-6, atol=1e-6)))


def test_cached_rational_hutchpp_logdet_matches_diagonal_oracle():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    intercept = log2 - 2.0 * slope
    coeffs = jnp.asarray([intercept, slope], dtype=jnp.float64)
    sketch = jnp.stack([_exact_interval(1.0), _exact_interval(0.0)], axis=0)[None, ...]
    residual = jnp.stack([_exact_interval(0.0), _exact_interval(1.0)], axis=0)[None, ...]

    metadata = jrb_mat.jrb_mat_logdet_rational_hutchpp_prepare_point(
        op,
        sketch,
        shifts=jnp.zeros((0,), dtype=jnp.float64),
        weights=jnp.zeros((0,), dtype=jnp.float64),
        polynomial_coefficients=coeffs,
        target_stderr=1e-4,
        min_probes=1,
        max_probes=4,
        block_size=1,
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
    exact = jnp.log(2.0) + jnp.log(3.0)
    _check(bool(metadata.gradient_supported))
    _check(bool(metadata.implicit_adjoint))
    _check(bool(metadata.cached_adjoint_supported))
    _check(int(cached.statistics.recommended_probe_count) >= 1)
    _check(bool(jnp.allclose(matrix_free_core.hutchpp_trace_from_metadata(cached), exact, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(direct, exact, rtol=1e-6, atol=1e-6)))


def test_cached_rational_hutchpp_logdet_has_residual_probe_gradient():
    a = _mat2(2.0, 0.0, 0.0, 3.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    intercept = log2 - 2.0 * slope
    coeffs = jnp.asarray([intercept, slope], dtype=jnp.float64)
    sketch = jnp.stack([_exact_interval(1.0), _exact_interval(0.0)], axis=0)[None, ...]
    metadata = jrb_mat.jrb_mat_logdet_rational_hutchpp_prepare_point(
        op,
        sketch,
        shifts=jnp.zeros((0,), dtype=jnp.float64),
        weights=jnp.zeros((0,), dtype=jnp.float64),
        polynomial_coefficients=coeffs,
    )

    def loss(t):
        residual = jnp.stack([jnp.stack([_exact_interval(0.0), _exact_interval(float(t))], axis=0)], axis=0)
        estimate = jrb_mat.jrb_mat_logdet_rational_hutchpp_from_metadata_point(metadata, residual)
        return matrix_free_core.hutchpp_trace_from_metadata(estimate)
    value = loss(jnp.asarray(1.0, dtype=jnp.float64))
    _check(bool(jnp.isfinite(value)))


def test_slq_heat_trace_gradient_matches_diagonal_oracle():
    op = jrb_mat.jrb_mat_dense_operator(_mat2(2.0, 0.0, 0.0, 3.0))
    p1 = _vec2(1.0, 1.0)
    p2 = _vec2(1.0, -1.0)
    probes = jnp.stack([p1, p2], axis=0)

    def loss(t):
        metadata = jrb_mat.jrb_mat_slq_prepare_point(op, probes, 2)
        return jrb_mat.jrb_mat_heat_trace_slq_from_metadata_point(metadata, t)

    g = jax.grad(loss)(jnp.asarray(0.5, dtype=jnp.float64))
    expected = -2.0 * jnp.exp(-1.0) - 3.0 * jnp.exp(-1.5)
    _check(bool(jnp.allclose(g, expected, rtol=1e-6, atol=1e-6)))


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


def test_jacobi_preconditioner_multi_shift_and_restarted_block_eigsh_surfaces():
    diag = jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64)
    dense = jnp.diag(diag)
    a = jax.vmap(jax.vmap(_exact_interval))(dense)
    rhs = _vec3(2.0, 8.0, 16.0)
    shifts = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    op = jrb_mat.jrb_mat_dense_operator(a)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(plan)

    applied = jrb_mat.matrix_free_core.preconditioner_plan_apply(
        prec,
        rhs,
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    _check(bool(jnp.allclose(applied, jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.float64), rtol=1e-6, atol=1e-6)))

    shifted = jrb_mat.jrb_mat_multi_shift_solve_point(plan, rhs, shifts, symmetric=True, preconditioner=prec, tol=1e-10)
    shifted_jit = jrb_mat.jrb_mat_multi_shift_solve_point_jit(plan, rhs, shifts, symmetric=True, preconditioner=prec, tol=1e-10)
    expected = jnp.stack(
        [
            jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.float64),
            jnp.asarray([2.0 / 3.0, 8.0 / 5.0, 16.0 / 9.0], dtype=jnp.float64),
        ],
        axis=0,
    )
    _check(shifted.shape == (2, 3, 2))
    _check(bool(jnp.allclose(di.midpoint(shifted), expected, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(shifted_jit), expected, rtol=1e-6, atol=1e-6)))

    vals_block, vecs_block = jrb_mat.jrb_mat_eigsh_block_point(plan, size=3, k=2, which="largest", block_size=2, subspace_iters=4)
    vals_restart, vecs_restart = jrb_mat.jrb_mat_eigsh_restarted_point(plan, size=3, k=2, which="largest", steps=2, restarts=2, block_size=2)
    vals_block_jit, vecs_block_jit = jrb_mat.jrb_mat_eigsh_block_point_jit(plan, size=3, k=2, which="largest", block_size=2, subspace_iters=4)
    vals_restart_jit, vecs_restart_jit = jrb_mat.jrb_mat_eigsh_restarted_point_jit(
        plan,
        size=3,
        k=2,
        which="largest",
        steps=2,
        restarts=2,
        block_size=2,
    )

    target = jnp.asarray([4.0, 8.0], dtype=jnp.float64)
    _check(bool(jnp.allclose(vals_block, target, rtol=2e-2, atol=5e-2)))
    _check(bool(jnp.allclose(vals_restart, target, rtol=2e-2, atol=5e-2)))
    _check(bool(jnp.allclose(vals_block_jit, target, rtol=2e-2, atol=5e-2)))
    _check(bool(jnp.allclose(vals_restart_jit, target, rtol=2e-2, atol=5e-2)))
    _check(vecs_block.shape == (3, 2))
    _check(vecs_restart.shape == (3, 2))
    _check(vecs_block_jit.shape == (3, 2))
    _check(vecs_restart_jit.shape == (3, 2))


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


def test_named_matrix_free_real_function_actions_match_diagonal_case():
    a = _mat2(4.0, 0.0, 0.0, 9.0)
    x = _vec2(1.0, -2.0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    log_action = jrb_mat.jrb_mat_log_action_lanczos_point(op, x, 2)
    sqrt_action = jrb_mat.jrb_mat_sqrt_action_lanczos_point(op, x, 2)
    root_action = jrb_mat.jrb_mat_root_action_lanczos_point(op, x, degree=2, steps=2)
    sign_action = jrb_mat.jrb_mat_sign_action_lanczos_point(op, x, 2)
    sin_action = jrb_mat.jrb_mat_sin_action_lanczos_point(op, x, 2)
    cosh_action, cosh_info = jrb_mat.jrb_mat_cosh_action_lanczos_with_diagnostics_point(op, x, 2)
    dense_log_action = jrb_mat.jrb_mat_log_action_lanczos_dense_point(a, x, 2)
    diag_action, diag_info = jrb_mat.jrb_mat_sqrt_action_lanczos_with_diagnostics_point(op, x, 2)

    x_mid = di.midpoint(x)
    _check(bool(jnp.allclose(di.midpoint(log_action), jnp.asarray([jnp.log(4.0), jnp.log(9.0)]) * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(sqrt_action), jnp.asarray([2.0, 3.0]) * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(root_action), di.midpoint(sqrt_action), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(sign_action), x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(sin_action), jnp.asarray([jnp.sin(4.0), jnp.sin(9.0)]) * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(cosh_action), jnp.asarray([jnp.cosh(4.0), jnp.cosh(9.0)]) * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(dense_log_action), di.midpoint(log_action), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(diag_action), di.midpoint(sqrt_action), rtol=1e-6, atol=1e-6)))
    _check(int(diag_info.steps) == 2)
    _check(int(cosh_info.steps) == 2)

    pow_action = jrb_mat.jrb_mat_pow_action_lanczos_point(op, x, exponent=2, steps=2)
    _check(bool(jnp.allclose(di.midpoint(pow_action), jnp.asarray([16.0, 81.0]) * x_mid, rtol=1e-6, atol=1e-6)))

    rational_action = jrb_mat.jrb_mat_rational_action_point(
        op,
        x,
        shifts=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        weights=jnp.asarray([0.5, -1.0], dtype=jnp.float64),
        polynomial_coefficients=jnp.asarray([1.0], dtype=jnp.float64),
        symmetric=True,
    )
    rational_basic = jrb_mat.jrb_mat_rational_action_basic(
        op,
        x,
        shifts=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        weights=jnp.asarray([0.5, -1.0], dtype=jnp.float64),
        polynomial_coefficients=jnp.asarray([1.0], dtype=jnp.float64),
        symmetric=True,
    )
    rational_diag = jnp.asarray(
        [
            1.0 + 0.5 / (4.0 - 1.0) - 1.0 / (4.0 - 2.0),
            1.0 + 0.5 / (9.0 - 1.0) - 1.0 / (9.0 - 2.0),
        ],
        dtype=jnp.float64,
    )
    _check(bool(jnp.allclose(di.midpoint(rational_action), rational_diag * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.all(di.contains(rational_basic, rational_action))))

    contour_log = jrb_mat.jrb_mat_log_action_contour_point(op, x, center=6.5 + 0.0j, radius=3.0, quadrature_order=32)
    contour_sqrt = jrb_mat.jrb_mat_sqrt_action_contour_point(op, x, center=6.5 + 0.0j, radius=3.0, quadrature_order=32)
    contour_root = jrb_mat.jrb_mat_root_action_contour_point(op, x, degree=2, center=6.5 + 0.0j, radius=3.0, quadrature_order=32)
    contour_sign = jrb_mat.jrb_mat_sign_action_contour_point(op, x, center=6.5 + 0.0j, radius=3.0, quadrature_order=32)
    contour_sin = jrb_mat.jrb_mat_sin_action_contour_point(op, x, center=6.5 + 0.0j, radius=3.0, quadrature_order=32)
    contour_cos = jrb_mat.jrb_mat_cos_action_contour_point(op, x, center=6.5 + 0.0j, radius=3.0, quadrature_order=32)
    _check(bool(jnp.allclose(di.midpoint(contour_log), di.midpoint(log_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(di.midpoint(contour_sqrt), di.midpoint(sqrt_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(di.midpoint(contour_root), di.midpoint(root_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(di.midpoint(contour_sign), di.midpoint(sign_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(di.midpoint(contour_sin), di.midpoint(sin_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(di.midpoint(contour_cos), jnp.asarray([jnp.cos(4.0), jnp.cos(9.0)]) * x_mid, rtol=1e-5, atol=1e-5)))


def test_real_solve_inverse_and_det_matrix_free_apis_match_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, 4.0)
    x = _vec2(3.0, -2.0)
    rhs = _vec2(6.0, -8.0)
    op = jrb_mat.jrb_mat_dense_operator(a)

    solved, info = jrb_mat.jrb_mat_solve_action_with_diagnostics_point(op, rhs, symmetric=True)
    inv_applied = jrb_mat.jrb_mat_inverse_action_point(op, rhs, symmetric=True)
    _check(bool(jnp.allclose(di.midpoint(solved), di.midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(inv_applied), di.midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(float(info.primal_residual) <= 1e-6)
    _check(int(info.structure_code) == int(jrb_mat.matrix_free_core.structure_code("symmetric")))
    _check(int(info.solver_code) == int(jrb_mat.matrix_free_core.solver_code("cg")))

    probes = jnp.stack([_vec2(1.0, 1.0), _vec2(1.0, -1.0)], axis=0)
    logdet = jrb_mat.jrb_mat_logdet_slq_point(op, probes, 2)
    det = jrb_mat.jrb_mat_det_slq_point(op, probes, 2)
    _check(bool(jnp.allclose(logdet, jnp.log(8.0), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(det, 8.0, rtol=1e-6, atol=1e-6)))

    bundle = jrb_mat.jrb_mat_logdet_solve_point(op, rhs, probes, 2, symmetric=True)
    _check(bool(jnp.allclose(bundle.logdet, logdet, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(bundle.solve), di.midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(bundle.aux.solve_diagnostics.converged))
    _check(bool(bundle.aux.logdet_diagnostics.converged))


def test_real_block_and_vblock_operator_plans_match_storage_matvec():
    dense = jnp.asarray([[2.0, 1.0, 0.0], [1.0, 3.0, -1.0], [0.0, -1.0, 4.0]], dtype=jnp.float64)
    block = srb_block_mat.srb_block_mat_from_dense_csr(dense[:2, :2], block_shape=(1, 1))
    vblock = srb_vblock_mat.srb_vblock_mat_from_dense_csr(
        dense,
        row_block_sizes=jnp.asarray([1, 2], dtype=jnp.int32),
        col_block_sizes=jnp.asarray([1, 2], dtype=jnp.int32),
    )
    x2 = jnp.stack([_exact_interval(1.0), _exact_interval(-2.0)], axis=0)
    x3 = jnp.stack([_exact_interval(1.0), _exact_interval(-2.0), _exact_interval(0.5)], axis=0)

    bplan = jrb_mat.jrb_mat_block_sparse_operator_plan_prepare(block)
    brplan = jrb_mat.jrb_mat_block_sparse_operator_rmatvec_plan_prepare(block)
    vplan = jrb_mat.jrb_mat_vblock_sparse_operator_plan_prepare(vblock)
    vrplan = jrb_mat.jrb_mat_vblock_sparse_operator_rmatvec_plan_prepare(vblock)

    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(bplan, x2)), dense[:2, :2] @ di.midpoint(x2))))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(brplan, x2)), dense[:2, :2].T @ di.midpoint(x2))))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(vplan, x3)), dense @ di.midpoint(x3))))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(vrplan, x3)), dense.T @ di.midpoint(x3))))


def test_real_minres_matrix_free_apis_match_indefinite_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, -3.0)
    x = _vec2(3.0, -2.0)
    rhs = _vec2(6.0, 6.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    solved, info = jrb_mat.jrb_mat_minres_solve_action_with_diagnostics_point(op, rhs)
    inv_applied = jrb_mat.jrb_mat_minres_inverse_action_point(op, rhs)
    solved_jit = jrb_mat.jrb_mat_minres_solve_action_point_jit(plan, rhs)
    inv_jit = jrb_mat.jrb_mat_minres_inverse_action_point_jit(plan, rhs)

    target = jnp.asarray([3.0, -2.0], dtype=jnp.float64)
    _check(bool(jnp.allclose(di.midpoint(solved), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(inv_applied), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(solved_jit), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(inv_jit), target, rtol=1e-6, atol=1e-6)))
    _check(float(info.primal_residual) <= 1e-6)
    _check(int(info.structure_code) == int(jrb_mat.matrix_free_core.structure_code("symmetric")))
    _check(int(info.solver_code) == int(jrb_mat.matrix_free_core.solver_code("minres")))


def test_real_preconditioned_minres_matrix_free_apis_match_indefinite_diagonal_case():
    a = _mat2(2.0, 0.0, 0.0, -3.0)
    rhs = _vec2(6.0, 6.0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(plan)

    solved, info = jrb_mat.jrb_mat_minres_solve_action_with_diagnostics_point(plan, rhs, preconditioner=prec)
    inv_applied = jrb_mat.jrb_mat_minres_inverse_action_point(plan, rhs, preconditioner=prec)
    solved_jit = jrb_mat.jrb_mat_minres_solve_action_point_jit(plan, rhs, preconditioner=prec)
    inv_jit = jrb_mat.jrb_mat_minres_inverse_action_point_jit(plan, rhs, preconditioner=prec)

    target = jnp.asarray([3.0, -2.0], dtype=jnp.float64)
    _check(bool(jnp.allclose(di.midpoint(solved), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(inv_applied), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(solved_jit), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(inv_jit), target, rtol=1e-6, atol=1e-6)))
    _check(float(info.primal_residual) <= 1e-6)


def test_sparse_operator_plans_cover_coo_csr_and_structured_aliases():
    dense = jnp.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    coo = sparse_common.SparseCOO(
        data=jnp.asarray([2.0, 1.0, 1.0, 3.0], dtype=jnp.float64),
        row=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        col=jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="jrb",
    )
    csr = sparse_common.SparseCSR(
        data=jnp.asarray([2.0, 1.0, 1.0, 3.0], dtype=jnp.float64),
        indices=jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        indptr=jnp.asarray([0, 2, 4], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="jrb",
    )
    x = _vec2(1.0, -1.0)
    x_mid = di.midpoint(x)
    coo_plan = jrb_mat.jrb_mat_sparse_operator_plan_prepare(coo)
    csr_plan = jrb_mat.jrb_mat_sparse_operator_plan_prepare(csr)
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(coo_plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(csr_plan, x)), dense @ x_mid)))

    a = _mat2(2.0, 1.0, 1.0, 3.0)
    sym_plan = jrb_mat.jrb_mat_symmetric_operator_plan_prepare(a)
    spd_plan = jrb_mat.jrb_mat_spd_operator_plan_prepare(a)
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(sym_plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(spd_plan, x)), dense @ x_mid)))


def test_real_matrix_free_plan_jit_surface_and_diagnostics_metadata():
    a = _mat2(2.0, 0.0, 0.0, 4.0)
    x = _vec2(1.0, -1.0)
    rhs = _vec2(2.0, -4.0)
    probes = jnp.stack([_vec2(1.0, 0.0), _vec2(0.0, 1.0)], axis=0)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    applied = jrb_mat.jrb_mat_operator_apply_point_jit(plan, x)
    r_applied = jrb_mat.jrb_mat_rmatvec_point_jit(a, x)
    solved = jrb_mat.jrb_mat_solve_action_point_jit(plan, rhs, symmetric=True)
    invd = jrb_mat.jrb_mat_inverse_action_point_jit(plan, rhs, symmetric=True)
    log_action = jrb_mat.jrb_mat_log_action_lanczos_point_jit(plan, x, 2)
    pow_action = jrb_mat.jrb_mat_pow_action_lanczos_point_jit(plan, x, exponent=2, steps=2)
    logdet = jrb_mat.jrb_mat_logdet_slq_point_jit(plan, probes, 2)
    det = jrb_mat.jrb_mat_det_slq_point_jit(plan, probes, 2)
    _, solve_diag = jrb_mat.jrb_mat_solve_action_with_diagnostics_point(plan, rhs, symmetric=True)

    _check(bool(jnp.allclose(di.midpoint(applied), jnp.asarray([2.0, -4.0]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(r_applied), jnp.asarray([2.0, -4.0]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(solved), jnp.asarray([1.0, -1.0]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(invd), jnp.asarray([1.0, -1.0]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(log_action), jnp.asarray([jnp.log(2.0), -jnp.log(4.0)]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(pow_action), jnp.asarray([4.0, -16.0]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet, jnp.log(8.0), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(det, 8.0, rtol=1e-6, atol=1e-6)))
    _check(float(solve_diag.primal_residual) <= 1e-6)
    _check(bool(jnp.all(jnp.isfinite(solve_diag.residual_history))))


def test_real_structured_matrix_free_aliases_match_base_paths():
    a = _mat2(2.0, 0.0, 0.0, 4.0)
    x = _vec2(1.0, -1.0)
    rhs = _vec2(2.0, -4.0)
    op = jrb_mat.jrb_mat_dense_operator(a)
    probes = jnp.stack([_vec2(1.0, 0.0), _vec2(0.0, 1.0)], axis=0)

    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_log_action_symmetric_point(op, x, 2)), di.midpoint(jrb_mat.jrb_mat_log_action_lanczos_point(op, x, 2)), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_sqrt_action_spd_point(op, x, 2)), di.midpoint(jrb_mat.jrb_mat_sqrt_action_lanczos_point(op, x, 2)), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(di.midpoint(jrb_mat.jrb_mat_expm_action_spd_point(op, x, steps=2)), di.midpoint(jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(op, x, steps=2, restarts=1)), rtol=1e-6, atol=1e-6)))


def test_jrb_mat_eigsh_operator_plan_matches_dense_diagonal_case():
    dense = jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 7.0], dtype=jnp.float64))
    a = di.interval(dense, dense)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    vals, vecs = jrb_mat.jrb_mat_eigsh_point(plan, size=4, k=2, which="smallest", steps=4)
    vals_jit, vecs_jit = jrb_mat.jrb_mat_eigsh_point_jit(plan, size=4, k=2, which="smallest", steps=4)
    expected = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    residual = dense @ vecs - vecs * vals[None, :]
    _check(bool(jnp.allclose(vals, expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(vals_jit, vals, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(residual, jnp.zeros_like(residual), rtol=1e-8, atol=1e-8)))
    _check(vecs.shape == (4, 2))
    _check(vecs_jit.shape == vecs.shape)


def test_jrb_mat_native_krylov_schur_davidson_shift_invert_and_contour_surfaces():
    dense = jnp.diag(jnp.asarray([1.0, 3.0, 7.0], dtype=jnp.float64))
    a = di.interval(dense, dense)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(plan)

    vals_ks, _ = jrb_mat.jrb_mat_eigsh_krylov_schur_point(plan, size=3, k=1, which="largest", steps=2, restarts=2, block_size=2)
    vals_dav, _ = jrb_mat.jrb_mat_eigsh_davidson_point(plan, size=3, k=1, which="largest", subspace_iters=3, block_size=2, preconditioner=prec)
    vals_jd, _ = jrb_mat.jrb_mat_eigsh_jacobi_davidson_point(plan, size=3, k=1, which="largest", subspace_iters=3, block_size=2, preconditioner=prec)
    vals_si, _ = jrb_mat.jrb_mat_eigsh_shift_invert_point(plan, size=3, shift=2.8, k=1, which="largest", steps=3, preconditioner=prec)
    vals_contour, _ = jrb_mat.jrb_mat_eigsh_contour_point(plan, size=3, center=3.0, radius=0.6, k=1, which="largest", quadrature_order=8, block_size=2, preconditioner=prec)

    _check(bool(jnp.allclose(vals_ks, jnp.asarray([7.0]), atol=1e-4)))
    _check(bool(jnp.allclose(vals_dav, jnp.asarray([7.0]), atol=1e-3)))
    _check(bool(jnp.allclose(vals_jd, jnp.asarray([7.0]), atol=1e-3)))
    _check(bool(jnp.allclose(di.midpoint(vals_si), jnp.asarray([3.0]), atol=5e-2)))
    _check(bool(jnp.allclose(di.midpoint(vals_contour), jnp.asarray([3.0]), atol=2e-1)))

    _, _, diag_ks = jrb_mat.jrb_mat_eigsh_krylov_schur_with_diagnostics_point(plan, size=3, k=1, which="largest", steps=2, restarts=2, block_size=2)
    _, _, diag_dav = jrb_mat.jrb_mat_eigsh_davidson_with_diagnostics_point(plan, size=3, k=1, which="largest", subspace_iters=3, block_size=2, preconditioner=prec)
    _check(int(diag_ks.locked_count) >= 1)
    _check(int(diag_ks.deflated_count) >= 1)
    _check(bool(diag_ks.converged))
    _check(bool(diag_dav.converged))
    _check(float(diag_dav.convergence_metric) <= 1e-3)
    _check(diag_dav.residual_history.shape[-1] >= 1)


def test_jrb_mat_generalized_eigsh_diagonal_surface():
    a_dense = jnp.diag(jnp.asarray([2.0, 6.0, 12.0], dtype=jnp.float64))
    b_dense = jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64))
    a = di.interval(a_dense, a_dense)
    b = di.interval(b_dense, b_dense)
    a_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    b_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(b)
    b_prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(b_plan)

    vals, vecs = jrb_mat.jrb_mat_geigsh_point(
        a_plan,
        b_plan,
        size=3,
        k=2,
        which="smallest",
        steps=3,
        b_preconditioner=b_prec,
    )
    vals_d, vecs_d, diag = jrb_mat.jrb_mat_geigsh_with_diagnostics_point(
        a_plan,
        b_plan,
        size=3,
        k=2,
        which="smallest",
        steps=3,
        b_preconditioner=b_prec,
        tol=1e-6,
    )

    _check(bool(jnp.allclose(vals, jnp.asarray([2.0, 3.0]), atol=1e-4)))
    _check(bool(jnp.allclose(vals_d, vals, atol=1e-4)))
    _check(vecs.shape == (3, 2))
    _check(vecs_d.shape == (3, 2))
    _check(bool(diag.converged))
    _check(float(diag.convergence_metric) <= 1e-6)


def test_jrb_mat_generalized_shift_invert_diagonal_surface():
    a_dense = jnp.diag(jnp.asarray([2.0, 6.0, 12.0], dtype=jnp.float64))
    b_dense = jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64))
    a = di.interval(a_dense, a_dense)
    b = di.interval(b_dense, b_dense)
    a_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    b_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(b)
    b_prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(b_plan)

    vals, vecs = jrb_mat.jrb_mat_geigsh_shift_invert_point(
        a_plan,
        b_plan,
        size=3,
        shift=2.8,
        k=1,
        which="largest",
        steps=3,
        preconditioner=b_prec,
    )
    vals_d, vecs_d, diag = jrb_mat.jrb_mat_geigsh_shift_invert_with_diagnostics_point(
        a_plan,
        b_plan,
        size=3,
        shift=2.8,
        k=1,
        which="largest",
        steps=3,
        preconditioner=b_prec,
        tol=1e-4,
    )

    _check(bool(jnp.allclose(di.midpoint(vals), jnp.asarray([3.0]), atol=5e-2)))
    _check(bool(jnp.allclose(di.midpoint(vals_d), di.midpoint(vals), atol=5e-2)))
    _check(vecs.shape == (3, 1, 2))
    _check(vecs_d.shape == (3, 1, 2))
    _check(bool(diag.converged))


def test_jrb_mat_polynomial_and_nonlinear_eig_surfaces():
    a0_dense = jnp.diag(jnp.asarray([4.0, 18.0, 40.0], dtype=jnp.float64))
    a1_dense = jnp.diag(jnp.asarray([-5.0, -9.0, -13.0], dtype=jnp.float64))
    a2_dense = jnp.eye(3, dtype=jnp.float64)
    coeffs = [
        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(a0_dense, a0_dense)),
        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(a1_dense, a1_dense)),
        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(a2_dense, a2_dense)),
    ]
    vals_p, vecs_p, diag_p = jrb_mat.jrb_mat_peigsh_with_diagnostics_point(
        coeffs,
        size=3,
        lambda0=2.8,
        newton_iters=4,
        eig_steps=3,
        tol=1e-6,
    )

    def mat_builder(lam):
        dense = jnp.diag(jnp.exp(jnp.asarray(lam, dtype=jnp.float64)) - jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64))
        return jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))

    def dmat_builder(lam):
        dense = jnp.eye(3, dtype=jnp.float64) * jnp.exp(jnp.asarray(lam, dtype=jnp.float64))
        return jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(dense, dense))

    vals_n, vecs_n, diag_n = jrb_mat.jrb_mat_neigsh_with_diagnostics_point(
        mat_builder,
        dmat_builder,
        size=3,
        lambda0=1.45,
        newton_iters=4,
        eig_steps=3,
        tol=1e-6,
    )

    _check(bool(jnp.allclose(di.midpoint(vals_p), jnp.asarray([3.0]), atol=5e-3)))
    _check(bool(jnp.allclose(di.midpoint(vals_n), jnp.asarray([jnp.log(4.0)]), atol=5e-3)))
    _check(vecs_p.shape == (3, 1, 2))
    _check(vecs_n.shape == (3, 1, 2))
    _check(bool(diag_p.converged))
    _check(bool(diag_n.converged))


def test_jrb_mat_restart_policy_handles_block_window_larger_than_k():
    dense = jnp.diag(jnp.asarray([1.0, 2.0, 5.0, 9.0], dtype=jnp.float64))
    a = di.interval(dense, dense)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)

    vals_restart, _ = jrb_mat.jrb_mat_eigsh_restarted_point(
        plan,
        size=4,
        k=1,
        which="largest",
        steps=2,
        restarts=3,
        block_size=3,
    )
    vals_ks, _, diag_ks = jrb_mat.jrb_mat_eigsh_krylov_schur_with_diagnostics_point(
        plan,
        size=4,
        k=1,
        which="smallest",
        steps=2,
        restarts=3,
        block_size=3,
        tol=1e-6,
    )

    _check(bool(jnp.allclose(vals_restart, jnp.asarray([9.0]), atol=1e-4)))
    _check(bool(jnp.allclose(vals_ks, jnp.asarray([1.0]), atol=1e-4)))
    _check(int(diag_ks.restart_count) == 3)
    _check(diag_ks.residual_history.shape[-1] >= 1)


def test_jrb_mat_davidson_and_jd_handle_block_window_larger_than_k():
    dense = jnp.diag(jnp.asarray([1.0, 2.0, 5.0, 9.0], dtype=jnp.float64))
    a = di.interval(dense, dense)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(plan)

    vals_dav, _ = jrb_mat.jrb_mat_eigsh_davidson_point(
        plan,
        size=4,
        k=1,
        which="largest",
        subspace_iters=3,
        block_size=3,
        preconditioner=prec,
        tol=1e-6,
    )
    vals_jd, _, diag_jd = jrb_mat.jrb_mat_eigsh_jacobi_davidson_with_diagnostics_point(
        plan,
        size=4,
        k=1,
        which="smallest",
        subspace_iters=3,
        block_size=3,
        preconditioner=prec,
        tol=1e-6,
    )

    _check(bool(jnp.allclose(vals_dav, jnp.asarray([9.0]), atol=1e-4)))
    _check(bool(jnp.allclose(vals_jd, jnp.asarray([1.0]), atol=1e-4)))
    _check(int(diag_jd.restart_count) == 3)
    _check(int(diag_jd.locked_count) >= 1)
    _check(int(diag_jd.deflated_count) >= 1)
    _check(diag_jd.residual_history.shape[-1] >= 1)
