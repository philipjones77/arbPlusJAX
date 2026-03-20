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


def test_sparse_bcoo_operator_and_adjoint_apply():
    dense = jnp.asarray(
        [
            [2.0 + 0.0j, 1.0 - 0.5j],
            [0.0 + 0.0j, 3.0 + 1.0j],
        ],
        dtype=jnp.complex128,
    )
    data = jnp.asarray([2.0 + 0.0j, 1.0 - 0.5j, 3.0 + 1.0j], dtype=jnp.complex128)
    indices = jnp.asarray([[0, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    bcoo = jcb_mat.sparse_common.SparseBCOO(data=data, indices=indices, rows=2, cols=2, algebra="jcb")
    x = _vec2(1.0 + 2.0j, -1.0 + 1.0j)
    op = jcb_mat.jcb_mat_bcoo_operator(bcoo)
    adj = jcb_mat.jcb_mat_bcoo_operator_adjoint(bcoo)

    got = jcb_mat.jcb_mat_operator_apply_point(op, x)
    got_adj = jcb_mat.jcb_mat_operator_apply_point(adj, x)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(got), dense @ acb_core.acb_midpoint(x))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(got_adj), jnp.conjugate(dense).T @ acb_core.acb_midpoint(x))))


def test_operator_plans_and_rmatvec_surface():
    a = _mat2(1.0 + 0.0j, 2.0 + 1.0j, 0.0 + 0.0j, 3.0 - 1.0j)
    x = _vec2(1.0 + 1.0j, -1.0 + 0.5j)
    dense = acb_core.acb_midpoint(a)
    x_mid = acb_core.acb_midpoint(x)

    rmat = jcb_mat.jcb_mat_rmatvec_point(a, x)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(rmat), dense.T @ x_mid)))

    dense_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    dense_rplan = jcb_mat.jcb_mat_dense_operator_rmatvec_plan_prepare(a)
    dense_aplan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(dense_plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(dense_rplan, x)), dense.T @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(dense_aplan, x)), jnp.conjugate(dense).T @ x_mid)))

    data = jnp.asarray([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j], dtype=jnp.complex128)
    indices = jnp.asarray([[0, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    bcoo = jcb_mat.sparse_common.SparseBCOO(data=data, indices=indices, rows=2, cols=2, algebra="jcb")
    plan = jcb_mat.jcb_mat_bcoo_operator_plan_prepare(bcoo)
    rplan = jcb_mat.jcb_mat_bcoo_operator_rmatvec_plan_prepare(bcoo)
    aplan = jcb_mat.jcb_mat_bcoo_operator_adjoint_plan_prepare(bcoo)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(rplan, x)), dense.T @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(aplan, x)), jnp.conjugate(dense).T @ x_mid)))

    coeffs = jnp.asarray([1.0 + 0.0j, 2.0 - 1.0j], dtype=jnp.complex128)
    poly_from_plan = jcb_mat.jcb_mat_poly_action_point(dense_plan, x, coeffs)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(poly_from_plan), x_mid + (2.0 - 1.0j) * (dense @ x_mid))))

    zero_dense = _mat2(0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
    zero_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(zero_dense)
    expm_from_plan = jcb_mat.jcb_mat_expm_action_point(zero_plan, x, terms=8)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(expm_from_plan), x_mid)))
    expm_from_plan_jit = jcb_mat.jcb_mat_expm_action_basic_jit(zero_plan, x, terms=8)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(expm_from_plan_jit), x_mid)))

    hpd_dense = _mat2(2.0 + 0.0j, 1.0 + 1.0j, 1.0 - 1.0j, 3.0 + 0.0j)
    hpd_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(hpd_dense)
    hpd_aplan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(hpd_dense)
    basis, H, beta0 = jcb_mat.jcb_mat_arnoldi_hessenberg_point(hpd_plan, x, steps=2)
    _check(basis.shape == (2, 2))
    _check(H.shape == (2, 2))
    _check(bool(jnp.isfinite(beta0)))

    dense_exp = jcb_mat.jcb_mat_dense_funm_general_eig_point(jnp.exp)
    funm_plan = jcb_mat.jcb_mat_funm_action_arnoldi_point(hpd_plan, x, dense_exp, 2, hpd_aplan)
    diag_funm, diag_info = jcb_mat.jcb_mat_funm_action_arnoldi_with_diagnostics_point(hpd_plan, x, dense_exp, 2, hpd_aplan)
    _check(funm_plan.shape == (2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(funm_plan), acb_core.acb_midpoint(diag_funm), rtol=1e-6, atol=1e-6)))
    _check(int(diag_info.steps) == 2)

    probes = jnp.stack([x, _vec2(1.0 + 0.0j, 1.0 + 0.0j)], axis=0)
    trace_value = jcb_mat.jcb_mat_trace_estimator_point(hpd_plan, probes, hpd_aplan)
    logdet_value = jcb_mat.jcb_mat_logdet_slq_point(hpd_plan, probes, 2, hpd_aplan)
    logdet_value_jit = jcb_mat.jcb_mat_logdet_slq_point_jit(hpd_plan, probes, 2, hpd_aplan)
    det_value_jit = jcb_mat.jcb_mat_det_slq_point_jit(hpd_plan, probes, 2, hpd_aplan)
    restarted = jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(hpd_plan, x, steps=2, restarts=2, adjoint_matvec=hpd_aplan)
    _check(bool(jnp.isfinite(jnp.real(trace_value)) and jnp.isfinite(jnp.imag(trace_value))))
    _check(bool(jnp.isfinite(jnp.real(logdet_value)) and jnp.isfinite(jnp.imag(logdet_value))))
    _check(bool(jnp.isfinite(jnp.real(logdet_value_jit)) and jnp.isfinite(jnp.imag(logdet_value_jit))))
    _check(bool(jnp.isfinite(jnp.real(det_value_jit)) and jnp.isfinite(jnp.imag(det_value_jit))))
    _check(bool(jnp.allclose(logdet_value_jit, logdet_value, rtol=1e-6, atol=1e-6)))
    _check(restarted.shape == (2, 4))

    hpd_data = jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j, 1.0 - 1.0j, 3.0 + 0.0j], dtype=jnp.complex128)
    hpd_indices = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
    hpd_bcoo = jcb_mat.sparse_common.SparseBCOO(data=hpd_data, indices=hpd_indices, rows=2, cols=2, algebra="jcb")
    hpd_bplan = jcb_mat.jcb_mat_bcoo_operator_plan_prepare(hpd_bcoo)
    hpd_baplan = jcb_mat.jcb_mat_bcoo_operator_adjoint_plan_prepare(hpd_bcoo)
    sparse_trace = jcb_mat.jcb_mat_trace_estimator_point(hpd_bplan, probes, hpd_baplan)
    sparse_logdet = jcb_mat.jcb_mat_logdet_slq_point(hpd_bplan, probes, 2, hpd_baplan)
    sparse_logdet_jit = jcb_mat.jcb_mat_logdet_slq_point_jit(hpd_bplan, probes, 2, hpd_baplan)
    sparse_det_jit = jcb_mat.jcb_mat_det_slq_point_jit(hpd_bplan, probes, 2, hpd_baplan)
    sparse_restarted = jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(hpd_bplan, x, steps=2, restarts=2, adjoint_matvec=hpd_baplan)
    _check(bool(jnp.isfinite(jnp.real(sparse_trace)) and jnp.isfinite(jnp.imag(sparse_trace))))
    _check(bool(jnp.isfinite(jnp.real(sparse_logdet)) and jnp.isfinite(jnp.imag(sparse_logdet))))
    _check(bool(jnp.isfinite(jnp.real(sparse_logdet_jit)) and jnp.isfinite(jnp.imag(sparse_logdet_jit))))
    _check(bool(jnp.isfinite(jnp.real(sparse_det_jit)) and jnp.isfinite(jnp.imag(sparse_det_jit))))
    _check(bool(jnp.allclose(sparse_logdet_jit, sparse_logdet, rtol=1e-6, atol=1e-6)))
    _check(sparse_restarted.shape == (2, 4))


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


def test_arnoldi_funm_action_custom_vjp_matches_under_jit_grad():
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

    arg = jnp.asarray(1.0, dtype=jnp.float64)
    eager = jax.grad(loss)(arg)
    jitted = jax.jit(jax.grad(loss))(arg)

    _check(bool(jnp.isfinite(jitted)))
    _check(bool(jnp.allclose(eager, jitted, rtol=1e-12, atol=1e-12)))


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


def test_named_matrix_free_complex_function_actions_match_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j)
    x = _vec2(1.0 + 0.5j, -2.0 + 1.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)

    log_action = jcb_mat.jcb_mat_log_action_arnoldi_point(op, x, 2, adj)
    sqrt_action = jcb_mat.jcb_mat_sqrt_action_arnoldi_point(op, x, 2, adj)
    root_action = jcb_mat.jcb_mat_root_action_arnoldi_point(op, x, degree=2, steps=2, adjoint_matvec=adj)
    sign_action = jcb_mat.jcb_mat_sign_action_arnoldi_point(op, x, 2, adj)
    sin_action = jcb_mat.jcb_mat_sin_action_arnoldi_point(op, x, 2, adj)
    cosh_action, cosh_info = jcb_mat.jcb_mat_cosh_action_arnoldi_with_diagnostics_point(op, x, 2, adj)
    dense_log_action = jcb_mat.jcb_mat_log_action_arnoldi_dense_point(a, x, 2)
    diag_action, diag_info = jcb_mat.jcb_mat_sqrt_action_arnoldi_with_diagnostics_point(op, x, 2, adj)

    x_mid = acb_core.acb_midpoint(x)
    expected_log = jnp.asarray([jnp.log(2.0 + 0.0j), jnp.log(1.0 + 1.0j)], dtype=jnp.complex128) * x_mid
    expected_sqrt = jnp.asarray([jnp.sqrt(2.0 + 0.0j), jnp.sqrt(1.0 + 1.0j)], dtype=jnp.complex128) * x_mid
    expected_sign = jnp.asarray([1.0 + 0.0j, (1.0 + 1.0j) / jnp.sqrt((1.0 + 1.0j) * (1.0 + 1.0j))], dtype=jnp.complex128) * x_mid
    _check(bool(jnp.allclose(acb_core.acb_midpoint(log_action), expected_log, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sqrt_action), expected_sqrt, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(root_action), acb_core.acb_midpoint(sqrt_action), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sign_action), expected_sign, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sin_action), jnp.asarray([jnp.sin(2.0 + 0.0j), jnp.sin(1.0 + 1.0j)], dtype=jnp.complex128) * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(cosh_action), jnp.asarray([jnp.cosh(2.0 + 0.0j), jnp.cosh(1.0 + 1.0j)], dtype=jnp.complex128) * x_mid, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(dense_log_action), acb_core.acb_midpoint(log_action), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(diag_action), acb_core.acb_midpoint(sqrt_action), rtol=1e-6, atol=1e-6)))
    _check(int(diag_info.steps) == 2)
    _check(int(cosh_info.steps) == 2)

    pow_action = jcb_mat.jcb_mat_pow_action_arnoldi_point(op, x, exponent=2, steps=2, adjoint_matvec=adj)
    expected_pow = jnp.asarray([(2.0 + 0.0j) ** 2, (1.0 + 1.0j) ** 2], dtype=jnp.complex128) * x_mid
    _check(bool(jnp.allclose(acb_core.acb_midpoint(pow_action), expected_pow, rtol=1e-6, atol=1e-6)))


def test_complex_solve_inverse_det_and_leja_matrix_free_apis_match_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    x = _vec2(1.0 + 2.0j, -2.0 + 1.0j)
    rhs = _vec2((2.0 + 0.0j) * (1.0 + 2.0j), (3.0 + 0.0j) * (-2.0 + 1.0j))
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)

    solved, info = jcb_mat.jcb_mat_solve_action_with_diagnostics_point(op, rhs, hermitian=True)
    inv_applied = jcb_mat.jcb_mat_inverse_action_point(op, rhs, hermitian=True)
    solved_h = jcb_mat.jcb_mat_solve_action_hermitian_point(op, rhs)
    inv_hpd = jcb_mat.jcb_mat_inverse_action_hpd_point(op, rhs)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved), acb_core.acb_midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_applied), acb_core.acb_midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved_h), acb_core.acb_midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_hpd), acb_core.acb_midpoint(x), rtol=1e-6, atol=1e-6)))
    _check(bool(info["converged"]))

    probes = jnp.stack([_vec2(1.0 + 0.0j, 1.0 + 0.0j), _vec2(1.0 + 0.0j, -1.0 + 0.0j)], axis=0)
    logdet = jcb_mat.jcb_mat_logdet_slq_point(op, probes, 2, adj)
    det = jcb_mat.jcb_mat_det_slq_point(op, probes, 2, adj)
    herm_logdet = jcb_mat.jcb_mat_logdet_slq_hermitian_point(op, probes, 2)
    hpd_det = jcb_mat.jcb_mat_det_slq_hpd_point(op, probes, 2)
    _check(bool(jnp.allclose(logdet, jnp.log(6.0 + 0.0j), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(det, 6.0 + 0.0j, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(herm_logdet, jnp.log(6.0 + 0.0j), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(hpd_det, 6.0 + 0.0j, rtol=1e-6, atol=1e-6)))

    leja_action = jcb_mat.jcb_mat_log_action_leja_point(op, x, degree=6, spectral_bounds=(2.0, 3.0))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(leja_action), jnp.asarray([jnp.log(2.0 + 0.0j), jnp.log(3.0 + 0.0j)]) * acb_core.acb_midpoint(x), rtol=1e-5, atol=1e-5)))

    leja_logdet = jcb_mat.jcb_mat_logdet_leja_hutchpp_point(op, probes, jnp.zeros((0, 2, 4), dtype=jnp.float64), degree=6, spectral_bounds=(2.0, 3.0))
    leja_det = jcb_mat.jcb_mat_det_leja_hutchpp_point(op, probes, jnp.zeros((0, 2, 4), dtype=jnp.float64), degree=6, spectral_bounds=(2.0, 3.0))
    _check(bool(jnp.allclose(leja_logdet, jnp.log(6.0 + 0.0j), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(leja_det, 6.0 + 0.0j, rtol=1e-5, atol=1e-5)))


def test_complex_sparse_operator_plans_cover_coo_csr_and_hermitian_aliases():
    dense = jnp.asarray([[2.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 3.0 + 0.0j]], dtype=jnp.complex128)
    coo = jcb_mat.sparse_common.SparseCOO(
        data=jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j, 1.0 - 1.0j, 3.0 + 0.0j], dtype=jnp.complex128),
        row=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        col=jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="jcb",
    )
    csr = jcb_mat.sparse_common.SparseCSR(
        data=jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j, 1.0 - 1.0j, 3.0 + 0.0j], dtype=jnp.complex128),
        indices=jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        indptr=jnp.asarray([0, 2, 4], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="jcb",
    )
    x = _vec2(1.0 + 0.5j, -1.0 + 1.0j)
    x_mid = acb_core.acb_midpoint(x)
    coo_plan = jcb_mat.jcb_mat_sparse_operator_plan_prepare(coo)
    csr_plan = jcb_mat.jcb_mat_sparse_operator_plan_prepare(csr)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(coo_plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(csr_plan, x)), dense @ x_mid)))

    a = _mat2(2.0 + 0.0j, 1.0 + 1.0j, 1.0 - 1.0j, 3.0 + 0.0j)
    herm_plan = jcb_mat.jcb_mat_hermitian_operator_plan_prepare(a)
    hpd_plan = jcb_mat.jcb_mat_hpd_operator_plan_prepare(a)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(herm_plan, x)), dense @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(hpd_plan, x)), dense @ x_mid)))
