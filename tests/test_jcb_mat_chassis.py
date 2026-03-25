import jax
import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import scb_block_mat
from arbplusjax import scb_vblock_mat

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


def test_logdet_slq_plan_jit_matches_point_across_step_buckets():
    a = jnp.stack(
        [
            jnp.stack([_box(4.0, 0.0), _box(1.0, 0.5), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(1.0, -0.5), _box(5.0, 0.0), _box(1.0, 0.25), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(1.0, -0.25), _box(6.0, 0.0), _box(1.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(1.0, 0.0), _box(7.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    x0 = jnp.stack([_box(1.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0)
    x1 = jnp.stack([_box(0.0, 0.0), _box(1.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0)
    probes = jnp.stack([x0, x1], axis=0)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    aplan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)

    for steps in (3, 4):
        point = jcb_mat.jcb_mat_logdet_slq_point(plan, probes, steps, aplan)
        compiled = jcb_mat.jcb_mat_logdet_slq_point_jit(plan, probes, steps, aplan)
        _check(bool(jnp.allclose(compiled, point, rtol=1e-6, atol=1e-6)))


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
    _check(float(logdet_diag.primal_residual) >= 0.0)
    _check(int(logdet_diag.solver_code) == int(jcb_mat.matrix_free_core.solver_code("arnoldi")))


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
    orth = jcb_mat.jcb_mat_orthogonal_normal_probes_like(p1, key=jax.random.PRNGKey(0), num=2)
    orth_mid = acb_core.acb_midpoint(orth)
    _check(orth.shape == (2, 2, 4))
    _check(bool(jnp.allclose(orth_mid @ jnp.conjugate(orth_mid.T), jnp.eye(2, dtype=jnp.complex128), atol=1e-6)))
    mean, variance, stderr = jcb_mat.jcb_mat_trace_estimator_probe_statistics_point(op, probes)
    recommended = jcb_mat.jcb_mat_trace_estimator_adaptive_probe_count(
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


def test_hermitian_slq_preparation_heat_trace_spectral_density_and_hutchpp_metadata_on_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    p1 = _vec2(1.0 + 0.0j, 1.0 + 0.0j)
    p2 = _vec2(1.0 + 0.0j, -1.0 + 0.0j)
    probes = jnp.stack([p1, p2], axis=0)

    metadata = jcb_mat.jcb_mat_slq_prepare_hermitian_point(op, probes, 2, target_stderr=1e-4, min_probes=2, max_probes=8, block_size=2)
    logdet = jcb_mat.jcb_mat_logdet_slq_hermitian_point(op, probes, 2)
    heat = jcb_mat.jcb_mat_heat_trace_slq_hermitian_from_metadata_point(metadata, 0.5)
    hist = jcb_mat.jcb_mat_spectral_density_slq_hermitian_from_metadata_point(
        metadata,
        jnp.asarray([1.0, 2.5, 4.0], dtype=jnp.float64),
        normalize=True,
    )

    _check(bool(jnp.allclose(metadata.statistics.mean, logdet, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(heat, jnp.exp(-1.0) + jnp.exp(-1.5), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(jnp.sum(hist), 1.0, atol=1e-6)))
    _check(int(metadata.statistics.recommended_probe_count) % 2 == 0)

    hutch = jcb_mat.jcb_mat_hutchpp_trace_with_metadata_point(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(op, v, 2),
        probes[:1],
        probes[1:],
        target_stderr=1e-4,
        min_probes=1,
        max_probes=4,
        block_size=1,
    )
    hutch_value = jcb_mat.jcb_mat_hutchpp_trace_estimate_point(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(op, v, 2),
        probes[:1],
        probes[1:],
    )
    _check(bool(jnp.allclose(hutch.low_rank_trace + hutch.residual_trace, hutch_value, rtol=1e-6, atol=1e-6)))

    deflation = jcb_mat.jcb_mat_deflated_operator_prepare_point(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(op, v, 2),
        probes[:1],
    )
    deflated = jcb_mat.jcb_mat_trace_estimate_deflated_point(
        lambda v: jcb_mat.jcb_mat_log_action_hermitian_point(op, v, 2),
        deflation,
        probes[1:],
        target_stderr=1e-4,
        min_probes=1,
        max_probes=4,
        block_size=1,
    )
    _check(bool(jnp.allclose(deflated.low_rank_trace + deflated.residual_trace, hutch_value, rtol=1e-6, atol=1e-6)))


def test_cached_complex_rational_hutchpp_logdet_matches_diagonal_oracle():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    intercept = log2 - 2.0 * slope
    coeffs = jnp.asarray([intercept, slope], dtype=jnp.float64)
    sketch = jnp.asarray([_vec2(1.0 + 0.0j, 0.0 + 0.0j)], dtype=jnp.complex128)
    residual = jnp.asarray([_vec2(0.0 + 0.0j, 1.0 + 0.0j)], dtype=jnp.complex128)

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
    exact = jnp.log(2.0 + 0.0j) + jnp.log(3.0 + 0.0j)
    _check(bool(jnp.allclose(matrix_free_core.hutchpp_trace_from_metadata(cached), exact, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(direct, exact, rtol=1e-6, atol=1e-6)))


def test_cached_complex_rational_hutchpp_logdet_has_residual_probe_gradient():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    slope = jnp.log(jnp.asarray(3.0 / 2.0, dtype=jnp.float64))
    intercept = log2 - 2.0 * slope
    coeffs = jnp.asarray([intercept, slope], dtype=jnp.float64)
    sketch = jnp.asarray([_vec2(1.0 + 0.0j, 0.0 + 0.0j)], dtype=jnp.complex128)
    metadata = jcb_mat.jcb_mat_logdet_rational_hutchpp_prepare_point(
        op,
        sketch,
        shifts=jnp.zeros((0,), dtype=jnp.complex128),
        weights=jnp.zeros((0,), dtype=jnp.complex128),
        polynomial_coefficients=coeffs,
        hermitian=True,
    )

    def loss(t):
        residual = jnp.asarray([_vec2(0.0 + 0.0j, t + 0.0j)], dtype=jnp.complex128)
        estimate = jcb_mat.jcb_mat_logdet_rational_hutchpp_from_metadata_point(metadata, residual)
        return jnp.real(matrix_free_core.hutchpp_trace_from_metadata(estimate))

    grad = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))
    _check(bool(jnp.allclose(grad, 2.0 * jnp.log(3.0), rtol=1e-6, atol=1e-6)))


def test_hermitian_slq_heat_trace_gradient_matches_diagonal_oracle():
    op = jcb_mat.jcb_mat_dense_operator(_mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j))
    p1 = _vec2(1.0 + 0.0j, 1.0 + 0.0j)
    p2 = _vec2(1.0 + 0.0j, -1.0 + 0.0j)
    probes = jnp.stack([p1, p2], axis=0)

    def loss(t):
        metadata = jcb_mat.jcb_mat_slq_prepare_hermitian_point(op, probes, 2)
        return jnp.real(jcb_mat.jcb_mat_heat_trace_slq_hermitian_from_metadata_point(metadata, t))

    g = jax.grad(loss)(jnp.asarray(0.5, dtype=jnp.float64))
    expected = -2.0 * jnp.exp(-1.0) - 3.0 * jnp.exp(-1.5)
    _check(bool(jnp.allclose(g, expected, rtol=1e-6, atol=1e-6)))


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


def test_jacobi_preconditioner_multi_shift_and_restarted_block_eigsh_surfaces():
    diag = jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64)
    dense = jnp.diag(diag).astype(jnp.complex128)
    a = jnp.stack(
        [
            jnp.stack([_box(2.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(4.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(8.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    rhs = jnp.stack([_box(2.0, 0.0), _box(8.0, 0.0), _box(16.0, 0.0)], axis=0)
    shifts = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(plan)

    applied = jcb_mat.matrix_free_core.preconditioner_plan_apply(
        prec,
        rhs,
        midpoint_vector=acb_core.acb_midpoint,
        sparse_bcoo_matvec=jcb_mat.sparse_common.sparse_bcoo_matvec,
        dtype=jnp.complex128,
    )
    _check(bool(jnp.allclose(applied, jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.complex128), rtol=1e-6, atol=1e-6)))

    shifted = jcb_mat.jcb_mat_multi_shift_solve_point(plan, rhs, shifts, hermitian=True, preconditioner=prec, tol=1e-10)
    shifted_jit = jcb_mat.jcb_mat_multi_shift_solve_point_jit(plan, rhs, shifts, hermitian=True, preconditioner=prec, tol=1e-10)
    expected = jnp.stack(
        [
            jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.complex128),
            jnp.asarray([2.0 / 3.0, 8.0 / 5.0, 16.0 / 9.0], dtype=jnp.complex128),
        ],
        axis=0,
    )
    _check(shifted.shape == (2, 3, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(shifted), expected, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(shifted_jit), expected, rtol=1e-6, atol=1e-6)))

    vals_block, vecs_block = jcb_mat.jcb_mat_eigsh_block_point(plan, size=3, k=2, which="largest", block_size=2, subspace_iters=4)
    vals_restart, vecs_restart = jcb_mat.jcb_mat_eigsh_restarted_point(plan, size=3, k=2, which="largest", steps=2, restarts=2, block_size=2)
    vals_block_jit, vecs_block_jit = jcb_mat.jcb_mat_eigsh_block_point_jit(plan, size=3, k=2, which="largest", block_size=2, subspace_iters=4)
    vals_restart_jit, vecs_restart_jit = jcb_mat.jcb_mat_eigsh_restarted_point_jit(
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

    contour_log = jcb_mat.jcb_mat_log_action_contour_point(op, x, center=1.5 + 0.5j, radius=1.25, quadrature_order=32)
    contour_sqrt = jcb_mat.jcb_mat_sqrt_action_contour_point(op, x, center=1.5 + 0.5j, radius=1.25, quadrature_order=32)
    contour_root = jcb_mat.jcb_mat_root_action_contour_point(op, x, degree=2, center=1.5 + 0.5j, radius=1.25, quadrature_order=32)
    contour_sign = jcb_mat.jcb_mat_sign_action_contour_point(op, x, center=1.5 + 0.5j, radius=1.25, quadrature_order=32)
    contour_sin = jcb_mat.jcb_mat_sin_action_contour_point(op, x, center=1.5 + 0.5j, radius=1.25, quadrature_order=32)
    contour_cos = jcb_mat.jcb_mat_cos_action_contour_point(op, x, center=1.5 + 0.5j, radius=1.25, quadrature_order=32)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(contour_log), acb_core.acb_midpoint(log_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(contour_sqrt), acb_core.acb_midpoint(sqrt_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(contour_root), acb_core.acb_midpoint(root_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(contour_sign), acb_core.acb_midpoint(sign_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(contour_sin), acb_core.acb_midpoint(sin_action), rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(contour_cos), jnp.asarray([jnp.cos(2.0 + 0.0j), jnp.cos(1.0 + 1.0j)], dtype=jnp.complex128) * x_mid, rtol=1e-5, atol=1e-5)))


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
    _check(float(info.primal_residual) <= 1e-6)
    _check(int(info.structure_code) == int(jcb_mat.matrix_free_core.structure_code("hermitian")))
    _check(int(info.solver_code) == int(jcb_mat.matrix_free_core.solver_code("cg")))

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

    rational_action = jcb_mat.jcb_mat_rational_action_point(
        op,
        x,
        shifts=jnp.asarray([1.0 + 0.0j, -1.0 + 0.5j], dtype=jnp.complex128),
        weights=jnp.asarray([0.5 + 0.0j, -0.25 + 0.5j], dtype=jnp.complex128),
        polynomial_coefficients=jnp.asarray([1.0 + 0.0j], dtype=jnp.complex128),
        hermitian=False,
    )
    rational_basic = jcb_mat.jcb_mat_rational_action_basic(
        op,
        x,
        shifts=jnp.asarray([1.0 + 0.0j, -1.0 + 0.5j], dtype=jnp.complex128),
        weights=jnp.asarray([0.5 + 0.0j, -0.25 + 0.5j], dtype=jnp.complex128),
        polynomial_coefficients=jnp.asarray([1.0 + 0.0j], dtype=jnp.complex128),
        hermitian=False,
    )
    x_mid_complex = acb_core.acb_midpoint(x)
    rational_diag = jnp.asarray(
        [
            1.0 + 0.5 / ((2.0 + 0.0j) - (1.0 + 0.0j)) + (-0.25 + 0.5j) / ((2.0 + 0.0j) - (-1.0 + 0.5j)),
            1.0 + 0.5 / ((3.0 + 0.0j) - (1.0 + 0.0j)) + (-0.25 + 0.5j) / ((3.0 + 0.0j) - (-1.0 + 0.5j)),
        ],
        dtype=jnp.complex128,
    )
    _check(bool(jnp.allclose(acb_core.acb_midpoint(rational_action), rational_diag * x_mid_complex, rtol=1e-5, atol=1e-5)))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(rational_basic), acb_core.acb_real(rational_action)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(rational_basic), acb_core.acb_imag(rational_action)))))


def test_complex_minres_matrix_free_apis_match_hermitian_indefinite_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -3.0 + 0.0j)
    x = _vec2(1.0 + 2.0j, -2.0 + 1.0j)
    rhs = _vec2((2.0 + 0.0j) * (1.0 + 2.0j), (-3.0 + 0.0j) * (-2.0 + 1.0j))
    op = jcb_mat.jcb_mat_dense_operator(a)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)

    solved, info = jcb_mat.jcb_mat_minres_solve_action_with_diagnostics_point(op, rhs)
    inv_applied = jcb_mat.jcb_mat_minres_inverse_action_point(op, rhs)
    solved_jit = jcb_mat.jcb_mat_minres_solve_action_point_jit(plan, rhs)
    inv_jit = jcb_mat.jcb_mat_minres_inverse_action_point_jit(plan, rhs)

    target = jnp.asarray([1.0 + 2.0j, -2.0 + 1.0j], dtype=jnp.complex128)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_applied), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved_jit), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_jit), target, rtol=1e-6, atol=1e-6)))
    _check(float(info.primal_residual) <= 1e-6)
    _check(int(info.structure_code) == int(jcb_mat.matrix_free_core.structure_code("hermitian")))
    _check(int(info.solver_code) == int(jcb_mat.matrix_free_core.solver_code("minres")))


def test_complex_preconditioned_minres_matrix_free_apis_match_hermitian_indefinite_diagonal_case():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -3.0 + 0.0j)
    rhs = _vec2((2.0 + 0.0j) * (1.0 + 2.0j), (-3.0 + 0.0j) * (-2.0 + 1.0j))
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(plan)

    solved, info = jcb_mat.jcb_mat_minres_solve_action_with_diagnostics_point(plan, rhs, preconditioner=prec)
    inv_applied = jcb_mat.jcb_mat_minres_inverse_action_point(plan, rhs, preconditioner=prec)
    solved_jit = jcb_mat.jcb_mat_minres_solve_action_point_jit(plan, rhs, preconditioner=prec)
    inv_jit = jcb_mat.jcb_mat_minres_inverse_action_point_jit(plan, rhs, preconditioner=prec)

    target = jnp.asarray([1.0 + 2.0j, -2.0 + 1.0j], dtype=jnp.complex128)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_applied), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved_jit), target, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_jit), target, rtol=1e-6, atol=1e-6)))
    _check(float(info.primal_residual) <= 1e-6)


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


def test_complex_block_and_vblock_operator_plans_match_storage_matvec():
    dense2 = jnp.asarray([[2.0 + 0.0j, 1.0 - 0.5j], [1.0 + 0.5j, 3.0 + 0.0j]], dtype=jnp.complex128)
    dense3 = jnp.asarray(
        [[2.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j], [1.0 + 0.5j, 3.0 + 0.0j, -1.0j], [0.0 + 0.0j, 1.0j, 4.0 + 0.0j]],
        dtype=jnp.complex128,
    )
    block = scb_block_mat.scb_block_mat_from_dense_csr(dense2, block_shape=(1, 1))
    vblock = scb_vblock_mat.scb_vblock_mat_from_dense_csr(
        dense3,
        row_block_sizes=jnp.asarray([1, 2], dtype=jnp.int32),
        col_block_sizes=jnp.asarray([1, 2], dtype=jnp.int32),
    )
    x2 = _vec2(1.0 + 0.5j, -1.0 + 1.0j)
    x3 = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, -1.0, 0.5]), jnp.asarray([1.0, -1.0, 0.5])),
        di.interval(jnp.asarray([0.5, 1.0, -0.25]), jnp.asarray([0.5, 1.0, -0.25])),
    )
    x2_mid = acb_core.acb_midpoint(x2)
    x3_mid = acb_core.acb_midpoint(x3)

    bplan = jcb_mat.jcb_mat_block_sparse_operator_plan_prepare(block)
    brplan = jcb_mat.jcb_mat_block_sparse_operator_rmatvec_plan_prepare(block)
    baplan = jcb_mat.jcb_mat_block_sparse_operator_adjoint_plan_prepare(block)
    vplan = jcb_mat.jcb_mat_vblock_sparse_operator_plan_prepare(vblock)
    vrplan = jcb_mat.jcb_mat_vblock_sparse_operator_rmatvec_plan_prepare(vblock)
    vaplan = jcb_mat.jcb_mat_vblock_sparse_operator_adjoint_plan_prepare(vblock)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(bplan, x2)), dense2 @ x2_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(brplan, x2)), dense2.T @ x2_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(baplan, x2)), jnp.conjugate(dense2).T @ x2_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(vplan, x3)), dense3 @ x3_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(vrplan, x3)), dense3.T @ x3_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(vaplan, x3)), jnp.conjugate(dense3).T @ x3_mid)))


def test_jcb_mat_eigsh_operator_plan_matches_dense_hermitian_case():
    dense = jnp.diag(jnp.asarray([1.0, 2.5, 4.5, 8.0], dtype=jnp.float64)).astype(jnp.complex128)
    a = acb_core.acb_box(di.interval(jnp.real(dense), jnp.real(dense)), di.interval(jnp.imag(dense), jnp.imag(dense)))
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    vals, vecs = jcb_mat.jcb_mat_eigsh_point(plan, size=4, k=2, which="largest", steps=4)
    vals_jit, vecs_jit = jcb_mat.jcb_mat_eigsh_point_jit(plan, size=4, k=2, which="largest", steps=4)
    expected = jnp.asarray([4.5, 8.0], dtype=jnp.float64)
    residual = dense @ vecs - vecs * vals[None, :]
    _check(bool(jnp.allclose(vals, expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(vals_jit, vals, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(residual, jnp.zeros_like(residual), rtol=1e-8, atol=1e-8)))
    _check(vecs.shape == (4, 2))
    _check(vecs_jit.shape == vecs.shape)


def test_jcb_mat_native_krylov_schur_davidson_shift_invert_and_contour_surfaces():
    dense = jnp.diag(jnp.asarray([1.0, 3.0, 7.0], dtype=jnp.float64)).astype(jnp.complex128)
    a = acb_core.acb_box(di.interval(jnp.real(dense), jnp.real(dense)), di.interval(jnp.imag(dense), jnp.imag(dense)))
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(plan)

    vals_ks, _ = jcb_mat.jcb_mat_eigsh_krylov_schur_point(plan, size=3, k=1, which="largest", steps=2, restarts=2, block_size=2)
    vals_dav, _ = jcb_mat.jcb_mat_eigsh_davidson_point(plan, size=3, k=1, which="largest", subspace_iters=3, block_size=2, preconditioner=prec)
    vals_jd, _ = jcb_mat.jcb_mat_eigsh_jacobi_davidson_point(plan, size=3, k=1, which="largest", subspace_iters=3, block_size=2, preconditioner=prec)
    vals_si, _ = jcb_mat.jcb_mat_eigsh_shift_invert_point(plan, size=3, shift=2.8 + 0.0j, k=1, which="largest", steps=3, preconditioner=prec)
    vals_contour, _ = jcb_mat.jcb_mat_eigsh_contour_point(plan, size=3, center=3.0 + 0.0j, radius=0.6, k=1, which="largest", quadrature_order=8, block_size=2, preconditioner=prec)

    _check(bool(jnp.allclose(vals_ks, jnp.asarray([7.0 + 0.0j]), atol=1e-4)))
    _check(bool(jnp.allclose(vals_dav, jnp.asarray([7.0 + 0.0j]), atol=1e-3)))
    _check(bool(jnp.allclose(vals_jd, jnp.asarray([7.0 + 0.0j]), atol=1e-3)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals_si), jnp.asarray([3.0 + 0.0j]), atol=5e-2)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals_contour), jnp.asarray([3.0 + 0.0j]), atol=2e-1)))


def test_complex_matrix_free_plan_jit_surface_and_diagnostics_metadata():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    x = _vec2(1.0 + 0.0j, -1.0 + 0.0j)
    rhs = _vec2(2.0 + 0.0j, -3.0 + 0.0j)
    probes = jnp.stack([_vec2(1.0 + 0.0j, 0.0 + 0.0j), _vec2(0.0 + 0.0j, 1.0 + 0.0j)], axis=0)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj_plan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)

    applied = jcb_mat.jcb_mat_operator_apply_point_jit(plan, x)
    r_applied = jcb_mat.jcb_mat_rmatvec_point_jit(a, x)
    solved = jcb_mat.jcb_mat_solve_action_point_jit(plan, rhs, hermitian=True)
    invd = jcb_mat.jcb_mat_inverse_action_point_jit(plan, rhs, hermitian=True)
    log_action = jcb_mat.jcb_mat_log_action_hermitian_point_jit(plan, x, 2, adj_plan)
    pow_action = jcb_mat.jcb_mat_pow_action_arnoldi_point_jit(plan, x, exponent=2, steps=2, adjoint_matvec=adj_plan)
    logdet = jcb_mat.jcb_mat_logdet_slq_point_jit(plan, probes, 2, adj_plan)
    det = jcb_mat.jcb_mat_det_slq_point_jit(plan, probes, 2, adj_plan)
    _, solve_diag = jcb_mat.jcb_mat_solve_action_with_diagnostics_point(plan, rhs, hermitian=True)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(applied), jnp.asarray([2.0 + 0.0j, -3.0 + 0.0j]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(r_applied), jnp.asarray([2.0 + 0.0j, -3.0 + 0.0j]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solved), jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(invd), jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(log_action), jnp.asarray([jnp.log(2.0 + 0.0j), -jnp.log(3.0 + 0.0j)]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(pow_action), jnp.asarray([4.0 + 0.0j, -9.0 + 0.0j]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(logdet, jnp.log(6.0 + 0.0j), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(det, 6.0 + 0.0j, rtol=1e-6, atol=1e-6)))
    _check(float(solve_diag.primal_residual) <= 1e-6)


def test_complex_hermitian_hpd_matrix_free_aliases_match_base_paths():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    x = _vec2(1.0 + 0.0j, -1.0 + 0.0j)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_sin_action_hermitian_point(op, x, 2, adj)), acb_core.acb_midpoint(jcb_mat.jcb_mat_sin_action_arnoldi_point(op, x, 2, adj)), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_cosh_action_hpd_point(op, x, 2, adj)), acb_core.acb_midpoint(jcb_mat.jcb_mat_cosh_action_hermitian_point(op, x, 2, adj)), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_pow_action_hpd_point(op, x, exponent=2, steps=2, adjoint_matvec=adj)), acb_core.acb_midpoint(jcb_mat.jcb_mat_pow_action_hermitian_point(op, x, exponent=2, steps=2, adjoint_matvec=adj)), rtol=1e-6, atol=1e-6)))
    _, diag = jcb_mat.jcb_mat_sin_action_hermitian_with_diagnostics_point(op, x, 2, adj)
    _check(not bool(diag.used_adjoint))
    _check(bool(diag.converged))


def test_complex_logdet_solve_and_eigsh_diagnostics_surfaces():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j)
    rhs = _vec2(2.0 + 0.0j, -3.0 + 0.0j)
    probes = jnp.stack([_vec2(1.0 + 0.0j, 0.0 + 0.0j), _vec2(0.0 + 0.0j, 1.0 + 0.0j)], axis=0)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)

    bundle = jcb_mat.jcb_mat_logdet_solve_point(plan, rhs, probes, 2, adj, hermitian=True)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(bundle.solve), jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j]), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(bundle.logdet, jnp.log(6.0 + 0.0j), rtol=1e-6, atol=1e-6)))
    _check(bool(bundle.aux.solve_diagnostics.converged))
    _check(bool(bundle.aux.logdet_diagnostics.converged))

    vals, vecs, diag = jcb_mat.jcb_mat_eigsh_with_diagnostics_point(plan, size=2, k=1, which="largest", steps=2)
    _check(bool(jnp.allclose(vals, jnp.asarray([3.0]), rtol=1e-6, atol=1e-6)))
    _check(vecs.shape == (2, 1))
    _check(bool(diag.converged))
    _check(int(diag.locked_count) >= 1)
    _check(int(diag.deflated_count) >= 1)
    _check(diag.residual_history.shape[-1] >= 1)


def test_jcb_mat_generalized_eigsh_diagonal_surface():
    a_mid = jnp.diag(jnp.asarray([2.0, 6.0, 12.0], dtype=jnp.float64)).astype(jnp.complex128)
    b_mid = jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)).astype(jnp.complex128)
    a = jnp.stack(
        [
            jnp.stack([_box(2.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(6.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(12.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    b = jnp.stack(
        [
            jnp.stack([_box(1.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(2.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(3.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    a_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    b_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(b)
    b_prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(b_plan)

    vals, vecs = jcb_mat.jcb_mat_geigsh_point(
        a_plan,
        b_plan,
        size=3,
        k=2,
        which="smallest",
        steps=3,
        b_preconditioner=b_prec,
    )
    vals_d, vecs_d, diag = jcb_mat.jcb_mat_geigsh_with_diagnostics_point(
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


def test_jcb_mat_generalized_shift_invert_diagonal_surface():
    a = jnp.stack(
        [
            jnp.stack([_box(2.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(6.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(12.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    b = jnp.stack(
        [
            jnp.stack([_box(1.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(2.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(3.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    a_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    b_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(b)
    b_prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(b_plan)

    vals, vecs = jcb_mat.jcb_mat_geigsh_shift_invert_point(
        a_plan,
        b_plan,
        size=3,
        shift=2.8 + 0.0j,
        k=1,
        which="largest",
        steps=3,
        preconditioner=b_prec,
    )
    vals_d, vecs_d, diag = jcb_mat.jcb_mat_geigsh_shift_invert_with_diagnostics_point(
        a_plan,
        b_plan,
        size=3,
        shift=2.8 + 0.0j,
        k=1,
        which="largest",
        steps=3,
        preconditioner=b_prec,
        tol=1e-4,
    )

    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals), jnp.asarray([3.0 + 0.0j]), atol=5e-2)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals_d), acb_core.acb_midpoint(vals), atol=5e-2)))
    _check(vecs.shape == (3, 1, 4))
    _check(vecs_d.shape == (3, 1, 4))
    _check(bool(diag.converged))


def test_jcb_mat_polynomial_and_nonlinear_eig_surfaces():
    def _as_box_dense(dense: jnp.ndarray) -> jnp.ndarray:
        return acb_core.acb_box(
            di.interval(jnp.real(dense), jnp.real(dense)),
            di.interval(jnp.imag(dense), jnp.imag(dense)),
        )

    a0_dense = jnp.diag(jnp.asarray([4.0, 18.0, 40.0], dtype=jnp.float64)).astype(jnp.complex128)
    a1_dense = jnp.diag(jnp.asarray([-5.0, -9.0, -13.0], dtype=jnp.float64)).astype(jnp.complex128)
    a2_dense = jnp.eye(3, dtype=jnp.complex128)
    coeffs = [
        jcb_mat.jcb_mat_dense_operator_plan_prepare(_as_box_dense(a0_dense)),
        jcb_mat.jcb_mat_dense_operator_plan_prepare(_as_box_dense(a1_dense)),
        jcb_mat.jcb_mat_dense_operator_plan_prepare(_as_box_dense(a2_dense)),
    ]
    vals_p, vecs_p, diag_p = jcb_mat.jcb_mat_peigsh_with_diagnostics_point(
        coeffs,
        size=3,
        lambda0=2.8 + 0.0j,
        newton_iters=4,
        eig_steps=3,
        tol=1e-6,
    )

    def mat_builder(lam):
        dense = jnp.diag(jnp.exp(jnp.real(jnp.asarray(lam, dtype=jnp.complex128))) - jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64)).astype(jnp.complex128)
        return jcb_mat.jcb_mat_dense_operator_plan_prepare(_as_box_dense(dense))

    def dmat_builder(lam):
        dense = (jnp.eye(3, dtype=jnp.float64) * jnp.exp(jnp.real(jnp.asarray(lam, dtype=jnp.complex128)))).astype(jnp.complex128)
        return jcb_mat.jcb_mat_dense_operator_plan_prepare(_as_box_dense(dense))

    vals_n, vecs_n, diag_n = jcb_mat.jcb_mat_neigsh_with_diagnostics_point(
        mat_builder,
        dmat_builder,
        size=3,
        lambda0=1.45 + 0.0j,
        newton_iters=4,
        eig_steps=3,
        tol=1e-6,
    )

    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(vals_p)), jnp.asarray([3.0]), atol=5e-3)))
    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(vals_n)), jnp.asarray([jnp.log(4.0)]), atol=5e-3)))
    _check(vecs_p.shape == (3, 1, 4))
    _check(vecs_n.shape == (3, 1, 4))
    _check(bool(diag_p.converged))
    _check(bool(diag_n.converged))


def test_jcb_mat_restart_policy_handles_block_window_larger_than_k():
    dense = jnp.diag(jnp.asarray([1.0, 2.0, 5.0, 9.0], dtype=jnp.float64)).astype(jnp.complex128)
    a = jnp.stack(
        [
            jnp.stack([_box(1.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(2.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(5.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0), _box(9.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)

    vals_restart, _ = jcb_mat.jcb_mat_eigsh_restarted_point(
        plan,
        size=4,
        k=1,
        which="largest",
        steps=2,
        restarts=3,
        block_size=3,
    )
    vals_ks, _, diag_ks = jcb_mat.jcb_mat_eigsh_krylov_schur_with_diagnostics_point(
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


def test_jcb_mat_davidson_and_jd_handle_block_window_larger_than_k():
    dense = jnp.diag(jnp.asarray([1.0, 2.0, 5.0, 9.0], dtype=jnp.float64)).astype(jnp.complex128)
    a = jnp.stack(
        [
            jnp.stack([_box(1.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(2.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(5.0, 0.0), _box(0.0, 0.0)], axis=0),
            jnp.stack([_box(0.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0), _box(9.0, 0.0)], axis=0),
        ],
        axis=0,
    )
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(plan)

    vals_dav, _ = jcb_mat.jcb_mat_eigsh_davidson_point(
        plan,
        size=4,
        k=1,
        which="largest",
        subspace_iters=3,
        block_size=3,
        preconditioner=prec,
        tol=1e-6,
    )
    vals_jd, _, diag_jd = jcb_mat.jcb_mat_eigsh_jacobi_davidson_with_diagnostics_point(
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
    _check(int(diag_jd.locked_count) >= 1)
    _check(diag_jd.residual_history.shape[-1] >= 1)
