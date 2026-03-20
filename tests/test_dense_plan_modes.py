import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import mat_wrappers

from tests._test_checks import _check


def test_arb_dense_matvec_plan_modes_and_batch_padding():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0]],
            [[0.0, 0.0], [3.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float64)
    batch = jnp.stack([a, a + 1.0], axis=0)

    point_plan = mat_wrappers.arb_mat_dense_matvec_plan_prepare_mode(a, impl="point", prec_bits=53)
    basic_plan = mat_wrappers.arb_mat_dense_matvec_plan_prepare_mode(a, impl="basic", prec_bits=53)
    rig_plan = mat_wrappers.arb_mat_dense_matvec_plan_prepare_mode(a, impl="rigorous", prec_bits=53)
    adapt_plan = mat_wrappers.arb_mat_dense_matvec_plan_prepare_mode(a, impl="adaptive", prec_bits=53)
    point_out = mat_wrappers.arb_mat_dense_matvec_plan_apply_mode(point_plan, x, impl="point", prec_bits=53)
    basic_out = mat_wrappers.arb_mat_dense_matvec_plan_apply_mode(basic_plan, x, impl="basic", prec_bits=53)
    rig_out = mat_wrappers.arb_mat_dense_matvec_plan_apply_mode(rig_plan, x, impl="rigorous", prec_bits=53)
    adapt_out = mat_wrappers.arb_mat_dense_matvec_plan_apply_mode(adapt_plan, x, impl="adaptive", prec_bits=53)
    padded_plan = mat_wrappers.arb_mat_dense_matvec_plan_prepare_batch_mode_padded(batch, pad_to=4, impl="basic", prec_bits=53)
    padded_out = arb_mat.arb_mat_dense_matvec_plan_apply_batch_padded(padded_plan, jnp.stack([x, x + 1.0], axis=0), pad_to=4)

    expected = di.midpoint(a) @ di.midpoint(x)
    _check(point_plan.rows == 2 and point_plan.cols == 2)
    _check(basic_plan.matrix.shape == (2, 2, 2))
    _check(rig_plan.matrix.shape == (2, 2, 2))
    _check(adapt_plan.matrix.shape == (2, 2, 2))
    _check(bool(jnp.allclose(point_out, expected)))
    _check(bool(jnp.allclose(di.midpoint(basic_out), expected)))
    _check(bool(jnp.all(di.contains(rig_out, basic_out))))
    _check(bool(jnp.all(di.contains(adapt_out, basic_out))))
    _check(padded_plan.matrix.shape == (4, 2, 2, 2))
    _check(padded_out.shape == (4, 2, 2))


def test_acb_dense_matvec_plan_modes_and_batch_padding():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.5, 0.5], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0, -1.0], [3.0, 3.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.5, 0.5]], dtype=jnp.float64)
    batch = jnp.stack([a, a], axis=0)

    point_plan = mat_wrappers.acb_mat_dense_matvec_plan_prepare_mode(a, impl="point", prec_bits=53)
    basic_plan = mat_wrappers.acb_mat_dense_matvec_plan_prepare_mode(a, impl="basic", prec_bits=53)
    rig_plan = mat_wrappers.acb_mat_dense_matvec_plan_prepare_mode(a, impl="rigorous", prec_bits=53)
    adapt_plan = mat_wrappers.acb_mat_dense_matvec_plan_prepare_mode(a, impl="adaptive", prec_bits=53)
    point_out = mat_wrappers.acb_mat_dense_matvec_plan_apply_mode(point_plan, x, impl="point", prec_bits=53)
    basic_out = mat_wrappers.acb_mat_dense_matvec_plan_apply_mode(basic_plan, x, impl="basic", prec_bits=53)
    rig_out = mat_wrappers.acb_mat_dense_matvec_plan_apply_mode(rig_plan, x, impl="rigorous", prec_bits=53)
    adapt_out = mat_wrappers.acb_mat_dense_matvec_plan_apply_mode(adapt_plan, x, impl="adaptive", prec_bits=53)
    padded_plan = mat_wrappers.acb_mat_dense_matvec_plan_prepare_batch_mode_padded(batch, pad_to=4, impl="basic", prec_bits=53)
    padded_out = acb_mat.acb_mat_dense_matvec_plan_apply_batch_padded(padded_plan, jnp.stack([x, x], axis=0), pad_to=4)

    expected = acb_core.acb_midpoint(a) @ acb_core.acb_midpoint(x)
    _check(point_plan.rows == 2 and point_plan.cols == 2)
    _check(basic_plan.matrix.shape == (2, 2, 4))
    _check(rig_plan.matrix.shape == (2, 2, 4))
    _check(adapt_plan.matrix.shape == (2, 2, 4))
    _check(bool(jnp.allclose(point_out, expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(basic_out), expected)))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(rig_out), acb_core.acb_real(basic_out)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(rig_out), acb_core.acb_imag(basic_out)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(adapt_out), acb_core.acb_real(basic_out)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(adapt_out), acb_core.acb_imag(basic_out)))))
    _check(padded_plan.matrix.shape == (4, 2, 2, 4))
    _check(padded_out.shape == (4, 2, 4))


def test_arb_dense_factorized_solve_ecosystem():
    a_mid = jnp.array([[3.0, 1.0], [2.0, 4.0]], dtype=jnp.float64)
    rhs_mid = jnp.array([[1.0], [2.0]], dtype=jnp.float64)
    add_mid = jnp.array([[0.5], [0.25]], dtype=jnp.float64)
    a = di.interval(a_mid, a_mid)
    rhs = di.interval(rhs_mid, rhs_mid)
    add = di.interval(add_mid, add_mid)
    plan = arb_mat.arb_mat_dense_lu_solve_plan_prepare(a)

    solve_t = mat_wrappers.arb_mat_solve_transpose_mode(plan, rhs, impl="basic", prec_bits=53)
    solve_add = mat_wrappers.arb_mat_solve_add_mode(plan, rhs, add, impl="basic", prec_bits=53)
    solve_t_add = mat_wrappers.arb_mat_solve_transpose_add_mode(plan, rhs, add, impl="basic", prec_bits=53)
    matsolve = mat_wrappers.arb_mat_mat_solve_mode(plan, rhs, impl="basic", prec_bits=53)
    matsolve_t = mat_wrappers.arb_mat_mat_solve_transpose_mode(plan, rhs, impl="basic", prec_bits=53)
    batch_t = mat_wrappers.arb_mat_solve_transpose_batch_mode_padded(plan, jnp.stack([rhs, rhs], axis=0), pad_to=4, impl="basic", prec_bits=53)

    expected_t = jnp.linalg.solve(a_mid.T, rhs_mid)
    expected = jnp.linalg.solve(a_mid, rhs_mid)
    _check(bool(jnp.allclose(di.midpoint(solve_t), expected_t)))
    _check(bool(jnp.allclose(di.midpoint(solve_add), expected + add_mid)))
    _check(bool(jnp.allclose(di.midpoint(solve_t_add), expected_t + add_mid)))
    _check(bool(jnp.allclose(di.midpoint(matsolve), expected)))
    _check(bool(jnp.allclose(di.midpoint(matsolve_t), expected_t)))
    _check(batch_t.shape == (4, 2, 1, 2))


def test_acb_dense_factorized_solve_ecosystem():
    a_mid = jnp.array([[3.0 + 0.0j, 1.0 - 0.5j], [2.0 + 0.25j, 4.0 + 0.0j]], dtype=jnp.complex128)
    rhs_mid = jnp.array([[1.0 + 0.25j], [2.0 - 0.5j]], dtype=jnp.complex128)
    add_mid = jnp.array([[0.5 - 0.1j], [0.25 + 0.2j]], dtype=jnp.complex128)
    a = acb_core.acb_box(di.interval(jnp.real(a_mid), jnp.real(a_mid)), di.interval(jnp.imag(a_mid), jnp.imag(a_mid)))
    rhs = acb_core.acb_box(di.interval(jnp.real(rhs_mid), jnp.real(rhs_mid)), di.interval(jnp.imag(rhs_mid), jnp.imag(rhs_mid)))
    add = acb_core.acb_box(di.interval(jnp.real(add_mid), jnp.real(add_mid)), di.interval(jnp.imag(add_mid), jnp.imag(add_mid)))
    plan = acb_mat.acb_mat_dense_lu_solve_plan_prepare(a)

    solve_t = mat_wrappers.acb_mat_solve_transpose_mode(plan, rhs, impl="basic", prec_bits=53)
    solve_add = mat_wrappers.acb_mat_solve_add_mode(plan, rhs, add, impl="basic", prec_bits=53)
    solve_t_add = mat_wrappers.acb_mat_solve_transpose_add_mode(plan, rhs, add, impl="basic", prec_bits=53)
    matsolve = mat_wrappers.acb_mat_mat_solve_mode(plan, rhs, impl="basic", prec_bits=53)
    matsolve_t = mat_wrappers.acb_mat_mat_solve_transpose_mode(plan, rhs, impl="basic", prec_bits=53)
    batch_t = mat_wrappers.acb_mat_solve_transpose_batch_mode_padded(plan, jnp.stack([rhs, rhs], axis=0), pad_to=4, impl="basic", prec_bits=53)

    expected_t = jnp.linalg.solve(a_mid.T, rhs_mid)
    expected = jnp.linalg.solve(a_mid, rhs_mid)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_t), expected_t)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_add), expected + add_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_t_add), expected_t + add_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(matsolve), expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(matsolve_t), expected_t)))
    _check(batch_t.shape == (4, 2, 1, 4))
