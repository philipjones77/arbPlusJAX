import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import mat_wrappers
from arbplusjax import double_interval as di

from tests._test_checks import _check


def test_arb_mat_det_trace_modes():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )

    det_basic = mat_wrappers.arb_mat_det_mode(a, impl="basic", prec_bits=53)
    det_rig = mat_wrappers.arb_mat_det_mode(a, impl="rigorous", prec_bits=53)
    det_adapt = mat_wrappers.arb_mat_det_mode(a, impl="adaptive", prec_bits=53)
    det_point = mat_wrappers.arb_mat_det_mode(a, impl="point", prec_bits=53)

    tr_basic = mat_wrappers.arb_mat_trace_mode(a, impl="basic", prec_bits=53)
    tr_rig = mat_wrappers.arb_mat_trace_mode(a, impl="rigorous", prec_bits=53)
    tr_adapt = mat_wrappers.arb_mat_trace_mode(a, impl="adaptive", prec_bits=53)
    tr_point = mat_wrappers.arb_mat_trace_mode(a, impl="point", prec_bits=53)

    a_mid = di.midpoint(a)

    _check(det_point.shape == ())
    _check(det_basic.shape == (2,))
    _check(det_rig.shape == (2,))
    _check(det_adapt.shape == (2,))
    _check(tr_point.shape == ())
    _check(tr_basic.shape == (2,))
    _check(tr_rig.shape == (2,))
    _check(tr_adapt.shape == (2,))
    _check(bool(jnp.allclose(det_point, jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(tr_point, jnp.trace(a_mid))))
    _check(bool(di.contains(det_rig, det_basic)))
    _check(bool(di.contains(tr_rig, tr_basic)))


def test_acb_mat_det_trace_modes():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.5, 0.5]],
        ],
        dtype=jnp.float64,
    )

    det_basic = mat_wrappers.acb_mat_det_mode(a, impl="basic", prec_bits=53)
    det_rig = mat_wrappers.acb_mat_det_mode(a, impl="rigorous", prec_bits=53)
    det_adapt = mat_wrappers.acb_mat_det_mode(a, impl="adaptive", prec_bits=53)
    det_point = mat_wrappers.acb_mat_det_mode(a, impl="point", prec_bits=53)

    tr_basic = mat_wrappers.acb_mat_trace_mode(a, impl="basic", prec_bits=53)
    tr_rig = mat_wrappers.acb_mat_trace_mode(a, impl="rigorous", prec_bits=53)
    tr_adapt = mat_wrappers.acb_mat_trace_mode(a, impl="adaptive", prec_bits=53)
    tr_point = mat_wrappers.acb_mat_trace_mode(a, impl="point", prec_bits=53)

    a_mid = acb_core.acb_midpoint(a)

    _check(det_point.shape == ())
    _check(det_basic.shape == (4,))
    _check(det_rig.shape == (4,))
    _check(det_adapt.shape == (4,))
    _check(tr_point.shape == ())
    _check(tr_basic.shape == (4,))
    _check(tr_rig.shape == (4,))
    _check(tr_adapt.shape == (4,))
    _check(bool(jnp.allclose(det_point, jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(tr_point, jnp.trace(a_mid))))
    _check(bool(di.contains(acb_core.acb_real(det_rig), acb_core.acb_real(det_basic))))
    _check(bool(di.contains(acb_core.acb_imag(det_rig), acb_core.acb_imag(det_basic))))
    _check(bool(di.contains(acb_core.acb_real(tr_rig), acb_core.acb_real(tr_basic))))
    _check(bool(di.contains(acb_core.acb_imag(tr_rig), acb_core.acb_imag(tr_basic))))


def test_complex_matrix_rmatvec_modes_and_cached_surface():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.25, 0.25], [1.0, 1.0, -0.5, -0.5], [0.0, 0.0, 0.0, 0.0]],
            [[-1.0, -1.0, 0.75, 0.75], [3.0, 3.0, 0.0, 0.0], [2.0, 2.0, 0.5, 0.5]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0, 0.5, 0.5], [-2.0, -2.0, 0.25, 0.25]], dtype=jnp.float64)
    point_cache = api.eval_point("acb_mat_rmatvec_cached_prepare", a)
    basic_cache = mat_wrappers.acb_mat_rmatvec_cached_prepare_mode(a, impl="basic", prec_bits=53)
    point = mat_wrappers.acb_mat_rmatvec_mode(a, x, impl="point", prec_bits=53)
    basic = mat_wrappers.acb_mat_rmatvec_mode(a, x, impl="basic", prec_bits=53)
    cached_point = mat_wrappers.acb_mat_rmatvec_cached_apply_mode(point_cache, x, impl="point", prec_bits=53)
    cached_basic = mat_wrappers.acb_mat_rmatvec_cached_apply_mode(basic_cache, x, impl="basic", prec_bits=53)
    expected = acb_core.acb_midpoint(a).T @ acb_core.acb_midpoint(x)

    _check(bool(jnp.allclose(point, expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(basic), expected)))
    _check(bool(jnp.allclose(cached_point, expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(cached_basic), expected)))


def test_matrix_point_api_fastpath():
    a = jnp.array(
        [
            [
                [[2.0, 2.0], [1.0, 1.0]],
                [[0.0, 0.0], [3.0, 3.0]],
            ],
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [4.0, 4.0]],
            ],
        ],
        dtype=jnp.float64,
    )
    out = api.eval_point_batch("arb_mat_det", a)
    _check(out.shape == (2,))
    _check(bool(jnp.allclose(out, jnp.linalg.det(di.midpoint(a)))))


def test_matrix_cached_matvec_and_sqr_modes():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0]],
            [[0.0, 0.0], [3.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float64)

    point_cache = api.eval_point("arb_mat_matvec_cached_prepare", a)
    basic_cache = a
    cached_point = mat_wrappers.arb_mat_matvec_cached_apply_mode(point_cache, x, impl="point", prec_bits=53)
    cached_basic = mat_wrappers.arb_mat_matvec_cached_apply_mode(basic_cache, x, impl="basic", prec_bits=53)
    sq_point = mat_wrappers.arb_mat_sqr_mode(a, impl="point", prec_bits=53)
    sq_basic = mat_wrappers.arb_mat_sqr_mode(a, impl="basic", prec_bits=53)
    sq_rig = mat_wrappers.arb_mat_sqr_mode(a, impl="rigorous", prec_bits=53)
    sq_adapt = mat_wrappers.arb_mat_sqr_mode(a, impl="adaptive", prec_bits=53)

    a_mid = di.midpoint(a)
    x_mid = di.midpoint(x)

    _check(cached_point.shape == (2,))
    _check(cached_basic.shape == (2, 2))
    _check(bool(jnp.allclose(cached_point, a_mid @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(cached_basic), a_mid @ x_mid)))
    _check(sq_point.shape == (2, 2))
    _check(sq_basic.shape == (2, 2, 2))
    _check(sq_rig.shape == (2, 2, 2))
    _check(sq_adapt.shape == (2, 2, 2))
    _check(bool(jnp.allclose(sq_point, a_mid @ a_mid)))
    _check(bool(jnp.all(di.contains(sq_rig, sq_basic))))


def test_matrix_rmatvec_modes_and_cached_surface():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[-1.0, -1.0], [3.0, 3.0], [2.0, 2.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [-2.0, -2.0]], dtype=jnp.float64)
    point_cache = api.eval_point("arb_mat_rmatvec_cached_prepare", a)
    basic_cache = mat_wrappers.arb_mat_rmatvec_cached_prepare_mode(a, impl="basic", prec_bits=53)
    point = mat_wrappers.arb_mat_rmatvec_mode(a, x, impl="point", prec_bits=53)
    basic = mat_wrappers.arb_mat_rmatvec_mode(a, x, impl="basic", prec_bits=53)
    cached_point = mat_wrappers.arb_mat_rmatvec_cached_apply_mode(point_cache, x, impl="point", prec_bits=53)
    cached_basic = mat_wrappers.arb_mat_rmatvec_cached_apply_mode(basic_cache, x, impl="basic", prec_bits=53)
    expected = di.midpoint(a).T @ di.midpoint(x)

    _check(bool(jnp.allclose(point, expected)))
    _check(bool(jnp.allclose(di.midpoint(basic), expected)))
    _check(bool(jnp.allclose(cached_point, expected)))
    _check(bool(jnp.allclose(di.midpoint(cached_basic), expected)))


def test_matrix_cached_prepare_modes_and_batch_api():
    a = jnp.array(
        [
            [
                [[2.0, 2.0], [1.0, 1.0]],
                [[0.0, 0.0], [3.0, 3.0]],
            ],
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [4.0, 4.0]],
            ],
        ],
        dtype=jnp.float64,
    )

    point = api.eval_point_batch("arb_mat_matvec_cached_prepare", a)
    basic = api.eval_interval_batch("arb_mat_matvec_cached_prepare", a, mode="basic", prec_bits=53)
    adaptive = api.eval_interval_batch("arb_mat_matvec_cached_prepare", a, mode="adaptive", prec_bits=53)
    rigorous = api.eval_interval_batch("arb_mat_matvec_cached_prepare", a, mode="rigorous", prec_bits=53)

    _check(point.shape == (2, 2, 2))
    _check(basic.shape == (2, 2, 2, 2))
    _check(adaptive.shape == (2, 2, 2, 2))
    _check(rigorous.shape == (2, 2, 2, 2))
    _check(bool(jnp.allclose(point, di.midpoint(a))))
    _check(bool(jnp.all(di.contains(rigorous, basic))))


def test_matrix_norm_modes_and_point_api():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[-1.0, -1.0], [3.0, 3.0], [2.0, 2.0]],
            [[0.5, 0.5], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    a_mid = di.midpoint(a)

    fro_point = api.eval_point("arb_mat_norm_fro", a)
    one_point = api.eval_point("arb_mat_norm_1", a)
    inf_point = api.eval_point("arb_mat_norm_inf", a)
    fro_basic = mat_wrappers.arb_mat_norm_fro_mode(a, impl="basic", prec_bits=53)
    one_rig = mat_wrappers.arb_mat_norm_1_mode(a, impl="rigorous", prec_bits=53)
    inf_adapt = mat_wrappers.arb_mat_norm_inf_mode(a, impl="adaptive", prec_bits=53)

    _check(bool(jnp.allclose(fro_point, jnp.linalg.norm(a_mid, ord="fro"))))
    _check(bool(jnp.allclose(one_point, jnp.linalg.norm(a_mid, ord=1))))
    _check(bool(jnp.allclose(inf_point, jnp.linalg.norm(a_mid, ord=jnp.inf))))
    _check(fro_basic.shape == (2,))
    _check(one_rig.shape == (2,))
    _check(inf_adapt.shape == (2,))
    _check(bool(di.contains(one_rig, mat_wrappers.arb_mat_norm_1_mode(a, impl="basic", prec_bits=53))))


def test_matrix_batch_fastpaths_and_constructors():
    batch = jnp.array(
        [
            [
                [[2.0, 2.0], [0.0, 0.0]],
                [[1.0, 1.0], [3.0, 3.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [4.0, 4.0]],
            ],
        ],
        dtype=jnp.float64,
    )

    eye = api.eval_point("arb_mat_identity", 2)
    zeros = api.eval_point("arb_mat_zero", 2)
    dets = api.eval_interval_batch("arb_mat_det", batch, prec_bits=53)
    fros = api.eval_point_batch("arb_mat_norm_fro", batch)

    _check(eye.shape == (2, 2))
    _check(zeros.shape == (2, 2))
    _check(bool(jnp.allclose(eye, jnp.eye(2))))
    _check(bool(jnp.allclose(zeros, jnp.zeros((2, 2)))))
    _check(dets.shape == (2, 2))
    _check(fros.shape == (2,))
    _check(bool(jnp.allclose(fros, jnp.linalg.norm(di.midpoint(batch), ord="fro", axis=(-2, -1)))))


def test_matrix_structure_and_lu_solve_modes():
    a = jnp.array(
        [
            [[4.0, 4.0], [1.0, 1.0]],
            [[2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[2.0, 2.0], [1.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    rhs = api.eval_interval("arb_mat_matmul", a, x, mode="basic")
    p, l, u = api.eval_interval("arb_mat_lu", a, mode="basic")
    sol = api.eval_interval("arb_mat_lu_solve", (p, l, u), rhs, mode="basic")
    t_point = api.eval_point("arb_mat_transpose", a)
    d_basic = api.eval_interval("arb_mat_diag", a, mode="basic")
    dm_point = api.eval_point("arb_mat_diag_matrix", api.eval_interval("arb_mat_diag", a, mode="basic"))

    a_mid = di.midpoint(a)
    x_mid = di.midpoint(x)

    _check(bool(jnp.allclose(di.midpoint(sol), x_mid)))
    _check(bool(jnp.allclose(t_point, a_mid.T)))
    _check(bool(jnp.allclose(di.midpoint(d_basic), jnp.diag(a_mid))))
    _check(bool(jnp.allclose(dm_point, jnp.diag(jnp.diag(a_mid)))))


def test_structure_rigorous_modes_are_exact_transforms():
    a = jnp.array(
        [
            [[0.9, 1.1], [1.8, 2.2]],
            [[-0.4, 0.5], [2.9, 3.1]],
        ],
        dtype=jnp.float64,
    )
    t_rig = api.eval_interval("arb_mat_transpose", a, mode="rigorous", prec_bits=53)
    d_rig = api.eval_interval("arb_mat_diag", a, mode="rigorous", prec_bits=53)
    dm_rig = api.eval_interval("arb_mat_diag_matrix", d_rig, mode="rigorous", prec_bits=53)

    _check(bool(jnp.allclose(t_rig, jnp.swapaxes(a, -3, -2))))
    _check(bool(jnp.allclose(d_rig, jnp.stack([a[0, 0, :], a[1, 1, :]], axis=0))))
    _check(bool(jnp.allclose(dm_rig[jnp.arange(2), jnp.arange(2), :], d_rig)))


def test_block_structure_modes_are_exact_transforms():
    a11 = jnp.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], dtype=jnp.float64)
    a12 = jnp.array([[[5.0, 5.0]], [[6.0, 6.0]]], dtype=jnp.float64)
    a21 = jnp.array([[[7.0, 7.0], [8.0, 8.0]]], dtype=jnp.float64)
    a22 = jnp.array([[[9.0, 9.0]]], dtype=jnp.float64)
    assembled_basic = api.eval_interval("arb_mat_block_assemble", ((a11, a12), (a21, a22)), mode="basic")
    diag_rig = api.eval_interval("arb_mat_block_diag", (a11, a22), mode="rigorous", prec_bits=53)
    extracted = api.eval_interval("arb_mat_block_extract", assembled_basic, (2, 1), (2, 1), 1, 0, mode="rigorous", prec_bits=53)

    expected = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]],
            [[3.0, 3.0], [4.0, 4.0], [6.0, 6.0]],
            [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
        ],
        dtype=jnp.float64,
    )
    expected_diag = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [9.0, 9.0]],
        ],
        dtype=jnp.float64,
    )

    _check(bool(jnp.allclose(assembled_basic, expected)))
    _check(bool(jnp.allclose(diag_rig, expected_diag)))
    _check(bool(jnp.allclose(extracted, expected[2:, :2, :])))


def test_dense_matvec_plan_modes_and_batch_padding():
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
