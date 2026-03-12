import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import acb_core
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
