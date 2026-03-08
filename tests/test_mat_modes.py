import jax.numpy as jnp

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

    tr_basic = mat_wrappers.arb_mat_trace_mode(a, impl="basic", prec_bits=53)
    tr_rig = mat_wrappers.arb_mat_trace_mode(a, impl="rigorous", prec_bits=53)
    tr_adapt = mat_wrappers.arb_mat_trace_mode(a, impl="adaptive", prec_bits=53)

    _check(det_basic.shape == (2,))
    _check(det_rig.shape == (2,))
    _check(det_adapt.shape == (2,))
    _check(tr_basic.shape == (2,))
    _check(tr_rig.shape == (2,))
    _check(tr_adapt.shape == (2,))
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

    tr_basic = mat_wrappers.acb_mat_trace_mode(a, impl="basic", prec_bits=53)
    tr_rig = mat_wrappers.acb_mat_trace_mode(a, impl="rigorous", prec_bits=53)
    tr_adapt = mat_wrappers.acb_mat_trace_mode(a, impl="adaptive", prec_bits=53)

    _check(det_basic.shape == (4,))
    _check(det_rig.shape == (4,))
    _check(det_adapt.shape == (4,))
    _check(tr_basic.shape == (4,))
    _check(tr_rig.shape == (4,))
    _check(tr_adapt.shape == (4,))
    _check(bool(di.contains(acb_core.acb_real(det_rig), acb_core.acb_real(det_basic))))
    _check(bool(di.contains(acb_core.acb_imag(det_rig), acb_core.acb_imag(det_basic))))
    _check(bool(di.contains(acb_core.acb_real(tr_rig), acb_core.acb_real(tr_basic))))
    _check(bool(di.contains(acb_core.acb_imag(tr_rig), acb_core.acb_imag(tr_basic))))
