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
