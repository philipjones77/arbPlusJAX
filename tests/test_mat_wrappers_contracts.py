from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import mat_common
from arbplusjax import mat_wrappers as mw
from arbplusjax import srb_mat


def test_mat_wrappers_dense_mode_dispatch_and_plan_contracts() -> None:
    a = di.interval(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), jnp.asarray([[1.1, 2.1], [3.1, 4.1]]))
    b = di.interval(jnp.asarray([[0.5, 0.25], [0.75, 1.5]]), jnp.asarray([[0.6, 0.35], [0.85, 1.6]]))
    v = di.interval(jnp.asarray([1.0, 2.0]), jnp.asarray([1.1, 2.1]))

    add_basic = mw.arb_mat_add_mode(a, b, impl="basic", prec_bits=80)
    add_baseline = mw.arb_mat_add_mode(a, b, impl="baseline", prec_bits=80)
    matvec_basic = mw.arb_mat_matvec_mode(a, v, impl="basic", prec_bits=80)
    plan = mw.arb_mat_dense_matvec_plan_prepare_mode(a, impl="basic", prec_bits=80)
    plan_out = mw.arb_mat_dense_matvec_plan_apply_mode(plan, v, impl="basic", prec_bits=80)

    assert add_basic.shape == (2, 2, 2)
    assert jnp.allclose(add_basic, add_baseline)
    assert matvec_basic.shape == (2, 2)
    assert isinstance(plan, mat_common.DenseMatvecPlan)
    assert plan_out.shape == (2, 2)


def test_mat_wrappers_sparse_mode_dispatch_reuses_sparse_surface() -> None:
    sp = srb_mat.srb_mat_identity(2)
    v = jnp.asarray([1.0, 2.0])

    point = mw.srb_mat_matvec_mode(sp, v, impl="point")
    basic = mw.srb_mat_matvec_mode(sp, v, impl="basic")
    rigorous = mw.srb_mat_matvec_mode(sp, v, impl="rigorous")

    assert jnp.array_equal(point, v)
    assert basic.shape == (2, 2)
    assert rigorous.shape == (2, 2)
    assert jnp.array_equal(di.midpoint(basic), v)
    assert jnp.array_equal(di.midpoint(rigorous), v)
