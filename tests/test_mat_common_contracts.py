from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import mat_common as mc


def test_mat_common_plan_types_and_pytree_roundtrip() -> None:
    mat = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    plan = mc.DenseMatvecPlan(matrix=mat, rows=2, cols=2, algebra="real")
    lu_plan = mc.DenseLUSolvePlan(p=jnp.arange(2), l=jnp.eye(2), u=mat, rows=2, algebra="real")
    chol_plan = mc.DenseCholeskySolvePlan(factor=jnp.eye(2), rows=2, algebra="real", structure="spd")

    assert mc.is_dense_plan_like(plan)
    assert mc.is_dense_plan_like(lu_plan)
    assert mc.is_dense_plan_like(chol_plan)
    assert mc.is_dense_plan_like((jnp.arange(2), jnp.eye(2), mat))
    assert not mc.is_dense_plan_like(mat)
    assert mc.is_batch_pad_candidate(jnp.asarray([1.0]))
    assert not mc.is_batch_pad_candidate(3.0)

    leaves, treedef = jax.tree_util.tree_flatten(plan)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, mc.DenseMatvecPlan)
    assert jnp.array_equal(restored.matrix, mat)


def test_mat_common_interval_and_box_shape_guards_and_constructors() -> None:
    ivec = di.interval(jnp.asarray([1.0, 2.0]), jnp.asarray([1.1, 2.1]))
    imat = di.interval(jnp.asarray([[1.0, 0.0], [0.0, 1.0]]), jnp.asarray([[1.1, 0.1], [0.1, 1.1]]))
    box = acb_core.acb_box(imat, imat)

    assert mc.as_interval_vector(ivec, "ivec").shape == (2, 2)
    assert mc.as_interval_matrix(imat, "imat").shape == (2, 2, 2)
    assert mc.as_interval_rect_matrix(imat, "imat").shape == (2, 2, 2)
    assert mc.as_box_matrix(box, "box").shape == (2, 2, 4)
    assert mc.as_box_rect_matrix(box, "box").shape == (2, 2, 4)

    full_iv = mc.full_interval_like(ivec)
    full_box = mc.full_box_like(box)
    assert jnp.all(jnp.isneginf(full_iv[..., 0]))
    assert jnp.all(jnp.isposinf(full_iv[..., 1]))
    assert jnp.all(jnp.isneginf(full_box[..., 0]))
    assert jnp.all(jnp.isposinf(full_box[..., 1]))

    point_iv = mc.interval_from_point(jnp.asarray([2.0, -1.0]))
    point_box = mc.box_from_point(jnp.asarray([2.0 + 1.0j, -1.0 - 3.0j]))
    assert jnp.all(point_iv[..., 0] <= jnp.asarray([2.0, -1.0]))
    assert jnp.all(point_iv[..., 1] >= jnp.asarray([2.0, -1.0]))
    assert jnp.all(acb_core.acb_real(point_box)[..., 0] <= jnp.asarray([2.0, -1.0]))
    assert jnp.all(acb_core.acb_real(point_box)[..., 1] >= jnp.asarray([2.0, -1.0]))

    with pytest.raises(ValueError):
        mc.as_interval_matrix(ivec, "bad")
    with pytest.raises(ValueError):
        mc.as_box_matrix(point_box, "bad")


def test_mat_common_overlap_finiteness_and_midpoint_symmetry_helpers() -> None:
    a = di.interval(jnp.asarray([0.0, 2.0]), jnp.asarray([1.0, 3.0]))
    b = di.interval(jnp.asarray([0.5, 4.0]), jnp.asarray([1.5, 5.0]))
    c = di.interval(jnp.asarray([5.0, 6.0]), jnp.asarray([6.0, 7.0]))

    assert jnp.array_equal(mc.interval_overlaps(a, b), jnp.asarray([True, False]))
    assert jnp.array_equal(mc.interval_equal(a, a), jnp.asarray([True, True]))
    assert jnp.array_equal(mc.interval_is_zero(di.interval(jnp.zeros(2), jnp.zeros(2))), jnp.asarray([True, True]))
    assert jnp.array_equal(mc.interval_is_finite(a), jnp.asarray([True, True]))

    box_a = acb_core.acb_box(a, a)
    box_b = acb_core.acb_box(b, b)
    assert jnp.array_equal(mc.box_overlaps(box_a, box_b), jnp.asarray([True, False]))
    assert jnp.array_equal(mc.box_equal(box_a, box_a), jnp.asarray([True, True]))
    assert jnp.array_equal(mc.box_is_finite(box_a), jnp.asarray([True, True]))

    sym = jnp.asarray([[1.0, 2.0], [2.0, 4.0]])
    herm = jnp.asarray([[1.0 + 0.0j, 2.0 - 1.0j], [2.0 + 1.0j, 3.0 + 0.0j]])
    nonsym = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert bool(mc.real_midpoint_is_symmetric(sym))
    assert bool(mc.complex_midpoint_is_hermitian(herm))
    assert not bool(mc.real_midpoint_is_symmetric(nonsym))
    assert jnp.allclose(mc.real_midpoint_symmetric_part(nonsym), jnp.asarray([[1.0, 2.5], [2.5, 4.0]]))
