import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import sparse_common as sc


def test_sparse_common_dense_bcoo_round_trip_and_matvec_contracts():
    dense = jnp.asarray([[2.0, 0.0], [1.0, 3.0]], dtype=jnp.float64)
    vec = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    rhs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)

    bcoo = sc.dense_to_sparse_bcoo(dense, algebra="srb")
    plan = sc.sparse_matvec_plan_from_sparse(bcoo, algebra="srb", label="test.sparse_common.plan")

    assert jnp.allclose(sc.sparse_bcoo_to_dense(bcoo, algebra="srb", label="test.sparse_common.to_dense"), dense)
    assert jnp.allclose(sc.sparse_bcoo_matvec(bcoo, vec, algebra="srb", label="test.sparse_common.matvec"), dense @ vec)
    assert jnp.allclose(sc.sparse_bcoo_matmul_dense_rhs(bcoo, rhs, algebra="srb", label="test.sparse_common.matmul_rhs"), dense @ rhs)
    assert jnp.allclose(sc.sparse_matvec_plan_apply(plan, vec, algebra="srb", label="test.sparse_common.plan_apply"), dense @ vec)


def test_sparse_common_add_and_batch_helpers_preserve_expected_shapes():
    x = sc.dense_to_sparse_bcoo(jnp.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float64), algebra="srb")
    y = sc.dense_to_sparse_bcoo(jnp.asarray([[0.0, 3.0], [4.0, 0.0]], dtype=jnp.float64), algebra="srb")
    added = sc.sparse_bcoo_add(x, y, algebra="srb", label="test.sparse_common.add")

    batch = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    padded = sc.pad_validated_batch(batch, pad_to=4, validate=lambda arr, _label: arr, label="test.sparse_common.pad")
    vm_fixed = sc.vmapped_batch_fixed(batch, validate=lambda arr, _label: arr, label="test.sparse_common.fixed", apply=lambda row: 2.0 * row)
    vm_padded = sc.vmapped_batch_padded(batch, pad_to=4, validate=lambda arr, _label: arr, label="test.sparse_common.padded", apply=lambda row: 2.0 * row)

    assert added.data.shape[0] == x.data.shape[0] + y.data.shape[0]
    assert padded.shape == (4, 2)
    assert vm_fixed.shape == batch.shape
    assert vm_padded.shape == (4, 2)


def test_sparse_common_interval_and_box_scaling_preserve_container_type():
    iv_data = jnp.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    interval_bcoo = sc.SparseIntervalBCOO(
        data=iv_data,
        indices=jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="srb",
    )
    scaled_iv = sc.sparse_interval_scale(interval_bcoo, jnp.asarray(2.0, dtype=jnp.float64), algebra="srb")

    box0 = acb_core.acb_box(di.interval(1.0, 1.0), di.interval(0.0, 0.0))
    box1 = acb_core.acb_box(di.interval(2.0, 2.0), di.interval(0.0, 0.0))
    box_bcoo = sc.SparseBoxBCOO(
        data=jnp.stack([box0, box1], axis=0),
        indices=jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="scb",
    )
    scaled_box = sc.sparse_box_scale(box_bcoo, jnp.asarray(2.0 + 0.0j, dtype=jnp.complex128), algebra="scb")

    assert isinstance(scaled_iv, sc.SparseIntervalBCOO)
    assert scaled_iv.data.shape == interval_bcoo.data.shape
    assert isinstance(scaled_box, sc.SparseBoxBCOO)
    assert scaled_box.data.shape == box_bcoo.data.shape
