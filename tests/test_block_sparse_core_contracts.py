import jax.numpy as jnp

from arbplusjax import block_sparse_core as bsc
from arbplusjax import sparse_common as sc


def _block_coo_ctor(data, row, col, *, shape, block_shape):
    return sc.BlockSparseCOO(
        data=jnp.asarray(data),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        block_rows=int(block_shape[0]),
        block_cols=int(block_shape[1]),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def _block_csr_ctor(data, indices, indptr, *, shape, block_shape):
    return sc.BlockSparseCSR(
        data=jnp.asarray(data),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        block_rows=int(block_shape[0]),
        block_cols=int(block_shape[1]),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def _as_coo(x, *, algebra, label):
    return sc.as_block_sparse_coo(x, algebra=algebra, label=label)


def _as_csr(x, *, algebra, label):
    return sc.as_block_sparse_csr(x, algebra=algebra, label=label)


def _coo_to_csr(x):
    return bsc.coo_to_csr(
        x,
        as_coo=_as_coo,
        algebra="srb",
        label="test.block_sparse.coo_to_csr",
        csr_ctor=_block_csr_ctor,
    )


def _csr_to_coo(x):
    return bsc.csr_to_coo(
        x,
        as_csr=_as_csr,
        algebra="srb",
        label="test.block_sparse.csr_to_coo",
        coo_ctor=_block_coo_ctor,
    )


def _to_dense(x):
    return bsc.to_dense(
        x,
        as_coo=_as_coo,
        csr_to_coo_fn=_csr_to_coo,
        algebra="srb",
        label="test.block_sparse.to_dense",
    )


def test_block_sparse_round_trip_shape_and_density_contracts():
    dense = jnp.asarray(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ],
        dtype=jnp.float64,
    )
    coo = bsc.from_dense_coo(
        dense,
        dtype=jnp.float64,
        block_shape=(2, 2),
        tol=0.0,
        label="test.block_sparse.from_dense",
        coo_ctor=_block_coo_ctor,
    )
    csr = _coo_to_csr(coo)

    assert bsc.shape(coo, as_coo=_as_coo, as_csr=_as_csr, algebra="srb", label="shape") == (4, 4)
    assert bsc.block_shape(coo, as_coo=_as_coo, as_csr=_as_csr, algebra="srb", label="block_shape") == (2, 2)
    assert bsc.nnzb(coo, as_coo=_as_coo, as_csr=_as_csr, algebra="srb", label="nnzb") == 2
    assert jnp.allclose(_to_dense(coo), dense)
    assert jnp.allclose(_to_dense(csr), dense)


def test_block_sparse_matvec_plan_and_dense_rhs_match_dense_reference():
    dense = jnp.asarray(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 6.0, 0.0],
            [0.0, 7.0, 0.0, 8.0],
        ],
        dtype=jnp.float64,
    )
    vec = jnp.asarray([1.0, -1.0, 2.0, -2.0], dtype=jnp.float64)
    rhs = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    coo = bsc.from_dense_coo(
        dense,
        dtype=jnp.float64,
        block_shape=(2, 2),
        tol=0.0,
        label="test.block_sparse.from_dense",
        coo_ctor=_block_coo_ctor,
    )
    plan = bsc.matvec_plan_prepare(
        coo,
        as_csr=_as_csr,
        coo_to_csr_fn=_coo_to_csr,
        algebra="srb",
        label="test.block_sparse.plan_prepare",
    )

    assert jnp.allclose(
        bsc.matvec(coo, vec, dtype=jnp.float64, as_coo=_as_coo, csr_to_coo_fn=_csr_to_coo, algebra="srb", label="test.block_sparse.matvec"),
        dense @ vec,
    )
    assert jnp.allclose(
        bsc.matvec_plan_apply(plan, vec, dtype=jnp.float64, algebra="srb", label="test.block_sparse.plan_apply"),
        dense @ vec,
    )
    assert jnp.allclose(
        bsc.matmul_dense_rhs(coo, rhs, dtype=jnp.float64, as_coo=_as_coo, csr_to_coo_fn=_csr_to_coo, algebra="srb", label="test.block_sparse.matmul_rhs"),
        dense @ rhs,
    )


def test_block_sparse_transpose_batch_and_diag_helpers_are_consistent():
    dense = jnp.asarray(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ],
        dtype=jnp.float64,
    )
    coo = bsc.from_dense_coo(
        dense,
        dtype=jnp.float64,
        block_shape=(2, 2),
        tol=0.0,
        label="test.block_sparse.from_dense",
        coo_ctor=_block_coo_ctor,
    )
    transposed = bsc.transpose(
        coo,
        as_coo=_as_coo,
        csr_to_coo_fn=_csr_to_coo,
        algebra="srb",
        label="test.block_sparse.transpose",
        coo_ctor=_block_coo_ctor,
    )

    assert jnp.allclose(_to_dense(transposed), dense.T)
    assert jnp.allclose(bsc.diag(coo, to_dense_fn=_to_dense), jnp.diag(dense))

    batch = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    assert bsc.batch_fixed(batch, dtype=jnp.float64, label="test.block_sparse.batch_fixed", apply=lambda x: 2.0 * x).shape == batch.shape
    assert bsc.batch_padded(batch, dtype=jnp.float64, pad_to=4, label="test.block_sparse.batch_padded", apply=lambda x: 2.0 * x).shape == (4, 2)
