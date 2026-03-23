import jax.numpy as jnp

from arbplusjax import sparse_common as sc
from arbplusjax import sparse_core as spc


def _coo(data, row, col, shape, algebra="srb"):
    return sc.SparseCOO(
        data=jnp.asarray(data),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra=algebra,
    )


def _csr(data, indices, indptr, shape, algebra="srb"):
    return sc.SparseCSR(
        data=jnp.asarray(data),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra=algebra,
    )


def _to_dense(x):
    if isinstance(x, sc.SparseCSR):
        x = spc.csr_to_coo(x)
    if isinstance(x, sc.SparseBCOO):
        dense = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
        return dense.at[x.indices[:, 0], x.indices[:, 1]].add(x.data)
    dense = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    return dense.at[x.row, x.col].add(x.data)


def _to_coo(x):
    return x if isinstance(x, sc.SparseCOO) else spc.csr_to_coo(x)


def _from_coo(data, row, col, *, shape):
    return _coo(data, row, col, shape)


def _identity_sparse_fn(n, *, dtype):
    return sc.dense_to_sparse_bcoo(jnp.eye(n, dtype=dtype), algebra="srb")


def _matmul_sparse_fn(x, y):
    return sc.dense_to_sparse_bcoo(_to_dense(x) @ _to_dense(y), algebra="srb")


def _trace_fn(x):
    return jnp.trace(_to_dense(x))


def test_sparse_core_swap_and_dense_lu_paths_match_reference_behaviour():
    dense = jnp.asarray([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    perm_expected, l_expected, u_expected = spc.dense_sparse_lu_partial_pivot(dense)
    perm_lax, l_lax, u_lax = spc.dense_sparse_lu_partial_pivot_lax(dense)

    assert jnp.array_equal(spc.swap_rows_dense(dense, 0, 1), spc.swap_rows_dense_lax(dense, 0, 1))
    assert jnp.array_equal(spc.swap_perm(jnp.asarray([0, 1], dtype=jnp.int32), 0, 1), spc.swap_perm_lax(jnp.asarray([0, 1], dtype=jnp.int32), 0, 1))
    assert jnp.array_equal(perm_expected, perm_lax)
    assert jnp.allclose(l_expected, l_lax)
    assert jnp.allclose(u_expected, u_lax)


def test_sparse_core_coalesce_diag_norm_and_structured_helpers():
    x = _coo(
        data=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        row=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        col=jnp.asarray([0, 0, 1, 0], dtype=jnp.int32),
        shape=(2, 2),
    )
    data_c, row_c, col_c = spc.coalesce_coo(x.data, x.row, x.col, rows=2, cols=2)
    coalesced = _coo(data_c, row_c, col_c, (2, 2))

    assert coalesced.data.shape[0] == 3
    assert jnp.allclose(spc.sparse_diag(coalesced, to_coo_fn=_to_coo, dtype=jnp.float64), jnp.asarray([3.0, 3.0], dtype=jnp.float64))
    assert jnp.isclose(spc.sparse_norm_1(coalesced, to_coo_fn=_to_coo), 7.0)
    assert jnp.isclose(spc.sparse_norm_inf(coalesced, to_coo_fn=_to_coo), 7.0)

    sym = spc.sparse_structured_part_real(coalesced, to_coo_fn=_to_coo, from_coo_fn=_from_coo)
    assert bool(spc.sparse_is_symmetric_structural(sym, to_coo_fn=_to_coo, rtol=1e-12, atol=1e-12))


def test_sparse_core_structural_pd_and_factor_helpers_match_dense_expectations():
    sym = _coo(
        data=jnp.asarray([4.0, 1.0, 1.0, 3.0], dtype=jnp.float64),
        row=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        col=jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        shape=(2, 2),
    )

    structured_part = lambda x: spc.sparse_structured_part_real(x, to_coo_fn=_to_coo, from_coo_fn=_from_coo)
    is_structured = lambda x: spc.sparse_is_symmetric_structural(x, to_coo_fn=_to_coo, rtol=1e-12, atol=1e-12)

    assert bool(spc.sparse_is_pd_structural(sym, to_coo_fn=_to_coo, structured_part_fn=structured_part, is_structured_fn=is_structured, dtype=jnp.float64, hermitian=False))
    cho_data, cho_row, cho_col = spc.sparse_cho_structural(sym, to_coo_fn=_to_coo, structured_part_fn=structured_part, is_structured_fn=is_structured, dtype=jnp.float64, hermitian=False)
    assert cho_data.shape[0] == cho_row.shape[0] == cho_col.shape[0]


def test_sparse_core_power_charpoly_and_exp_helpers_return_dense_reference_outputs():
    dense = jnp.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=jnp.float64)
    base = sc.dense_to_sparse_bcoo(dense, algebra="srb")

    squared = spc.sparse_dense_power_ui(base, 2, to_bcoo_fn=lambda x, label: x, matmul_sparse_fn=_matmul_sparse_fn, to_dense_fn=_to_dense, identity_sparse_fn=_identity_sparse_fn)
    exp_dense = spc.sparse_dense_exp_taylor(base, to_bcoo_fn=lambda x, label: x, matmul_sparse_fn=_matmul_sparse_fn, to_dense_fn=_to_dense, identity_sparse_fn=_identity_sparse_fn, terms=12)
    coeffs = spc.sparse_charpoly_from_traces(base, to_bcoo_fn=lambda x, label: x, matmul_sparse_fn=_matmul_sparse_fn, trace_fn=_trace_fn, identity_sparse_fn=_identity_sparse_fn)

    assert jnp.allclose(squared, dense @ dense)
    assert jnp.all(jnp.isfinite(exp_dense))
    assert coeffs.shape == (3,)
