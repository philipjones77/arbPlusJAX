import jax.numpy as jnp

from arbplusjax import api

from tests._test_checks import _check


def test_variable_block_sparse_point_api():
    dense = jnp.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 4.0, 0.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 8.0, 9.0],
        ],
        dtype=jnp.float64,
    )
    row_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    col_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    rhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    x = api.eval_point("srb_vblock_mat_from_dense_csr", dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)
    lu = api.eval_point("srb_vblock_mat_lu", x)
    plan = api.eval_point("srb_vblock_mat_matvec_cached_prepare", x)
    vecs = jnp.stack([rhs, rhs + 1.0], axis=0)

    _check(bool(jnp.allclose(api.eval_point("srb_vblock_mat_to_dense", x), dense)))
    _check(bool(jnp.allclose(api.eval_point("srb_vblock_mat_matvec_cached_apply", plan, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(api.eval_point_batch("srb_vblock_mat_matvec", x, vecs), vecs @ dense.T)))
    _check(bool(jnp.allclose(api.eval_point_batch("srb_vblock_mat_matvec_cached_apply", plan, vecs), vecs @ dense.T)))
    _check(bool(jnp.allclose(api.eval_point("srb_vblock_mat_lu_solve", lu, rhs), jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
