import jax.numpy as jnp

from arbplusjax import api

from tests._test_checks import _check


def test_block_sparse_point_api():
    real_dense = jnp.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 10.0],
        ],
        dtype=jnp.float64,
    )
    complex_dense = real_dense.astype(jnp.complex128) + 1j * jnp.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.25, 0.0],
            [0.0, 0.0, -0.25, 0.0],
        ],
        dtype=jnp.float64,
    )
    r = api.eval_point("srb_block_mat_from_dense_csr", real_dense, block_shape=(2, 2))
    c = api.eval_point("scb_block_mat_from_dense_csr", complex_dense, block_shape=(2, 2))
    rv = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float64)
    cv = rv.astype(jnp.complex128)
    rplan = api.eval_point("srb_block_mat_matvec_cached_prepare", r)
    cplan = api.eval_point("scb_block_mat_matvec_cached_prepare", c)

    _check(bool(jnp.allclose(api.eval_point("srb_block_mat_to_dense", r), real_dense)))
    _check(bool(jnp.allclose(api.eval_point("scb_block_mat_to_dense", c), complex_dense)))
    _check(bool(jnp.allclose(api.eval_point("srb_block_mat_matvec", r, rv), real_dense @ rv)))
    _check(bool(jnp.allclose(api.eval_point("scb_block_mat_matvec", c, cv), complex_dense @ cv)))
    _check(bool(jnp.allclose(api.eval_point("srb_block_mat_matvec_cached_apply", rplan, rv), real_dense @ rv)))
    _check(bool(jnp.allclose(api.eval_point("scb_block_mat_matvec_cached_apply", cplan, cv), complex_dense @ cv)))
