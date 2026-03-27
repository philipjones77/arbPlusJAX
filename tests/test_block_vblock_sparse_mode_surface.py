import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import mat_wrappers
from arbplusjax import scb_block_mat
from arbplusjax import scb_vblock_mat
from arbplusjax import srb_block_mat
from arbplusjax import srb_vblock_mat

from tests._test_checks import _check


def test_real_block_sparse_mode_surface_matches_dense_reference() -> None:
    dense = jnp.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 1.0],
            [0.0, 0.0, 2.0, 6.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float64)
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    block = srb_block_mat.srb_block_mat_from_dense_csr(dense, block_shape=(2, 2))

    _check(bool(jnp.allclose(mat_wrappers.srb_block_mat_to_dense_mode(block, impl="point"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.srb_block_mat_to_dense_mode(block, impl="basic"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.srb_block_mat_matvec_mode(block, rhs, impl="point"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.srb_block_mat_matvec_mode(block, rhs, impl="basic"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.srb_block_mat_det_mode(block, impl="basic"), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))

    cache = mat_wrappers.srb_block_mat_matvec_cached_prepare_mode(block, impl="adaptive")
    cached = mat_wrappers.srb_block_mat_matvec_cached_apply_mode(cache, rhs, impl="rigorous")
    batch = mat_wrappers.srb_block_mat_matvec_batch_mode_padded(block, rhs_batch, pad_to=4, impl="adaptive")
    solve = mat_wrappers.srb_block_mat_solve_mode(block, rhs, impl="basic", method="lu")

    _check(bool(jnp.allclose(cached, dense @ rhs)))
    _check(batch.shape == (4, 4))
    _check(bool(jnp.allclose(batch[:2], rhs_batch @ dense.T)))
    _check(bool(jnp.allclose(solve, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))


def test_complex_block_sparse_mode_surface_matches_dense_reference() -> None:
    dense = jnp.array(
        [
            [4.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [2.0 + 0.25j, 3.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 5.0 + 0.0j, 1.0 + 0.25j],
            [0.0 + 0.0j, 0.0 + 0.0j, 2.0 - 0.25j, 6.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.array([1.0 + 0.5j, -1.0 + 0.0j, 0.5 - 0.25j, 2.0 + 0.0j], dtype=jnp.complex128)
    rhs_batch = jnp.stack([rhs, rhs + (0.5 - 0.25j)], axis=0)
    block = scb_block_mat.scb_block_mat_from_dense_csr(dense, block_shape=(2, 2))

    _check(bool(jnp.allclose(mat_wrappers.scb_block_mat_to_dense_mode(block, impl="point"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.scb_block_mat_to_dense_mode(block, impl="basic"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.scb_block_mat_matvec_mode(block, rhs, impl="point"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.scb_block_mat_matvec_mode(block, rhs, impl="basic"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.scb_block_mat_det_mode(block, impl="basic"), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))

    cache = mat_wrappers.scb_block_mat_matvec_cached_prepare_mode(block, impl="adaptive")
    adj_cache = mat_wrappers.scb_block_mat_adjoint_matvec_cached_prepare_mode(block, impl="adaptive")
    cached = mat_wrappers.scb_block_mat_matvec_cached_apply_mode(cache, rhs, impl="rigorous")
    adj_cached = mat_wrappers.scb_block_mat_adjoint_matvec_cached_apply_mode(adj_cache, rhs, impl="basic")
    batch = mat_wrappers.scb_block_mat_matvec_batch_mode_padded(block, rhs_batch, pad_to=4, impl="adaptive")
    solve = mat_wrappers.scb_block_mat_solve_mode(block, rhs, impl="basic", method="lu")

    _check(bool(jnp.allclose(cached, dense @ rhs)))
    _check(bool(jnp.allclose(adj_cached, jnp.conjugate(dense).T @ rhs)))
    _check(batch.shape == (4, 4))
    _check(bool(jnp.allclose(batch[:2], rhs_batch @ dense.T)))
    _check(bool(jnp.allclose(solve, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))


def test_real_vblock_sparse_mode_surface_matches_dense_reference() -> None:
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
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    vblock = srb_vblock_mat.srb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)

    _check(bool(jnp.allclose(mat_wrappers.srb_vblock_mat_to_dense_mode(vblock, impl="point"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.srb_vblock_mat_to_dense_mode(vblock, impl="basic"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.srb_vblock_mat_matvec_mode(vblock, rhs, impl="point"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.srb_vblock_mat_matvec_mode(vblock, rhs, impl="basic"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.srb_vblock_mat_det_mode(vblock, impl="basic"), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))

    cache = mat_wrappers.srb_vblock_mat_matvec_cached_prepare_mode(vblock, impl="adaptive")
    cached = mat_wrappers.srb_vblock_mat_matvec_cached_apply_mode(cache, rhs, impl="rigorous")
    batch = mat_wrappers.srb_vblock_mat_matvec_batch_mode_padded(vblock, rhs_batch, pad_to=4, impl="adaptive")
    solve = mat_wrappers.srb_vblock_mat_solve_mode(vblock, rhs, impl="basic", method="lu")

    _check(bool(jnp.allclose(cached, dense @ rhs)))
    _check(batch.shape == (4, 4))
    _check(bool(jnp.allclose(batch[:2], rhs_batch @ dense.T)))
    _check(bool(jnp.allclose(solve, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))


def test_complex_vblock_sparse_mode_surface_matches_dense_reference() -> None:
    dense = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.25j, 3.0 + 0.0j, 4.0 + 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 5.0 - 0.25j, 6.0 + 0.0j, 7.0 + 0.25j],
            [0.0 + 0.0j, 0.0 + 0.0j, 8.0 - 0.5j, 9.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    row_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    col_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    rhs = jnp.array([1.0 + 0.5j, 2.0 - 0.25j, 3.0 + 0.0j, 4.0 - 0.5j], dtype=jnp.complex128)
    rhs_batch = jnp.stack([rhs, rhs + (0.5 - 0.25j)], axis=0)
    vblock = scb_vblock_mat.scb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)

    _check(bool(jnp.allclose(mat_wrappers.scb_vblock_mat_to_dense_mode(vblock, impl="point"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.scb_vblock_mat_to_dense_mode(vblock, impl="basic"), dense)))
    _check(bool(jnp.allclose(mat_wrappers.scb_vblock_mat_matvec_mode(vblock, rhs, impl="point"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.scb_vblock_mat_matvec_mode(vblock, rhs, impl="basic"), dense @ rhs)))
    _check(bool(jnp.allclose(mat_wrappers.scb_vblock_mat_det_mode(vblock, impl="basic"), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))

    cache = mat_wrappers.scb_vblock_mat_matvec_cached_prepare_mode(vblock, impl="adaptive")
    adj_cache = mat_wrappers.scb_vblock_mat_adjoint_matvec_cached_prepare_mode(vblock, impl="adaptive")
    cached = mat_wrappers.scb_vblock_mat_matvec_cached_apply_mode(cache, rhs, impl="rigorous")
    adj_cached = mat_wrappers.scb_vblock_mat_adjoint_matvec_cached_apply_mode(adj_cache, rhs, impl="basic")
    batch = mat_wrappers.scb_vblock_mat_matvec_batch_mode_padded(vblock, rhs_batch, pad_to=4, impl="adaptive")
    solve = mat_wrappers.scb_vblock_mat_solve_mode(vblock, rhs, impl="basic", method="lu")

    _check(bool(jnp.allclose(cached, dense @ rhs)))
    _check(bool(jnp.allclose(adj_cached, jnp.conjugate(dense).T @ rhs)))
    _check(batch.shape == (4, 4))
    _check(bool(jnp.allclose(batch[:2], rhs_batch @ dense.T)))
    _check(bool(jnp.allclose(solve, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
