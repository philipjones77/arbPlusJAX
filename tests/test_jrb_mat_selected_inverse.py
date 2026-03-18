import jax
import jax.numpy as jnp

from arbplusjax import jrb_mat
from arbplusjax import sparse_common

from tests._test_checks import _check


def test_selected_inverse_sparse_diagonal_exactness_contract():
    diag = jnp.asarray([1.0, 2.0, 4.0, 8.0], dtype=jnp.float64)
    bcoo = sparse_common.dense_to_sparse_bcoo(jnp.diag(diag), algebra="jrb")
    est, diag_info = jrb_mat.jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(
        bcoo,
        overlap=0,
        block_size=2,
        correction_probes=0,
    )
    exact = 1.0 / diag
    _check(bool(jnp.allclose(est, exact, rtol=1e-12, atol=1e-12)))
    _check(int(diag_info.algorithm_code) == 5)
    _check(int(diag_info.partition_count) == 2)
    _check(int(diag_info.max_local_size) == 2)
    _check(not bool(diag_info.correction_used))


def test_selected_inverse_full_overlap_matches_exact_inverse_diagonal():
    dense = jnp.asarray(
        [
            [4.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    bcoo = sparse_common.dense_to_sparse_bcoo(dense, algebra="jrb")
    est, diag_info = jrb_mat.jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(
        bcoo,
        overlap=8,
        block_size=2,
        correction_probes=0,
    )
    exact = jnp.diag(jnp.linalg.inv(dense))
    _check(bool(jnp.allclose(est, exact, rtol=1e-10, atol=1e-10)))
    _check(int(diag_info.max_local_size) == dense.shape[0])


def test_selected_inverse_stochastic_correction_improves_local_estimate():
    dense = jnp.asarray(
        [
            [2.5, -0.9, 0.0, 0.0, 0.0, 0.0],
            [-0.9, 2.5, -0.9, 0.0, 0.0, 0.0],
            [0.0, -0.9, 2.5, -0.9, 0.0, 0.0],
            [0.0, 0.0, -0.9, 2.5, -0.9, 0.0],
            [0.0, 0.0, 0.0, -0.9, 2.5, -0.9],
            [0.0, 0.0, 0.0, 0.0, -0.9, 2.5],
        ],
        dtype=jnp.float64,
    )
    bcoo = sparse_common.dense_to_sparse_bcoo(dense, algebra="jrb")
    local_only = jrb_mat.jrb_mat_bcoo_inverse_diagonal_point(
        bcoo,
        overlap=0,
        block_size=2,
        correction_probes=0,
    )
    corrected, diag_info = jrb_mat.jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(
        bcoo,
        overlap=0,
        block_size=2,
        correction_probes=128,
        key=jax.random.PRNGKey(0),
        tol=1e-7,
        maxiter=32,
    )
    exact = jnp.diag(jnp.linalg.inv(dense))
    local_err = jnp.linalg.norm(local_only - exact)
    corrected_err = jnp.linalg.norm(corrected - exact)
    _check(bool(corrected_err < local_err))
    _check(bool(diag_info.correction_used))
    _check(int(diag_info.correction_probe_count) == 128)
    _check(int(diag_info.converged_probe_count) > 0)
    _check(bool(jnp.isfinite(diag_info.final_residual_max)))
