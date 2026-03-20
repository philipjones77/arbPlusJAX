import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import mat_wrappers
from arbplusjax import scb_mat
from arbplusjax import srb_mat

from tests._test_checks import _check


def _real_formats(dense):
    return {
        "coo": srb_mat.srb_mat_from_dense_coo(dense),
        "csr": srb_mat.srb_mat_from_dense_csr(dense),
        "bcoo": srb_mat.srb_mat_from_dense_bcoo(dense),
    }


def _complex_formats(dense):
    return {
        "coo": scb_mat.scb_mat_from_dense_coo(dense),
        "csr": scb_mat.scb_mat_from_dense_csr(dense),
        "bcoo": scb_mat.scb_mat_from_dense_bcoo(dense),
    }


def test_srb_all_formats_point_and_basic_modes():
    dense = jnp.array(
        [
            [6.0, 1.0, 0.0],
            [1.0, 5.0, 0.5],
            [0.0, 0.5, 4.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    rhs_cols = jnp.stack([rhs, rhs + 1.0], axis=1)
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    expected = jnp.linalg.solve(dense, rhs)
    expected_cols = jnp.linalg.solve(dense, rhs_cols)
    expected_square = dense @ dense

    for storage, sparse in _real_formats(dense).items():
        _check(bool(jnp.allclose(mat_wrappers.srb_mat_to_dense_mode(sparse, impl="point"), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_to_dense_mode(sparse, impl="basic")), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_transpose_mode(sparse, impl="basic")), dense.T, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_symmetric_part_mode(sparse, impl="basic")), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(mat_wrappers.srb_mat_is_symmetric_mode(sparse, impl="point")))
        _check(bool(mat_wrappers.srb_mat_is_symmetric_mode(sparse, impl="basic")))
        _check(bool(mat_wrappers.srb_mat_is_spd_mode(sparse, impl="point")))
        _check(bool(mat_wrappers.srb_mat_is_spd_mode(sparse, impl="basic")))
        _check(bool(jnp.allclose(mat_wrappers.srb_mat_trace_mode(sparse, impl="point"), jnp.trace(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_trace_mode(sparse, impl="basic")), jnp.trace(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(mat_wrappers.srb_mat_det_mode(sparse, impl="point"), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_det_mode(sparse, impl="basic")), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_norm_fro_mode(sparse, impl="basic")), jnp.linalg.norm(dense, ord="fro"), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_norm_1_mode(sparse, impl="basic")), jnp.linalg.norm(dense, ord=1), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_norm_inf_mode(sparse, impl="basic")), jnp.linalg.norm(dense, ord=jnp.inf), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(mat_wrappers.srb_mat_matvec_mode(sparse, rhs, impl="point"), dense @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_matvec_mode(sparse, rhs, impl="basic")), dense @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(mat_wrappers.srb_mat_rmatvec_mode(sparse, rhs, impl="point"), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_rmatvec_mode(sparse, rhs, impl="basic")), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_matmul_dense_rhs_mode(sparse, rhs_cols, impl="basic")), dense @ rhs_cols, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_sqr_mode(sparse, impl="basic")), expected_square, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_inv_mode(sparse, impl="basic")), jnp.linalg.inv(dense), rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_charpoly_mode(sparse, impl="basic")), jnp.poly(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_pow_ui_mode(sparse, 2, impl="basic")), expected_square, rtol=1e-8, atol=1e-8)))
        eigvals_point, eigvecs_point = mat_wrappers.srb_mat_eigh_mode(sparse, impl="point")
        eigvals_basic, eigvecs_basic = mat_wrappers.srb_mat_eigh_mode(sparse, impl="basic")
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_exp_mode(sparse, impl="basic")), mat_wrappers.srb_mat_exp_mode(sparse, impl="point"), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_eigvalsh_mode(sparse, impl="basic")), mat_wrappers.srb_mat_eigvalsh_mode(sparse, impl="point"), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(eigvals_basic), eigvals_point, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(eigvecs_basic), eigvecs_point, rtol=1e-8, atol=1e-8)))

        cache_point = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        cache_basic = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        rcache_point = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="point")
        rcache_basic = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")
        spd_plan = mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(sparse, impl="basic")
        lu_plan = mat_wrappers.srb_mat_lu_solve_plan_prepare_mode(sparse, impl="basic")
        chol = mat_wrappers.srb_mat_cho_mode(sparse, impl="basic")
        ldl_l, ldl_d = mat_wrappers.srb_mat_ldl_mode(sparse, impl="basic")

        _check(getattr(cache_point, "storage", None) == storage)
        _check(getattr(cache_basic, "storage", None) == storage)
        _check(getattr(rcache_point, "storage", None) == storage)
        _check(getattr(rcache_basic, "storage", None) == storage)
        _check(bool(jnp.allclose(di.midpoint(chol) @ di.midpoint(chol).T, dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(ldl_l) @ jnp.diag(di.midpoint(ldl_d)) @ di.midpoint(ldl_l).T, dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_matvec_cached_apply_mode(cache_basic, rhs, impl="basic")), dense @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(mat_wrappers.srb_mat_rmatvec_cached_apply_mode(rcache_point, rhs, impl="point"), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_rmatvec_cached_apply_mode(rcache_basic, rhs, impl="basic")), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_spd_solve_mode(sparse, rhs, impl="basic")), expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_spd_solve_plan_apply_mode(spd_plan, rhs, impl="basic")), expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_solve_lu_mode(lu_plan, rhs, impl="basic")), expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_solve_transpose_mode(lu_plan, rhs, impl="basic")), jnp.linalg.solve(dense.T, rhs), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_solve_add_mode(lu_plan, rhs, rhs, impl="basic")), rhs + expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(mat_wrappers.srb_mat_mat_solve_mode(spd_plan, rhs_cols, impl="basic")), expected_cols, rtol=1e-8, atol=1e-8)))
        _check(mat_wrappers.srb_mat_matvec_batch_mode_padded(sparse, rhs_batch, pad_to=4, impl="point").shape == (4, 3))

        api_det = api.eval_interval("srb_mat_det", sparse, mode="basic", prec_bits=80)
        api_solve = api.eval_interval("srb_mat_spd_solve", sparse, rhs, mode="basic", prec_bits=80)
        _check(bool(jnp.allclose(di.midpoint(api_det), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(api_solve), expected, rtol=1e-8, atol=1e-8)))


def test_scb_all_formats_point_and_basic_modes():
    base = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.5 + 0.0j, 0.25 + 0.1j],
            [0.5 - 0.2j, 0.0 + 0.0j, 1.75 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    dense = jnp.conj(base.T) @ base + jnp.eye(3, dtype=jnp.complex128) * 2.0
    rhs = jnp.array([1.0 + 0.5j, 2.0 - 0.5j, 3.0 + 0.25j], dtype=jnp.complex128)
    rhs_cols = jnp.stack([rhs, rhs + (0.5 - 0.25j)], axis=1)
    rhs_batch = jnp.stack([rhs, rhs + (0.5 - 0.25j)], axis=0)
    expected = jnp.linalg.solve(dense, rhs)
    expected_cols = jnp.linalg.solve(dense, rhs_cols)
    expected_square = dense @ dense

    for storage, sparse in _complex_formats(dense).items():
        _check(bool(jnp.allclose(mat_wrappers.scb_mat_to_dense_mode(sparse, impl="point"), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_to_dense_mode(sparse, impl="basic")), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_transpose_mode(sparse, impl="basic")), dense.T, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_conjugate_transpose_mode(sparse, impl="basic")), jnp.conj(dense.T), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_hermitian_part_mode(sparse, impl="basic")), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(mat_wrappers.scb_mat_is_hermitian_mode(sparse, impl="point")))
        _check(bool(mat_wrappers.scb_mat_is_hpd_mode(sparse, impl="basic")))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_trace_mode(sparse, impl="basic")), jnp.trace(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_det_mode(sparse, impl="basic")), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_norm_fro_mode(sparse, impl="basic")), jnp.linalg.norm(dense, ord="fro"), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_norm_1_mode(sparse, impl="basic")), jnp.linalg.norm(dense, ord=1), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_norm_inf_mode(sparse, impl="basic")), jnp.linalg.norm(dense, ord=jnp.inf), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_matvec_mode(sparse, rhs, impl="basic")), dense @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(mat_wrappers.scb_mat_rmatvec_mode(sparse, rhs, impl="point"), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_rmatvec_mode(sparse, rhs, impl="basic")), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_matmul_dense_rhs_mode(sparse, rhs_cols, impl="basic")), dense @ rhs_cols, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_sqr_mode(sparse, impl="basic")), expected_square, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_inv_mode(sparse, impl="basic")), jnp.linalg.inv(dense), rtol=1e-6, atol=1e-6)))
        eigvals_point, eigvecs_point = mat_wrappers.scb_mat_eigh_mode(sparse, impl="point")
        eigvals_basic, eigvecs_basic = mat_wrappers.scb_mat_eigh_mode(sparse, impl="basic")
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_charpoly_mode(sparse, impl="basic")), jnp.poly(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_pow_ui_mode(sparse, 2, impl="basic")), expected_square, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_exp_mode(sparse, impl="basic")), mat_wrappers.scb_mat_exp_mode(sparse, impl="point"), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_eigvalsh_mode(sparse, impl="basic")), mat_wrappers.scb_mat_eigvalsh_mode(sparse, impl="point"), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(eigvals_basic), eigvals_point, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(eigvecs_basic), eigvecs_point, rtol=1e-8, atol=1e-8)))

        cache_point = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        cache_basic = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        rcache_point = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="point")
        rcache_basic = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")
        hpd_plan = mat_wrappers.scb_mat_hpd_solve_plan_prepare_mode(sparse, impl="basic")
        lu_plan = mat_wrappers.scb_mat_lu_solve_plan_prepare_mode(sparse, impl="basic")
        chol = mat_wrappers.scb_mat_cho_mode(sparse, impl="basic")
        ldl_l, ldl_d = mat_wrappers.scb_mat_ldl_mode(sparse, impl="basic")

        _check(getattr(cache_point, "storage", None) == storage)
        _check(getattr(cache_basic, "storage", None) == storage)
        _check(getattr(rcache_point, "storage", None) == storage)
        _check(getattr(rcache_basic, "storage", None) == storage)
        _check(bool(jnp.allclose(acb_core.acb_midpoint(chol) @ jnp.conj(acb_core.acb_midpoint(chol).T), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(ldl_l) @ jnp.diag(acb_core.acb_midpoint(ldl_d)) @ jnp.conj(acb_core.acb_midpoint(ldl_l).T), dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_matvec_cached_apply_mode(cache_basic, rhs, impl="basic")), dense @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(mat_wrappers.scb_mat_rmatvec_cached_apply_mode(rcache_point, rhs, impl="point"), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_rmatvec_cached_apply_mode(rcache_basic, rhs, impl="basic")), dense.T @ rhs, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_hpd_solve_mode(sparse, rhs, impl="basic")), expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_hpd_solve_plan_apply_mode(hpd_plan, rhs, impl="basic")), expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_solve_lu_mode(lu_plan, rhs, impl="basic")), expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_solve_transpose_mode(lu_plan, rhs, impl="basic")), jnp.linalg.solve(dense.T, rhs), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_solve_add_mode(lu_plan, rhs, rhs, impl="basic")), rhs + expected, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_wrappers.scb_mat_mat_solve_mode(hpd_plan, rhs_cols, impl="basic")), expected_cols, rtol=1e-8, atol=1e-8)))

        api_det = api.eval_interval("scb_mat_det", sparse, mode="basic", prec_bits=80)
        api_solve = api.eval_interval("scb_mat_hpd_solve", sparse, rhs, mode="basic", prec_bits=80)
        _check(bool(jnp.allclose(acb_core.acb_midpoint(api_det), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(api_solve), expected, rtol=1e-8, atol=1e-8)))


def test_sparse_higher_function_jit_surface_and_exports():
    real_dense = jnp.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    complex_base = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.25j],
            [0.5 + 0.25j, 1.75 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    complex_dense = jnp.conj(complex_base.T) @ complex_base + jnp.eye(2, dtype=jnp.complex128)

    sreal = srb_mat.srb_mat_from_dense_csr(real_dense)
    scomplex = scb_mat.scb_mat_from_dense_csr(complex_dense)

    _check("srb_mat_charpoly" in srb_mat.__all__)
    _check("srb_mat_exp_jit" in srb_mat.__all__)
    _check("srb_mat_rmatvec_cached_apply_jit" in srb_mat.__all__)
    _check("scb_mat_charpoly" in scb_mat.__all__)
    _check("scb_mat_exp_jit" in scb_mat.__all__)
    _check("scb_mat_rmatvec_cached_apply_jit" in scb_mat.__all__)

    _check(bool(jnp.allclose(srb_mat.srb_mat_charpoly_jit(sreal), srb_mat.srb_mat_charpoly(sreal), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_pow_ui_jit(sreal, 2), srb_mat.srb_mat_pow_ui(sreal, 2), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_exp_jit(sreal), srb_mat.srb_mat_exp(sreal), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_eigvalsh_jit(sreal), srb_mat.srb_mat_eigvalsh(sreal), rtol=1e-8, atol=1e-8)))
    evals_r, evecs_r = srb_mat.srb_mat_eigh_jit(sreal)
    evals_r_ref, evecs_r_ref = srb_mat.srb_mat_eigh(sreal)
    _check(bool(jnp.allclose(evals_r, evals_r_ref, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(jnp.abs(evecs_r), jnp.abs(evecs_r_ref), rtol=1e-8, atol=1e-8)))

    _check(bool(jnp.allclose(scb_mat.scb_mat_charpoly_jit(scomplex), scb_mat.scb_mat_charpoly(scomplex), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_pow_ui_jit(scomplex, 2), scb_mat.scb_mat_pow_ui(scomplex, 2), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_exp_jit(scomplex), scb_mat.scb_mat_exp(scomplex), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_eigvalsh_jit(scomplex), scb_mat.scb_mat_eigvalsh(scomplex), rtol=1e-8, atol=1e-8)))
    evals_c, evecs_c = scb_mat.scb_mat_eigh_jit(scomplex)
    evals_c_ref, evecs_c_ref = scb_mat.scb_mat_eigh(scomplex)
    _check(bool(jnp.allclose(evals_c, evals_c_ref, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(jnp.abs(evecs_c), jnp.abs(evecs_c_ref), rtol=1e-8, atol=1e-8)))
