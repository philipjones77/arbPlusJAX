import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import mat_common
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
        _check(di.midpoint(eigvecs_basic).shape == eigvecs_point.shape)

        cache_point = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        cache_basic = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        rcache_point = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="point")
        rcache_basic = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")
        spd_plan = mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(sparse, impl="basic")
        lu_plan = mat_wrappers.srb_mat_lu_solve_plan_prepare_mode(sparse, impl="basic")
        _check(isinstance(spd_plan, mat_common.DenseCholeskySolvePlan))
        _check(isinstance(lu_plan, mat_common.DenseLUSolvePlan))
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
        _check(acb_core.acb_midpoint(eigvecs_basic).shape == eigvecs_point.shape)

        cache_point = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        cache_basic = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        rcache_point = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="point")
        rcache_basic = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")
        hpd_plan = mat_wrappers.scb_mat_hpd_solve_plan_prepare_mode(sparse, impl="basic")
        lu_plan = mat_wrappers.scb_mat_lu_solve_plan_prepare_mode(sparse, impl="basic")
        _check(isinstance(hpd_plan, mat_common.DenseCholeskySolvePlan))
        _check(isinstance(lu_plan, mat_common.DenseLUSolvePlan))
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
    _check(evecs_r.shape == evecs_r_ref.shape)

    _check(bool(jnp.allclose(scb_mat.scb_mat_charpoly_jit(scomplex), scb_mat.scb_mat_charpoly(scomplex), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_pow_ui_jit(scomplex, 2), scb_mat.scb_mat_pow_ui(scomplex, 2), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_exp_jit(scomplex), scb_mat.scb_mat_exp(scomplex), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_eigvalsh_jit(scomplex), scb_mat.scb_mat_eigvalsh(scomplex), rtol=1e-8, atol=1e-8)))
    evals_c, evecs_c = scb_mat.scb_mat_eigh_jit(scomplex)
    evals_c_ref, evecs_c_ref = scb_mat.scb_mat_eigh(scomplex)
    _check(bool(jnp.allclose(evals_c, evals_c_ref, rtol=1e-8, atol=1e-8)))
    _check(evecs_c.shape == evecs_c_ref.shape)


def test_sparse_operator_plan_specialization_uses_native_storage_for_coo_and_csr():
    real_dense = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    complex_base = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.25j],
            [0.5 + 0.25j, 1.75 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    complex_dense = jnp.conj(complex_base.T) @ complex_base + jnp.eye(2, dtype=jnp.complex128)
    rv = jnp.array([1.0, -2.0], dtype=jnp.float64)
    cv = jnp.array([1.0 + 0.5j, -2.0 + 0.25j], dtype=jnp.complex128)

    for storage, sparse in _real_formats(real_dense).items():
        plan = srb_mat.srb_mat_operator_plan_prepare(sparse)
        sym_plan = jrb_mat.jrb_mat_symmetric_operator_plan_prepare(sparse)
        spd_plan = jrb_mat.jrb_mat_spd_operator_plan_prepare(sparse)
        _check(plan.kind == ("sparse_bcoo" if storage == "bcoo" else "shell"))
        _check(sym_plan.kind == plan.kind)
        _check(spd_plan.kind == plan.kind)
        out = jrb_mat.jrb_mat_operator_plan_apply(plan, di.interval(rv, rv))
        sym_out = jrb_mat.jrb_mat_operator_plan_apply(sym_plan, di.interval(rv, rv))
        spd_out = jrb_mat.jrb_mat_operator_plan_apply(spd_plan, di.interval(rv, rv))
        _check(bool(jnp.allclose(di.midpoint(out), real_dense @ rv, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(sym_out), real_dense @ rv, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(spd_out), real_dense @ rv, rtol=1e-8, atol=1e-8)))

    for storage, sparse in _complex_formats(complex_dense).items():
        plan = scb_mat.scb_mat_operator_plan_prepare(sparse)
        herm_plan = jcb_mat.jcb_mat_hermitian_operator_plan_prepare(sparse)
        hpd_plan = jcb_mat.jcb_mat_hpd_operator_plan_prepare(sparse)
        _check(plan.kind == ("sparse_bcoo" if storage == "bcoo" else "shell"))
        _check(herm_plan.kind == plan.kind)
        _check(hpd_plan.kind == plan.kind)
        out = jcb_mat.jcb_mat_operator_plan_apply(plan, acb_core.acb_box(di.interval(cv.real, cv.real), di.interval(cv.imag, cv.imag)))
        herm_out = jcb_mat.jcb_mat_operator_plan_apply(herm_plan, acb_core.acb_box(di.interval(cv.real, cv.real), di.interval(cv.imag, cv.imag)))
        hpd_out = jcb_mat.jcb_mat_operator_plan_apply(hpd_plan, acb_core.acb_box(di.interval(cv.real, cv.real), di.interval(cv.imag, cv.imag)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(out), complex_dense @ cv, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(herm_out), complex_dense @ cv, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(hpd_out), complex_dense @ cv, rtol=1e-8, atol=1e-8)))


def test_sparse_preconditioner_and_multi_shift_plan_specialization():
    real_dense = jnp.diag(jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64))
    complex_dense = jnp.diag(jnp.asarray([2.0 + 0.0j, 4.0 + 0.0j, 8.0 + 0.0j], dtype=jnp.complex128))
    rhs_r = di.interval(jnp.asarray([2.0, 8.0, 16.0], dtype=jnp.float64), jnp.asarray([2.0, 8.0, 16.0], dtype=jnp.float64))
    rhs_c = acb_core.acb_box(
        di.interval(jnp.asarray([2.0, 8.0, 16.0], dtype=jnp.float64), jnp.asarray([2.0, 8.0, 16.0], dtype=jnp.float64)),
        di.interval(jnp.zeros((3,), dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64)),
    )
    shifts = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    expected_r = jnp.stack(
        [
            jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.float64),
            jnp.asarray([2.0 / 3.0, 8.0 / 5.0, 16.0 / 9.0], dtype=jnp.float64),
        ],
        axis=0,
    )
    expected_c = expected_r.astype(jnp.complex128)

    for storage, sparse in _real_formats(real_dense).items():
        op_plan = srb_mat.srb_mat_operator_plan_prepare(sparse)
        prec = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(op_plan)
        _check(prec.kind == "diagonal")
        prec_applied = jrb_mat.matrix_free_core.preconditioner_plan_apply(
            prec,
            rhs_r,
            midpoint_vector=di.midpoint,
            sparse_bcoo_matvec=jrb_mat.sparse_common.sparse_bcoo_matvec,
            dtype=jnp.float64,
        )
        _check(bool(jnp.allclose(prec_applied, jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.float64), rtol=1e-8, atol=1e-8)))
        shifted_plan = jrb_mat.jrb_mat_multi_shift_solve_plan_prepare(sparse, shifts, symmetric=True, preconditioner=prec)
        _check(shifted_plan.operator.kind == ("sparse_bcoo" if storage == "bcoo" else "shell"))
        shifted = jrb_mat.jrb_mat_multi_shift_solve_point(sparse, rhs_r, shifts, symmetric=True, preconditioner=prec, tol=1e-10)
        _check(bool(jnp.allclose(di.midpoint(shifted), expected_r, rtol=1e-6, atol=1e-6)))

    for storage, sparse in _complex_formats(complex_dense).items():
        op_plan = scb_mat.scb_mat_operator_plan_prepare(sparse)
        prec = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(op_plan)
        _check(prec.kind == "diagonal")
        prec_applied = jcb_mat.matrix_free_core.preconditioner_plan_apply(
            prec,
            rhs_c,
            midpoint_vector=acb_core.acb_midpoint,
            sparse_bcoo_matvec=jcb_mat.sparse_common.sparse_bcoo_matvec,
            dtype=jnp.complex128,
        )
        _check(bool(jnp.allclose(prec_applied, jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.complex128), rtol=1e-8, atol=1e-8)))
        shifted_plan = jcb_mat.jcb_mat_multi_shift_solve_plan_prepare(sparse, shifts, hermitian=True, preconditioner=prec)
        _check(shifted_plan.operator.kind == ("sparse_bcoo" if storage == "bcoo" else "shell"))
        shifted = jcb_mat.jcb_mat_multi_shift_solve_point(sparse, rhs_c, shifts, hermitian=True, preconditioner=prec, tol=1e-10)
        _check(bool(jnp.allclose(acb_core.acb_midpoint(shifted), expected_c, rtol=1e-6, atol=1e-6)))


def test_sparse_solve_preconditioner_specialization_for_repeated_gmres_and_minres():
    real_general = jnp.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    real_spd = jnp.diag(jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64))
    complex_general = jnp.array(
        [
            [4.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 3.0 + 0.0j, 1.0 + 0.25j],
            [0.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    complex_hpd = jnp.diag(jnp.asarray([2.0 + 0.0j, 4.0 + 0.0j, 8.0 + 0.0j], dtype=jnp.complex128))
    rhs_r = di.interval(
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
    )
    rhs_c = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64), jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)),
        di.interval(jnp.asarray([0.5, -0.25, 0.25], dtype=jnp.float64), jnp.asarray([0.5, -0.25, 0.25], dtype=jnp.float64)),
    )

    expected_rg = jnp.linalg.solve(real_general, di.midpoint(rhs_r))
    expected_rs = jnp.linalg.solve(real_spd, di.midpoint(rhs_r))
    expected_cg = jnp.linalg.solve(complex_general, acb_core.acb_midpoint(rhs_c))
    expected_ch = jnp.linalg.solve(complex_hpd, acb_core.acb_midpoint(rhs_c))

    for storage, sparse in _real_formats(real_general).items():
        op_plan = srb_mat.srb_mat_operator_plan_prepare(sparse)
        prec = jrb_mat.jrb_mat_lu_preconditioner_plan_prepare(op_plan)
        _check(prec.kind == "sparse_lu_solve")
        tprec = jrb_mat.matrix_free_core.preconditioner_transpose_plan(prec, algebra="jrb", conjugate=False)
        _check(tprec is not None and tprec.kind == "sparse_lu_solve")
        solved = jrb_mat.jrb_mat_solve_action_point(op_plan, rhs_r, symmetric=False, preconditioner=prec, tol=1e-10, maxiter=10)
        _check(bool(jnp.allclose(di.midpoint(solved), expected_rg, rtol=1e-8, atol=1e-8)))

    for storage, sparse in _real_formats(real_spd).items():
        op_plan = srb_mat.srb_mat_operator_plan_prepare(sparse)
        prec = jrb_mat.jrb_mat_structured_preconditioner_plan_prepare(op_plan, symmetric=True)
        _check(prec.kind == "sparse_cholesky_solve")
        tprec = jrb_mat.matrix_free_core.preconditioner_transpose_plan(prec, algebra="jrb", conjugate=False)
        _check(tprec is not None and tprec.kind == "sparse_cholesky_solve")
        solved = jrb_mat.jrb_mat_minres_solve_action_point(op_plan, rhs_r, preconditioner=prec, tol=1e-10, maxiter=10)
        _check(bool(jnp.allclose(di.midpoint(solved), expected_rs, rtol=1e-8, atol=1e-8)))

    for storage, sparse in _complex_formats(complex_general).items():
        op_plan = scb_mat.scb_mat_operator_plan_prepare(sparse)
        prec = jcb_mat.jcb_mat_lu_preconditioner_plan_prepare(op_plan)
        _check(prec.kind == "sparse_lu_solve")
        tprec = jcb_mat.matrix_free_core.preconditioner_transpose_plan(prec, algebra="jcb", conjugate=True)
        _check(tprec is not None and tprec.kind == "sparse_lu_solve")
        solved = jcb_mat.jcb_mat_solve_action_point(op_plan, rhs_c, hermitian=False, preconditioner=prec, tol=1e-10, maxiter=10)
        _check(bool(jnp.allclose(acb_core.acb_midpoint(solved), expected_cg, rtol=1e-8, atol=1e-8)))

    for storage, sparse in _complex_formats(complex_hpd).items():
        op_plan = scb_mat.scb_mat_operator_plan_prepare(sparse)
        prec = jcb_mat.jcb_mat_structured_preconditioner_plan_prepare(op_plan, hermitian=True)
        _check(prec.kind == "sparse_cholesky_solve")
        tprec = jcb_mat.matrix_free_core.preconditioner_transpose_plan(prec, algebra="jcb", conjugate=True)
        _check(tprec is not None and tprec.kind == "sparse_cholesky_solve")
        solved = jcb_mat.jcb_mat_minres_solve_action_point(op_plan, rhs_c, preconditioner=prec, tol=1e-10, maxiter=10)
        _check(bool(jnp.allclose(acb_core.acb_midpoint(solved), expected_ch, rtol=1e-8, atol=1e-8)))


def test_sparse_eigsh_uses_matrix_free_operator_surface_without_dense_bridge(monkeypatch):
    real_dense = jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 6.0], dtype=jnp.float64))
    complex_dense = jnp.diag(jnp.asarray([1.5, 3.0, 5.0, 7.5], dtype=jnp.float64)).astype(jnp.complex128)

    def _fail_real(*args, **kwargs):
        raise AssertionError("srb_mat eigsh should not rebuild a dense interval matrix")

    def _fail_complex(*args, **kwargs):
        raise AssertionError("scb_mat eigsh should not rebuild a dense box matrix")

    monkeypatch.setattr(srb_mat, "_dense_interval_matrix", _fail_real)
    monkeypatch.setattr(scb_mat, "_dense_box_matrix", _fail_complex)

    for sparse in _real_formats(real_dense).values():
        vals, vecs = srb_mat.srb_mat_eigsh(sparse, k=2, which="smallest", steps=4)
        vals_basic, vecs_basic = mat_wrappers.srb_mat_eigsh_mode(sparse, k=2, which="smallest", steps=4, impl="basic")
        residual = real_dense @ vecs - vecs * vals[None, :]
        _check(bool(jnp.allclose(vals, jnp.asarray([1.0, 2.0], dtype=jnp.float64), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(residual, jnp.zeros_like(residual), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(vals_basic), vals, rtol=1e-8, atol=1e-8)))
        _check(di.midpoint(vecs_basic).shape == vecs.shape)

    for sparse in _complex_formats(complex_dense).values():
        vals, vecs = scb_mat.scb_mat_eigsh(sparse, k=2, which="largest", steps=4)
        vals_basic, vecs_basic = mat_wrappers.scb_mat_eigsh_mode(sparse, k=2, which="largest", steps=4, impl="basic")
        residual = complex_dense @ vecs - vecs * vals[None, :]
        _check(bool(jnp.allclose(vals, jnp.asarray([5.0, 7.5], dtype=jnp.float64), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(residual, jnp.zeros_like(residual), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(vals_basic), vals, rtol=1e-8, atol=1e-8)))
        _check(acb_core.acb_midpoint(vecs_basic).shape == vecs.shape)


def test_sparse_basic_core_entrypoints_avoid_dense_bridge(monkeypatch):
    real_dense = jnp.array(
        [
            [5.0, 1.0, 0.0],
            [1.0, 4.0, 0.5],
            [0.0, 0.5, 3.0],
        ],
        dtype=jnp.float64,
    )
    complex_base = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.25j, 0.0 + 0.0j],
            [0.5 + 0.25j, 1.75 + 0.0j, 0.2 - 0.1j],
            [0.0 + 0.0j, 0.2 + 0.1j, 1.5 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    complex_dense = jnp.conj(complex_base.T) @ complex_base + jnp.eye(3, dtype=jnp.complex128)
    rhs_real = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    rhs_complex = jnp.array([1.0 + 0.0j, 2.0 - 0.5j, 3.0 + 0.25j], dtype=jnp.complex128)

    def _fail_real(*args, **kwargs):
        raise AssertionError("srb_mat basic sparse core entrypoints should not rebuild a dense interval matrix")

    def _fail_complex(*args, **kwargs):
        raise AssertionError("scb_mat basic sparse core entrypoints should not rebuild a dense box matrix")

    monkeypatch.setattr(srb_mat, "_dense_interval_matrix", _fail_real)
    monkeypatch.setattr(scb_mat, "_dense_box_matrix", _fail_complex)

    for sparse in _real_formats(real_dense).values():
        chol = mat_wrappers.srb_mat_cho_mode(sparse, impl="basic")
        ldl_l, ldl_d = mat_wrappers.srb_mat_ldl_mode(sparse, impl="basic")
        tri = srb_mat.srb_mat_triangular_solve_basic(srb_mat.srb_mat_cho(sparse), rhs_real, lower=True)
        solve = mat_wrappers.srb_mat_solve_mode(sparse, rhs_real, impl="basic")
        det = mat_wrappers.srb_mat_det_mode(sparse, impl="basic")
        inv = mat_wrappers.srb_mat_inv_mode(sparse, impl="basic")
        sqr = mat_wrappers.srb_mat_sqr_mode(sparse, impl="basic")

        _check(bool(jnp.allclose(di.midpoint(chol) @ di.midpoint(chol).T, real_dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(ldl_l) @ jnp.diag(di.midpoint(ldl_d)) @ di.midpoint(ldl_l).T, real_dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(tri), jnp.linalg.solve(jnp.linalg.cholesky(real_dense), rhs_real), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(solve), jnp.linalg.solve(real_dense, rhs_real), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(det), jnp.linalg.det(real_dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(di.midpoint(inv), jnp.linalg.inv(real_dense), rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(di.midpoint(sqr), real_dense @ real_dense, rtol=1e-8, atol=1e-8)))

    for sparse in _complex_formats(complex_dense).values():
        chol = mat_wrappers.scb_mat_cho_mode(sparse, impl="basic")
        ldl_l, ldl_d = mat_wrappers.scb_mat_ldl_mode(sparse, impl="basic")
        tri = scb_mat.scb_mat_triangular_solve_basic(scb_mat.scb_mat_cho(sparse), rhs_complex, lower=True)
        solve = mat_wrappers.scb_mat_solve_mode(sparse, rhs_complex, impl="basic")
        det = mat_wrappers.scb_mat_det_mode(sparse, impl="basic")
        inv = mat_wrappers.scb_mat_inv_mode(sparse, impl="basic")
        sqr = mat_wrappers.scb_mat_sqr_mode(sparse, impl="basic")

        _check(bool(jnp.allclose(acb_core.acb_midpoint(chol) @ jnp.conj(acb_core.acb_midpoint(chol).T), complex_dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(ldl_l) @ jnp.diag(acb_core.acb_midpoint(ldl_d)) @ jnp.conj(acb_core.acb_midpoint(ldl_l).T), complex_dense, rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(tri), jnp.linalg.solve(jnp.linalg.cholesky(complex_dense), rhs_complex), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(solve), jnp.linalg.solve(complex_dense, rhs_complex), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(det), jnp.linalg.det(complex_dense), rtol=1e-8, atol=1e-8)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(inv), jnp.linalg.inv(complex_dense), rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(acb_core.acb_midpoint(sqr), complex_dense @ complex_dense, rtol=1e-8, atol=1e-8)))


def test_sparse_native_policy_diagnostics_expose_dense_lift_vs_sparse_native_paths():
    real_dense = jnp.diag(jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64))
    complex_dense = jnp.diag(jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float64)).astype(jnp.complex128)

    sreal = srb_mat.srb_mat_from_dense_csr(real_dense)
    scomplex = scb_mat.scb_mat_from_dense_csr(complex_dense)

    _, pow_diag_r = srb_mat.srb_mat_pow_ui_with_diagnostics(sreal, 2)
    _, exp_diag_r = srb_mat.srb_mat_exp_with_diagnostics(sreal)
    _, eigval_diag_r = srb_mat.srb_mat_eigvalsh_with_diagnostics(sreal)
    _, eigh_diag_r = srb_mat.srb_mat_eigh_with_diagnostics(sreal)
    _, eigsh_diag_r = srb_mat.srb_mat_eigsh_with_diagnostics(sreal, k=2, which="largest", steps=2)

    _, pow_diag_c = scb_mat.scb_mat_pow_ui_with_diagnostics(scomplex, 2)
    _, exp_diag_c = scb_mat.scb_mat_exp_with_diagnostics(scomplex)
    _, eigval_diag_c = scb_mat.scb_mat_eigvalsh_with_diagnostics(scomplex)
    _, eigh_diag_c = scb_mat.scb_mat_eigh_with_diagnostics(scomplex)
    _, eigsh_diag_c = scb_mat.scb_mat_eigsh_with_diagnostics(scomplex, k=2, which="largest", steps=2)

    for diag in (pow_diag_r, exp_diag_r, eigval_diag_r, eigh_diag_r, pow_diag_c, exp_diag_c, eigval_diag_c, eigh_diag_c):
        _check(not bool(diag.dense_lift_used))
        _check(bool(diag.sparse_native))
        _check(not bool(diag.preserves_sparse_output))
        _check(int(diag.rows) == 3)
        _check(int(diag.cols) == 3)
        _check(int(diag.nnz) == 3)

    for diag in (eigval_diag_r, eigh_diag_r, eigval_diag_c, eigh_diag_c, eigsh_diag_r, eigsh_diag_c):
        _check(bool(diag.structured_input_required))

    for diag in (eigsh_diag_r, eigsh_diag_c):
        _check(bool(diag.sparse_native))
        _check(not bool(diag.dense_lift_used))
        _check(not bool(diag.preserves_sparse_output))
