import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import mat_wrappers
from arbplusjax import scb_mat
from arbplusjax import srb_mat

from tests._test_checks import _check


def test_srb_sparse_mode_surface_and_api():
    dense = jnp.array(
        [
            [5.0, 1.0, 0.0],
            [1.0, 4.0, 0.5],
            [0.0, 0.5, 3.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    rhs_cols = jnp.stack([rhs, rhs + 1.0], axis=1)
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    sparse = srb_mat.srb_mat_from_dense_csr(dense)

    det_point = mat_wrappers.srb_mat_det_mode(sparse, impl="point", prec_bits=80)
    det_basic = mat_wrappers.srb_mat_det_mode(sparse, impl="basic", prec_bits=80)
    det_rig = mat_wrappers.srb_mat_det_mode(sparse, impl="rigorous", prec_bits=80)
    det_adapt = mat_wrappers.srb_mat_det_mode(sparse, impl="adaptive", prec_bits=80)
    tr_point = mat_wrappers.srb_mat_trace_mode(sparse, impl="point", prec_bits=80)
    tr_basic = mat_wrappers.srb_mat_trace_mode(sparse, impl="basic", prec_bits=80)
    tr_rig = mat_wrappers.srb_mat_trace_mode(sparse, impl="rigorous", prec_bits=80)
    tr_adapt = mat_wrappers.srb_mat_trace_mode(sparse, impl="adaptive", prec_bits=80)
    sym_basic = mat_wrappers.srb_mat_symmetric_part_mode(sparse, impl="basic", prec_bits=80)

    _check(bool(jnp.allclose(det_point, jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(tr_point, jnp.trace(dense), rtol=1e-8, atol=1e-8)))
    _check(det_basic.shape == (2,))
    _check(det_rig.shape == (2,))
    _check(det_adapt.shape == (2,))
    _check(tr_basic.shape == (2,))
    _check(tr_rig.shape == (2,))
    _check(tr_adapt.shape == (2,))
    _check(bool(jnp.allclose(di.midpoint(det_basic), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(tr_basic), jnp.trace(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(di.contains(det_rig, det_basic)))
    _check(bool(di.contains(tr_rig, tr_basic)))
    _check(bool(jnp.allclose(di.midpoint(sym_basic), dense, rtol=1e-8, atol=1e-8)))

    spd_plan = mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(sparse, impl="adaptive", prec_bits=80)
    lu_plan = mat_wrappers.srb_mat_lu_solve_plan_prepare_mode(sparse, impl="basic", prec_bits=80)
    solve_basic = mat_wrappers.srb_mat_spd_solve_mode(sparse, rhs, impl="basic", prec_bits=80)
    solve_rig = mat_wrappers.srb_mat_spd_solve_plan_apply_mode(spd_plan, rhs, impl="rigorous", prec_bits=80)
    solve_lu = mat_wrappers.srb_mat_solve_lu_mode(lu_plan, rhs, impl="adaptive", prec_bits=80)
    solve_t = mat_wrappers.srb_mat_solve_transpose_mode(lu_plan, rhs, impl="basic", prec_bits=80)
    solve_add = mat_wrappers.srb_mat_solve_add_mode(lu_plan, rhs, rhs, impl="basic", prec_bits=80)
    mat_solve = mat_wrappers.srb_mat_mat_solve_mode(spd_plan, rhs_cols, impl="adaptive", prec_bits=80)
    vec_batch = mat_wrappers.srb_mat_matvec_batch_mode_padded(sparse, rhs_batch, pad_to=4, impl="adaptive", prec_bits=80)
    solve_batch = mat_wrappers.srb_mat_spd_solve_plan_apply_batch_mode_padded(spd_plan, rhs_batch, pad_to=4, impl="rigorous", prec_bits=80)

    expected = jnp.linalg.solve(dense, rhs)
    expected_cols = jnp.linalg.solve(dense, rhs_cols)
    _check(bool(jnp.allclose(di.midpoint(solve_basic), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(solve_rig), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(solve_lu), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(solve_t), jnp.linalg.solve(dense.T, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(solve_add), rhs + expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(mat_solve), expected_cols, rtol=1e-8, atol=1e-8)))
    _check(vec_batch.shape == (4, 3, 2))
    _check(solve_batch.shape == (4, 3, 2))

    api_det = api.eval_interval("srb_mat_det", sparse, mode="adaptive", prec_bits=80)
    api_solve = api.eval_interval("srb_mat_spd_solve", sparse, rhs, mode="rigorous", prec_bits=80)
    api_batch = api.eval_interval_batch("srb_mat_spd_solve_plan_apply", spd_plan, rhs_batch, mode="basic", prec_bits=80, pad_to=4)

    _check(bool(jnp.allclose(di.midpoint(api_det), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(di.midpoint(api_solve), expected, rtol=1e-8, atol=1e-8)))
    _check(api_batch.shape == (2, 3, 2))


def test_scb_sparse_mode_surface_and_api():
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
    sparse = scb_mat.scb_mat_from_dense_csr(dense)

    det_point = mat_wrappers.scb_mat_det_mode(sparse, impl="point", prec_bits=80)
    det_basic = mat_wrappers.scb_mat_det_mode(sparse, impl="basic", prec_bits=80)
    det_rig = mat_wrappers.scb_mat_det_mode(sparse, impl="rigorous", prec_bits=80)
    tr_point = mat_wrappers.scb_mat_trace_mode(sparse, impl="point", prec_bits=80)
    tr_basic = mat_wrappers.scb_mat_trace_mode(sparse, impl="basic", prec_bits=80)
    tr_rig = mat_wrappers.scb_mat_trace_mode(sparse, impl="rigorous", prec_bits=80)
    herm_basic = mat_wrappers.scb_mat_hermitian_part_mode(sparse, impl="basic", prec_bits=80)

    _check(bool(jnp.allclose(det_point, jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(tr_point, jnp.trace(dense), rtol=1e-8, atol=1e-8)))
    _check(det_basic.shape == (4,))
    _check(det_rig.shape == (4,))
    _check(tr_basic.shape == (4,))
    _check(tr_rig.shape == (4,))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(det_basic), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(tr_basic), jnp.trace(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(di.contains(acb_core.acb_real(det_rig), acb_core.acb_real(det_basic))))
    _check(bool(di.contains(acb_core.acb_imag(det_rig), acb_core.acb_imag(det_basic))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(herm_basic), dense, rtol=1e-8, atol=1e-8)))

    hpd_plan = mat_wrappers.scb_mat_hpd_solve_plan_prepare_mode(sparse, impl="adaptive", prec_bits=80)
    lu_plan = mat_wrappers.scb_mat_lu_solve_plan_prepare_mode(sparse, impl="basic", prec_bits=80)
    solve_basic = mat_wrappers.scb_mat_hpd_solve_mode(sparse, rhs, impl="basic", prec_bits=80)
    solve_rig = mat_wrappers.scb_mat_hpd_solve_plan_apply_mode(hpd_plan, rhs, impl="rigorous", prec_bits=80)
    solve_lu = mat_wrappers.scb_mat_solve_lu_mode(lu_plan, rhs, impl="adaptive", prec_bits=80)
    solve_t = mat_wrappers.scb_mat_solve_transpose_mode(lu_plan, rhs, impl="basic", prec_bits=80)
    solve_add = mat_wrappers.scb_mat_solve_add_mode(lu_plan, rhs, rhs, impl="basic", prec_bits=80)
    mat_solve = mat_wrappers.scb_mat_mat_solve_mode(hpd_plan, rhs_cols, impl="adaptive", prec_bits=80)
    vec_batch = mat_wrappers.scb_mat_matvec_batch_mode_padded(sparse, rhs_batch, pad_to=4, impl="adaptive", prec_bits=80)
    solve_batch = mat_wrappers.scb_mat_hpd_solve_plan_apply_batch_mode_padded(hpd_plan, rhs_batch, pad_to=4, impl="rigorous", prec_bits=80)

    expected = jnp.linalg.solve(dense, rhs)
    expected_cols = jnp.linalg.solve(dense, rhs_cols)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_basic), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_rig), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_lu), expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_t), jnp.linalg.solve(dense.T, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_add), rhs + expected, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(mat_solve), expected_cols, rtol=1e-8, atol=1e-8)))
    _check(vec_batch.shape == (4, 3, 4))
    _check(solve_batch.shape == (4, 3, 4))

    api_det = api.eval_interval("scb_mat_det", sparse, mode="adaptive", prec_bits=80)
    api_solve = api.eval_interval("scb_mat_hpd_solve", sparse, rhs, mode="rigorous", prec_bits=80)
    api_batch = api.eval_interval_batch("scb_mat_hpd_solve_plan_apply", hpd_plan, rhs_batch, mode="basic", prec_bits=80, pad_to=4)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(api_det), jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(api_solve), expected, rtol=1e-8, atol=1e-8)))
    _check(api_batch.shape == (2, 3, 4))
