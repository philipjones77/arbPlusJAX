import jax.numpy as jnp

from arbplusjax import scb_mat
from arbplusjax import srb_mat

from tests._test_checks import _check


def test_srb_sparse_structured_surface():
    dense = jnp.array(
        [
            [5.0, 1.0, 0.0],
            [1.0, 4.0, 0.5],
            [0.0, 0.5, 3.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    sparse = srb_mat.srb_mat_from_dense_csr(dense)

    _check(bool(srb_mat.srb_mat_is_symmetric(sparse)))
    _check(bool(srb_mat.srb_mat_is_spd(sparse)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(srb_mat.srb_mat_symmetric_part(sparse)), dense)))

    chol = srb_mat.srb_mat_cho(sparse)
    ldl_l, ldl_d = srb_mat.srb_mat_ldl(sparse)
    spd_plan = srb_mat.srb_mat_spd_solve_plan_prepare(sparse)
    lu_plan = srb_mat.srb_mat_lu_solve_plan_prepare(sparse)

    chol_dense = srb_mat.srb_mat_to_dense(chol)
    ldl_l_dense = srb_mat.srb_mat_to_dense(ldl_l)
    sol_spd = srb_mat.srb_mat_spd_solve_plan_apply(spd_plan, rhs)
    sol_lu = srb_mat.srb_mat_lu_solve_plan_apply(lu_plan, rhs)
    sol_t = srb_mat.srb_mat_solve_transpose(lu_plan, rhs)
    sol_add = srb_mat.srb_mat_solve_add(lu_plan, rhs, rhs)
    sol_batch = srb_mat.srb_mat_spd_solve_plan_apply_batch_padded(spd_plan, rhs_batch, pad_to=4)

    _check(bool(jnp.allclose(chol_dense @ chol_dense.T, dense, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(ldl_l_dense @ jnp.diag(ldl_d) @ ldl_l_dense.T, dense, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_spd, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_lu, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_t, jnp.linalg.solve(dense.T, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_add, rhs + jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(sol_batch.shape == (4, 3))


def test_scb_sparse_structured_surface():
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
    rhs_batch = jnp.stack([rhs, rhs + (0.5 - 0.25j)], axis=0)
    sparse = scb_mat.scb_mat_from_dense_csr(dense)

    _check(bool(scb_mat.scb_mat_is_hermitian(sparse)))
    _check(bool(scb_mat.scb_mat_is_hpd(sparse)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(scb_mat.scb_mat_hermitian_part(sparse)), dense)))

    chol = scb_mat.scb_mat_cho(sparse)
    ldl_l, ldl_d = scb_mat.scb_mat_ldl(sparse)
    hpd_plan = scb_mat.scb_mat_hpd_solve_plan_prepare(sparse)
    lu_plan = scb_mat.scb_mat_lu_solve_plan_prepare(sparse)

    chol_dense = scb_mat.scb_mat_to_dense(chol)
    ldl_l_dense = scb_mat.scb_mat_to_dense(ldl_l)
    sol_hpd = scb_mat.scb_mat_hpd_solve_plan_apply(hpd_plan, rhs)
    sol_lu = scb_mat.scb_mat_lu_solve_plan_apply(lu_plan, rhs)
    sol_t = scb_mat.scb_mat_solve_transpose(lu_plan, rhs)
    sol_add = scb_mat.scb_mat_solve_add(lu_plan, rhs, rhs)
    sol_batch = scb_mat.scb_mat_hpd_solve_plan_apply_batch_padded(hpd_plan, rhs_batch, pad_to=4)

    _check(bool(jnp.allclose(chol_dense @ jnp.conj(chol_dense.T), dense, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(ldl_l_dense @ jnp.diag(ldl_d) @ jnp.conj(ldl_l_dense.T), dense, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_hpd, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_lu, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_t, jnp.linalg.solve(dense.T, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol_add, rhs + jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(sol_batch.shape == (4, 3))
