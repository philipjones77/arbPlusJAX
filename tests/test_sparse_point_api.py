import jax.numpy as jnp
import pytest

from arbplusjax import api
from arbplusjax import scb_mat
from arbplusjax import srb_mat

from tests._test_checks import _check


def test_srb_point_api_and_batch_fastpaths():
    dense = jnp.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    sparse = srb_mat.srb_mat_from_dense_csr(dense)
    vec = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float64)
    vecs = jnp.stack([vec, vec + 1.0], axis=0)
    plan = api.eval_point("srb_mat_matvec_cached_prepare", sparse)
    solve_rhs = jnp.stack([jnp.array([2.0, 6.0, 10.0]), jnp.array([4.0, 9.0, 20.0])], axis=0)
    lower = srb_mat.srb_mat_from_dense_csr(jnp.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, -1.0, 4.0]], dtype=jnp.float64))
    tri_rhs = jnp.stack([jnp.array([2.0, 7.0, 5.0]), jnp.array([4.0, 10.0, 9.0])], axis=0)

    _check(bool(jnp.allclose(api.eval_point("srb_mat_matvec", sparse, vec), dense @ vec)))
    _check(bool(jnp.allclose(api.eval_point("srb_mat_matvec_cached_apply", plan, vec), dense @ vec)))
    _check(bool(jnp.allclose(api.eval_point_batch("srb_mat_matvec", sparse, vecs), vecs @ dense.T)))
    _check(bool(jnp.allclose(api.eval_point_batch("srb_mat_matvec_cached_apply", plan, vecs), vecs @ dense.T)))
    _check(bool(jnp.allclose(api.eval_point_batch("srb_mat_solve", srb_mat.srb_mat_from_dense_bcoo(jnp.diag(jnp.array([2.0, 3.0, 5.0]))), solve_rhs, method="gmres"), solve_rhs / jnp.array([2.0, 3.0, 5.0]))))
    tri_expected = jnp.stack([jnp.linalg.solve(srb_mat.srb_mat_to_dense(lower), tri_rhs[0]), jnp.linalg.solve(srb_mat.srb_mat_to_dense(lower), tri_rhs[1])], axis=0)
    _check(bool(jnp.allclose(api.eval_point_batch("srb_mat_triangular_solve", lower, tri_rhs, lower=True), tri_expected)))
    _check(bool(jnp.allclose(api.eval_point("srb_mat_trace", sparse), jnp.trace(dense))))
    _check(bool(jnp.allclose(api.eval_point("srb_mat_norm_fro", sparse), jnp.linalg.norm(dense, ord="fro"))))
    p, l, u = api.eval_point("srb_mat_lu", sparse)
    lu_sol = api.eval_point("srb_mat_lu_solve", (p, l, u), vec)
    _check(bool(jnp.allclose(api.eval_point("srb_mat_to_dense", p) @ dense, api.eval_point("srb_mat_to_dense", l) @ api.eval_point("srb_mat_to_dense", u))))
    _check(bool(jnp.allclose(lu_sol, jnp.linalg.solve(dense, vec), rtol=1e-6, atol=1e-6)))
    qr = api.eval_point("srb_mat_qr", sparse)
    q = api.eval_point("srb_mat_qr_explicit_q", qr)
    r = api.eval_point("srb_mat_to_dense", api.eval_point("srb_mat_qr_r", qr))
    qr_sol = api.eval_point("srb_mat_qr_solve", qr, vec)
    _check(bool(jnp.allclose(q @ r, dense, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(qr_sol, jnp.linalg.solve(dense, vec), rtol=1e-6, atol=1e-6)))


def test_scb_point_api_and_batch_fastpaths():
    dense = jnp.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j, 1.0 - 0.25j],
            [0.0 + 0.0j, 3.0 + 1.0j, 0.0 + 0.0j],
            [4.0 + 0.5j, 0.0 + 0.0j, 5.0 - 1.0j],
        ],
        dtype=jnp.complex128,
    )
    sparse = scb_mat.scb_mat_from_dense_csr(dense)
    vec = jnp.array([1.0 + 0.0j, -1.0 + 0.25j, 0.5 - 1.0j], dtype=jnp.complex128)
    vecs = jnp.stack([vec, vec + (1.0 - 0.5j)], axis=0)
    plan = api.eval_point("scb_mat_matvec_cached_prepare", sparse)
    solve_rhs = jnp.stack(
        [
            jnp.array([2.0 + 1.0j, 6.0 - 1.0j, 10.0 + 0.5j]),
            jnp.array([4.0 + 0.0j, 9.0 + 1.0j, 20.0 - 0.5j]),
        ],
        axis=0,
    )
    diag = scb_mat.scb_mat_from_dense_bcoo(jnp.diag(jnp.array([2.0 + 0.5j, 3.0 - 0.5j, 5.0 + 0.0j], dtype=jnp.complex128)))

    _check(bool(jnp.allclose(api.eval_point("scb_mat_matvec", sparse, vec), dense @ vec)))
    _check(bool(jnp.allclose(api.eval_point("scb_mat_matvec_cached_apply", plan, vec), dense @ vec)))
    _check(bool(jnp.allclose(api.eval_point_batch("scb_mat_matvec", sparse, vecs), vecs @ dense.T)))
    _check(bool(jnp.allclose(api.eval_point_batch("scb_mat_matvec_cached_apply", plan, vecs), vecs @ dense.T)))
    expected = jnp.stack([jnp.linalg.solve(scb_mat.scb_mat_to_dense(diag), solve_rhs[0]), jnp.linalg.solve(scb_mat.scb_mat_to_dense(diag), solve_rhs[1])], axis=0)
    _check(bool(jnp.allclose(api.eval_point_batch("scb_mat_solve", diag, solve_rhs, method="gmres"), expected)))
    _check(bool(jnp.allclose(api.eval_point("scb_mat_trace", sparse), jnp.trace(dense))))
    _check(bool(jnp.allclose(api.eval_point("scb_mat_norm_fro", sparse), jnp.linalg.norm(dense, ord="fro"))))
    p, l, u = api.eval_point("scb_mat_lu", sparse)
    lu_sol = api.eval_point("scb_mat_lu_solve", (p, l, u), vec)
    _check(bool(jnp.allclose(api.eval_point("scb_mat_to_dense", p) @ dense, api.eval_point("scb_mat_to_dense", l) @ api.eval_point("scb_mat_to_dense", u))))
    _check(bool(jnp.allclose(lu_sol, jnp.linalg.solve(dense, vec), rtol=1e-6, atol=1e-6)))
    qr = api.eval_point("scb_mat_qr", sparse)
    q = api.eval_point("scb_mat_qr_explicit_q", qr)
    r = api.eval_point("scb_mat_to_dense", api.eval_point("scb_mat_qr_r", qr))
    qr_sol = api.eval_point("scb_mat_qr_solve", qr, vec)
    _check(bool(jnp.allclose(q @ r, dense, rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(qr_sol, jnp.linalg.solve(dense, vec), rtol=1e-6, atol=1e-6)))


def test_srb_sparse_lu_and_solves_avoid_dense_fallback_for_all_formats(monkeypatch: pytest.MonkeyPatch):
    dense = jnp.array(
        [
            [3.0, 1.0, 0.0],
            [2.0, 4.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    expected = jnp.linalg.solve(dense, rhs)
    expected_t = jnp.linalg.solve(dense.T, rhs)
    cases = (
        srb_mat.srb_mat_from_dense_coo(dense),
        srb_mat.srb_mat_from_dense_csr(dense),
        srb_mat.srb_mat_from_dense_bcoo(dense),
    )

    def _forbid_dense(*args, **kwargs):
        raise AssertionError("srb_mat_to_dense should not be used by sparse LU/solve paths")

    monkeypatch.setattr(srb_mat, "srb_mat_to_dense", _forbid_dense)

    for sparse in cases:
        p, l, u = srb_mat.srb_mat_lu(sparse)
        sol = srb_mat.srb_mat_solve(sparse, rhs)
        sol_t = srb_mat.srb_mat_solve_transpose(sparse, rhs)
        lu_sol = srb_mat.srb_mat_lu_solve((p, l, u), rhs)
        _check(bool(jnp.allclose(sol, expected, rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(sol_t, expected_t, rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(lu_sol, expected, rtol=1e-6, atol=1e-6)))


def test_scb_sparse_lu_and_solves_avoid_dense_fallback_for_all_formats(monkeypatch: pytest.MonkeyPatch):
    dense = jnp.array(
        [
            [3.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j],
            [2.0 + 0.25j, 4.0 + 0.0j, 1.0 + 0.5j],
            [0.0 + 0.0j, 1.0 - 0.25j, 2.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.array([1.0 + 0.5j, 2.0 - 0.25j, 3.0 + 0.75j], dtype=jnp.complex128)
    expected = jnp.linalg.solve(dense, rhs)
    expected_t = jnp.linalg.solve(dense.T, rhs)
    cases = (
        scb_mat.scb_mat_from_dense_coo(dense),
        scb_mat.scb_mat_from_dense_csr(dense),
        scb_mat.scb_mat_from_dense_bcoo(dense),
    )

    def _forbid_dense(*args, **kwargs):
        raise AssertionError("scb_mat_to_dense should not be used by sparse LU/solve paths")

    monkeypatch.setattr(scb_mat, "scb_mat_to_dense", _forbid_dense)

    for sparse in cases:
        p, l, u = scb_mat.scb_mat_lu(sparse)
        sol = scb_mat.scb_mat_solve(sparse, rhs)
        sol_t = scb_mat.scb_mat_solve_transpose(sparse, rhs)
        lu_sol = scb_mat.scb_mat_lu_solve((p, l, u), rhs)
        _check(bool(jnp.allclose(sol, expected, rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(sol_t, expected_t, rtol=1e-6, atol=1e-6)))
        _check(bool(jnp.allclose(lu_sol, expected, rtol=1e-6, atol=1e-6)))
