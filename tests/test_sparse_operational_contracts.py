import jax
import jax.numpy as jnp
import pytest

from arbplusjax import api
from arbplusjax import mat_wrappers
from arbplusjax import scb_mat
from arbplusjax import sparse_common as sc
from arbplusjax import srb_mat


def _real_cases():
    dense = jnp.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 2.0],
            [1.5, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    return dense, (
        srb_mat.srb_mat_from_dense_coo(dense),
        srb_mat.srb_mat_from_dense_csr(dense),
        srb_mat.srb_mat_from_dense_bcoo(dense),
    )


def _complex_cases():
    dense = jnp.array(
        [
            [4.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 3.0 + 0.25j, 2.0 + 0.1j],
            [1.5 + 0.5j, 0.0 + 0.0j, 5.0 - 0.25j],
        ],
        dtype=jnp.complex128,
    )
    return dense, (
        scb_mat.scb_mat_from_dense_coo(dense),
        scb_mat.scb_mat_from_dense_csr(dense),
        scb_mat.scb_mat_from_dense_bcoo(dense),
    )


def _forbid_real_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("dense fallback is forbidden on sparse operational paths")

    monkeypatch.setattr(srb_mat, "srb_mat_to_dense", _boom)
    monkeypatch.setattr(sc, "sparse_interval_to_dense", _boom)


def _forbid_complex_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("dense fallback is forbidden on sparse operational paths")

    monkeypatch.setattr(scb_mat, "scb_mat_to_dense", _boom)
    monkeypatch.setattr(sc, "sparse_box_to_dense", _boom)


def test_srb_sparse_point_and_basic_operational_paths_avoid_dense_fallback(monkeypatch: pytest.MonkeyPatch):
    dense, cases = _real_cases()
    vec = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)
    vecs = jnp.stack([vec, vec + 1.0], axis=0)
    expected = dense @ vec
    expected_t = dense.T @ vec
    _forbid_real_dense(monkeypatch)

    point_bound = api.bind_point_batch_jit("srb_mat_matvec_cached_apply", dtype="float64", pad_to=4)
    basic_bound = api.bind_interval_batch_jit("srb_mat_matvec_cached_apply", mode="basic", dtype="float64", pad_to=4, backend="cpu")

    for sparse in cases:
        point_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        basic_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        basic_rplan = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")

        assert jnp.allclose(mat_wrappers.srb_mat_matvec_mode(sparse, vec, impl="point"), expected)
        assert jnp.allclose(mat_wrappers.srb_mat_rmatvec_mode(sparse, vec, impl="point"), expected_t)
        assert jnp.allclose(mat_wrappers.srb_mat_matvec_cached_apply_mode(point_plan, vec, impl="point"), expected)
        assert jnp.allclose(mat_wrappers.srb_mat_rmatvec_cached_apply_mode(mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="point"), vec, impl="point"), expected_t)

        basic_out = mat_wrappers.srb_mat_matvec_mode(sparse, vec, impl="basic")
        basic_rout = mat_wrappers.srb_mat_rmatvec_mode(sparse, vec, impl="basic")
        basic_cached = mat_wrappers.srb_mat_matvec_cached_apply_mode(basic_plan, vec, impl="basic")
        basic_rcached = mat_wrappers.srb_mat_rmatvec_cached_apply_mode(basic_rplan, vec, impl="basic")

        assert basic_out.shape == (3, 2)
        assert basic_rout.shape == (3, 2)
        assert basic_cached.shape == (3, 2)
        assert basic_rcached.shape == (3, 2)
        assert jnp.allclose(point_bound(point_plan, vecs), vecs @ dense.T)
        assert basic_bound(basic_plan, vecs).shape == (2, 3, 2)


def test_scb_sparse_point_and_basic_operational_paths_avoid_dense_fallback(monkeypatch: pytest.MonkeyPatch):
    dense, cases = _complex_cases()
    vec = jnp.array([1.0 + 0.2j, -0.5 + 0.1j, 0.25 - 0.4j], dtype=jnp.complex128)
    vecs = jnp.stack([vec, vec + (0.5 - 0.25j)], axis=0)
    expected = dense @ vec
    expected_t = dense.T @ vec
    _forbid_complex_dense(monkeypatch)

    point_bound = api.bind_point_batch_jit("scb_mat_matvec_cached_apply", dtype="float64", pad_to=4)
    basic_bound = api.bind_interval_batch_jit("scb_mat_matvec_cached_apply", mode="basic", dtype="float64", pad_to=4, backend="cpu")

    for sparse in cases:
        point_plan = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        basic_plan = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        basic_rplan = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")

        assert jnp.allclose(mat_wrappers.scb_mat_matvec_mode(sparse, vec, impl="point"), expected)
        assert jnp.allclose(mat_wrappers.scb_mat_rmatvec_mode(sparse, vec, impl="point"), expected_t)
        assert jnp.allclose(mat_wrappers.scb_mat_matvec_cached_apply_mode(point_plan, vec, impl="point"), expected)
        assert jnp.allclose(mat_wrappers.scb_mat_rmatvec_cached_apply_mode(mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="point"), vec, impl="point"), expected_t)

        basic_out = mat_wrappers.scb_mat_matvec_mode(sparse, vec, impl="basic")
        basic_rout = mat_wrappers.scb_mat_rmatvec_mode(sparse, vec, impl="basic")
        basic_cached = mat_wrappers.scb_mat_matvec_cached_apply_mode(basic_plan, vec, impl="basic")
        basic_rcached = mat_wrappers.scb_mat_rmatvec_cached_apply_mode(basic_rplan, vec, impl="basic")

        assert basic_out.shape == (3, 4)
        assert basic_rout.shape == (3, 4)
        assert basic_cached.shape == (3, 4)
        assert basic_rcached.shape == (3, 4)
        assert jnp.allclose(point_bound(point_plan, vecs), vecs @ dense.T)
        assert basic_bound(basic_plan, vecs).shape == (2, 3, 4)


def test_sparse_operational_binders_expose_policy_and_diagnostics():
    dense, (sparse, *_rest) = _real_cases()
    vecs = jnp.stack(
        [
            jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64),
            jnp.array([0.5, 0.0, -1.0], dtype=jnp.float64),
        ],
        axis=0,
    )
    point_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="point")
    basic_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="basic")

    point_bound = api.bind_point_batch_jit_with_diagnostics("srb_mat_matvec_cached_apply", dtype="float64", pad_to=4, backend="cpu")
    basic_bound = api.bind_interval_batch_jit_with_diagnostics("srb_mat_matvec_cached_apply", mode="basic", dtype="float64", pad_to=4, backend="cpu")

    point_out, point_diag = point_bound(point_plan, vecs)
    basic_out, basic_diag = basic_bound(basic_plan, vecs)

    assert point_out.shape == (2, 3)
    assert basic_out.shape == (2, 3, 2)
    assert point_diag.chosen_backend == "cpu"
    assert basic_diag.chosen_backend == "cpu"
    assert point_diag.batch_size == 2
    assert basic_diag.batch_size == 2
    assert point_diag.effective_pad_to == 4
    assert basic_diag.effective_pad_to == 4
    assert jnp.allclose(point_out[:2], vecs @ dense.T)
