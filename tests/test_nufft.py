import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import dft
from arbplusjax import nufft


def _rand_points(n: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random(n), dtype=jnp.float64)


def _rand_points_nd(n: int, ndim: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((n, ndim)), dtype=jnp.float64)


def _rand_complex(n: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n), dtype=jnp.complex128)


def test_nufft_type1_direct_matches_uniform_dft():
    values = _rand_complex(16, seed=11)
    points = jnp.arange(16, dtype=jnp.float64) / 16.0
    got = nufft.nufft_type1_direct(points, values, 16)
    ref = dft.dft(values)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_nufft_lanczos_type1_close_to_direct():
    points = _rand_points(48, seed=21)
    values = _rand_complex(48, seed=22)
    ref = nufft.nufft_type1_direct(points, values, 24)
    got = nufft.nufft_type1(points, values, 24, method="lanczos")
    rel = float(jnp.linalg.norm(got - ref) / jnp.maximum(jnp.linalg.norm(ref), 1e-12))
    assert rel < 2.0e-3


def test_nufft_lanczos_type2_close_to_direct():
    points = _rand_points(48, seed=31)
    modes = _rand_complex(24, seed=32)
    ref = nufft.nufft_type2_direct(points, modes)
    got = nufft.nufft_type2(points, modes, method="lanczos")
    rel = float(jnp.linalg.norm(got - ref) / jnp.maximum(jnp.linalg.norm(ref), 1e-12))
    assert rel < 2.0e-3


def test_nufft_lanczos_type1_type2_are_adjoint():
    points = _rand_points(40, seed=41)
    modes = _rand_complex(20, seed=42)
    probe = _rand_complex(40, seed=43)
    left = jnp.vdot(nufft.nufft_type2(points, modes, method="lanczos"), probe)
    right = jnp.vdot(modes, nufft.nufft_type1(points, probe, 20, method="lanczos"))
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-12, atol=1e-12)


def test_nufft_diagnostics_report_grid_path():
    points = _rand_points(96, seed=51)
    values = _rand_complex(96, seed=52)
    _, diagnostics = nufft.nufft_type1_with_diagnostics(points, values, 64, method="auto")
    assert diagnostics["method"] == "lanczos"
    assert diagnostics["grid_size"] >= 64
    assert diagnostics["kernel_width"] == 8


def test_nufft_cached_type1_and_batch_match_uncached():
    points = _rand_points(48, seed=531)
    values = _rand_complex(48, seed=532)
    values_batch = jnp.stack([values, (1.0 - 0.25j) * values], axis=0)
    plan = nufft.nufft_type1_cached_prepare(points, 24, method="lanczos")

    got = nufft.nufft_type1_cached_apply(plan, values)
    got_diag, diagnostics = nufft.nufft_type1_cached_apply_with_diagnostics(plan, values)
    batch = nufft.nufft_type1_cached_apply_batch_fixed(plan, values_batch)
    padded = nufft.nufft_type1_cached_apply_batch_padded(plan, values_batch, pad_to=4)
    ref = nufft.nufft_type1(points, values, 24, method="lanczos")
    ref_batch = jax.vmap(lambda row: nufft.nufft_type1(points, row, 24, method="lanczos"))(values_batch)

    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got_diag), np.asarray(got), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(batch), np.asarray(ref_batch), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(padded), np.asarray(ref_batch), rtol=1e-12, atol=1e-12)
    assert diagnostics["method"] == "lanczos"
    assert diagnostics["mode_shape"] == (24,)


def test_nufft_cached_type2_and_batch_match_uncached():
    points = _rand_points_nd(30, 2, seed=541)
    modes = _rand_complex(56, seed=542).reshape(8, 7)
    modes_batch = jnp.stack([modes, (0.5 + 0.1j) * modes], axis=0)
    plan = nufft.nufft_type2_nd_cached_prepare(points, (8, 7), method="lanczos")

    got = nufft.nufft_type2_cached_apply(plan, modes)
    got_diag, diagnostics = nufft.nufft_type2_cached_apply_with_diagnostics(plan, modes)
    batch = nufft.nufft_type2_cached_apply_batch_fixed(plan, modes_batch)
    padded = nufft.nufft_type2_cached_apply_batch_padded(plan, modes_batch, pad_to=4)
    ref = nufft.nufft_type2_2d(points, modes, method="lanczos")
    ref_batch = jax.vmap(lambda grid: nufft.nufft_type2_2d(points, grid, method="lanczos"))(modes_batch)

    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got_diag), np.asarray(got), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(batch), np.asarray(ref_batch), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(padded), np.asarray(ref_batch), rtol=1e-12, atol=1e-12)
    assert diagnostics["method"] == "lanczos"
    assert diagnostics["mode_shape"] == (8, 7)


def test_nufft_grad_path_is_finite():
    points = _rand_points(24, seed=61)
    modes = _rand_complex(12, seed=62)

    def loss(re_im: jax.Array) -> jax.Array:
        z = re_im[0::2] + 1j * re_im[1::2]
        y = nufft.nufft_type2(points, z, method="lanczos")
        return jnp.real(jnp.vdot(y, y))

    vec = jnp.stack([jnp.real(modes), jnp.imag(modes)], axis=1).reshape(-1)
    grad = jax.grad(loss)(vec)
    assert grad.shape == vec.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_nufft_type1_2d_direct_matches_uniform_fft():
    values = _rand_complex(12, seed=71).reshape(3, 4)
    points = jnp.stack(jnp.meshgrid(jnp.arange(3) / 3.0, jnp.arange(4) / 4.0, indexing="ij"), axis=-1).reshape(-1, 2)
    got = nufft.nufft_type1_2d_direct(points, values.reshape(-1), (3, 4))
    ref = dft.dft2(values)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_nufft_type1_3d_direct_matches_uniform_fft():
    values = _rand_complex(24, seed=72).reshape(2, 3, 4)
    points = jnp.stack(
        jnp.meshgrid(jnp.arange(2) / 2.0, jnp.arange(3) / 3.0, jnp.arange(4) / 4.0, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    got = nufft.nufft_type1_3d_direct(points, values.reshape(-1), (2, 3, 4))
    ref = dft.dft3(values)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_nufft_lanczos_type1_2d_close_to_direct():
    points = _rand_points_nd(32, 2, seed=81)
    values = _rand_complex(32, seed=82)
    ref = nufft.nufft_type1_2d_direct(points, values, (8, 7))
    got = nufft.nufft_type1_2d(points, values, (8, 7), method="lanczos")
    rel = float(jnp.linalg.norm(got - ref) / jnp.maximum(jnp.linalg.norm(ref), 1e-12))
    assert rel < 2.0e-3


def test_nufft_lanczos_type2_2d_close_to_direct_and_adjoint():
    points = _rand_points_nd(30, 2, seed=83)
    modes = _rand_complex(56, seed=84).reshape(8, 7)
    probe = _rand_complex(30, seed=85)
    ref = nufft.nufft_type2_2d_direct(points, modes)
    got = nufft.nufft_type2_2d(points, modes, method="lanczos")
    rel = float(jnp.linalg.norm(got - ref) / jnp.maximum(jnp.linalg.norm(ref), 1e-12))
    assert rel < 2.0e-3
    left = jnp.vdot(got, probe)
    right = jnp.vdot(modes, nufft.nufft_type1_2d(points, probe, (8, 7), method="lanczos"))
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-12, atol=1e-12)


def test_nufft_lanczos_type1_3d_and_type2_3d():
    points = _rand_points_nd(20, 3, seed=91)
    values = _rand_complex(20, seed=92)
    modes = _rand_complex(60, seed=93).reshape(5, 4, 3)
    probe = _rand_complex(20, seed=94)

    ref1 = nufft.nufft_type1_3d_direct(points, values, (5, 4, 3))
    got1 = nufft.nufft_type1_3d(points, values, (5, 4, 3), method="lanczos")
    rel1 = float(jnp.linalg.norm(got1 - ref1) / jnp.maximum(jnp.linalg.norm(ref1), 1e-12))
    assert rel1 < 2.0e-3

    ref2 = nufft.nufft_type2_3d_direct(points, modes)
    got2 = nufft.nufft_type2_3d(points, modes, method="lanczos")
    rel2 = float(jnp.linalg.norm(got2 - ref2) / jnp.maximum(jnp.linalg.norm(ref2), 1e-12))
    assert rel2 < 2.0e-3

    left = jnp.vdot(got2, probe)
    right = jnp.vdot(modes, nufft.nufft_type1_3d(points, probe, (5, 4, 3), method="lanczos"))
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-12, atol=1e-12)


def test_nufft_nd_diagnostics_report_shape_and_method():
    points = _rand_points_nd(96, 2, seed=101)
    values = _rand_complex(96, seed=102)
    _, diagnostics = nufft.nufft_type1_nd_with_diagnostics(points, values, (32, 24), method="auto")
    assert diagnostics["method"] == "lanczos"
    assert diagnostics["mode_shape"] == (32, 24)
    assert diagnostics["grid_shape"][0] >= 32
    assert diagnostics["grid_shape"][1] >= 24
    assert diagnostics["ndim"] == 2
