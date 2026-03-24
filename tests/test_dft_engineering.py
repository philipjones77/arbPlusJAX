from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import acb_core
from arbplusjax import dft
from arbplusjax import dft_wrappers as dw
from arbplusjax import double_interval as di


def _rand_complex(n: int, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    re = rng.normal(size=n)
    im = rng.normal(size=n)
    return jnp.asarray(re + 1j * im, dtype=jnp.complex128)


def _to_point_boxes(z: jax.Array) -> jax.Array:
    zr = jnp.real(z)
    zi = jnp.imag(z)
    return jnp.stack([zr, zr, zi, zi], axis=-1)


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_dft_point_cached_plan_reuse_and_padded_batch_paths_match_direct_kernels() -> None:
    x0 = _rand_complex(11, seed=201)
    x1 = _rand_complex(11, seed=202)
    x2 = _rand_complex(11, seed=203)
    xs = jnp.stack([x0, x1, x2], axis=0)
    plan = dft.dft_matvec_cached_prepare_point(11)

    fixed = dft.dft_matvec_batch_fixed_point(xs)
    padded = dft.dft_matvec_batch_padded_point(xs, pad_to=8)
    cached_fixed = dft.dft_matvec_cached_apply_batch_fixed_point(plan, xs)
    cached_padded = dft.dft_matvec_cached_apply_batch_padded_point(plan, xs, pad_to=8)
    diag_out, diagnostics = dft.dft_matvec_cached_apply_point_with_diagnostics(plan, x0)

    np.testing.assert_allclose(np.asarray(fixed), np.asarray(jax.vmap(dft.dft_jit)(xs)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(padded), np.asarray(fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_fixed), np.asarray(fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_padded), np.asarray(fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(dft.dft_matvec_cached_apply_batch_fixed_point(plan, xs)), np.asarray(cached_fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(diag_out), np.asarray(dft.dft_jit(x0)), rtol=1e-12, atol=1e-12)
    assert diagnostics["method"] == "bluestein"
    assert diagnostics["mode"] == "point"


def test_dft_basic_cached_plan_reuse_and_wrapper_modes_have_containment() -> None:
    z0 = _rand_complex(11, seed=211)
    z1 = _rand_complex(11, seed=212)
    boxes = jnp.stack([_to_point_boxes(z0), _to_point_boxes(z1)], axis=0)
    plan = dft.dft_matvec_cached_prepare_basic(11)

    fixed = dft.dft_matvec_batch_fixed_basic(boxes)
    padded = dft.dft_matvec_batch_padded_basic(boxes, pad_to=4)
    cached_fixed = dft.dft_matvec_cached_apply_batch_fixed_basic(plan, boxes)
    cached_padded = dft.dft_matvec_cached_apply_batch_padded_basic(plan, boxes, pad_to=4)
    diag_out, diagnostics = dft.dft_matvec_cached_apply_basic_with_diagnostics(plan, boxes[0])

    np.testing.assert_allclose(np.asarray(fixed), np.asarray(jax.vmap(dft.acb_dft_jit)(boxes)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(padded), np.asarray(fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_fixed), np.asarray(fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_padded), np.asarray(fixed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(diag_out), np.asarray(dft.acb_dft_jit(boxes[0])), rtol=1e-12, atol=1e-12)
    assert diagnostics["method"] == "bluestein"
    assert diagnostics["mode"] == "basic"

    basic = dw.acb_dft_mode(boxes[0], impl="basic", prec_bits=80)
    rigorous = dw.acb_dft_mode(boxes[0], impl="rigorous", prec_bits=80)
    inverse_basic = dw.acb_dft_inverse_mode(boxes[0], impl="basic", prec_bits=80)
    inverse_rigorous = dw.acb_dft_inverse_mode(boxes[0], impl="rigorous", prec_bits=80)

    assert _contains_box(rigorous, basic)
    assert _contains_box(inverse_rigorous, inverse_basic)


def test_dft_wrapper_convolution_modes_remain_aligned_and_rigorous_contains_basic() -> None:
    f = _to_point_boxes(_rand_complex(8, seed=221))
    g = _to_point_boxes(_rand_complex(8, seed=222))

    basic_naive = dw.acb_dft_convol_naive_mode(f, g, impl="basic", prec_bits=80)
    basic_dft = dw.acb_dft_convol_dft_mode(f, g, impl="basic", prec_bits=80)
    basic_rad2 = dw.acb_dft_convol_rad2_mode(f, g, impl="basic", prec_bits=80)
    rigorous = dw.acb_dft_convol_mode(f, g, impl="rigorous", prec_bits=80)

    np.testing.assert_allclose(np.asarray(basic_dft), np.asarray(basic_naive), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(basic_rad2), np.asarray(basic_naive), rtol=1e-12, atol=1e-12)
    assert _contains_box(rigorous, basic_dft)


def test_dft_point_and_basic_ad_smoke() -> None:
    def point_loss(re_im: jax.Array) -> jax.Array:
        z = re_im[0::2] + 1j * re_im[1::2]
        y = dft.dft_matvec_cached_apply_point(dft.dft_matvec_cached_prepare_point(z.shape[0]), z)
        return jnp.real(jnp.vdot(y, y))

    x = _rand_complex(8, seed=231)
    vec = jnp.stack([jnp.real(x), jnp.imag(x)], axis=1).reshape(-1)
    g = jax.grad(point_loss)(vec)
    assert g.shape == vec.shape
    assert bool(jnp.all(jnp.isfinite(g)))

    def _box(xr, xi):
        xr = jnp.asarray(xr, dtype=jnp.float64)
        xi = jnp.asarray(xi, dtype=jnp.float64)
        return jnp.stack((xr, xr, xi, xi), axis=-1)

    def basic_loss(t: jax.Array) -> jax.Array:
        z = jnp.stack((t, 0.5 * t), axis=0)
        x = _box(z, jnp.zeros_like(z))
        y = acb_core.acb_midpoint(dft.acb_dft_jit(x))
        return jnp.real(jnp.vdot(y, y))

    basic_grad = jax.grad(basic_loss)(jnp.asarray([0.25, -0.5, 1.0, 0.75], dtype=jnp.float64))
    assert basic_grad.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(basic_grad)))
