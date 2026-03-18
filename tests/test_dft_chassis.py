import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import dft


from tests._test_checks import _check
def _rand_complex(n: int, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    re = rng.normal(size=n)
    im = rng.normal(size=n)
    return jnp.asarray(re + 1j * im, dtype=jnp.complex128)


def test_dft_roundtrip_jit():
    x = _rand_complex(8, seed=11)
    y = dft.dft_jit(x)
    z = dft.idft_jit(y)
    np.testing.assert_allclose(np.asarray(z), np.asarray(x), rtol=1e-12, atol=1e-12)


def test_prime_length_dft_matches_fft_and_precomp():
    x = _rand_complex(11, seed=14)
    precomp = dft.make_dft_precomp(11)
    y_main = dft.dft_jit(x)
    y_precomp = dft.dft_bluestein_precomp(x, precomp=precomp)
    np.testing.assert_allclose(np.asarray(y_main), np.asarray(jnp.fft.fft(x)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y_precomp), np.asarray(jnp.fft.fft(x)), rtol=1e-12, atol=1e-12)


def test_dft2_and_dft3_roundtrip_match_fftnd():
    x2 = _rand_complex(35, seed=15).reshape(5, 7)
    y2 = dft.dft2_jit(x2)
    z2 = dft.idft2_jit(y2)
    np.testing.assert_allclose(np.asarray(y2), np.asarray(jnp.fft.fftn(x2)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(z2), np.asarray(x2), rtol=1e-12, atol=1e-12)

    x3 = _rand_complex(60, seed=16).reshape(3, 4, 5)
    y3 = dft.dft3_jit(x3)
    z3 = dft.idft3_jit(y3)
    np.testing.assert_allclose(np.asarray(y3), np.asarray(jnp.fft.fftn(x3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(z3), np.asarray(x3), rtol=1e-12, atol=1e-12)


def test_dft_nd_axis_subset_matches_separable_fft():
    x = _rand_complex(60, seed=17).reshape(3, 4, 5)
    got = dft.dft_nd_jit(x, axes=(1, 2))
    ref = jnp.fft.fftn(x, axes=(1, 2))
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_dft_good_size_is_smooth():
    n = dft.dft_good_size(29)
    m = n
    for p in (2, 3, 5):
        while m % p == 0:
            m //= p
    _check(n >= 29)
    _check(m == 1)


def test_dft_matvec_cached_and_batch_point():
    x0 = _rand_complex(11, seed=141)
    x1 = _rand_complex(11, seed=142)
    xs = jnp.stack([x0, x1], axis=0)
    plan = dft.dft_matvec_cached_prepare_point(11)

    y = dft.dft_matvec_cached_apply_point(plan, x0)
    y_diag, diagnostics = dft.dft_matvec_cached_apply_point_with_diagnostics(plan, x0)
    batch = dft.dft_matvec_batch_fixed_point(xs)
    cached_batch = dft.dft_matvec_cached_apply_batch_fixed_point(plan, xs)
    padded = dft.dft_matvec_batch_padded_point(xs, pad_to=4)
    cached_padded = dft.dft_matvec_cached_apply_batch_padded_point(plan, xs, pad_to=4)

    np.testing.assert_allclose(np.asarray(y), np.asarray(dft.dft_jit(x0)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y_diag), np.asarray(y), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(batch), np.asarray(jax.vmap(dft.dft_jit)(xs)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_batch), np.asarray(batch), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(padded), np.asarray(batch), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_padded), np.asarray(batch), rtol=1e-12, atol=1e-12)
    _check(diagnostics["method"] == "bluestein")
    _check(diagnostics["mode"] == "point")


def test_dft_matvec_cached_and_batch_basic():
    z0 = _rand_complex(11, seed=151)
    z1 = _rand_complex(11, seed=152)
    xb = jnp.stack([_to_point_boxes(z0), _to_point_boxes(z1)], axis=0)
    plan = dft.dft_matvec_cached_prepare_basic(11)

    y = dft.dft_matvec_cached_apply_basic(plan, xb[0])
    y_diag, diagnostics = dft.dft_matvec_cached_apply_basic_with_diagnostics(plan, xb[0])
    batch = dft.dft_matvec_batch_fixed_basic(xb)
    cached_batch = dft.dft_matvec_cached_apply_batch_fixed_basic(plan, xb)
    padded = dft.dft_matvec_batch_padded_basic(xb, pad_to=4)
    cached_padded = dft.dft_matvec_cached_apply_batch_padded_basic(plan, xb, pad_to=4)

    np.testing.assert_allclose(np.asarray(y), np.asarray(dft.acb_dft_jit(xb[0])), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y_diag), np.asarray(y), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(batch), np.asarray(jax.vmap(dft.acb_dft_jit)(xb)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_batch), np.asarray(batch), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(padded), np.asarray(batch), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cached_padded), np.asarray(batch), rtol=1e-12, atol=1e-12)
    _check(diagnostics["method"] == "bluestein")
    _check(diagnostics["mode"] == "basic")


def test_product_api_matches_main_dft():
    x = _rand_complex(12, seed=12)
    y = dft.dft_jit(x)
    yp = dft.dft_prod_jit(x, cyc=(3, 4))
    np.testing.assert_allclose(np.asarray(yp), np.asarray(y), rtol=1e-12, atol=1e-12)


def test_convolution_variants_agree():
    f = _rand_complex(8, seed=21)
    g = _rand_complex(8, seed=22)
    yn = dft.convol_circular_naive_jit(f, g)
    yd = dft.convol_circular_dft_jit(f, g)
    yr = dft.convol_circular_rad2_jit(f, g)
    np.testing.assert_allclose(np.asarray(yd), np.asarray(yn), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(yr), np.asarray(yn), rtol=1e-12, atol=1e-12)


def test_grad_path_real_loss():
    x = _rand_complex(8, seed=31)

    def loss(re_im: jax.Array) -> jax.Array:
        z = re_im[0::2] + 1j * re_im[1::2]
        y = dft.dft_jit(z)
        return jnp.real(jnp.vdot(y, y))

    vec = jnp.stack([jnp.real(x), jnp.imag(x)], axis=1).reshape(-1)
    g = jax.grad(loss)(vec)
    _check(g.shape == vec.shape)
    _check(bool(jnp.all(jnp.isfinite(g))))


def _to_point_boxes(z: jax.Array) -> jax.Array:
    zr = jnp.real(z)
    zi = jnp.imag(z)
    return jnp.stack([zr, zr, zi, zi], axis=-1)


def test_acb_dft_roundtrip_and_order():
    z = _rand_complex(8, seed=77)
    x = _to_point_boxes(z)
    y = dft.acb_dft_jit(x)
    w = dft.acb_idft_jit(y)
    _check(y.shape == (8, 4))
    _check(bool(jnp.all(y[:, 0] <= y[:, 1])))
    _check(bool(jnp.all(y[:, 2] <= y[:, 3])))
    np.testing.assert_allclose(np.asarray(w[:, 0]), np.asarray(jnp.real(z)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w[:, 2]), np.asarray(jnp.imag(z)), rtol=1e-12, atol=1e-12)


def test_acb_prime_length_exact_boxes_match_midpoint_fft():
    z = _rand_complex(11, seed=79)
    x = _to_point_boxes(z)
    y = dft.acb_dft_jit(x)
    target = jnp.fft.fft(z)
    np.testing.assert_allclose(np.asarray(y[:, 0]), np.asarray(jnp.real(target)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y[:, 2]), np.asarray(jnp.imag(target)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y[:, 1] - y[:, 0]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(y[:, 3] - y[:, 2]), 0.0, rtol=0.0, atol=0.0)


def test_acb_dft2_and_dft3_roundtrip_for_point_boxes():
    z2 = _rand_complex(35, seed=91).reshape(5, 7)
    x2 = _to_point_boxes(z2)
    y2 = dft.acb_dft2_jit(x2)
    w2 = dft.acb_idft2_jit(y2)
    np.testing.assert_allclose(np.asarray(y2[..., 0]), np.asarray(jnp.real(jnp.fft.fftn(z2))), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y2[..., 2]), np.asarray(jnp.imag(jnp.fft.fftn(z2))), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w2[..., 0]), np.asarray(jnp.real(z2)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w2[..., 2]), np.asarray(jnp.imag(z2)), rtol=1e-12, atol=1e-12)

    z3 = _rand_complex(60, seed=92).reshape(3, 4, 5)
    x3 = _to_point_boxes(z3)
    y3 = dft.acb_dft3_jit(x3)
    w3 = dft.acb_idft3_jit(y3)
    np.testing.assert_allclose(np.asarray(y3[..., 0]), np.asarray(jnp.real(jnp.fft.fftn(z3))), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y3[..., 2]), np.asarray(jnp.imag(jnp.fft.fftn(z3))), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w3[..., 0]), np.asarray(jnp.real(z3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(w3[..., 2]), np.asarray(jnp.imag(z3)), rtol=1e-12, atol=1e-12)


def test_acb_convolution_variants_agree():
    f = _to_point_boxes(_rand_complex(8, seed=81))
    g = _to_point_boxes(_rand_complex(8, seed=82))
    yn = dft.acb_convol_circular_naive_jit(f, g)
    yd = dft.acb_convol_circular_dft_jit(f, g)
    yr = dft.acb_convol_circular_rad2_jit(f, g)
    np.testing.assert_allclose(np.asarray(yd), np.asarray(yn), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(yr), np.asarray(yn), rtol=1e-12, atol=1e-12)
