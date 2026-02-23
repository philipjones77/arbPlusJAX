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


def test_acb_convolution_variants_agree():
    f = _to_point_boxes(_rand_complex(8, seed=81))
    g = _to_point_boxes(_rand_complex(8, seed=82))
    yn = dft.acb_convol_circular_naive_jit(f, g)
    yd = dft.acb_convol_circular_dft_jit(f, g)
    yr = dft.acb_convol_circular_rad2_jit(f, g)
    np.testing.assert_allclose(np.asarray(yd), np.asarray(yn), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(yr), np.asarray(yn), rtol=1e-12, atol=1e-12)
