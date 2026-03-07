import jax
import jax.numpy as jnp

from arbplusjax import ball_wrappers
from arbplusjax import bessel_kernels as bk
from arbplusjax import double_interval as di
from arbplusjax import hypgeom

from tests._test_checks import _check


_REAL_POINT_VARIANTS = {
    "j": lambda nu, z: bk.real_bessel_eval_j(nu, z),
    "i": lambda nu, z: bk.real_bessel_eval_i(nu, z),
    "y": lambda nu, z: bk.real_bessel_eval_y(nu, z),
    "k": lambda nu, z: bk.real_bessel_eval_k(nu, z),
    "i_scaled": lambda nu, z: jnp.exp(-z) * bk.real_bessel_eval_i(nu, z),
    "k_scaled": lambda nu, z: jnp.exp(z) * bk.real_bessel_eval_k(nu, z),
}

_COMPLEX_POINT_VARIANTS = {
    "j": lambda nu, z: bk.complex_bessel_series(nu, z, -1.0),
    "i": lambda nu, z: bk.complex_bessel_series(nu, z, 1.0),
    "y": lambda nu, z: bk.complex_bessel_y(nu, z),
    "k": lambda nu, z: bk.complex_bessel_k(nu, z),
    "i_scaled": lambda nu, z: jnp.exp(-z) * bk.complex_bessel_series(nu, z, 1.0),
    "k_scaled": lambda nu, z: jnp.exp(z) * bk.complex_bessel_k(nu, z),
}

_REAL_WRAPPER_VARIANTS = {
    "j": lambda nu, z: ball_wrappers.arb_ball_bessel_j(nu, z, prec_bits=80),
    "i": lambda nu, z: ball_wrappers.arb_ball_bessel_i(nu, z, prec_bits=80),
    "y": lambda nu, z: ball_wrappers.arb_ball_bessel_y(nu, z, prec_bits=80),
    "k": lambda nu, z: ball_wrappers.arb_ball_bessel_k(nu, z, prec_bits=80),
    "i_scaled": lambda nu, z: ball_wrappers.arb_ball_bessel_i_scaled(nu, z, prec_bits=80),
    "k_scaled": lambda nu, z: ball_wrappers.arb_ball_bessel_k_scaled(nu, z, prec_bits=80),
    "j_adaptive": lambda nu, z: ball_wrappers.arb_ball_bessel_j_adaptive(nu, z, prec_bits=80, samples=5),
    "i_adaptive": lambda nu, z: ball_wrappers.arb_ball_bessel_i_adaptive(nu, z, prec_bits=80, samples=5),
    "y_adaptive": lambda nu, z: ball_wrappers.arb_ball_bessel_y_adaptive(nu, z, prec_bits=80, samples=5),
    "k_adaptive": lambda nu, z: ball_wrappers.arb_ball_bessel_k_adaptive(nu, z, prec_bits=80, samples=5),
    "i_scaled_adaptive": lambda nu, z: ball_wrappers.arb_ball_bessel_i_scaled_adaptive(nu, z, prec_bits=80, samples=5),
    "k_scaled_adaptive": lambda nu, z: ball_wrappers.arb_ball_bessel_k_scaled_adaptive(nu, z, prec_bits=80, samples=5),
}

_COMPLEX_WRAPPER_VARIANTS = {
    "j": lambda nu, z: ball_wrappers.acb_ball_bessel_j(nu, z, prec_bits=80),
    "i": lambda nu, z: ball_wrappers.acb_ball_bessel_i(nu, z, prec_bits=80),
    "y": lambda nu, z: ball_wrappers.acb_ball_bessel_y(nu, z, prec_bits=80),
    "k": lambda nu, z: ball_wrappers.acb_ball_bessel_k(nu, z, prec_bits=80),
    "i_scaled": lambda nu, z: ball_wrappers.acb_ball_bessel_i_scaled(nu, z, prec_bits=80),
    "k_scaled": lambda nu, z: ball_wrappers.acb_ball_bessel_k_scaled(nu, z, prec_bits=80),
    "j_adaptive": lambda nu, z: ball_wrappers.acb_ball_bessel_j_adaptive(nu, z, prec_bits=80, samples=5),
    "i_adaptive": lambda nu, z: ball_wrappers.acb_ball_bessel_i_adaptive(nu, z, prec_bits=80, samples=5),
    "y_adaptive": lambda nu, z: ball_wrappers.acb_ball_bessel_y_adaptive(nu, z, prec_bits=80, samples=5),
    "k_adaptive": lambda nu, z: ball_wrappers.acb_ball_bessel_k_adaptive(nu, z, prec_bits=80, samples=5),
    "i_scaled_adaptive": lambda nu, z: ball_wrappers.acb_ball_bessel_i_scaled_adaptive(nu, z, prec_bits=80, samples=5),
    "k_scaled_adaptive": lambda nu, z: ball_wrappers.acb_ball_bessel_k_scaled_adaptive(nu, z, prec_bits=80, samples=5),
}


def test_bessel_kernels_preserve_float32_and_complex64():
    nu_r = jnp.asarray(0.5, dtype=jnp.float32)
    z_r = jnp.asarray(3.0, dtype=jnp.float32)
    out_r = bk.real_bessel_eval_j(nu_r, z_r)
    _check(out_r.dtype == jnp.float32)

    nu_c = jnp.asarray(0.5 + 0.1j, dtype=jnp.complex64)
    z_c = jnp.asarray(2.0 + 0.3j, dtype=jnp.complex64)
    out_c = bk.complex_bessel_series(nu_c, z_c, 1.0)
    _check(out_c.dtype == jnp.complex64)


def test_bessel_point_kernels_are_jittable_and_differentiable():
    nu = jnp.asarray(0.4, dtype=jnp.float32)

    jit_j = jax.jit(lambda z: bk.real_bessel_eval_j(nu, z))
    jit_i = jax.jit(lambda z: bk.real_bessel_eval_i(nu, z))

    z = jnp.asarray(2.5, dtype=jnp.float32)
    yj = jit_j(z)
    yi = jit_i(z)
    _check(jnp.isfinite(yj))
    _check(jnp.isfinite(yi))

    gj = jax.grad(lambda t: bk.real_bessel_eval_j(nu, t))(z)
    gi = jax.grad(lambda t: bk.real_bessel_eval_i(nu, t))(z)
    _check(jnp.isfinite(gj))
    _check(jnp.isfinite(gi))


def test_bessel_point_kernels_have_universal_real_ad_audit():
    nu = jnp.asarray(0.4, dtype=jnp.float32)
    z = jnp.asarray(2.5, dtype=jnp.float32)
    for name, fn in _REAL_POINT_VARIANTS.items():
        jit_fn = jax.jit(lambda t, impl=fn: impl(nu, t))
        y = jit_fn(z)
        g = jax.grad(lambda t, impl=fn: impl(nu, t))(z)
        _check(jnp.isfinite(y), f"{name} point value not finite")
        _check(jnp.isfinite(g), f"{name} point grad not finite")


def test_bessel_point_kernels_have_universal_complex_ad_audit():
    nu = jnp.asarray(0.35 + 0.1j, dtype=jnp.complex64)
    z0 = jnp.asarray([2.0, 0.3], dtype=jnp.float32)

    for name, fn in _COMPLEX_POINT_VARIANTS.items():
        def vec(v, impl=fn):
            z = jnp.asarray(v[0] + 1j * v[1], dtype=jnp.complex64)
            w = impl(nu, z)
            return jnp.asarray([jnp.real(w), jnp.imag(w)], dtype=jnp.float32)

        jac = jax.jacfwd(vec)(z0)
        _check(jnp.all(jnp.isfinite(jac)), f"{name} complex point jacobian not finite")


def test_bessel_batch_jit_paths_remain_callable():
    nu = jnp.stack(
        [
            jnp.asarray([0.2, 0.3], dtype=jnp.float32),
            jnp.asarray([0.25, 0.35], dtype=jnp.float32),
        ],
        axis=0,
    )
    z = jnp.stack(
        [
            jnp.asarray([1.5, 2.0], dtype=jnp.float32),
            jnp.asarray([1.8, 2.3], dtype=jnp.float32),
        ],
        axis=0,
    )
    out = hypgeom.arb_hypgeom_bessel_j_batch_jit(nu, z, mode="sample")
    _check(out.shape == nu.shape)
    _check(jnp.all(jnp.isfinite(out)))


def test_bessel_padded_batch_matches_unpadded():
    nu = jnp.stack(
        [
            jnp.asarray([0.2, 0.3], dtype=jnp.float32),
            jnp.asarray([0.25, 0.35], dtype=jnp.float32),
            jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        ],
        axis=0,
    )
    z = jnp.stack(
        [
            jnp.asarray([1.5, 2.0], dtype=jnp.float32),
            jnp.asarray([1.8, 2.3], dtype=jnp.float32),
            jnp.asarray([2.1, 2.6], dtype=jnp.float32),
        ],
        axis=0,
    )
    plain = hypgeom.arb_hypgeom_bessel_k_batch_prec_jit(nu, z, prec_bits=80, mode="sample")
    padded = hypgeom.arb_hypgeom_bessel_k_batch_padded_prec_jit(nu, z, pad_to=8, prec_bits=80, mode="sample")
    _check(plain.shape == padded.shape)
    _check(jnp.allclose(plain, padded, equal_nan=True))


def test_bessel_fixed_batch_matches_padded():
    nu = jnp.stack(
        [
            jnp.asarray([0.2, 0.3], dtype=jnp.float32),
            jnp.asarray([0.25, 0.35], dtype=jnp.float32),
            jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        ],
        axis=0,
    )
    z = jnp.stack(
        [
            jnp.asarray([1.5, 2.0], dtype=jnp.float32),
            jnp.asarray([1.8, 2.3], dtype=jnp.float32),
            jnp.asarray([2.1, 2.6], dtype=jnp.float32),
        ],
        axis=0,
    )
    padded = hypgeom.arb_hypgeom_bessel_k_batch_padded_prec_jit(nu, z, pad_to=8, prec_bits=80, mode="sample")
    nu_fixed = jnp.concatenate([nu, jnp.repeat(nu[-1:, :], 5, axis=0)], axis=0)
    z_fixed = jnp.concatenate([z, jnp.repeat(z[-1:, :], 5, axis=0)], axis=0)
    fixed = hypgeom.arb_hypgeom_bessel_k_batch_fixed_prec(nu_fixed, z_fixed, prec_bits=80, mode="sample")
    _check(fixed.shape[0] == 8)
    _check(jnp.allclose(padded, fixed[: nu.shape[0]], equal_nan=True))


def test_bessel_wrapper_audit_real_is_universal():
    radius = jnp.asarray(0.02, dtype=jnp.float32)

    def make_iv(mu):
        return di.interval(mu - radius, mu + radius)

    def mid(iv):
        return 0.5 * (iv[0] + iv[1])

    nu0 = jnp.asarray(0.4, dtype=jnp.float32)
    z0 = jnp.asarray(2.5, dtype=jnp.float32)

    for name, fn in _REAL_WRAPPER_VARIANTS.items():
        jit_fn = jax.jit(lambda t, impl=fn: mid(impl(make_iv(nu0), make_iv(t))))
        val = jit_fn(z0)
        grad = jax.grad(lambda t, impl=fn: mid(impl(make_iv(nu0), make_iv(t))))(z0)
        _check(jnp.isfinite(val), f"{name} wrapper value not finite")
        _check(jnp.isfinite(grad), f"{name} wrapper grad not finite")


def test_bessel_wrapper_audit_complex_is_universal():
    r = jnp.asarray(0.02, dtype=jnp.float32)

    def make_box(v):
        re = v[0]
        im = v[1]
        return jnp.asarray([re - r, re + r, im - r, im + r], dtype=jnp.float32)

    nu = jnp.asarray([0.33, 0.37, 0.08, 0.12], dtype=jnp.float32)
    z0 = jnp.asarray([2.0, 0.3], dtype=jnp.float32)

    def mid_re_im(box):
        return jnp.asarray([0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])], dtype=jnp.float32)

    for name, fn in _COMPLEX_WRAPPER_VARIANTS.items():
        def vec(v, impl=fn):
            return mid_re_im(impl(nu, make_box(v)))

        jac = jax.jacfwd(vec)(z0)
        _check(jnp.all(jnp.isfinite(jac)), f"{name} complex wrapper jacobian not finite")
