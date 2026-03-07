import jax
import jax.numpy as jnp

from arbplusjax import ball_wrappers
from arbplusjax import bessel_kernels as bk
from arbplusjax import double_interval as di
from arbplusjax import hypgeom

from tests._test_checks import _check


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


def test_bessel_ad_audit_real_yk_and_scaled_variants():
    nu = jnp.asarray(0.4, dtype=jnp.float32)
    z = jnp.asarray(2.5, dtype=jnp.float32)

    gy = jax.grad(lambda t: bk.real_bessel_eval_y(nu, t))(z)
    gk = jax.grad(lambda t: bk.real_bessel_eval_k(nu, t))(z)
    gis = jax.grad(lambda t: jnp.exp(-t) * bk.real_bessel_eval_i(nu, t))(z)
    gks = jax.grad(lambda t: jnp.exp(t) * bk.real_bessel_eval_k(nu, t))(z)
    _check(jnp.isfinite(gy))
    _check(jnp.isfinite(gk))
    _check(jnp.isfinite(gis))
    _check(jnp.isfinite(gks))


def test_bessel_ad_audit_complex_yk_and_scaled_variants():
    nu = jnp.asarray(0.35 + 0.1j, dtype=jnp.complex64)
    z0 = jnp.asarray([2.0, 0.3], dtype=jnp.float32)

    def complex_k_vec(v):
        z = jnp.asarray(v[0] + 1j * v[1], dtype=jnp.complex64)
        w = bk.complex_bessel_k(nu, z)
        return jnp.asarray([jnp.real(w), jnp.imag(w)], dtype=jnp.float32)

    def complex_ks_vec(v):
        z = jnp.asarray(v[0] + 1j * v[1], dtype=jnp.complex64)
        w = jnp.exp(z) * bk.complex_bessel_k(nu, z)
        return jnp.asarray([jnp.real(w), jnp.imag(w)], dtype=jnp.float32)

    jk = jax.jacfwd(complex_k_vec)(z0)
    jks = jax.jacfwd(complex_ks_vec)(z0)
    _check(jnp.all(jnp.isfinite(jk)))
    _check(jnp.all(jnp.isfinite(jks)))


def test_bessel_wrapper_audit_real_yk_and_scaled_variants():
    radius = jnp.asarray(0.02, dtype=jnp.float32)

    def make_iv(mu):
        return di.interval(mu - radius, mu + radius)

    def mid(iv):
        return 0.5 * (iv[0] + iv[1])

    nu0 = jnp.asarray(0.4, dtype=jnp.float32)
    z0 = jnp.asarray(2.5, dtype=jnp.float32)

    fy = jax.jit(lambda t: mid(ball_wrappers.arb_ball_bessel_y(make_iv(nu0), make_iv(t), prec_bits=80)))
    fk = jax.jit(lambda t: mid(ball_wrappers.arb_ball_bessel_k(make_iv(nu0), make_iv(t), prec_bits=80)))
    fys = jax.jit(lambda t: mid(ball_wrappers.arb_ball_bessel_y_adaptive(make_iv(nu0), make_iv(t), prec_bits=80, samples=5)))
    fks = jax.jit(lambda t: mid(ball_wrappers.arb_ball_bessel_k_scaled_adaptive(make_iv(nu0), make_iv(t), prec_bits=80, samples=5)))

    vy = fy(z0)
    vk = fk(z0)
    vys = fys(z0)
    vks = fks(z0)
    _check(jnp.isfinite(vy))
    _check(jnp.isfinite(vk))
    _check(jnp.isfinite(vys))
    _check(jnp.isfinite(vks))

    gy = jax.grad(lambda t: mid(ball_wrappers.arb_ball_bessel_y(make_iv(nu0), make_iv(t), prec_bits=80)))(z0)
    gk = jax.grad(lambda t: mid(ball_wrappers.arb_ball_bessel_k(make_iv(nu0), make_iv(t), prec_bits=80)))(z0)
    _check(jnp.isfinite(gy))
    _check(jnp.isfinite(gk))


def test_bessel_wrapper_audit_complex_yk_and_scaled_variants():
    r = jnp.asarray(0.02, dtype=jnp.float32)

    def make_box(v):
        re = v[0]
        im = v[1]
        return jnp.asarray([re - r, re + r, im - r, im + r], dtype=jnp.float32)

    nu = jnp.asarray([0.33, 0.37, 0.08, 0.12], dtype=jnp.float32)
    z0 = jnp.asarray([2.0, 0.3], dtype=jnp.float32)

    def mid_re_im(box):
        return jnp.asarray([0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])], dtype=jnp.float32)

    def ky(v):
        return mid_re_im(ball_wrappers.acb_ball_bessel_y(nu, make_box(v), prec_bits=80))

    def kk(v):
        return mid_re_im(ball_wrappers.acb_ball_bessel_k(nu, make_box(v), prec_bits=80))

    def kks(v):
        return mid_re_im(ball_wrappers.acb_ball_bessel_k_scaled_adaptive(nu, make_box(v), prec_bits=80, samples=5))

    jy = jax.jacfwd(ky)(z0)
    jk = jax.jacfwd(kk)(z0)
    jks = jax.jacfwd(kks)(z0)
    _check(jnp.all(jnp.isfinite(jy)))
    _check(jnp.all(jnp.isfinite(jk)))
    _check(jnp.all(jnp.isfinite(jks)))
