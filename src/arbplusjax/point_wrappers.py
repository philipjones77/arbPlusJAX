from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import acb_core
from . import double_interval as di
from . import hypgeom
from . import elementary as el

jax.config.update("jax_enable_x64", True)


# Point-only kernels (no interval or outward rounding)


def _point_interval(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    return di.interval(arr, arr)


def _point_box(z: jax.Array) -> jax.Array:
    zz = jnp.asarray(z)
    return acb_core.acb_box(di.interval(jnp.real(zz), jnp.real(zz)), di.interval(jnp.imag(zz), jnp.imag(zz)))


def _scalarize_unary_complex(fn):
    @jax.jit
    def wrapped(z: jax.Array) -> jax.Array:
        flat = jnp.ravel(jnp.asarray(z, dtype=jnp.complex128))
        out = jax.vmap(lambda t: acb_core.acb_midpoint(fn(_point_box(t))))(flat)
        return out.reshape(jnp.shape(z))

    return wrapped


def _scalarize_binary_complex(fn):
    @jax.jit
    def wrapped(x: jax.Array, y: jax.Array) -> jax.Array:
        xx = jnp.asarray(x, dtype=jnp.complex128)
        yy = jnp.asarray(y, dtype=jnp.complex128)
        flat_x = jnp.ravel(xx)
        flat_y = jnp.ravel(yy)
        out = jax.vmap(lambda a, b: acb_core.acb_midpoint(fn(_point_box(a), _point_box(b))))(flat_x, flat_y)
        return out.reshape(jnp.shape(xx))

    return wrapped


def _vmap_complex_scalar(fn):
    @jax.jit
    def wrapped(z: jax.Array) -> jax.Array:
        zz = jnp.asarray(z, dtype=jnp.complex128)
        flat = jnp.ravel(zz)
        out = jax.vmap(fn)(flat)
        return out.reshape(jnp.shape(zz))

    return wrapped

@partial(jax.jit, static_argnames=())
def arb_exp_point(x: jax.Array) -> jax.Array:
    return jnp.exp(x)


@partial(jax.jit, static_argnames=())
def arb_log_point(x: jax.Array) -> jax.Array:
    return jnp.log(x)


@partial(jax.jit, static_argnames=())
def arb_sqrt_point(x: jax.Array) -> jax.Array:
    return jnp.sqrt(x)


@partial(jax.jit, static_argnames=())
def arb_sin_point(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@partial(jax.jit, static_argnames=())
def arb_cos_point(x: jax.Array) -> jax.Array:
    return jnp.cos(x)


@partial(jax.jit, static_argnames=())
def arb_tan_point(x: jax.Array) -> jax.Array:
    return jnp.tan(x)


@partial(jax.jit, static_argnames=())
def arb_sinh_point(x: jax.Array) -> jax.Array:
    return jnp.sinh(x)


@partial(jax.jit, static_argnames=())
def arb_cosh_point(x: jax.Array) -> jax.Array:
    return jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def arb_tanh_point(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


@partial(jax.jit, static_argnames=())
def arb_abs_point(x: jax.Array) -> jax.Array:
    return jnp.abs(x)


@partial(jax.jit, static_argnames=())
def arb_add_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


@partial(jax.jit, static_argnames=())
def arb_sub_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x - y


@partial(jax.jit, static_argnames=())
def arb_mul_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@partial(jax.jit, static_argnames=())
def arb_div_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x / y


@partial(jax.jit, static_argnames=())
def arb_inv_point(x: jax.Array) -> jax.Array:
    return 1.0 / x


@partial(jax.jit, static_argnames=())
def arb_fma_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x * y + z


@partial(jax.jit, static_argnames=())
def arb_log1p_point(x: jax.Array) -> jax.Array:
    return jnp.log1p(x)


@partial(jax.jit, static_argnames=())
def arb_expm1_point(x: jax.Array) -> jax.Array:
    return jnp.expm1(x)


@partial(jax.jit, static_argnames=())
def arb_sin_cos_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sin(x), jnp.cos(x)


@partial(jax.jit, static_argnames=())
def arb_sinh_cosh_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sinh(x), jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def arb_sin_pi_point(x: jax.Array) -> jax.Array:
    return jnp.sin(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def arb_cos_pi_point(x: jax.Array) -> jax.Array:
    return jnp.cos(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def arb_tan_pi_point(x: jax.Array) -> jax.Array:
    return jnp.tan(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def arb_sinc_point(x: jax.Array) -> jax.Array:
    return el.sinc(x)


@partial(jax.jit, static_argnames=())
def arb_sinc_pi_point(x: jax.Array) -> jax.Array:
    return el.sinc_pi(x)


@partial(jax.jit, static_argnames=())
def arb_asin_point(x: jax.Array) -> jax.Array:
    return jnp.arcsin(x)


@partial(jax.jit, static_argnames=())
def arb_acos_point(x: jax.Array) -> jax.Array:
    return jnp.arccos(x)


@partial(jax.jit, static_argnames=())
def arb_atan_point(x: jax.Array) -> jax.Array:
    return jnp.arctan(x)


@partial(jax.jit, static_argnames=())
def arb_asinh_point(x: jax.Array) -> jax.Array:
    return jnp.arcsinh(x)


@partial(jax.jit, static_argnames=())
def arb_acosh_point(x: jax.Array) -> jax.Array:
    return jnp.arccosh(x)


@partial(jax.jit, static_argnames=())
def arb_atanh_point(x: jax.Array) -> jax.Array:
    return jnp.arctanh(x)


@partial(jax.jit, static_argnames=())
def arb_sign_point(x: jax.Array) -> jax.Array:
    return jnp.sign(x)


@partial(jax.jit, static_argnames=())
def arb_pow_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(x, y)


@partial(jax.jit, static_argnames=("n",))
def arb_pow_ui_point(x: jax.Array, n: int) -> jax.Array:
    return jnp.power(x, n)


@partial(jax.jit, static_argnames=("k",))
def arb_root_ui_point(x: jax.Array, k: int) -> jax.Array:
    xx = jnp.asarray(x)
    kf = jnp.asarray(k, dtype=xx.dtype if jnp.issubdtype(xx.dtype, jnp.floating) else jnp.float64)
    root_abs = jnp.power(jnp.abs(xx), 1.0 / kf)
    if (k % 2) == 1:
        return jnp.sign(xx) * root_abs
    return jnp.where(xx < 0, jnp.nan, root_abs)


@partial(jax.jit, static_argnames=())
def arb_cbrt_point(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.power(jnp.abs(x), 1.0 / 3.0)


@partial(jax.jit, static_argnames=())
def arb_pow_fmpz_point(x: jax.Array, n: jax.Array | int) -> jax.Array:
    return jnp.power(x, jnp.asarray(n))


@partial(jax.jit, static_argnames=())
def arb_pow_fmpq_point(x: jax.Array, p: jax.Array, q: jax.Array) -> jax.Array:
    return jnp.power(x, jnp.asarray(p) / jnp.asarray(q))


@partial(jax.jit, static_argnames=("k",))
def arb_root_point(x: jax.Array, k: int) -> jax.Array:
    return arb_root_ui_point(x, k)


@partial(jax.jit, static_argnames=())
def arb_lgamma_point(x: jax.Array) -> jax.Array:
    return lax.lgamma(x)


@partial(jax.jit, static_argnames=())
def arb_rgamma_point(x: jax.Array) -> jax.Array:
    return jnp.exp(-lax.lgamma(x))


@partial(jax.jit, static_argnames=())
def acb_abs_point(z: jax.Array) -> jax.Array:
    return jnp.abs(z)


@partial(jax.jit, static_argnames=())
def acb_add_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


@partial(jax.jit, static_argnames=())
def acb_sub_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x - y


@partial(jax.jit, static_argnames=())
def acb_mul_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@partial(jax.jit, static_argnames=())
def acb_div_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return x / y


@partial(jax.jit, static_argnames=())
def acb_inv_point(x: jax.Array) -> jax.Array:
    return 1.0 / x


@partial(jax.jit, static_argnames=())
def acb_fma_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x * y + z


@partial(jax.jit, static_argnames=())
def acb_log1p_point(x: jax.Array) -> jax.Array:
    return jnp.log1p(x)


@partial(jax.jit, static_argnames=())
def acb_expm1_point(x: jax.Array) -> jax.Array:
    return jnp.expm1(x)


@partial(jax.jit, static_argnames=())
def acb_sin_cos_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sin(x), jnp.cos(x)


@partial(jax.jit, static_argnames=())
def acb_asin_point(x: jax.Array) -> jax.Array:
    return jnp.arcsin(x)


@partial(jax.jit, static_argnames=())
def acb_acos_point(x: jax.Array) -> jax.Array:
    return jnp.arccos(x)


@partial(jax.jit, static_argnames=())
def acb_atan_point(x: jax.Array) -> jax.Array:
    return jnp.arctan(x)


@partial(jax.jit, static_argnames=())
def acb_asinh_point(x: jax.Array) -> jax.Array:
    return jnp.arcsinh(x)


@partial(jax.jit, static_argnames=())
def acb_acosh_point(x: jax.Array) -> jax.Array:
    return jnp.arccosh(x)


@partial(jax.jit, static_argnames=())
def acb_atanh_point(x: jax.Array) -> jax.Array:
    return jnp.arctanh(x)


@partial(jax.jit, static_argnames=())
def acb_exp_point(x: jax.Array) -> jax.Array:
    return jnp.exp(x)


@partial(jax.jit, static_argnames=())
def acb_log_point(x: jax.Array) -> jax.Array:
    return jnp.log(x)


@partial(jax.jit, static_argnames=())
def acb_sqrt_point(x: jax.Array) -> jax.Array:
    return jnp.sqrt(x)


@partial(jax.jit, static_argnames=())
def acb_rsqrt_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.sqrt(x)


@partial(jax.jit, static_argnames=())
def acb_sin_point(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@partial(jax.jit, static_argnames=())
def acb_cos_point(x: jax.Array) -> jax.Array:
    return jnp.cos(x)


@partial(jax.jit, static_argnames=())
def acb_tan_point(x: jax.Array) -> jax.Array:
    return jnp.tan(x)


@partial(jax.jit, static_argnames=())
def acb_cot_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.tan(x)


@partial(jax.jit, static_argnames=())
def acb_sinh_point(x: jax.Array) -> jax.Array:
    return jnp.sinh(x)


@partial(jax.jit, static_argnames=())
def acb_cosh_point(x: jax.Array) -> jax.Array:
    return jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def acb_tanh_point(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


@partial(jax.jit, static_argnames=())
def acb_sech_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.cosh(x)


@partial(jax.jit, static_argnames=())
def acb_csch_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.sinh(x)


@partial(jax.jit, static_argnames=())
def acb_sin_pi_point(x: jax.Array) -> jax.Array:
    return jnp.sin(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_cos_pi_point(x: jax.Array) -> jax.Array:
    return jnp.cos(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_sin_cos_pi_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sin(el.pi_like(x) * x), jnp.cos(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_tan_pi_point(x: jax.Array) -> jax.Array:
    return jnp.tan(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_cot_pi_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.tan(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_csc_pi_point(x: jax.Array) -> jax.Array:
    return 1.0 / jnp.sin(el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_sinc_point(x: jax.Array) -> jax.Array:
    return jnp.where(x == 0.0, 1.0 + 0.0j, jnp.sin(x) / x)


@partial(jax.jit, static_argnames=())
def acb_sinc_pi_point(x: jax.Array) -> jax.Array:
    pix = el.pi_like(x) * x
    return jnp.where(x == 0.0, 1.0 + 0.0j, jnp.sin(pix) / pix)


@partial(jax.jit, static_argnames=())
def acb_exp_pi_i_point(x: jax.Array) -> jax.Array:
    return jnp.exp(1j * el.pi_like(x) * x)


@partial(jax.jit, static_argnames=())
def acb_exp_invexp_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    ex = jnp.exp(x)
    return ex, 1.0 / ex


@partial(jax.jit, static_argnames=())
def acb_addmul_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x + y * z


@partial(jax.jit, static_argnames=())
def acb_submul_point(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return x - y * z


@partial(jax.jit, static_argnames=())
def acb_pow_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(x, y)


@partial(jax.jit, static_argnames=())
def acb_pow_arb_point(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(x, y)


@partial(jax.jit, static_argnames=("n",))
def acb_pow_ui_point(x: jax.Array, n: int) -> jax.Array:
    return jnp.power(x, n)


@partial(jax.jit, static_argnames=("n",))
def acb_pow_si_point(x: jax.Array, n: int) -> jax.Array:
    return jnp.power(x, n)


@partial(jax.jit, static_argnames=())
def acb_pow_fmpz_point(x: jax.Array, n: int | jax.Array) -> jax.Array:
    return jnp.power(x, jnp.asarray(n))


@partial(jax.jit, static_argnames=())
def acb_sqr_point(x: jax.Array) -> jax.Array:
    return x * x


@partial(jax.jit, static_argnames=("k",))
def acb_root_ui_point(x: jax.Array, k: int) -> jax.Array:
    return jnp.power(x, 1.0 / jnp.asarray(k, dtype=jnp.asarray(x).real.dtype))


@partial(jax.jit, static_argnames=())
def acb_gamma_point(x: jax.Array) -> jax.Array:
    return _vmap_complex_scalar(lambda t: jnp.exp(acb_core._complex_loggamma(t)))(x)


@partial(jax.jit, static_argnames=())
def acb_rgamma_point(x: jax.Array) -> jax.Array:
    return _vmap_complex_scalar(lambda t: jnp.exp(-acb_core._complex_loggamma(t)))(x)


@partial(jax.jit, static_argnames=())
def acb_lgamma_point(x: jax.Array) -> jax.Array:
    return _vmap_complex_scalar(acb_core._complex_loggamma)(x)


@partial(jax.jit, static_argnames=())
def acb_log_sin_pi_point(x: jax.Array) -> jax.Array:
    return jnp.log(jnp.sin(el.pi_like(x) * x))


acb_digamma_point = _scalarize_unary_complex(acb_core.acb_digamma)
acb_zeta_point = _scalarize_unary_complex(acb_core.acb_zeta)


@jax.jit
def acb_hurwitz_zeta_point(s: jax.Array, a: jax.Array) -> jax.Array:
    ss = jnp.ravel(jnp.asarray(s, dtype=jnp.complex128))
    aa = jnp.ravel(jnp.asarray(a, dtype=jnp.complex128))
    out = jax.vmap(lambda x, y: acb_core.acb_midpoint(acb_core.acb_hurwitz_zeta(_point_box(x), _point_box(y))))(ss, aa)
    return out.reshape(jnp.shape(s))


@partial(jax.jit, static_argnames=("n",))
def acb_polygamma_point(n: int, x: jax.Array) -> jax.Array:
    flat = jnp.ravel(jnp.asarray(x, dtype=jnp.complex128))
    out = jax.vmap(lambda t: acb_core.acb_midpoint(acb_core.acb_polygamma(n, _point_box(t))))(flat)
    return out.reshape(jnp.shape(x))


@partial(jax.jit, static_argnames=("n",))
def acb_bernoulli_poly_ui_point(n: int, x: jax.Array) -> jax.Array:
    flat = jnp.ravel(jnp.asarray(x, dtype=jnp.complex128))
    out = jax.vmap(lambda t: acb_core.acb_midpoint(acb_core.acb_bernoulli_poly_ui(n, _point_box(t))))(flat)
    return out.reshape(jnp.shape(x))


acb_polylog_point = _scalarize_binary_complex(acb_core.acb_polylog)


@partial(jax.jit, static_argnames=("s",))
def acb_polylog_si_point(s: int, z: jax.Array) -> jax.Array:
    flat = jnp.ravel(jnp.asarray(z, dtype=jnp.complex128))
    out = jax.vmap(lambda t: acb_core.acb_midpoint(acb_core.acb_polylog_si(s, _point_box(t))))(flat)
    return out.reshape(jnp.shape(z))


acb_agm_point = _scalarize_binary_complex(acb_core.acb_agm)
acb_agm1_point = _scalarize_unary_complex(acb_core.acb_agm1)
acb_agm1_cpx_point = _scalarize_unary_complex(acb_core.acb_agm1_cpx)


@partial(jax.jit, static_argnames=())
def arb_gamma_point(x: jax.Array) -> jax.Array:
    return jnp.exp(lax.lgamma(x))


@partial(jax.jit, static_argnames=())
def arb_erf_point(x: jax.Array) -> jax.Array:
    return hypgeom._real_erf_series(x)


@partial(jax.jit, static_argnames=())
def arb_erfc_point(x: jax.Array) -> jax.Array:
    return 1.0 - hypgeom._real_erf_series(x)


@partial(jax.jit, static_argnames=())
def arb_bessel_j_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_j(nu, z)


@partial(jax.jit, static_argnames=())
def arb_bessel_y_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_y(nu, z)


@partial(jax.jit, static_argnames=())
def arb_bessel_i_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_i(nu, z)


@partial(jax.jit, static_argnames=())
def arb_bessel_k_point(nu: jax.Array, z: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_k(nu, z)


__all__ = [
    "arb_exp_point",
    "arb_log_point",
    "arb_sqrt_point",
    "arb_sin_point",
    "arb_cos_point",
    "arb_tan_point",
    "arb_sinh_point",
    "arb_cosh_point",
    "arb_tanh_point",
    "arb_abs_point",
    "arb_add_point",
    "arb_sub_point",
    "arb_mul_point",
    "arb_div_point",
    "arb_inv_point",
    "arb_fma_point",
    "arb_log1p_point",
    "arb_expm1_point",
    "arb_sin_cos_point",
    "acb_abs_point",
    "acb_add_point",
    "acb_sub_point",
    "acb_mul_point",
    "acb_div_point",
    "acb_inv_point",
    "acb_fma_point",
    "acb_log1p_point",
    "acb_expm1_point",
    "acb_sin_cos_point",
    "arb_gamma_point",
    "arb_erf_point",
    "arb_erfc_point",
    "arb_bessel_j_point",
    "arb_bessel_y_point",
    "arb_bessel_i_point",
    "arb_bessel_k_point",
]

__all__.extend(
    [
        "arb_sinh_cosh_point",
        "arb_sin_pi_point",
        "arb_cos_pi_point",
        "arb_tan_pi_point",
        "arb_sinc_point",
        "arb_sinc_pi_point",
        "arb_asin_point",
        "arb_acos_point",
        "arb_atan_point",
        "arb_asinh_point",
        "arb_acosh_point",
        "arb_atanh_point",
        "arb_sign_point",
        "arb_pow_point",
        "arb_pow_ui_point",
        "arb_pow_fmpz_point",
        "arb_pow_fmpq_point",
        "arb_root_ui_point",
        "arb_root_point",
        "arb_cbrt_point",
        "arb_lgamma_point",
        "arb_rgamma_point",
        "acb_exp_point",
        "acb_log_point",
        "acb_sqrt_point",
        "acb_rsqrt_point",
        "acb_sin_point",
        "acb_cos_point",
        "acb_tan_point",
        "acb_cot_point",
        "acb_sinh_point",
        "acb_cosh_point",
        "acb_tanh_point",
        "acb_asin_point",
        "acb_acos_point",
        "acb_atan_point",
        "acb_asinh_point",
        "acb_acosh_point",
        "acb_atanh_point",
        "acb_sech_point",
        "acb_csch_point",
        "acb_sin_pi_point",
        "acb_cos_pi_point",
        "acb_sin_cos_pi_point",
        "acb_tan_pi_point",
        "acb_cot_pi_point",
        "acb_csc_pi_point",
        "acb_sinc_point",
        "acb_sinc_pi_point",
        "acb_exp_pi_i_point",
        "acb_exp_invexp_point",
        "acb_addmul_point",
        "acb_submul_point",
        "acb_pow_point",
        "acb_pow_arb_point",
        "acb_pow_ui_point",
        "acb_pow_si_point",
        "acb_pow_fmpz_point",
        "acb_sqr_point",
        "acb_root_ui_point",
        "acb_gamma_point",
        "acb_rgamma_point",
        "acb_lgamma_point",
        "acb_log_sin_pi_point",
        "acb_digamma_point",
        "acb_zeta_point",
        "acb_hurwitz_zeta_point",
        "acb_polygamma_point",
        "acb_bernoulli_poly_ui_point",
        "acb_polylog_point",
        "acb_polylog_si_point",
        "acb_agm_point",
        "acb_agm1_point",
        "acb_agm1_cpx_point",
    ]
)
