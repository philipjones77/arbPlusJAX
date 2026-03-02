from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import hypgeom
from . import elementary as el

jax.config.update("jax_enable_x64", True)


# Point-only kernels (no interval or outward rounding)

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
    root_abs = jnp.power(jnp.abs(xx), 1.0 / jnp.float64(k))
    if (k % 2) == 1:
        return jnp.sign(xx) * root_abs
    return jnp.where(xx < 0, jnp.nan, root_abs)


@partial(jax.jit, static_argnames=())
def arb_cbrt_point(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.power(jnp.abs(x), 1.0 / 3.0)


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
def arb_gamma_point(x: jax.Array) -> jax.Array:
    val = jnp.exp(hypgeom._complex_loggamma(x))
    if jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating):
        return val
    return jnp.real(val)


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
        "arb_root_ui_point",
        "arb_cbrt_point",
        "acb_asin_point",
        "acb_acos_point",
        "acb_atan_point",
        "acb_asinh_point",
        "acb_acosh_point",
        "acb_atanh_point",
    ]
)
