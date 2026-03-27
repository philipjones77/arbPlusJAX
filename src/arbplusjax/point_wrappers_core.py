from __future__ import annotations
from functools import partial
import importlib

import jax
import jax.numpy as jnp
from jax import lax

from . import acb_core
from . import checks
from . import double_interval as di
from . import elementary as el
from . import kernel_helpers as kh
from .lazy_imports import lazy_module_proxy
from .kernel_helpers import scalarize_binary_complex, scalarize_unary_complex, vmap_complex_scalar

hypgeom = lazy_module_proxy("hypgeom", package=__package__)

_LAZY_FAMILY_EXPORTS = {
    "acb_dirichlet_zeta_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_zeta_point"),
    "acb_dirichlet_eta_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_eta_point"),
    "acb_modular_j_point": ("point_wrappers_dirichlet_modular", "acb_modular_j_point"),
    "acb_dirichlet_zeta_batch_fixed_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_zeta_batch_fixed_point"),
    "acb_dirichlet_zeta_batch_padded_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_zeta_batch_padded_point"),
    "acb_dirichlet_eta_batch_fixed_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_eta_batch_fixed_point"),
    "acb_dirichlet_eta_batch_padded_point": ("point_wrappers_dirichlet_modular", "acb_dirichlet_eta_batch_padded_point"),
    "acb_modular_j_batch_fixed_point": ("point_wrappers_dirichlet_modular", "acb_modular_j_batch_fixed_point"),
    "acb_modular_j_batch_padded_point": ("point_wrappers_dirichlet_modular", "acb_modular_j_batch_padded_point"),
    "acb_elliptic_k_point": ("point_wrappers_elliptic", "acb_elliptic_k_point"),
    "acb_elliptic_e_point": ("point_wrappers_elliptic", "acb_elliptic_e_point"),
    "acb_elliptic_k_batch_fixed_point": ("point_wrappers_elliptic", "acb_elliptic_k_batch_fixed_point"),
    "acb_elliptic_k_batch_padded_point": ("point_wrappers_elliptic", "acb_elliptic_k_batch_padded_point"),
    "acb_elliptic_e_batch_fixed_point": ("point_wrappers_elliptic", "acb_elliptic_e_batch_fixed_point"),
    "acb_elliptic_e_batch_padded_point": ("point_wrappers_elliptic", "acb_elliptic_e_batch_padded_point"),
}



# Point-only kernels (no interval or outward rounding)


def _broadcast_flatten(*args):
    arrs = [jnp.asarray(arg) for arg in args]
    bcast = jnp.broadcast_arrays(*arrs)
    shape = bcast[0].shape
    return tuple(jnp.ravel(a) for a in bcast), shape


def _vectorize_real_scalar(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out = jax.vmap(fn)(*flats)
    return out.reshape(shape)


def _vectorize_complex_scalar(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out = jax.vmap(fn)(*flats)
    return out.reshape(shape)


def _vectorize_real_scalar_tuple2(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out1, out2 = jax.vmap(fn)(*flats)
    return out1.reshape(shape), out2.reshape(shape)


def _vectorize_complex_scalar_tuple2(fn, *args):
    flats, shape = _broadcast_flatten(*args)
    out1, out2 = jax.vmap(fn)(*flats)
    return out1.reshape(shape), out2.reshape(shape)


def _pad_args_repeat_last(args, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)


def _fixed_unary_point(fn, x: jax.Array, **kwargs):
    return fn(x, **kwargs)


def _padded_unary_point(fn, x: jax.Array, *, pad_to: int, **kwargs):
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return fn(*call_args, **kwargs)




def _real_laguerre_l_scalar(n: int, m: jax.Array, x: jax.Array) -> jax.Array:
    def body(k, acc):
        kf = jnp.asarray(k, dtype=jnp.asarray(x).dtype)
        coeff = jnp.exp(
            hypgeom._gammaln_real(n + m + 1.0)
            - hypgeom._gammaln_real(n - k + 1.0)
            - hypgeom._gammaln_real(m + k + 1.0)
        )
        term = coeff * jnp.power(-x, kf) / jnp.exp(hypgeom._gammaln_real(kf + 1.0))
        return acc + term

    return lax.fori_loop(0, n + 1, body, jnp.asarray(0.0, dtype=jnp.asarray(x).dtype))


def _real_hermite_h_scalar(n: int, x: jax.Array) -> jax.Array:
    if n == 0:
        return jnp.asarray(1.0, dtype=jnp.asarray(x).dtype)
    if n == 1:
        return jnp.asarray(2.0, dtype=jnp.asarray(x).dtype) * x
    h0 = jnp.asarray(1.0, dtype=jnp.asarray(x).dtype)
    h1 = jnp.asarray(2.0, dtype=jnp.asarray(x).dtype) * x

    def body(k, state):
        h_prev, h_curr = state
        h_next = 2.0 * x * h_curr - 2.0 * jnp.asarray(k - 1, dtype=jnp.asarray(x).dtype) * h_prev
        return h_curr, h_next

    _, hn = lax.fori_loop(2, n + 1, body, (h0, h1))
    return hn


def _real_pfq_scalar(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool, n_terms: int) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    z = jnp.asarray(z, dtype=jnp.result_type(z, jnp.float64))

    def body(k, state):
        term, s = state
        k1 = jnp.asarray(k + 1, dtype=z.dtype)
        num = jnp.prod(a + k) if a.size else jnp.asarray(1.0, dtype=z.dtype)
        den = jnp.prod(b + k) if b.size else jnp.asarray(1.0, dtype=z.dtype)
        term = term * (num / den) * (z / k1)
        return term, s + term

    term0 = jnp.asarray(1.0, dtype=z.dtype)
    _, out = lax.fori_loop(0, n_terms - 1, body, (term0, term0))
    return jnp.where(reciprocal, 1.0 / out, out)


def _complex_pfq_scalar(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool, n_terms: int) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    z = jnp.asarray(z, dtype=jnp.result_type(z, jnp.complex128))

    def body(k, state):
        term, s = state
        k1 = jnp.asarray(k + 1, dtype=z.real.dtype)
        num = jnp.prod(a + k) if a.size else jnp.asarray(1.0 + 0.0j, dtype=z.dtype)
        den = jnp.prod(b + k) if b.size else jnp.asarray(1.0 + 0.0j, dtype=z.dtype)
        term = term * (num / den) * (z / k1)
        return term, s + term

    term0 = jnp.asarray(1.0 + 0.0j, dtype=z.dtype)
    _, out = lax.fori_loop(0, n_terms - 1, body, (term0, term0))
    return jnp.where(reciprocal, 1.0 / out, out)


def _complex_digamma_scalar(z: jax.Array) -> jax.Array:
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    h = jnp.asarray(1e-6 + 0.0j, dtype=zz.dtype)
    two = jnp.asarray(2.0, dtype=real_dtype)
    return (acb_core._complex_loggamma(zz + h) - acb_core._complex_loggamma(zz - h)) / (two * h)


def _complex_zeta_scalar(s: jax.Array, n_terms: int = 64) -> jax.Array:
    ss = el.as_complex(s)
    real_dtype = el.real_dtype_from_complex_dtype(ss.dtype)
    n = jnp.arange(1, n_terms + 1, dtype=real_dtype)
    return jnp.sum(jnp.exp(-ss * jnp.log(n)))


def _complex_hurwitz_zeta_scalar(
    s: jax.Array,
    a: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    ss = el.as_complex(s)
    aa = el.as_complex(a)
    real_dtype = el.real_dtype_from_complex_dtype(ss.dtype)
    re_s = jnp.real(ss)
    eps = jnp.asarray(1e-12, dtype=real_dtype)
    tail_target = eps * jnp.maximum(re_s - 1.0, jnp.asarray(1e-12, dtype=real_dtype))
    base = jnp.power(tail_target, 1.0 / jnp.maximum(1.0 - re_s, jnp.asarray(1e-12, dtype=real_dtype)))
    n_est = jnp.ceil(base + 1.0)
    n_eff = jnp.where(re_s > 1.1, n_est, jnp.asarray(terms, dtype=real_dtype))
    n_eff = jnp.clip(n_eff, jnp.asarray(min_terms, dtype=real_dtype), jnp.asarray(max_terms, dtype=real_dtype))
    ks = jnp.arange(max_terms, dtype=real_dtype)
    mask = ks < n_eff
    terms_arr = jnp.power(aa + ks, -ss)
    return jnp.sum(jnp.where(mask, terms_arr, jnp.zeros_like(terms_arr)))


def _complex_polygamma_scalar(
    m: int,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    if m == 0:
        return _complex_digamma_scalar(zz)
    re_z = jnp.real(zz)
    m_float = jnp.asarray(float(m), dtype=real_dtype)
    eps = jnp.asarray(1e-12, dtype=real_dtype)
    tail_target = eps * jnp.maximum(m_float, jnp.asarray(1.0, dtype=real_dtype))
    base = jnp.power(tail_target, -1.0 / jnp.maximum(m_float, eps))
    n_est = jnp.ceil(base - re_z)
    n_eff = jnp.where(m_float > 0, n_est, jnp.asarray(terms, dtype=real_dtype))
    n_eff = jnp.clip(n_eff, jnp.asarray(min_terms, dtype=real_dtype), jnp.asarray(max_terms, dtype=real_dtype))
    ks = jnp.arange(max_terms, dtype=real_dtype)
    mask = ks < n_eff
    factorial = jnp.exp(lax.lgamma(m_float + 1.0))
    series_terms = jnp.power(zz + ks, -(m_float + 1.0))
    series = jnp.sum(jnp.where(mask, series_terms, jnp.zeros_like(series_terms)))
    sign = -1.0 if (m + 1) % 2 else 1.0
    return jnp.asarray(sign, dtype=real_dtype) * factorial * series


def _complex_bernoulli_poly_ui_scalar(n: int, z: jax.Array) -> jax.Array:
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    if n == 0:
        return jnp.asarray(1.0 + 0.0j, dtype=zz.dtype)
    if n == 1:
        return zz - jnp.asarray(0.5, dtype=real_dtype)
    if n == 2:
        return zz * zz - zz + jnp.asarray(1.0 / 6.0, dtype=real_dtype)
    if n == 3:
        return zz * zz * zz - jnp.asarray(1.5, dtype=real_dtype) * zz * zz + jnp.asarray(0.5, dtype=real_dtype) * zz
    if n == 4:
        return zz**4 - jnp.asarray(2.0, dtype=real_dtype) * zz**3 + zz * zz - jnp.asarray(1.0 / 30.0, dtype=real_dtype)
    return jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=zz.dtype)


def _complex_polylog_scalar(
    s: jax.Array,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    ss = el.as_complex(s)
    zz = el.as_complex(z)
    real_dtype = el.real_dtype_from_complex_dtype(zz.dtype)
    absz = jnp.abs(zz)
    eps = jnp.asarray(1e-12, dtype=real_dtype)
    base = jnp.ceil(jnp.asarray(8.0, dtype=real_dtype) / jnp.maximum(jnp.asarray(1.0, dtype=real_dtype) - absz, eps))
    n_eff = jnp.where(absz < 1.0, base, jnp.asarray(terms, dtype=real_dtype))
    n_eff = jnp.clip(n_eff, jnp.asarray(min_terms, dtype=real_dtype), jnp.asarray(max_terms, dtype=real_dtype))
    ks = jnp.arange(1, max_terms + 1, dtype=real_dtype)
    mask = ks <= n_eff
    series_terms = jnp.power(zz, ks) / jnp.power(ks, ss)
    series = jnp.sum(jnp.where(mask, series_terms, jnp.zeros_like(series_terms)))
    nanv = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=zz.dtype)
    return jnp.where(absz < 1.0, series, nanv)


def _complex_polylog_si_scalar(s: int, z: jax.Array, terms: int = 64, max_terms: int = 512, min_terms: int = 32) -> jax.Array:
    zz = el.as_complex(z)
    sval = jnp.asarray(float(s) + 0.0j, dtype=zz.dtype)
    return _complex_polylog_scalar(sval, zz, terms=terms, max_terms=max_terms, min_terms=min_terms)


def _complex_agm_scalar(a: jax.Array, b: jax.Array, iters: int = 10) -> jax.Array:
    aa = el.as_complex(a)
    bb = el.as_complex(b)

    def body(_, state):
        x, y = state
        return (0.5 * (x + y), jnp.sqrt(x * y))

    out, _ = lax.fori_loop(0, iters, body, (aa, bb))
    return out


def _pad_point_batch_last(args, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)

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
    return el.sin_pi(x)


@partial(jax.jit, static_argnames=())
def arb_cos_pi_point(x: jax.Array) -> jax.Array:
    return el.cos_pi(x)


@partial(jax.jit, static_argnames=())
def arb_tan_pi_point(x: jax.Array) -> jax.Array:
    return el.tan_pi(x)


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
    return el.sin_pi(x)


@partial(jax.jit, static_argnames=())
def acb_cos_pi_point(x: jax.Array) -> jax.Array:
    return el.cos_pi(x)


@partial(jax.jit, static_argnames=())
def acb_sin_cos_pi_point(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return el.sin_pi(x), el.cos_pi(x)


@partial(jax.jit, static_argnames=())
def acb_tan_pi_point(x: jax.Array) -> jax.Array:
    return el.tan_pi(x)


@partial(jax.jit, static_argnames=())
def acb_cot_pi_point(x: jax.Array) -> jax.Array:
    return 1.0 / el.tan_pi(x)


@partial(jax.jit, static_argnames=())
def acb_csc_pi_point(x: jax.Array) -> jax.Array:
    return 1.0 / el.sin_pi(x)


@partial(jax.jit, static_argnames=())
def acb_sinc_point(x: jax.Array) -> jax.Array:
    return jnp.where(x == 0.0, 1.0 + 0.0j, jnp.sin(x) / x)


@partial(jax.jit, static_argnames=())
def acb_sinc_pi_point(x: jax.Array) -> jax.Array:
    return el.sinc_pi(x)


@partial(jax.jit, static_argnames=())
def acb_exp_pi_i_point(x: jax.Array) -> jax.Array:
    return el.exp_pi_i(x)


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
    return vmap_complex_scalar(lambda t: jnp.exp(acb_core._complex_loggamma(t)))(x)


@partial(jax.jit, static_argnames=())
def acb_rgamma_point(x: jax.Array) -> jax.Array:
    return vmap_complex_scalar(lambda t: jnp.exp(-acb_core._complex_loggamma(t)))(x)


@partial(jax.jit, static_argnames=())
def acb_lgamma_point(x: jax.Array) -> jax.Array:
    return vmap_complex_scalar(acb_core._complex_loggamma)(x)


@partial(jax.jit, static_argnames=())
def acb_log_sin_pi_point(x: jax.Array) -> jax.Array:
    return el.log_sin_pi(x)


acb_digamma_point = scalarize_unary_complex(_complex_digamma_scalar)
acb_zeta_point = scalarize_unary_complex(_complex_zeta_scalar)


@jax.jit
def acb_hurwitz_zeta_point(s: jax.Array, a: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(_complex_hurwitz_zeta_scalar, s, a)


@partial(jax.jit, static_argnames=("n",))
def acb_polygamma_point(n: int, x: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(partial(_complex_polygamma_scalar, n), x)


@partial(jax.jit, static_argnames=("n",))
def acb_bernoulli_poly_ui_point(n: int, x: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(partial(_complex_bernoulli_poly_ui_scalar, n), x)


acb_polylog_point = scalarize_binary_complex(_complex_polylog_scalar)


@partial(jax.jit, static_argnames=("s",))
def acb_polylog_si_point(s: int, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(partial(_complex_polylog_si_scalar, s), z)


acb_agm_point = scalarize_binary_complex(_complex_agm_scalar)
acb_agm1_point = scalarize_unary_complex(lambda x: _complex_agm_scalar(jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(x).dtype), x))
acb_agm1_cpx_point = scalarize_unary_complex(lambda x: _complex_agm_scalar(jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(x).dtype), x))


def __getattr__(name: str):
    if name.startswith(("arb_hypgeom_", "acb_hypgeom_")):
        mod = importlib.import_module(".point_wrappers_hypgeom", package=__package__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    entry = _LAZY_FAMILY_EXPORTS.get(name)
    if entry is None:
        raise AttributeError(name)
    module_name, attr_name = entry
    mod = importlib.import_module(f".{module_name}", package=__package__)
    value = getattr(mod, attr_name)
    globals()[name] = value
    return value


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

__all__ = sorted(
    name
    for name, value in globals().items()
    if not name.startswith('_') and callable(value)
)
