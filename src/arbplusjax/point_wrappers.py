from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import acb_core
from . import double_interval as di
from . import hypgeom
from . import elementary as el
from .kernel_helpers import point_box, scalarize_binary_complex, scalarize_unary_complex, vmap_complex_scalar

jax.config.update("jax_enable_x64", True)


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


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_0f1_point(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda aa, zz: hypgeom._real_hyp0f1_scalar(aa, zz), a, z)
    if regularized:
        out = out * arb_rgamma_point(a)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_1f1_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda aa, bb, zz: hypgeom._real_hyp1f1_regime(aa, bb, zz), a, b, z)
    if regularized:
        out = out * arb_rgamma_point(b)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_m_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1_point(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_2f1_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda aa, bb, cc, zz: hypgeom._real_hyp2f1_regime(aa, bb, cc, zz), a, b, c, z)
    if regularized:
        out = out * arb_rgamma_point(c)
    return out


@partial(jax.jit, static_argnames=())
def arb_hypgeom_u_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda aa, bb, zz: hypgeom._real_hypu_regime(aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_gamma_lower_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda ss, zz: hypgeom._gammainc_real(ss, zz), s, z)
    if not regularized:
        out = out * hypgeom._gamma_real(s)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_gamma_upper_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(lambda ss, zz: hypgeom._gammaincc_real(ss, zz), s, z)
    if not regularized:
        out = out * hypgeom._gamma_real(s)
    return out


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_legendre_p_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_real_scalar(lambda mm, zz: hypgeom._real_legendre_p_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_legendre_q_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_real_scalar(lambda mm, zz: hypgeom._real_legendre_q_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_jacobi_p_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda aa, bb, zz: hypgeom._real_jacobi_p_scalar(n, aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_gegenbauer_c_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda ll, zz: hypgeom._real_gegenbauer_c_scalar(n, ll, zz), lam, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_chebyshev_t_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    nf = jnp.asarray(n, dtype=x.dtype)
    return lax.cond(jnp.all(jnp.abs(x) <= 1.0), lambda t: jnp.cos(nf * jnp.arccos(t)), lambda t: jnp.cosh(nf * jnp.arccosh(t)), x)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_chebyshev_u_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    nf = jnp.asarray(n + 1, dtype=x.dtype)
    def in_range(t):
        ang = jnp.arccos(t)
        return jnp.sin(nf * ang) / jnp.sin(ang)
    def out_range(t):
        ach = jnp.arccosh(jnp.abs(t))
        return jnp.sinh(nf * ach) / jnp.sinh(ach)
    return lax.cond(jnp.all(jnp.abs(x) <= 1.0), in_range, out_range, x)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_laguerre_l_point(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda mm, zz: _real_laguerre_l_scalar(n, mm, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_hermite_h_point(n: int, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda zz: _real_hermite_h_scalar(n, zz), z)


@partial(jax.jit, static_argnames=("reciprocal", "n_terms"))
def arb_hypgeom_pfq_point(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    z_arr = jnp.asarray(z)
    if a_arr.ndim <= 1 and b_arr.ndim <= 1 and z_arr.ndim == 0:
        return _real_pfq_scalar(a_arr, b_arr, z_arr, reciprocal=reciprocal, n_terms=n_terms)
    if z_arr.ndim == 0:
        z_arr = jnp.broadcast_to(z_arr, (a_arr.shape[0],))
    return jax.vmap(lambda aa, bb, zz: _real_pfq_scalar(aa, bb, zz, reciprocal=reciprocal, n_terms=n_terms))(a_arr, b_arr, z_arr)


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


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_0f1_point(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda aa, zz: hypgeom._complex_hyp0f1_scalar(aa, zz), a, z)
    if regularized:
        out = out * acb_rgamma_point(a)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_1f1_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda aa, bb, zz: hypgeom._complex_hyp1f1_regime(aa, bb, zz), a, b, z)
    if regularized:
        out = out * acb_rgamma_point(b)
    return out


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_m_point(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1_point(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_2f1_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda aa, bb, cc, zz: hypgeom._complex_hyp2f1_regime(aa, bb, cc, zz), a, b, c, z)
    if regularized:
        out = out * acb_rgamma_point(c)
    return out


@partial(jax.jit, static_argnames=())
def acb_hypgeom_u_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda aa, bb, zz: hypgeom._complex_hypu_regime(aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_gamma_lower_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda ss, zz: hypgeom._complex_gamma_lower_scalar(ss, zz), s, z)
    if regularized:
        out = out / jnp.exp(acb_lgamma_point(s))
    return out


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_gamma_upper_point(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(lambda ss, zz: hypgeom._complex_gamma_upper_scalar(ss, zz), s, z)
    if regularized:
        out = out / jnp.exp(acb_lgamma_point(s))
    return out


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_legendre_p_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_complex_scalar(lambda mm, zz: hypgeom._complex_legendre_p_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_legendre_q_point(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    del type
    return _vectorize_complex_scalar(lambda mm, zz: hypgeom._complex_legendre_q_scalar(n, zz), m, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_jacobi_p_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda aa, bb, zz: hypgeom._complex_jacobi_p_scalar(n, aa, bb, zz), a, b, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_gegenbauer_c_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda ll, zz: hypgeom._complex_gegenbauer_c_scalar(n, ll, zz), lam, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_chebyshev_t_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    nf = jnp.asarray(n, dtype=x.real.dtype)
    return jnp.cos(nf * jnp.arccos(x))


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_chebyshev_u_point(n: int, z: jax.Array) -> jax.Array:
    x = jnp.asarray(z)
    theta = jnp.arccos(x)
    nf = jnp.asarray(n + 1, dtype=x.real.dtype)
    return jnp.sin(nf * theta) / jnp.sin(theta)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_laguerre_l_point(n: int, a: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda aa, zz: jnp.exp(hypgeom._complex_loggamma(n + aa + 1.0) - hypgeom._complex_loggamma(n + 1.0) - hypgeom._complex_loggamma(aa + 1.0)) * hypgeom._complex_hyp1f1_scalar(-jnp.asarray(n, dtype=zz.real.dtype), aa + 1.0, zz), a, z)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_hermite_h_point(n: int, z: jax.Array) -> jax.Array:
    def scalar(w):
        if n == 0:
            return jnp.asarray(1.0 + 0.0j, dtype=jnp.asarray(w).dtype)
        if n == 1:
            return 2.0 * w
        h0 = jnp.asarray(1.0 + 0.0j, dtype=jnp.asarray(w).dtype)
        h1 = 2.0 * w

        def body(k, state):
            h_prev, h_curr = state
            h_next = 2.0 * w * h_curr - 2.0 * jnp.asarray(k - 1, dtype=jnp.asarray(w).real.dtype) * h_prev
            return h_curr, h_next

        _, hn = lax.fori_loop(2, n + 1, body, (h0, h1))
        return hn

    return _vectorize_complex_scalar(scalar, z)


@partial(jax.jit, static_argnames=("reciprocal", "n_terms"))
def acb_hypgeom_pfq_point(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    z_arr = jnp.asarray(z)
    if a_arr.ndim <= 1 and b_arr.ndim <= 1 and z_arr.ndim == 0:
        return _complex_pfq_scalar(a_arr, b_arr, z_arr, reciprocal=reciprocal, n_terms=n_terms)
    if z_arr.ndim == 0:
        z_arr = jnp.broadcast_to(z_arr, (a_arr.shape[0],))
    return jax.vmap(lambda aa, bb, zz: _complex_pfq_scalar(aa, bb, zz, reciprocal=reciprocal, n_terms=n_terms))(a_arr, b_arr, z_arr)


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


acb_digamma_point = scalarize_unary_complex(acb_core.acb_digamma)
acb_zeta_point = scalarize_unary_complex(acb_core.acb_zeta)


@jax.jit
def acb_hurwitz_zeta_point(s: jax.Array, a: jax.Array) -> jax.Array:
    ss = jnp.ravel(jnp.asarray(s, dtype=jnp.complex128))
    aa = jnp.ravel(jnp.asarray(a, dtype=jnp.complex128))
    out = jax.vmap(lambda x, y: acb_core.acb_midpoint(acb_core.acb_hurwitz_zeta(point_box(x), point_box(y))))(ss, aa)
    return out.reshape(jnp.shape(s))


@partial(jax.jit, static_argnames=("n",))
def acb_polygamma_point(n: int, x: jax.Array) -> jax.Array:
    flat = jnp.ravel(jnp.asarray(x, dtype=jnp.complex128))
    out = jax.vmap(lambda t: acb_core.acb_midpoint(acb_core.acb_polygamma(n, point_box(t))))(flat)
    return out.reshape(jnp.shape(x))


@partial(jax.jit, static_argnames=("n",))
def acb_bernoulli_poly_ui_point(n: int, x: jax.Array) -> jax.Array:
    flat = jnp.ravel(jnp.asarray(x, dtype=jnp.complex128))
    out = jax.vmap(lambda t: acb_core.acb_midpoint(acb_core.acb_bernoulli_poly_ui(n, point_box(t))))(flat)
    return out.reshape(jnp.shape(x))


acb_polylog_point = scalarize_binary_complex(acb_core.acb_polylog)


@partial(jax.jit, static_argnames=("s",))
def acb_polylog_si_point(s: int, z: jax.Array) -> jax.Array:
    flat = jnp.ravel(jnp.asarray(z, dtype=jnp.complex128))
    out = jax.vmap(lambda t: acb_core.acb_midpoint(acb_core.acb_polylog_si(s, point_box(t))))(flat)
    return out.reshape(jnp.shape(z))


acb_agm_point = scalarize_binary_complex(acb_core.acb_agm)
acb_agm1_point = scalarize_unary_complex(acb_core.acb_agm1)
acb_agm1_cpx_point = scalarize_unary_complex(acb_core.acb_agm1_cpx)


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
