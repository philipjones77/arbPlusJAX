from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import point_wrappers_core as core


hypgeom = core.hypgeom
di = core.di
_pad_args_repeat_last = core._pad_args_repeat_last
_real_hermite_h_scalar = core._real_hermite_h_scalar
_real_laguerre_l_scalar = core._real_laguerre_l_scalar
_real_pfq_scalar = core._real_pfq_scalar
_vectorize_real_scalar = core._vectorize_real_scalar
_vectorize_real_scalar_tuple2 = core._vectorize_real_scalar_tuple2
arb_gamma_point = core.arb_gamma_point
arb_rgamma_point = core.arb_rgamma_point


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
    z_arr = jnp.asarray(z, dtype=jnp.result_type(z, jnp.float64))

    def _exact_interval(x):
        xx = jnp.asarray(x, dtype=jnp.result_type(x, jnp.float64))
        return di.interval(xx, xx)

    def scalar(aa, bb, zz):
        return di.midpoint(
            hypgeom.arb_hypgeom_pfq(
                _exact_interval(aa),
                _exact_interval(bb),
                _exact_interval(zz),
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
        )

    if a_arr.ndim <= 1 and b_arr.ndim <= 1 and z_arr.ndim == 0:
        return scalar(a_arr, b_arr, z_arr)
    if z_arr.ndim == 0:
        z_arr = jnp.broadcast_to(z_arr, (a_arr.shape[0],))
    return jax.vmap(scalar)(a_arr, b_arr, z_arr)


def arb_hypgeom_0f1_batch_fixed_point(a: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_0f1_point(a, z, regularized=regularized)


def arb_hypgeom_0f1_batch_padded_point(a: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, z), pad_to)
    return arb_hypgeom_0f1_point(*call_args, regularized=regularized)


def arb_hypgeom_1f1_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1_point(a, b, z, regularized=regularized)


def arb_hypgeom_1f1_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_1f1_point(*call_args, regularized=regularized)


def arb_hypgeom_m_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_m_point(a, b, z, regularized=regularized)


def arb_hypgeom_m_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_m_point(*call_args, regularized=regularized)


def arb_hypgeom_2f1_batch_fixed_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_2f1_point(a, b, c, z, regularized=regularized)


def arb_hypgeom_2f1_batch_padded_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, c, z), pad_to)
    return arb_hypgeom_2f1_point(*call_args, regularized=regularized)


def arb_hypgeom_u_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_u_point(a, b, z)


def arb_hypgeom_u_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_u_point(*call_args)


def arb_hypgeom_gamma_lower_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_gamma_lower_point(s, z, regularized=regularized)


def arb_hypgeom_gamma_lower_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return arb_hypgeom_gamma_lower_point(*call_args, regularized=regularized)


def arb_hypgeom_gamma_upper_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_gamma_upper_point(s, z, regularized=regularized)


def arb_hypgeom_gamma_upper_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return arb_hypgeom_gamma_upper_point(*call_args, regularized=regularized)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_gamma_point(x: jax.Array) -> jax.Array:
    return arb_gamma_point(x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erf_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_erf_series, x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfc_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: 1.0 - hypgeom._real_erf_series(t), x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfi_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_erfi, x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfinv_point(x: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_erfinv_scalar, x)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_erfcinv_point(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    return arb_hypgeom_erfinv_point(1.0 - arr)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_ei_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_ei_scalar, z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_si_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: hypgeom._real_si_ci_scalar(t)[0], z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_ci_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: hypgeom._real_si_ci_scalar(t)[1], z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_shi_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: 0.5 * (hypgeom._real_ei_scalar(t) - hypgeom._real_ei_scalar(-t)), z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_chi_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(lambda t: 0.5 * (hypgeom._real_ei_scalar(t) + hypgeom._real_ei_scalar(-t)), z)


@partial(jax.jit, static_argnames=("offset",))
def arb_hypgeom_li_point(z: jax.Array, offset: int = 0) -> jax.Array:
    offset_term = jnp.asarray(0.0, dtype=jnp.asarray(z).dtype)
    if offset > 0:
        offset_term = hypgeom._real_ei_scalar(jnp.log(jnp.asarray(offset, dtype=jnp.asarray(z).dtype)))
    return _vectorize_real_scalar(lambda t: hypgeom._real_ei_scalar(jnp.log(t)) - offset_term, z)


@partial(jax.jit, static_argnames=())
def arb_hypgeom_dilog_point(z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(hypgeom._real_dilog_scalar, z)


@partial(jax.jit, static_argnames=("normalized",))
def arb_hypgeom_fresnel_point(z: jax.Array, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return _vectorize_real_scalar_tuple2(lambda t: hypgeom._real_fresnel_scalar(t, normalized), z)


def arb_hypgeom_pfq_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    return arb_hypgeom_pfq_point(a, b, z, reciprocal=reciprocal, n_terms=n_terms)


def arb_hypgeom_pfq_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_pfq_point(*call_args, reciprocal=reciprocal, n_terms=n_terms)


def arb_hypgeom_gamma_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_gamma_point(x)


def arb_hypgeom_gamma_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_gamma_point(*call_args)


def arb_hypgeom_erf_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erf_point(x)


def arb_hypgeom_erf_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erf_point(*call_args)


def arb_hypgeom_erfc_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfc_point(x)


def arb_hypgeom_erfc_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfc_point(*call_args)


def arb_hypgeom_erfi_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfi_point(x)


def arb_hypgeom_erfi_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfi_point(*call_args)


def arb_hypgeom_erfinv_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfinv_point(x)


def arb_hypgeom_erfinv_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfinv_point(*call_args)


def arb_hypgeom_erfcinv_batch_fixed_point(x: jax.Array) -> jax.Array:
    return arb_hypgeom_erfcinv_point(x)


def arb_hypgeom_erfcinv_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return arb_hypgeom_erfcinv_point(*call_args)


def arb_hypgeom_ei_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_ei_point(z)


def arb_hypgeom_ei_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_ei_point(*call_args)


def arb_hypgeom_si_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_si_point(z)


def arb_hypgeom_si_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_si_point(*call_args)


def arb_hypgeom_ci_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_ci_point(z)


def arb_hypgeom_ci_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_ci_point(*call_args)


def arb_hypgeom_shi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_shi_point(z)


def arb_hypgeom_shi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_shi_point(*call_args)


def arb_hypgeom_chi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_chi_point(z)


def arb_hypgeom_chi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_chi_point(*call_args)


def arb_hypgeom_li_batch_fixed_point(z: jax.Array, *, offset: int = 0) -> jax.Array:
    return arb_hypgeom_li_point(z, offset=offset)


def arb_hypgeom_li_batch_padded_point(z: jax.Array, *, pad_to: int, offset: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_li_point(*call_args, offset=offset)


def arb_hypgeom_dilog_batch_fixed_point(z: jax.Array) -> jax.Array:
    return arb_hypgeom_dilog_point(z)


def arb_hypgeom_dilog_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_dilog_point(*call_args)


def arb_hypgeom_fresnel_batch_fixed_point(z: jax.Array, *, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_fresnel_point(z, normalized=normalized)


def arb_hypgeom_fresnel_batch_padded_point(z: jax.Array, *, pad_to: int, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_fresnel_point(*call_args, normalized=normalized)


def arb_hypgeom_legendre_p_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return arb_hypgeom_legendre_p_point(n, m, z, type=type)


def arb_hypgeom_legendre_p_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return arb_hypgeom_legendre_p_point(n, *call_args, type=type)


def arb_hypgeom_legendre_q_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return arb_hypgeom_legendre_q_point(n, m, z, type=type)


def arb_hypgeom_legendre_q_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return arb_hypgeom_legendre_q_point(n, *call_args, type=type)


def arb_hypgeom_jacobi_p_batch_fixed_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_jacobi_p_point(n, a, b, z)


def arb_hypgeom_jacobi_p_batch_padded_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return arb_hypgeom_jacobi_p_point(n, *call_args)


def arb_hypgeom_gegenbauer_c_batch_fixed_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_gegenbauer_c_point(n, lam, z)


def arb_hypgeom_gegenbauer_c_batch_padded_point(n: int, lam: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((lam, z), pad_to)
    return arb_hypgeom_gegenbauer_c_point(n, *call_args)


def arb_hypgeom_chebyshev_t_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return arb_hypgeom_chebyshev_t_point(n, z)


def arb_hypgeom_chebyshev_t_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_chebyshev_t_point(n, *call_args)


def arb_hypgeom_chebyshev_u_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return arb_hypgeom_chebyshev_u_point(n, z)


def arb_hypgeom_chebyshev_u_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_chebyshev_u_point(n, *call_args)


def arb_hypgeom_laguerre_l_batch_fixed_point(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_laguerre_l_point(n, m, z)


def arb_hypgeom_laguerre_l_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return arb_hypgeom_laguerre_l_point(n, *call_args)


def arb_hypgeom_hermite_h_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return arb_hypgeom_hermite_h_point(n, z)


def arb_hypgeom_hermite_h_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return arb_hypgeom_hermite_h_point(n, *call_args)


__all__ = sorted(name for name in globals() if name.startswith("arb_hypgeom_"))
