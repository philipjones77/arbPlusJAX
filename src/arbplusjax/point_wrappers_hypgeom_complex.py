from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import point_wrappers_core as core


hypgeom = core.hypgeom
acb_core = core.acb_core
di = core.di
_pad_args_repeat_last = core._pad_args_repeat_last
_complex_pfq_scalar = core._complex_pfq_scalar
_vectorize_complex_scalar = core._vectorize_complex_scalar
_vectorize_complex_scalar_tuple2 = core._vectorize_complex_scalar_tuple2
acb_gamma_point = core.acb_gamma_point
acb_rgamma_point = core.acb_rgamma_point
acb_lgamma_point = core.acb_lgamma_point


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
    z_arr = jnp.asarray(z, dtype=jnp.result_type(z, jnp.complex128))

    def _box_exact(x):
        xx = jnp.asarray(x, dtype=jnp.result_type(x, jnp.complex128))
        return acb_core.acb_box(
            di.interval(jnp.real(xx), jnp.real(xx)),
            di.interval(jnp.imag(xx), jnp.imag(xx)),
        )

    def scalar(aa, bb, zz):
        return acb_core.acb_midpoint(
            hypgeom.acb_hypgeom_pfq(
                _box_exact(aa),
                _box_exact(bb),
                _box_exact(zz),
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
        )

    if a_arr.ndim <= 1 and b_arr.ndim <= 1 and z_arr.ndim == 0:
        return scalar(a_arr, b_arr, z_arr)
    if z_arr.ndim == 0:
        z_arr = jnp.broadcast_to(z_arr, (a_arr.shape[0],))
    return jax.vmap(scalar)(a_arr, b_arr, z_arr)


def acb_hypgeom_0f1_batch_fixed_point(a: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_0f1_point(a, z, regularized=regularized)


def acb_hypgeom_0f1_batch_padded_point(a: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, z), pad_to)
    return acb_hypgeom_0f1_point(*call_args, regularized=regularized)


def acb_hypgeom_1f1_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1_point(a, b, z, regularized=regularized)


def acb_hypgeom_1f1_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_1f1_point(*call_args, regularized=regularized)


def acb_hypgeom_m_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_m_point(a, b, z, regularized=regularized)


def acb_hypgeom_m_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_m_point(*call_args, regularized=regularized)


def acb_hypgeom_2f1_batch_fixed_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_2f1_point(a, b, c, z, regularized=regularized)


def acb_hypgeom_2f1_batch_padded_point(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, c, z), pad_to)
    return acb_hypgeom_2f1_point(*call_args, regularized=regularized)


def acb_hypgeom_u_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_u_point(a, b, z)


def acb_hypgeom_u_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_u_point(*call_args)


def acb_hypgeom_gamma_lower_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_gamma_lower_point(s, z, regularized=regularized)


def acb_hypgeom_gamma_lower_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return acb_hypgeom_gamma_lower_point(*call_args, regularized=regularized)


def acb_hypgeom_gamma_upper_batch_fixed_point(s: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_gamma_upper_point(s, z, regularized=regularized)


def acb_hypgeom_gamma_upper_batch_padded_point(s: jax.Array, z: jax.Array, *, pad_to: int, regularized: bool = False) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((s, z), pad_to)
    return acb_hypgeom_gamma_upper_point(*call_args, regularized=regularized)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_gamma_point(x: jax.Array) -> jax.Array:
    return acb_gamma_point(x)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_erf_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_erf_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_erfc_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_erfc_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_erfi_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_erfi_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_ei_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_ei_series, z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_si_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_si_ci_series(w)[0], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_ci_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_si_ci_series(w)[1], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_shi_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_shi_chi_series(w)[0], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_chi_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_shi_chi_series(w)[1], z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_li_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(lambda w: hypgeom._complex_ei_series(jnp.log(w)), z)


@partial(jax.jit, static_argnames=())
def acb_hypgeom_dilog_point(z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(hypgeom._complex_dilog_series, z)


@partial(jax.jit, static_argnames=("normalized",))
def acb_hypgeom_fresnel_point(z: jax.Array, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return _vectorize_complex_scalar_tuple2(lambda w: hypgeom._complex_fresnel(w, normalized), z)


def acb_hypgeom_pfq_batch_fixed_point(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    return acb_hypgeom_pfq_point(a, b, z, reciprocal=reciprocal, n_terms=n_terms)


def acb_hypgeom_pfq_batch_padded_point(a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_pfq_point(*call_args, reciprocal=reciprocal, n_terms=n_terms)


def acb_hypgeom_gamma_batch_fixed_point(x: jax.Array) -> jax.Array:
    return acb_hypgeom_gamma_point(x)


def acb_hypgeom_gamma_batch_padded_point(x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((x,), pad_to)
    return acb_hypgeom_gamma_point(*call_args)


def acb_hypgeom_erf_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_erf_point(z)


def acb_hypgeom_erf_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_erf_point(*call_args)


def acb_hypgeom_erfc_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_erfc_point(z)


def acb_hypgeom_erfc_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_erfc_point(*call_args)


def acb_hypgeom_erfi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_erfi_point(z)


def acb_hypgeom_erfi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_erfi_point(*call_args)


def acb_hypgeom_ei_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_ei_point(z)


def acb_hypgeom_ei_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_ei_point(*call_args)


def acb_hypgeom_si_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_si_point(z)


def acb_hypgeom_si_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_si_point(*call_args)


def acb_hypgeom_ci_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_ci_point(z)


def acb_hypgeom_ci_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_ci_point(*call_args)


def acb_hypgeom_shi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_shi_point(z)


def acb_hypgeom_shi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_shi_point(*call_args)


def acb_hypgeom_chi_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_chi_point(z)


def acb_hypgeom_chi_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_chi_point(*call_args)


def acb_hypgeom_li_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_li_point(z)


def acb_hypgeom_li_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_li_point(*call_args)


def acb_hypgeom_dilog_batch_fixed_point(z: jax.Array) -> jax.Array:
    return acb_hypgeom_dilog_point(z)


def acb_hypgeom_dilog_batch_padded_point(z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_dilog_point(*call_args)


def acb_hypgeom_fresnel_batch_fixed_point(z: jax.Array, *, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    return acb_hypgeom_fresnel_point(z, normalized=normalized)


def acb_hypgeom_fresnel_batch_padded_point(z: jax.Array, *, pad_to: int, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_fresnel_point(*call_args, normalized=normalized)


def acb_hypgeom_legendre_p_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return acb_hypgeom_legendre_p_point(n, m, z, type=type)


def acb_hypgeom_legendre_p_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return acb_hypgeom_legendre_p_point(n, *call_args, type=type)


def acb_hypgeom_legendre_q_batch_fixed_point(n: int, m: jax.Array, z: jax.Array, *, type: int = 0) -> jax.Array:
    return acb_hypgeom_legendre_q_point(n, m, z, type=type)


def acb_hypgeom_legendre_q_batch_padded_point(n: int, m: jax.Array, z: jax.Array, *, pad_to: int, type: int = 0) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((m, z), pad_to)
    return acb_hypgeom_legendre_q_point(n, *call_args, type=type)


def acb_hypgeom_jacobi_p_batch_fixed_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_jacobi_p_point(n, a, b, z)


def acb_hypgeom_jacobi_p_batch_padded_point(n: int, a: jax.Array, b: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, b, z), pad_to)
    return acb_hypgeom_jacobi_p_point(n, *call_args)


def acb_hypgeom_gegenbauer_c_batch_fixed_point(n: int, lam: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_gegenbauer_c_point(n, lam, z)


def acb_hypgeom_gegenbauer_c_batch_padded_point(n: int, lam: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((lam, z), pad_to)
    return acb_hypgeom_gegenbauer_c_point(n, *call_args)


def acb_hypgeom_chebyshev_t_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return acb_hypgeom_chebyshev_t_point(n, z)


def acb_hypgeom_chebyshev_t_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_chebyshev_t_point(n, *call_args)


def acb_hypgeom_chebyshev_u_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return acb_hypgeom_chebyshev_u_point(n, z)


def acb_hypgeom_chebyshev_u_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_chebyshev_u_point(n, *call_args)


def acb_hypgeom_laguerre_l_batch_fixed_point(n: int, a: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_laguerre_l_point(n, a, z)


def acb_hypgeom_laguerre_l_batch_padded_point(n: int, a: jax.Array, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((a, z), pad_to)
    return acb_hypgeom_laguerre_l_point(n, *call_args)


def acb_hypgeom_hermite_h_batch_fixed_point(n: int, z: jax.Array) -> jax.Array:
    return acb_hypgeom_hermite_h_point(n, z)


def acb_hypgeom_hermite_h_batch_padded_point(n: int, z: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = _pad_args_repeat_last((z,), pad_to)
    return acb_hypgeom_hermite_h_point(n, *call_args)


__all__ = sorted(name for name in globals() if name.startswith("acb_hypgeom_"))
