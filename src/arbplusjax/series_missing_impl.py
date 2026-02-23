from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy.special as jsp

from . import double_interval as di
from . import acb_core
from . import series_utils as su
from . import hypgeom

jax.config.update('jax_enable_x64', True)

def _full_interval_coeffs(length: int) -> jax.Array:
    return jnp.tile(jnp.array([-jnp.inf, jnp.inf], dtype=jnp.float64), (length, 1))

def _full_box_coeffs(length: int) -> jax.Array:
    return jnp.tile(jnp.array([-jnp.inf, jnp.inf, -jnp.inf, jnp.inf], dtype=jnp.float64), (length, 1))

def _lambertw(z: jax.Array, iters: int = 12) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    w = jnp.where(jnp.abs(z) < 1.0, z, jnp.log(z))
    for _ in range(iters):
        ew = jnp.exp(w)
        f = w * ew - z
        wp1 = w + 1.0
        denom = ew * wp1 - (wp1 + 1.0) * f / (2.0 * wp1)
        w = w - f / denom
    return w

def _theta3(tau: jax.Array, terms: int = 12) -> jax.Array:
    tau = jnp.asarray(tau, dtype=jnp.complex128)
    q = jnp.exp(1j * jnp.pi * tau)
    ns = jnp.arange(1, terms + 1, dtype=jnp.float64)
    return 1.0 + 2.0 * jnp.sum(q ** (ns * ns))


def _qseries_simple(tau: jax.Array, terms: int = 12) -> jax.Array:
    tau = jnp.asarray(tau, dtype=jnp.complex128)
    q = jnp.exp(2j * jnp.pi * tau)
    ns = jnp.arange(1, terms + 1, dtype=jnp.float64)
    return jnp.sum(q ** ns)


def _powsum_simple(z: jax.Array, power: int = 1, terms: int = 12) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    ks = jnp.arange(1, terms + 1, dtype=jnp.float64)
    return jnp.sum((ks ** power) * (z ** ks))


def _zeta_series_complex(s: jax.Array, terms: int = 32) -> jax.Array:
    s = jnp.asarray(s, dtype=jnp.complex128)
    n = jnp.arange(1, terms + 1, dtype=jnp.float64)
    return jnp.sum(jnp.exp(-s * jnp.log(n)))


def _polylog(s: jax.Array, z: jax.Array, terms: int = 40) -> jax.Array:
    s = jnp.asarray(s, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    ks = jnp.arange(1, terms + 1, dtype=jnp.float64)
    return jnp.sum(jnp.power(z, ks) / jnp.power(ks, s))

def _series_unary_real(fn, h, length):
    vals = su.series_from_poly_real(fn, h, length)
    return su.coeffs_to_intervals(vals)

def _series_unary_complex(fn, h, length):
    vals = su.series_from_poly_complex(fn, h, length)
    return su.coeffs_to_boxes(vals)

def _jet_real(fn, x, length):
    x = di.as_interval(x)
    mid = di.midpoint(x)
    vals = su.taylor_series_unary_real(fn, mid, length)
    return su.coeffs_to_intervals(vals)

def _jet_complex_cauchy(fn, x, length, bound_fn):
    x = acb_core.as_acb_box(x)
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    mid = acb_core.acb_midpoint(x)
    r = 0.5 * jnp.maximum(re[1] - re[0], im[1] - im[0])
    vals = su.taylor_series_unary_complex(fn, mid, length)
    M = bound_fn(mid, r, length)
    # Cauchy bound: |a_k| <= M / r^k
    ks = jnp.arange(length, dtype=jnp.float64)
    r_safe = jnp.where(r == 0.0, 1.0, r)
    eps = jnp.where(r == 0.0, 0.0, M / (r_safe ** ks))
    re_vals = jnp.real(vals)
    im_vals = jnp.imag(vals)
    return jnp.stack([
        di._below(re_vals - eps), di._above(re_vals + eps),
        di._below(im_vals - eps), di._above(im_vals + eps)
    ], axis=-1)


def _jet_real_cauchy(fn, x, length, bound_fn):
    x = di.as_interval(x)
    mid = di.midpoint(x)
    r = 0.5 * (x[1] - x[0])
    vals = su.taylor_series_unary_real(fn, mid, length)
    M = bound_fn(mid, r, length)
    ks = jnp.arange(length, dtype=jnp.float64)
    r_safe = jnp.where(r == 0.0, 1.0, r)
    eps = jnp.where(r == 0.0, 0.0, M / (r_safe ** ks))
    return jnp.stack([di._below(vals - eps), di._above(vals + eps)], axis=-1)


def _theta3_bound(mid, r, terms):
    im_min = jnp.imag(mid) - r
    q = jnp.exp(-jnp.pi * im_min)
    ns = jnp.arange(1, terms + 1, dtype=jnp.float64)
    partial = jnp.sum(q ** (ns * ns))
    tail = jnp.where(q < 1.0, (q ** ((terms + 1) ** 2)) / (1.0 - q), jnp.inf)
    bound = 1.0 + 2.0 * (partial + tail)
    return jnp.where(im_min <= 0.0, jnp.inf, bound)


def _qseries_bound(mid, r, terms):
    im_min = jnp.imag(mid) - r
    q = jnp.exp(-2.0 * jnp.pi * im_min)
    partial = q * (1.0 - q ** terms) / (1.0 - q)
    tail = q ** (terms + 1) / (1.0 - q)
    bound = partial + tail
    return jnp.where(jnp.logical_or(im_min <= 0.0, q >= 1.0), jnp.inf, bound)


def _powsum_bound(mid, r, terms, power=1):
    r_abs = jnp.abs(mid) + r
    ks = jnp.arange(1, terms + 1, dtype=jnp.float64)
    partial = jnp.sum((ks ** power) * (r_abs ** ks))
    tail = (jnp.float64(terms + 1) ** power) * (r_abs ** (terms + 1)) / (1.0 - r_abs)
    bound = partial + tail
    return jnp.where(r_abs >= 1.0, jnp.inf, bound)


def _exp_bound(mid, r, terms):
    return jnp.exp(jnp.abs(mid) + r)


def _jet_complex(fn, x, length):
    x = acb_core.as_acb_box(x)
    mid = acb_core.acb_midpoint(x)
    vals = su.taylor_series_unary_complex(fn, mid, length)
    return su.coeffs_to_boxes(vals)

def _fn_from_name(name: str):
    n = name
    if 'log1p' in n:
        return jnp.log1p
    if 'log' in n and 'log1p' not in n and 'log_rising' not in n:
        return jnp.log
    if 'exp' in n and 'expi' not in n and 'exp_pi' not in n:
        return jnp.exp
    if 'sin' in n and 'sinh' not in n and 'asin' not in n and 'sinc' not in n and 'sin_cos' not in n:
        return jnp.sin
    if 'cos' in n and 'cosh' not in n and 'acos' not in n and 'sin_cos' not in n:
        return jnp.cos
    if 'tan' in n and 'tanh' not in n and 'atan' not in n:
        return jnp.tan
    if 'sinh' in n and 'sinh_cosh' not in n:
        return jnp.sinh
    if 'cosh' in n and 'sinh_cosh' not in n:
        return jnp.cosh
    if 'tanh' in n:
        return jnp.tanh
    if 'rsqrt' in n:
        return lambda x: 1.0 / jnp.sqrt(x)
    if 'sqrt' in n:
        return jnp.sqrt
    if 'asin' in n:
        return jnp.arcsin
    if 'acos' in n:
        return jnp.arccos
    if 'atan' in n:
        return jnp.arctan
    if 'fresnel' in n:
        return lambda x: hypgeom._fresnel_eval(x, False)[0]
    if 'erfc' in n:

        return jsp.erfc
    if 'erf' in n:
        return jsp.erf
    if 'si' in n:
        return lambda x: hypgeom._si_ci_from_series(x)[0]
    if 'ci' in n:
        return lambda x: hypgeom._si_ci_from_series(x)[1]
    if 'shi' in n:
        return lambda x: 0.5 * (jsp.expi(x) - jsp.expi(-x))
    if 'chi' in n:
        return lambda x: 0.5 * (jsp.expi(x) + jsp.expi(-x))
    if 'li' in n:
        return lambda x: jsp.expi(jnp.log(x))
    if 'digamma' in n:
        return jsp.digamma
    if 'gamma' in n and 'lgamma' not in n and 'rgamma' not in n:
        return jsp.gamma
    if 'lgamma' in n:
        return jsp.gammaln
    if 'rgamma' in n:
        return lambda x: 1.0 / jsp.gamma(x)
    if 'zeta' in n:
        return lambda x: _zeta_series_complex(x, terms=32)
    if 'lambertw' in n:
        return _lambertw
    if 'airy' in n:
        return lambda x: hypgeom._airy_series(x, -1.0)[0]
    if 'polylog' in n:
        return lambda x: _polylog(2.0 + 0.0j, x)
    if 'pfq' in n:
        return jnp.exp
    if 'sinc_pi' in n:
        return lambda x: jnp.where(x == 0.0, 1.0, jnp.sin(jnp.pi * x) / (jnp.pi * x))
    if 'sinc' in n:
        return lambda x: jnp.where(x == 0.0, 1.0, jnp.sin(x) / x)
    if 'sin_pi' in n:
        return lambda x: jnp.sin(jnp.pi * x)
    if 'cos_pi' in n:
        return lambda x: jnp.cos(jnp.pi * x)
    if 'cot_pi' in n:
        return lambda x: jnp.cos(jnp.pi * x) / jnp.sin(jnp.pi * x)
    if 'exp_pi_i' in n:
        return lambda x: jnp.exp(1j * jnp.pi * x)
    if ('expi' in n or 'ei' in n) and 'acb' in n:
        return hypgeom._complex_ei_series
    if 'expi' in n or 'ei' in n:
        return jsp.expi
    return None

def _arb_poly_compose(f, g, length):
    f = di.midpoint(di.as_interval(f))
    g = di.midpoint(di.as_interval(g))
    vals = su.series_compose(f, g, length)
    return su.coeffs_to_intervals(vals)

def _acb_poly_compose(f, g, length):
    f = acb_core.as_acb_box(f)
    g = acb_core.as_acb_box(g)
    fm = di.midpoint(acb_core.acb_real(f)) + 1j * di.midpoint(acb_core.acb_imag(f))
    gm = di.midpoint(acb_core.acb_real(g)) + 1j * di.midpoint(acb_core.acb_imag(g))
    vals = su.series_compose(fm, gm, length)
    return su.coeffs_to_boxes(vals)

def _arb_poly_inv(f, length):
    fm = di.midpoint(di.as_interval(f))
    vals = su.series_inv(fm, length)
    return su.coeffs_to_intervals(vals)

def _acb_poly_inv(f, length):
    f = acb_core.as_acb_box(f)
    fm = di.midpoint(acb_core.acb_real(f)) + 1j * di.midpoint(acb_core.acb_imag(f))
    vals = su.series_inv(fm, length)
    return su.coeffs_to_boxes(vals)

def _arb_poly_div(a, b, length):
    am = di.midpoint(di.as_interval(a))
    bm = di.midpoint(di.as_interval(b))
    vals = su.series_div(am, bm, length)
    return su.coeffs_to_intervals(vals)

def _acb_poly_div(a, b, length):
    a = acb_core.as_acb_box(a)
    b = acb_core.as_acb_box(b)
    am = di.midpoint(acb_core.acb_real(a)) + 1j * di.midpoint(acb_core.acb_imag(a))
    bm = di.midpoint(acb_core.acb_real(b)) + 1j * di.midpoint(acb_core.acb_imag(b))
    vals = su.series_div(am, bm, length)
    return su.coeffs_to_boxes(vals)

def _arb_poly_revert(f, length):
    fm = di.midpoint(di.as_interval(f))
    vals = su.series_revert(fm, length)
    return su.coeffs_to_intervals(vals)

def _acb_poly_revert(f, length):
    f = acb_core.as_acb_box(f)
    fm = di.midpoint(acb_core.acb_real(f)) + 1j * di.midpoint(acb_core.acb_imag(f))
    vals = su.series_revert(fm, length)
    return su.coeffs_to_boxes(vals)

def _arb_poly_pow(f, p, length):
    fm = di.midpoint(di.as_interval(f))
    pm = di.midpoint(di.as_interval(p)) if hasattr(p, 'shape') else jnp.float64(p)
    vals = su.series_pow_scalar(fm, pm, length)
    return su.coeffs_to_intervals(vals)

def _acb_poly_pow(f, p, length):
    f = acb_core.as_acb_box(f)
    fm = di.midpoint(acb_core.acb_real(f)) + 1j * di.midpoint(acb_core.acb_imag(f))
    pm = p
    vals = su.series_pow_scalar(fm, pm, length)
    return su.coeffs_to_boxes(vals)

def _acb_dirichlet_hardy_theta_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "_acb_dirichlet_hardy_theta_series" else _full_interval_coeffs(length)


def _acb_dirichlet_hardy_z_series(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def _acb_dirichlet_l_series(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def _acb_elliptic_k_series(m, length=8, *args, **kwargs):
    from . import acb_elliptic
    f = acb_elliptic.acb_elliptic_k
    return _jet_complex(lambda z: acb_core.acb_midpoint(f(acb_core.as_acb_box(jnp.array([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], dtype=jnp.float64)))), m, length)


def _acb_elliptic_p_series(m, length=8, *args, **kwargs):
    from . import acb_elliptic
    f = acb_elliptic.acb_elliptic_k
    return _jet_complex(lambda z: acb_core.acb_midpoint(f(acb_core.as_acb_box(jnp.array([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], dtype=jnp.float64)))), m, length)


def _acb_hypgeom_airy_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_airy_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_airy_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_airy_series") else _jet_real(f, z, length)


def _acb_hypgeom_beta_lower_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_beta_lower_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_beta_lower_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_beta_lower_series") else _jet_real(f, z, length)


def _acb_hypgeom_chi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_chi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_chi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_chi_series") else _jet_real(f, z, length)


def _acb_hypgeom_ci_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_ci_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_ci_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_ci_series") else _jet_real(f, z, length)


def _acb_hypgeom_coulomb_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_coulomb_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_coulomb_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_coulomb_series") else _jet_real(f, z, length)


def _acb_hypgeom_ei_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_ei_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_ei_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_ei_series") else _jet_real(f, z, length)


def _acb_hypgeom_erf_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_erf_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_erf_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_erf_series") else _jet_real(f, z, length)


def _acb_hypgeom_erfc_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_erfc_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_erfc_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_erfc_series") else _jet_real(f, z, length)


def _acb_hypgeom_erfi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_erfi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_erfi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_erfi_series") else _jet_real(f, z, length)


def _acb_hypgeom_fresnel_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_fresnel_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_fresnel_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_fresnel_series") else _jet_real(f, z, length)


def _acb_hypgeom_gamma_lower_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_gamma_lower_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_gamma_lower_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_gamma_lower_series") else _jet_real(f, z, length)


def _acb_hypgeom_gamma_upper_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_gamma_upper_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_gamma_upper_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_gamma_upper_series") else _jet_real(f, z, length)


def _acb_hypgeom_li_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_li_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_li_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_li_series") else _jet_real(f, z, length)


def _acb_hypgeom_shi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_shi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_shi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_shi_series") else _jet_real(f, z, length)


def _acb_hypgeom_si_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_acb_hypgeom_si_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_acb_hypgeom_si_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_acb_hypgeom_si_series") else _jet_real(f, z, length)


def _acb_modular_theta_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "_acb_modular_theta_series" else _full_interval_coeffs(length)


def _acb_poly_agm1_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_agm1_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_atan_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_atan_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_binomial_pow_acb_series(a, p, length, *args, **kwargs):
    return _acb_poly_pow(a, p, length)


def _acb_poly_compose_series(f, g, length, *args, **kwargs):
    return _acb_poly_compose(f, g, length)


def _acb_poly_compose_series_brent_kung(f, g, length, *args, **kwargs):
    return _acb_poly_compose(f, g, length)


def _acb_poly_compose_series_horner(f, g, length, *args, **kwargs):
    return _acb_poly_compose(f, g, length)


def _acb_poly_cos_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_cos_pi_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_cos_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_cos_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_cosh_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_cosh_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_cot_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_cot_pi_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_digamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_digamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_div_series(a, b, length, *args, **kwargs):
    return _acb_poly_div(a, b, length)


def _acb_poly_elliptic_k_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_elliptic_k_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_elliptic_p_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_elliptic_p_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_erf_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_erf_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_exp_pi_i_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_exp_pi_i_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_exp_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_exp_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_exp_series_basecase(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_exp_series_basecase")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_gamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_gamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_inv_series(a, length, *args, **kwargs):
    return _acb_poly_inv(a, length)


def _acb_poly_lambertw_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_lambertw_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_lgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_lgamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_log1p_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_log1p_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_log_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_log_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_polylog_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_polylog_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_pow_acb_series(a, p, length, *args, **kwargs):
    return _acb_poly_pow(a, p, length)


def _acb_poly_pow_series(a, p, length, *args, **kwargs):
    return _acb_poly_pow(a, p, length)


def _acb_poly_powsum_one_series_sieved(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "_acb_poly_powsum_one_series_sieved" else _full_interval_coeffs(length)


def _acb_poly_powsum_series_naive(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "_acb_poly_powsum_series_naive" else _full_interval_coeffs(length)


def _acb_poly_powsum_series_naive_threaded(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "_acb_poly_powsum_series_naive_threaded" else _full_interval_coeffs(length)


def _acb_poly_revert_series(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def _acb_poly_revert_series_lagrange(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def _acb_poly_revert_series_lagrange_fast(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def _acb_poly_revert_series_newton(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def _acb_poly_rgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_rgamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_rising_ui_series(x, n=2, length=8, *args, **kwargs):
    f = lambda t: jnp.exp(jsp.gammaln(t + n) - jsp.gammaln(t))
    return _jet_complex(f, x, length)


def _acb_poly_rsqrt_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_rsqrt_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_sin_cos_pi_series(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sin_cos_series(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sin_cos_series_basecase(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sin_cos_series_tangent(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sin_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_sin_pi_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_sin_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_sin_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_sinc_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_sinc_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_sinh_cosh_series(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sinh_cosh(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sinh_cosh_series_basecase(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sinh_cosh(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sinh_cosh_series_exponential(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sinh_cosh(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)


def _acb_poly_sinh_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_sinh_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_sqrt_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_sqrt_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_tan_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_tan_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_zeta_cpx_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_zeta_cpx_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _acb_poly_zeta_series(h, length, *args, **kwargs):
    f = _fn_from_name("_acb_poly_zeta_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def _arb_hypgeom_airy_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_airy_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_airy_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_airy_series") else _jet_real(f, z, length)


def _arb_hypgeom_beta_lower_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_beta_lower_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_beta_lower_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_beta_lower_series") else _jet_real(f, z, length)


def _arb_hypgeom_chi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_chi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_chi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_chi_series") else _jet_real(f, z, length)


def _arb_hypgeom_ci_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_ci_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_ci_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_ci_series") else _jet_real(f, z, length)


def _arb_hypgeom_coulomb_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_coulomb_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_coulomb_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_coulomb_series") else _jet_real(f, z, length)


def _arb_hypgeom_ei_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_ei_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_ei_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_ei_series") else _jet_real(f, z, length)


def _arb_hypgeom_erf_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_erf_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_erf_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_erf_series") else _jet_real(f, z, length)


def _arb_hypgeom_erfc_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_erfc_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_erfc_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_erfc_series") else _jet_real(f, z, length)


def _arb_hypgeom_erfi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_erfi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_erfi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_erfi_series") else _jet_real(f, z, length)


def _arb_hypgeom_fresnel_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_fresnel_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_fresnel_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_fresnel_series") else _jet_real(f, z, length)


def _arb_hypgeom_gamma_lower_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_gamma_lower_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_gamma_lower_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_gamma_lower_series") else _jet_real(f, z, length)


def _arb_hypgeom_gamma_upper_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_gamma_upper_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_gamma_upper_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_gamma_upper_series") else _jet_real(f, z, length)


def _arb_hypgeom_li_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_li_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_li_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_li_series") else _jet_real(f, z, length)


def _arb_hypgeom_shi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_shi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_shi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_shi_series") else _jet_real(f, z, length)


def _arb_hypgeom_si_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("_arb_hypgeom_si_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "_arb_hypgeom_si_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "_arb_hypgeom_si_series") else _jet_real(f, z, length)


def _arb_poly_acos_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_acos_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_asin_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_asin_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_atan_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_atan_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_binomial_pow_arb_series(a, p, length, *args, **kwargs):
    return _arb_poly_pow(a, p, length)


def _arb_poly_compose_series(f, g, length, *args, **kwargs):
    return _arb_poly_compose(f, g, length)


def _arb_poly_compose_series_brent_kung(f, g, length, *args, **kwargs):
    return _arb_poly_compose(f, g, length)


def _arb_poly_compose_series_horner(f, g, length, *args, **kwargs):
    return _arb_poly_compose(f, g, length)


def _arb_poly_cos_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_cos_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_cos_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_cos_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_cosh_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_cosh_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_cot_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_cot_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_digamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_digamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_div_series(a, b, length, *args, **kwargs):
    return _arb_poly_div(a, b, length)


def _arb_poly_exp_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_exp_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_exp_series_basecase(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_exp_series_basecase")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_gamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_gamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_inv_series(a, length, *args, **kwargs):
    return _arb_poly_inv(a, length)


def _arb_poly_lambertw_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_lambertw_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_lgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_lgamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_log1p_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_log1p_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_log_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_log_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_pow_arb_series(a, p, length, *args, **kwargs):
    return _arb_poly_pow(a, p, length)


def _arb_poly_pow_series(a, p, length, *args, **kwargs):
    return _arb_poly_pow(a, p, length)


def _arb_poly_revert_series(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def _arb_poly_revert_series_lagrange(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def _arb_poly_revert_series_lagrange_fast(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def _arb_poly_revert_series_newton(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def _arb_poly_rgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_rgamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_riemann_siegel_theta_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "_arb_poly_riemann_siegel_theta_series" else _full_interval_coeffs(length)


def _arb_poly_riemann_siegel_z_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_riemann_siegel_z_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_rising_ui_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_interval_coeffs(length)


def _arb_poly_rsqrt_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_rsqrt_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_sin_cos_pi_series(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sin_cos_series(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sin_cos_series_basecase(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sin_cos_series_tangent(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sin_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_sin_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_sin_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_sin_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_sinc_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_sinc_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_sinc_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_sinc_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_sinh_cosh_series(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sinh_cosh(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sinh_cosh_series_basecase(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sinh_cosh(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sinh_cosh_series_exponential(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sinh_cosh(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)


def _arb_poly_sinh_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_sinh_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_sqrt_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_sqrt_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_tan_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_tan_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def _arb_poly_zeta_series(h, length, *args, **kwargs):
    f = _fn_from_name("_arb_poly_zeta_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def acb_dirichlet_hardy_theta_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_dirichlet_hardy_theta_series" else _full_interval_coeffs(length)


def acb_dirichlet_hardy_z_series(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def acb_dirichlet_hardy_z_series_prec(s, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_dirichlet_hardy_z_series(s, length=length, *args, **kwargs), prec_bits)


def acb_dirichlet_l_jet(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def acb_dirichlet_l_jet_prec(s, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_dirichlet_l_jet(s, length=length, *args, **kwargs), prec_bits)


def acb_dirichlet_l_series(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def acb_dirichlet_l_series_prec(s, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_dirichlet_l_series(s, length=length, *args, **kwargs), prec_bits)


def acb_dirichlet_qseries_arb(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_dirichlet_qseries_arb" else _full_interval_coeffs(length)


def acb_dirichlet_qseries_arb_powers_naive(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_dirichlet_qseries_arb_powers_naive" else _full_interval_coeffs(length)


def acb_dirichlet_qseries_arb_powers_smallorder(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_dirichlet_qseries_arb_powers_smallorder" else _full_interval_coeffs(length)


def acb_dirichlet_zeta_jet(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def acb_dirichlet_zeta_jet_prec(s, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_dirichlet_zeta_jet(s, length=length, *args, **kwargs), prec_bits)


def acb_dirichlet_zeta_jet_rs(s, length=8, *args, **kwargs):
    f = _fn_from_name('zeta')
    return _jet_complex(f, s, length)


def acb_dirichlet_zeta_jet_rs_prec(s, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_dirichlet_zeta_jet_rs(s, length=length, *args, **kwargs), prec_bits)


def acb_elliptic_k_jet(m, length=8, *args, **kwargs):
    from . import acb_elliptic
    f = acb_elliptic.acb_elliptic_k
    return _jet_complex(lambda z: acb_core.acb_midpoint(f(acb_core.as_acb_box(jnp.array([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], dtype=jnp.float64)))), m, length)


def acb_elliptic_k_jet_prec(m, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_elliptic_k_jet(m, length=length, *args, **kwargs), prec_bits)


def acb_elliptic_k_series(m, length=8, *args, **kwargs):
    from . import acb_elliptic
    f = acb_elliptic.acb_elliptic_k
    return _jet_complex(lambda z: acb_core.acb_midpoint(f(acb_core.as_acb_box(jnp.array([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], dtype=jnp.float64)))), m, length)


def acb_elliptic_k_series_prec(m, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_elliptic_k_series(m, length=length, *args, **kwargs), prec_bits)


def acb_elliptic_p_jet(m, length=8, *args, **kwargs):
    from . import acb_elliptic
    f = acb_elliptic.acb_elliptic_k
    return _jet_complex(lambda z: acb_core.acb_midpoint(f(acb_core.as_acb_box(jnp.array([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], dtype=jnp.float64)))), m, length)


def acb_elliptic_p_jet_prec(m, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_elliptic_p_jet(m, length=length, *args, **kwargs), prec_bits)


def acb_elliptic_p_series(m, length=8, *args, **kwargs):
    from . import acb_elliptic
    f = acb_elliptic.acb_elliptic_k
    return _jet_complex(lambda z: acb_core.acb_midpoint(f(acb_core.as_acb_box(jnp.array([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], dtype=jnp.float64)))), m, length)


def acb_elliptic_p_series_prec(m, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_elliptic_p_series(m, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_2f1_series_direct(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_2f1_series_direct")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_2f1_series_direct" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_2f1_series_direct") else _jet_real(f, z, length)


def acb_hypgeom_2f1_series_direct_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_2f1_series_direct":
        return acb_core.acb_box_round_prec(acb_hypgeom_2f1_series_direct(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_2f1_series_direct(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_airy_jet(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_airy_jet")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_airy_jet" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_airy_jet") else _jet_real(f, z, length)


def acb_hypgeom_airy_jet_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_airy_jet":
        return acb_core.acb_box_round_prec(acb_hypgeom_airy_jet(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_airy_jet(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_airy_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_airy_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_airy_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_airy_series") else _jet_real(f, z, length)


def acb_hypgeom_airy_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_airy_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_airy_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_airy_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_bessel_k_0f1_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_bessel_k_0f1_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_bessel_k_0f1_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_bessel_k_0f1_series") else _jet_real(f, z, length)


def acb_hypgeom_bessel_k_0f1_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_bessel_k_0f1_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_bessel_k_0f1_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_bessel_k_0f1_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_beta_lower_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_beta_lower_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_beta_lower_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_beta_lower_series") else _jet_real(f, z, length)


def acb_hypgeom_beta_lower_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_beta_lower_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_beta_lower_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_beta_lower_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_chi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_chi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_chi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_chi_series") else _jet_real(f, z, length)


def acb_hypgeom_chi_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_chi_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_chi_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_chi_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_ci_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_ci_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_ci_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_ci_series") else _jet_real(f, z, length)


def acb_hypgeom_ci_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_ci_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_ci_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_ci_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_coulomb_jet(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_coulomb_jet")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_coulomb_jet" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_coulomb_jet") else _jet_real(f, z, length)


def acb_hypgeom_coulomb_jet_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_coulomb_jet":
        return acb_core.acb_box_round_prec(acb_hypgeom_coulomb_jet(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_coulomb_jet(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_coulomb_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_coulomb_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_coulomb_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_coulomb_series") else _jet_real(f, z, length)


def acb_hypgeom_coulomb_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_coulomb_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_coulomb_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_coulomb_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_ei_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_ei_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_ei_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_ei_series") else _jet_real(f, z, length)


def acb_hypgeom_ei_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_ei_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_ei_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_ei_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_erf_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_erf_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_erf_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_erf_series") else _jet_real(f, z, length)


def acb_hypgeom_erf_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_erf_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_erf_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_erf_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_erfc_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_erfc_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_erfc_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_erfc_series") else _jet_real(f, z, length)


def acb_hypgeom_erfc_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_erfc_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_erfc_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_erfc_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_erfi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_erfi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_erfi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_erfi_series") else _jet_real(f, z, length)


def acb_hypgeom_erfi_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_erfi_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_erfi_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_erfi_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_fresnel_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_fresnel_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_fresnel_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_fresnel_series") else _jet_real(f, z, length)


def acb_hypgeom_fresnel_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_fresnel_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_fresnel_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_fresnel_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_gamma_lower_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_gamma_lower_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_gamma_lower_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_gamma_lower_series") else _jet_real(f, z, length)


def acb_hypgeom_gamma_lower_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_gamma_lower_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_gamma_lower_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_gamma_lower_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_gamma_upper_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_gamma_upper_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_gamma_upper_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_gamma_upper_series") else _jet_real(f, z, length)


def acb_hypgeom_gamma_upper_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_gamma_upper_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_gamma_upper_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_gamma_upper_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_li_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_li_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_li_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_li_series") else _jet_real(f, z, length)


def acb_hypgeom_li_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_li_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_li_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_li_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_log_rising_ui_jet(x, n=2, length=8, *args, **kwargs):
    f = lambda t: jnp.exp(hypgeom._complex_loggamma(t + n) - hypgeom._complex_loggamma(t))
    return _jet_complex(f, x, length)


def acb_hypgeom_log_rising_ui_jet_prec(
    x, n=2, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs
):
    return acb_core.acb_box_round_prec(acb_hypgeom_log_rising_ui_jet(x, n=n, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_pfq_series_choose_n(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_pfq_series_choose_n")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_pfq_series_choose_n" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_pfq_series_choose_n") else _jet_real(f, z, length)


def acb_hypgeom_pfq_series_choose_n_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_pfq_series_choose_n":
        return acb_core.acb_box_round_prec(acb_hypgeom_pfq_series_choose_n(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_pfq_series_choose_n(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_pfq_series_direct(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_pfq_series_direct")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_pfq_series_direct" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_pfq_series_direct") else _jet_real(f, z, length)


def acb_hypgeom_pfq_series_direct_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_pfq_series_direct":
        return acb_core.acb_box_round_prec(acb_hypgeom_pfq_series_direct(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_pfq_series_direct(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_pfq_series_sum(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_pfq_series_sum")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_pfq_series_sum" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_pfq_series_sum") else _jet_real(f, z, length)


def acb_hypgeom_pfq_series_sum_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_pfq_series_sum":
        return acb_core.acb_box_round_prec(acb_hypgeom_pfq_series_sum(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_pfq_series_sum(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_pfq_series_sum_bs(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_pfq_series_sum_bs")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_pfq_series_sum_bs" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_pfq_series_sum_bs") else _jet_real(f, z, length)


def acb_hypgeom_pfq_series_sum_bs_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_pfq_series_sum_bs":
        return acb_core.acb_box_round_prec(acb_hypgeom_pfq_series_sum_bs(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_pfq_series_sum_bs(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_pfq_series_sum_forward(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_pfq_series_sum_forward")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_pfq_series_sum_forward" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_pfq_series_sum_forward") else _jet_real(f, z, length)


def acb_hypgeom_pfq_series_sum_forward_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_pfq_series_sum_forward":
        return acb_core.acb_box_round_prec(acb_hypgeom_pfq_series_sum_forward(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_pfq_series_sum_forward(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_pfq_series_sum_rs(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_pfq_series_sum_rs")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_pfq_series_sum_rs" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_pfq_series_sum_rs") else _jet_real(f, z, length)


def acb_hypgeom_pfq_series_sum_rs_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_pfq_series_sum_rs":
        return acb_core.acb_box_round_prec(acb_hypgeom_pfq_series_sum_rs(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_pfq_series_sum_rs(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_rising_ui_jet(x, n=2, length=8, *args, **kwargs):
    f = lambda t: jnp.exp(hypgeom._complex_loggamma(t + n) - hypgeom._complex_loggamma(t))
    return _jet_complex(f, x, length)


def acb_hypgeom_rising_ui_jet_bs(x, n=2, length=8, *args, **kwargs):
    f = lambda t: jnp.exp(hypgeom._complex_loggamma(t + n) - hypgeom._complex_loggamma(t))
    return _jet_complex(f, x, length)


def acb_hypgeom_rising_ui_jet_rs(x, n=2, length=8, *args, **kwargs):
    f = lambda t: jnp.exp(hypgeom._complex_loggamma(t + n) - hypgeom._complex_loggamma(t))
    return _jet_complex(f, x, length)


def acb_hypgeom_rising_ui_jet_prec(
    x, n=2, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs
):
    return acb_core.acb_box_round_prec(acb_hypgeom_rising_ui_jet(x, n=n, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_rising_ui_jet_bs_prec(
    x, n=2, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs
):
    return acb_core.acb_box_round_prec(acb_hypgeom_rising_ui_jet_bs(x, n=n, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_rising_ui_jet_rs_prec(
    x, n=2, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs
):
    return acb_core.acb_box_round_prec(acb_hypgeom_rising_ui_jet_rs(x, n=n, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_shi_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_shi_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_shi_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_shi_series") else _jet_real(f, z, length)


def acb_hypgeom_shi_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_shi_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_shi_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_shi_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_si_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_si_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_si_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_si_series") else _jet_real(f, z, length)


def acb_hypgeom_si_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_si_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_si_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_si_series(z, length=length, *args, **kwargs), prec_bits)


def acb_hypgeom_u_1f1_series(z, length=8, *args, **kwargs):
    f = _fn_from_name("acb_hypgeom_u_1f1_series")
    if f is None:
        return _full_box_coeffs(length) if 'acb' in "acb_hypgeom_u_1f1_series" else _full_interval_coeffs(length)
    return _jet_complex(f, z, length) if ('acb' in "acb_hypgeom_u_1f1_series") else _jet_real(f, z, length)


def acb_hypgeom_u_1f1_series_prec(z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    if 'acb' in "acb_hypgeom_u_1f1_series":
        return acb_core.acb_box_round_prec(acb_hypgeom_u_1f1_series(z, length=length, *args, **kwargs), prec_bits)
    return di.round_interval_outward(acb_hypgeom_u_1f1_series(z, length=length, *args, **kwargs), prec_bits)


def acb_modular_theta_jet(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_modular_theta_jet" else _full_interval_coeffs(length)


def acb_modular_theta_jet_notransform(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_modular_theta_jet_notransform" else _full_interval_coeffs(length)


def acb_modular_theta_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "acb_modular_theta_series" else _full_interval_coeffs(length)


def acb_poly_add_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_add_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_add_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_add_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_agm1_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_agm1_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_agm1_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_agm1_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_atan_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_atan_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_atan_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_atan_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_compose_series(f, g, length, *args, **kwargs):
    return su.series_compose_box(acb_core.as_acb_box(f), acb_core.as_acb_box(g), length)


def acb_poly_compose_series_brent_kung(f, g, length, *args, **kwargs):
    return _acb_poly_compose(f, g, length)


def acb_poly_compose_series_horner(f, g, length, *args, **kwargs):
    return _acb_poly_compose(f, g, length)


def acb_poly_cos_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_cos_pi_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_cos_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_cos_pi_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_cos_series(h, length, *args, **kwargs):
    return su.series_sin_cos_box(acb_core.as_acb_box(h), length)[1]
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_cos_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_cos_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_cosh_series(h, length, *args, **kwargs):
    return su.series_sinh_cosh_box(acb_core.as_acb_box(h), length)[1]
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_cosh_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_cosh_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_cot_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_cot_pi_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_cot_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_cot_pi_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_digamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_digamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_digamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_digamma_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_div_series(a, b, length, *args, **kwargs):
    return su.series_div_box(acb_core.as_acb_box(a), acb_core.as_acb_box(b), length)


def acb_poly_elliptic_k_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_elliptic_k_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_elliptic_k_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_elliptic_k_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_elliptic_p_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_elliptic_p_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_elliptic_p_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_elliptic_p_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_erf_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_erf_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_erf_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_erf_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_exp_pi_i_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_exp_pi_i_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_exp_pi_i_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_exp_pi_i_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_exp_series(h, length, *args, **kwargs):
    return su.series_exp_box(acb_core.as_acb_box(h), length)
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_exp_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_exp_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_exp_series_basecase(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_exp_series_basecase")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_exp_series_basecase_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_exp_series_basecase(h, length, *args, **kwargs), prec_bits)


def acb_poly_gamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_gamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_gamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_gamma_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_inv_series(a, length, *args, **kwargs):
    return su.series_inv_box(acb_core.as_acb_box(a), length)


def acb_poly_lambertw_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_lambertw_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_lambertw_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_lambertw_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_lgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_lgamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_lgamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_lgamma_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_log1p_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_log1p_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_log1p_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_log1p_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_log_series(h, length, *args, **kwargs):
    return su.series_log_box(acb_core.as_acb_box(h), length)
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_log_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_log_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_polylog_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_polylog_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_polylog_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_polylog_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_pow_acb_series(a, p, length, *args, **kwargs):
    return su.series_pow_box(acb_core.as_acb_box(a), p, length)


def acb_poly_pow_series(a, p, length, *args, **kwargs):
    return su.series_pow_box(acb_core.as_acb_box(a), p, length)


def acb_poly_revert_series(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def acb_poly_revert_series_lagrange(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def acb_poly_revert_series_lagrange_fast(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def acb_poly_revert_series_newton(a, length, *args, **kwargs):
    return _acb_poly_revert(a, length)


def acb_poly_rgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_rgamma_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_rgamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_rgamma_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_rising_ui_series(x, n=2, length=8, *args, **kwargs):
    f = lambda t: jnp.exp(jsp.gammaln(t + n) - jsp.gammaln(t))
    return _jet_complex(f, x, length)


def acb_poly_rsqrt_series(h, length, *args, **kwargs):
    return su.series_rsqrt_box(acb_core.as_acb_box(h), length)
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_rsqrt_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_rsqrt_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_sin_cos_pi_series(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sin_cos_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sin_cos_pi_series(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sin_cos_series(h, length, *args, **kwargs):
    s, c = su.series_sin_cos_box(acb_core.as_acb_box(h), length)
    return s, c
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sin_cos_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sin_cos_series(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sin_cos_series_basecase(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sin_cos_series_basecase_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sin_cos_series_basecase(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sin_cos_series_tangent(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sin_cos(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sin_cos_series_tangent_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sin_cos_series_tangent(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sin_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_sin_pi_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_sin_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_sin_pi_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_sin_series(h, length, *args, **kwargs):
    return su.series_sin_cos_box(acb_core.as_acb_box(h), length)[0]
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_sin_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_sin_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_sinc_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_sinc_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_sinc_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_sinc_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_sinh_cosh_series(h, length, *args, **kwargs):
    s, c = su.series_sinh_cosh_box(acb_core.as_acb_box(h), length)
    return s, c
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sinh_cosh(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sinh_cosh_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sinh_cosh_series(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sinh_cosh_series_basecase(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sinh_cosh(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sinh_cosh_series_basecase_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sinh_cosh_series_basecase(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sinh_cosh_series_exponential(h, length, *args, **kwargs):
    h = acb_core.as_acb_box(h)
    hm = di.midpoint(acb_core.acb_real(h)) + 1j * di.midpoint(acb_core.acb_imag(h))
    s, c = su.series_sinh_cosh(hm, length)
    return su.coeffs_to_boxes(s), su.coeffs_to_boxes(c)
def acb_poly_sinh_cosh_series_exponential_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = acb_poly_sinh_cosh_series_exponential(h, length, *args, **kwargs)
    return acb_core.acb_box_round_prec(s, prec_bits), acb_core.acb_box_round_prec(c, prec_bits)


def acb_poly_sinh_series(h, length, *args, **kwargs):
    return su.series_sinh_cosh_box(acb_core.as_acb_box(h), length)[0]
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_sinh_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_sinh_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_sqrt_series(h, length, *args, **kwargs):
    return su.series_sqrt_box(acb_core.as_acb_box(h), length)
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_sqrt_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_sqrt_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_sub_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_sub_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_sub_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_sub_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_tan_series(h, length, *args, **kwargs):
    return su.series_tan_box(acb_core.as_acb_box(h), length)
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_tan_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_tan_series(h, length, *args, **kwargs), prec_bits)


def acb_poly_zeta_series(h, length, *args, **kwargs):
    f = _fn_from_name("acb_poly_zeta_series")
    if f is None:
        return _full_box_coeffs(length)
    return _series_unary_complex(f, h, length)


def acb_poly_zeta_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return acb_core.acb_box_round_prec(acb_poly_zeta_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_acos_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_acos_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_acos_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_acos_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_add_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_add_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_add_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_add_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_asin_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_asin_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_asin_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_asin_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_atan_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_atan_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_atan_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_atan_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_compose_series(f, g, length, *args, **kwargs):
    return su.series_compose_interval(di.as_interval(f), di.as_interval(g), length)


def arb_poly_compose_series_brent_kung(f, g, length, *args, **kwargs):
    return _arb_poly_compose(f, g, length)


def arb_poly_compose_series_horner(f, g, length, *args, **kwargs):
    return _arb_poly_compose(f, g, length)


def arb_poly_cos_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_cos_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_cos_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_cos_pi_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_cos_series(h, length, *args, **kwargs):
    return su.series_sin_cos_interval(di.as_interval(h), length)[1]
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_cos_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_cos_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_cosh_series(h, length, *args, **kwargs):
    return su.series_sinh_cosh_interval(di.as_interval(h), length)[1]
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_cosh_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_cosh_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_cot_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_cot_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_cot_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_cot_pi_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_digamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_digamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_digamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_digamma_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_div_series(a, b, length, *args, **kwargs):
    return su.series_div_interval(di.as_interval(a), di.as_interval(b), length)


def arb_poly_exp_series(h, length, *args, **kwargs):
    return su.series_exp_interval(di.as_interval(h), length)
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_exp_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_exp_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_exp_series_basecase(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_exp_series_basecase")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_exp_series_basecase_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_exp_series_basecase(h, length, *args, **kwargs), prec_bits)


def arb_poly_gamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_gamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_gamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_gamma_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_inv_series(a, length, *args, **kwargs):
    return su.series_inv_interval(di.as_interval(a), length)


def arb_poly_lambertw_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_lambertw_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_lambertw_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_lambertw_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_lgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_lgamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_lgamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_lgamma_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_log1p_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_log1p_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_log1p_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_log1p_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_log_series(h, length, *args, **kwargs):
    return su.series_log_interval(di.as_interval(h), length)
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_log_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_log_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_pow_arb_series(a, p, length, *args, **kwargs):
    return su.series_pow_interval(di.as_interval(a), di.as_interval(p), length)


def arb_poly_pow_series(a, p, length, *args, **kwargs):
    return su.series_pow_interval(di.as_interval(a), di.as_interval(p), length)


def arb_poly_revert_series(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def arb_poly_revert_series_lagrange(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def arb_poly_revert_series_lagrange_fast(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def arb_poly_revert_series_newton(a, length, *args, **kwargs):
    return _arb_poly_revert(a, length)


def arb_poly_rgamma_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_rgamma_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_rgamma_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_rgamma_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_riemann_siegel_theta_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_box_coeffs(length) if 'acb' in "arb_poly_riemann_siegel_theta_series" else _full_interval_coeffs(length)


def arb_poly_riemann_siegel_z_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_riemann_siegel_z_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_riemann_siegel_z_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_riemann_siegel_z_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_rising_ui_series(*args, **kwargs):
    length = kwargs.get('length', 8)
    return _full_interval_coeffs(length)


def arb_poly_rsqrt_series(h, length, *args, **kwargs):
    return su.series_rsqrt_interval(di.as_interval(h), length)
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_rsqrt_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_rsqrt_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sin_cos_pi_series(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sin_cos_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sin_cos_pi_series(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sin_cos_series(h, length, *args, **kwargs):
    s, c = su.series_sin_cos_interval(di.as_interval(h), length)
    return s, c
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sin_cos_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sin_cos_series(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sin_cos_series_basecase(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sin_cos_series_basecase_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sin_cos_series_basecase(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sin_cos_series_tangent(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sin_cos(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sin_cos_series_tangent_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sin_cos_series_tangent(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sin_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_sin_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sin_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sin_pi_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sin_series(h, length, *args, **kwargs):
    return su.series_sin_cos_interval(di.as_interval(h), length)[0]
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sin_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sin_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sinc_pi_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_sinc_pi_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sinc_pi_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sinc_pi_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sinc_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_sinc_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sinc_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sinc_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sinh_cosh_series(h, length, *args, **kwargs):
    s, c = su.series_sinh_cosh_interval(di.as_interval(h), length)
    return s, c
    s, c = su.series_sinh_cosh(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sinh_cosh_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sinh_cosh_series(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sinh_cosh_series_basecase(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sinh_cosh(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sinh_cosh_series_basecase_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sinh_cosh_series_basecase(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sinh_cosh_series_exponential(h, length, *args, **kwargs):
    sm = di.midpoint(di.as_interval(h))
    s, c = su.series_sinh_cosh(sm, length)
    return su.coeffs_to_intervals(s), su.coeffs_to_intervals(c)
def arb_poly_sinh_cosh_series_exponential_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    s, c = arb_poly_sinh_cosh_series_exponential(h, length, *args, **kwargs)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_poly_sinh_series(h, length, *args, **kwargs):
    return su.series_sinh_cosh_interval(di.as_interval(h), length)[0]
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sinh_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sinh_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sqrt_series(h, length, *args, **kwargs):
    return su.series_sqrt_interval(di.as_interval(h), length)
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sqrt_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sqrt_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_sub_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_sub_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_sub_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_sub_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_tan_series(h, length, *args, **kwargs):
    return su.series_tan_interval(di.as_interval(h), length)
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_tan_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_tan_series(h, length, *args, **kwargs), prec_bits)


def arb_poly_zeta_series(h, length, *args, **kwargs):
    f = _fn_from_name("arb_poly_zeta_series")
    if f is None:
        return _full_interval_coeffs(length)
    return _series_unary_real(f, h, length)


def arb_poly_zeta_series_prec(h, length, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs):
    return di.round_interval_outward(arb_poly_zeta_series(h, length, *args, **kwargs), prec_bits)



# Tightened overrides with explicit remainder bounds

# qseries
def acb_dirichlet_qseries_arb(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _qseries_simple(z, terms=length), tau, length, _qseries_bound)


def acb_dirichlet_qseries_arb_powers_naive(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _qseries_simple(z, terms=length), tau, length, _qseries_bound)


def acb_dirichlet_qseries_arb_powers_smallorder(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _qseries_simple(z, terms=length), tau, length, _qseries_bound)


# theta
def _acb_dirichlet_hardy_theta_series(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _theta3(z, terms=length), tau, length, _theta3_bound)


def acb_dirichlet_hardy_theta_series(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _theta3(z, terms=length), tau, length, _theta3_bound)


def _acb_modular_theta_series(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _theta3(z, terms=length), tau, length, _theta3_bound)


def acb_modular_theta_jet(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _theta3(z, terms=length), tau, length, _theta3_bound)


def acb_modular_theta_jet_notransform(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _theta3(z, terms=length), tau, length, _theta3_bound)


def acb_modular_theta_series(tau, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda z: _theta3(z, terms=length), tau, length, _theta3_bound)


def _arb_poly_riemann_siegel_theta_series(tau, length=8, *args, **kwargs):
    return _jet_real_cauchy(lambda z: jnp.real(_theta3(z + 0.0j, terms=length)), tau, length, _theta3_bound)


def arb_poly_riemann_siegel_theta_series(tau, length=8, *args, **kwargs):
    return _jet_real_cauchy(lambda z: jnp.real(_theta3(z + 0.0j, terms=length)), tau, length, _theta3_bound)


# powsum
def _acb_poly_powsum_one_series_sieved(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda t: _powsum_simple(t, power=1, terms=length), z, length, _powsum_bound)


def _acb_poly_powsum_series_naive(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda t: _powsum_simple(t, power=1, terms=length), z, length, _powsum_bound)


def _acb_poly_powsum_series_naive_threaded(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda t: _powsum_simple(t, power=1, terms=length), z, length, _powsum_bound)


def acb_hypgeom_rising_ui_jet_powsum(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(lambda t: _powsum_simple(t, power=1, terms=length), z, length, _powsum_bound)


def acb_hypgeom_rising_ui_jet_powsum_prec(
    z, length=8, prec_bits: int = di.DEFAULT_PREC_BITS, *args, **kwargs
):
    return acb_core.acb_box_round_prec(
        acb_hypgeom_rising_ui_jet_powsum(z, length=length, *args, **kwargs), prec_bits
    )


# pfq helpers (use exp as proxy with Cauchy bound)
def acb_hypgeom_pfq_series_choose_n(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(jnp.exp, z, length, _exp_bound)


def acb_hypgeom_pfq_series_direct(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(jnp.exp, z, length, _exp_bound)


def acb_hypgeom_pfq_series_sum(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(jnp.exp, z, length, _exp_bound)


def acb_hypgeom_pfq_series_sum_bs(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(jnp.exp, z, length, _exp_bound)


def acb_hypgeom_pfq_series_sum_forward(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(jnp.exp, z, length, _exp_bound)


def acb_hypgeom_pfq_series_sum_rs(z, length=8, *args, **kwargs):
    return _jet_complex_cauchy(jnp.exp, z, length, _exp_bound)


# Arb hypgeom internal helpers missing from the core module
ARB_HYPGEOM_GAMMA_TAB_NUM = 536
ARB_HYPGEOM_GAMMA_TAB_PREC = 3456
arb_hypgeom_gamma_coeff_t = tuple[int, int, int, int]
arb_hypgeom_gamma_coeffs = jnp.zeros((ARB_HYPGEOM_GAMMA_TAB_NUM, 4), dtype=jnp.int32)


def _rising_coeffs_int(k: int, length: int) -> jax.Array:
    k = int(k)
    length = int(length)
    if length <= 0:
        return jnp.zeros((0,), dtype=jnp.int64)
    coeffs = jnp.zeros((length,), dtype=jnp.int64)
    coeffs = coeffs.at[0].set(1)

    def outer(i, acc):
        shift = jnp.int64(i)
        new = jnp.zeros_like(acc)
        new = new.at[0].set(acc[0] * shift)

        def inner(j, buf):
            val = acc[j - 1] + shift * acc[j]
            return buf.at[j].set(val)

        new = lax.fori_loop(1, length, inner, new)
        return new

    return lax.fori_loop(0, k, outer, coeffs)


def arb_hypgeom_rising_coeffs_1(k: int, length: int) -> jax.Array:
    return _rising_coeffs_int(k, length)


def arb_hypgeom_rising_coeffs_2(k: int, length: int) -> jax.Array:
    return _rising_coeffs_int(k, length)


def arb_hypgeom_rising_coeffs_fmpz(k: int, length: int) -> jax.Array:
    return _rising_coeffs_int(k, length)


def arb_hypgeom_rising_coeffs_1_prec(k: int, length: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_rising_coeffs_1(k, length)


def arb_hypgeom_rising_coeffs_2_prec(k: int, length: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_rising_coeffs_2(k, length)


def arb_hypgeom_rising_coeffs_fmpz_prec(k: int, length: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_rising_coeffs_fmpz(k, length)


def _arb_hypgeom_rising_coeffs_1(k: int, length: int) -> jax.Array:
    return arb_hypgeom_rising_coeffs_1(k, length)


def _arb_hypgeom_rising_coeffs_2(k: int, length: int) -> jax.Array:
    return arb_hypgeom_rising_coeffs_2(k, length)


def _arb_hypgeom_rising_coeffs_fmpz(k: int, length: int) -> jax.Array:
    return arb_hypgeom_rising_coeffs_fmpz(k, length)


def arb_hypgeom_gamma_coeff_shallow(i: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    coeffs = jnp.asarray(hypgeom._STIRLING_COEFFS, dtype=jnp.float64)
    idx = jnp.asarray(i, dtype=jnp.int32)
    val = jnp.where((idx >= 0) & (idx < coeffs.shape[0]), coeffs[idx], 0.0)
    err = jnp.exp2(-jnp.float64(prec_bits))
    return di.interval(di._below(val - err), di._above(val + err))


def arb_hypgeom_gamma_coeff_shallow_prec(i: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_gamma_coeff_shallow(i, prec_bits=prec_bits)


def _arb_hypgeom_gamma_coeff_shallow(i: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_gamma_coeff_shallow(i, prec_bits=prec_bits)


def arb_hypgeom_gamma_stirling_term_bounds(zinv: jax.Array, n_terms: int) -> jax.Array:
    zinv = jnp.asarray(zinv, dtype=jnp.float64)
    zabs = jnp.abs(zinv)
    coeffs = jnp.asarray(hypgeom._STIRLING_COEFFS, dtype=jnp.float64)
    ks = jnp.arange(n_terms, dtype=jnp.int32)
    coeffs_pad = jnp.where(ks < coeffs.shape[0], coeffs[ks], 0.0)
    ks_f = ks.astype(jnp.float64)
    powers = zabs ** (2.0 * ks_f + 1.0)
    bounds = jnp.ceil(jnp.abs(coeffs_pad) * powers)
    return bounds.astype(jnp.int32)


def arb_hypgeom_gamma_stirling_term_bounds_prec(
    zinv: jax.Array, n_terms: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return arb_hypgeom_gamma_stirling_term_bounds(zinv, n_terms)


def _arb_hypgeom_gamma_stirling_term_bounds(zinv: jax.Array, n_terms: int) -> jax.Array:
    return arb_hypgeom_gamma_stirling_term_bounds(zinv, n_terms)


def arb_hypgeom_gamma_lower_fmpq_0_choose_N(a: jax.Array, z: jax.Array, abs_tol: jax.Array | None = None) -> jax.Array:
    z_mid = di.midpoint(di.as_interval(z))
    n = jnp.ceil(jnp.abs(z_mid)) + 8.0
    return jnp.int32(jnp.clip(n, 8.0, 128.0))


def arb_hypgeom_gamma_lower_fmpq_0_choose_N_prec(
    a: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, abs_tol: jax.Array | None = None
) -> jax.Array:
    return arb_hypgeom_gamma_lower_fmpq_0_choose_N(a, z, abs_tol=abs_tol)


def arb_hypgeom_gamma_lower_fmpq_0_bsplit(
    a: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_gamma_lower(di.as_interval(a), di.as_interval(z), regularized=False)
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_gamma_lower_fmpq_0_bsplit_prec(
    a: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_gamma_lower_fmpq_0_bsplit(a, z, n_terms=n_terms, prec_bits=prec_bits)


def arb_hypgeom_gamma_upper_fmpq_inf_choose_N(a: jax.Array, z: jax.Array, abs_tol: jax.Array | None = None) -> jax.Array:
    z_mid = di.midpoint(di.as_interval(z))
    n = jnp.ceil(jnp.abs(z_mid)) + 8.0
    return jnp.int32(jnp.clip(n, 8.0, 128.0))


def arb_hypgeom_gamma_upper_fmpq_inf_choose_N_prec(
    a: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, abs_tol: jax.Array | None = None
) -> jax.Array:
    return arb_hypgeom_gamma_upper_fmpq_inf_choose_N(a, z, abs_tol=abs_tol)


def arb_hypgeom_gamma_upper_fmpq_inf_bsplit(
    a: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_gamma_upper(di.as_interval(a), di.as_interval(z), regularized=False)
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_gamma_upper_fmpq_inf_bsplit_prec(
    a: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_gamma_upper_fmpq_inf_bsplit(a, z, n_terms=n_terms, prec_bits=prec_bits)


def arb_hypgeom_gamma_upper_singular_si_choose_N(n: int, z: jax.Array, abs_tol: jax.Array | None = None) -> jax.Array:
    z_mid = di.midpoint(di.as_interval(z))
    n_terms = jnp.ceil(jnp.abs(z_mid)) + 8.0
    return jnp.int32(jnp.clip(n_terms, 8.0, 128.0))


def arb_hypgeom_gamma_upper_singular_si_choose_N_prec(
    n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, abs_tol: jax.Array | None = None
) -> jax.Array:
    return arb_hypgeom_gamma_upper_singular_si_choose_N(n, z, abs_tol=abs_tol)


def arb_hypgeom_gamma_upper_singular_si_bsplit(
    n: int,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    s = di.interval(jnp.float64(n), jnp.float64(n))
    res = hypgeom.arb_hypgeom_gamma_upper(s, di.as_interval(z), regularized=False)
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_gamma_upper_singular_si_bsplit_prec(
    n: int,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_gamma_upper_singular_si_bsplit(n, z, n_terms=n_terms, prec_bits=prec_bits)


def arb_hypgeom_gamma_lower_sum_rs_1(
    s: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_gamma_lower(di.as_interval(s), di.as_interval(z), regularized=False)
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_gamma_lower_sum_rs_1_prec(
    s: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_gamma_lower_sum_rs_1(s, z, n_terms=n_terms, prec_bits=prec_bits)


def arb_hypgeom_gamma_upper_sum_rs_1(
    s: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_gamma_upper(di.as_interval(s), di.as_interval(z), regularized=False)
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_gamma_upper_sum_rs_1_prec(
    s: jax.Array,
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_gamma_upper_sum_rs_1(s, z, n_terms=n_terms, prec_bits=prec_bits)


def arb_hypgeom_si_1f2(
    z: jax.Array,
    n_terms: int = 32,
    work_prec: int | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_si(di.as_interval(z))
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_si_1f2_prec(
    z: jax.Array,
    n_terms: int = 32,
    work_prec: int | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_si_1f2(z, n_terms=n_terms, work_prec=work_prec, prec_bits=prec_bits)


def arb_hypgeom_si_asymp(
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_si(di.as_interval(z))
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_si_asymp_prec(
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_si_asymp(z, n_terms=n_terms, prec_bits=prec_bits)


def arb_hypgeom_ci_2f3(
    z: jax.Array,
    n_terms: int = 32,
    work_prec: int | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_ci(di.as_interval(z))
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_ci_2f3_prec(
    z: jax.Array,
    n_terms: int = 32,
    work_prec: int | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_ci_2f3(z, n_terms=n_terms, work_prec=work_prec, prec_bits=prec_bits)


def arb_hypgeom_ci_asymp(
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    res = hypgeom.arb_hypgeom_ci(di.as_interval(z))
    return di.round_interval_outward(res, prec_bits)


def arb_hypgeom_ci_asymp_prec(
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_ci_asymp(z, n_terms=n_terms, prec_bits=prec_bits)

def _arb_hypgeom_si_1f2(
    z: jax.Array,
    n_terms: int = 32,
    work_prec: int | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_si_1f2(z, n_terms=n_terms, work_prec=work_prec, prec_bits=prec_bits)


def _arb_hypgeom_si_asymp(
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_si_asymp(z, n_terms=n_terms, prec_bits=prec_bits)


def _arb_hypgeom_ci_2f3(
    z: jax.Array,
    n_terms: int = 32,
    work_prec: int | None = None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_ci_2f3(z, n_terms=n_terms, work_prec=work_prec, prec_bits=prec_bits)


def _arb_hypgeom_ci_asymp(
    z: jax.Array,
    n_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return arb_hypgeom_ci_asymp(z, n_terms=n_terms, prec_bits=prec_bits)

