from __future__ import annotations

"""BarnesDoubleGamma.jl-lineage Barnes and double-gamma families.

These functions were ported from `BarnesDoubleGamma.jl`. Under the current
repo naming policy, non-Arb/FLINT implementation lineages must expose a short
provider prefix in the public name, so this module exports the `bdg_*` family.

Provenance:
- classification: alternative
- module lineage: Julia/BarnesDoubleGamma.jl-derived implementation family
- naming policy: see docs/standards/function_naming.md
- registry report: see docs/status/reports/function_implementation_index.md
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import barnesg
from . import double_interval as di
from . import acb_core
from . import wrappers_common as wc
from . import precision
from . import checks
from . import elementary as el
from . import kernel_helpers as kh

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "alternative",
    "base_names": ("barnesdoublegamma", "barnesgamma2", "normalizeddoublegamma", "double_sine"),
    "preferred_prefix": "bdg",
    "module_lineage": "Julia/BarnesDoubleGamma.jl-derived implementation family",
    "naming_policy": "docs/standards/function_naming.md",
    "registry_report": "docs/status/reports/function_implementation_index.md",
}
def _complex_loggamma(z: jax.Array) -> jax.Array:
    return barnesg._complex_loggamma(z)


def _digamma(z: jax.Array) -> jax.Array:
    cdt = el.complex_dtype_from(z)
    h = jnp.asarray(1e-6 + 0.0j, dtype=cdt)
    return (_complex_loggamma(z + h) - _complex_loggamma(z - h)) / (2.0 * h)


def _trigamma(z: jax.Array) -> jax.Array:
    cdt = el.complex_dtype_from(z)
    h = jnp.asarray(1e-5 + 0.0j, dtype=cdt)
    return (_digamma(z + h) - _digamma(z - h)) / (2.0 * h)


def _integrate_trapz(fn, a: jax.Array | float, b: jax.Array | float, n: int) -> jax.Array:
    dt = jnp.result_type(jnp.asarray(a).dtype, jnp.asarray(b).dtype)
    if not jnp.issubdtype(dt, jnp.floating):
        dt = jnp.float32 if dt == jnp.complex64 else jnp.float64
    xs = jnp.linspace(jnp.asarray(a, dtype=dt), jnp.asarray(b, dtype=dt), n)
    vals = fn(xs)
    dx = xs[1:] - xs[:-1]
    return jnp.sum(0.5 * (vals[1:] + vals[:-1]) * dx)


def _integrandC(x: jax.Array, tau: jax.Array) -> jax.Array:
    rdt = jnp.result_type(jnp.asarray(x).dtype, jnp.asarray(jnp.real(tau)).dtype)
    one = jnp.asarray(1.0, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    one_minus_tau = one - tau
    sinh_x = jnp.sinh(x)
    sinh_taux = jnp.sinh(tau * x)
    exp1 = jnp.exp(one_minus_tau * x)
    exp2 = jnp.exp(-two * x)
    exp3 = jnp.exp(x)
    term1 = exp1 / (two * sinh_x * sinh_taux)
    term2 = exp2 / (tau * x) * (exp3 / (two * sinh_x) + one - tau / two)
    return term1 - term2


def _integrandD(x: jax.Array, tau: jax.Array) -> jax.Array:
    rdt = jnp.result_type(jnp.asarray(x).dtype, jnp.asarray(jnp.real(tau)).dtype)
    one = jnp.asarray(1.0, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    return x * jnp.exp((one - tau) * x) / (jnp.sinh(x) * jnp.sinh(tau * x)) - jnp.exp(-two * x) / (tau * x)


def _modularC(tau: jax.Array, prec_bits: int) -> jax.Array:
    cdt = el.complex_dtype_from(tau)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    P = max(int(prec_bits // 2), 8)
    one = jnp.asarray(1.0, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    three = jnp.asarray(3.0, dtype=rdt)
    five = jnp.asarray(5.0, dtype=rdt)
    six = jnp.asarray(6.0, dtype=rdt)
    nine = jnp.asarray(9.0, dtype=rdt)
    twelve = jnp.asarray(12.0, dtype=rdt)
    twenty = jnp.asarray(20.0, dtype=rdt)
    two_seventy = jnp.asarray(270.0, dtype=rdt)
    cutoff = jnp.exp2(-jnp.asarray(P, dtype=rdt) / three)
    upper = jnp.maximum(twenty, five * jnp.abs(tau))

    def fn(x):
        return _integrandC(x, tau)

    val = _integrate_trapz(fn, cutoff, upper, 256)
    c0 = (two / tau - three / two + tau / six) * cutoff + (five / twelve - one / tau + tau / twelve) * cutoff ** 2
    c0 = c0 + (jnp.asarray(4.0, dtype=rdt) / (nine * tau) - two / nine + tau / jnp.asarray(54.0, dtype=rdt) - (tau ** 3) / two_seventy) * cutoff ** 3
    return (one / (two * tau)) * jnp.log(jnp.asarray(2.0 * el.PI, dtype=rdt)) - val - c0


def _modularD(tau: jax.Array, prec_bits: int) -> jax.Array:
    cdt = el.complex_dtype_from(tau)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    upper = jnp.maximum(jnp.asarray(20.0, dtype=rdt), jnp.asarray(5.0, dtype=rdt) * jnp.abs(tau))

    def fn(x):
        return _integrandD(x, tau)

    return _integrate_trapz(fn, jnp.asarray(1e-8, dtype=rdt), upper, 256)


def _modularcoeff_a(tau: jax.Array, prec_bits: int) -> jax.Array:
    cdt = el.complex_dtype_from(tau)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    modc = _modularC(tau, prec_bits)
    half = jnp.asarray(0.5, dtype=rdt)
    two_pi = jnp.asarray(2.0 * el.PI, dtype=cdt)
    return half * tau * jnp.log(two_pi * tau) + half * jnp.log(tau) - tau * modc


def _modularcoeff_b(tau: jax.Array, prec_bits: int) -> jax.Array:
    cdt = el.complex_dtype_from(tau)
    modd = _modularD(tau, prec_bits)
    tau = jnp.asarray(tau, dtype=cdt)
    return -tau * jnp.log(tau) - tau * tau * modd


def _evalpoly(z: jax.Array, coeffs: jax.Array) -> jax.Array:
    cdt = el.complex_dtype_from(z, coeffs)
    coeffs = jnp.asarray(coeffs, dtype=cdt)

    def body(i, acc):
        c = coeffs[coeffs.shape[0] - 1 - i]
        return acc * z + c

    return lax.fori_loop(0, coeffs.shape[0], body, jnp.zeros((), dtype=cdt))


def _polynomial_Pns(M: int, tau_pows, taup1_pows, factorials):
    cdt = el.complex_dtype_from(tau_pows[0], taup1_pows[0])
    rdt = el.real_dtype_from_complex_dtype(cdt)
    coeffs = [jnp.zeros((n,), dtype=cdt) for n in range(1, M + 1)]
    tau_factors = []
    for k in range(1, M + 1):
        tf = (taup1_pows[k + 1] - jnp.asarray(1.0, dtype=rdt) - tau_pows[k + 1]) / factorials[k + 1] / tau_pows[0]
        tau_factors.append(tf)
    for n in range(1, M + 1):
        c = coeffs[n - 1]
        c = c.at[n - 1].set(jnp.asarray(1.0, dtype=rdt) / factorials[n + 1])
        for j in range(0, n - 1):
            acc = jnp.asarray(0.0 + 0.0j, dtype=cdt)
            for k in range(1, n - j):
                acc = acc + tau_factors[k - 1] * coeffs[n - k - 1][j]
            c = c.at[j].set(-acc)
        coeffs[n - 1] = c
    return coeffs


def _rest_RMN(z: jax.Array, tau: jax.Array, M: int, N: int, tau_pows, taup1_pows, Nm_tau_pows, factorials, poly):
    cdt = el.complex_dtype_from(z, tau)
    coeffs_sum = [jnp.zeros((), dtype=cdt)]
    coeffs_sum.extend(factorials[k - 1] * _evalpoly(z, poly[k - 1]) for k in range(1, M + 1))
    m_tau = -tau
    coeffs_sum_arr = jnp.stack(coeffs_sum)
    return jnp.sum(jnp.asarray(Nm_tau_pows, dtype=cdt) * coeffs_sum_arr) / m_tau


def _log_barnesdoublegamma_mid(z: jax.Array, tau: jax.Array, prec_bits: int) -> jax.Array:
    cdt = el.complex_dtype_from(z, tau)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    tau = jnp.asarray(tau, dtype=cdt)
    z = jnp.asarray(z, dtype=cdt)
    M = max(2, int(jnp.floor(0.5 / jnp.log(jnp.float64(20.0)) * max(prec_bits, 32))))
    N = max(20, 50 * M)

    logtau = jnp.log(tau)
    moda = _modularcoeff_a(tau, prec_bits)
    modb = _modularcoeff_b(tau, prec_bits)

    m = jnp.arange(1, N + 1, dtype=rdt)
    mt = m * tau
    loggammas = _complex_loggamma(mt)
    digammas = _digamma(mt)
    trigammas = _trigamma(mt)

    zsq2 = jnp.asarray(0.5, dtype=rdt) * z * z
    res = -_complex_loggamma(z) - logtau
    res = res + moda * (z / tau)
    res = res + modb * (zsq2 / (tau * tau))

    res = res + jnp.sum(loggammas - _complex_loggamma(z + mt) + z * digammas + zsq2 * trigammas)

    k_idx = jnp.arange(1, M + 3, dtype=rdt)
    tau_pows = (-tau) ** k_idx
    taup1_pows = (-tau + jnp.asarray(1.0, dtype=rdt)) ** k_idx
    nm_base = -(jnp.asarray(1.0, dtype=rdt) / (jnp.asarray(N, dtype=rdt) * tau))
    Nm_tau_pows = jnp.concatenate(
        [jnp.ones((1,), dtype=cdt), jnp.cumprod(jnp.full((M,), nm_base, dtype=cdt))]
    )
    factorials = jnp.exp(lax.lgamma(jnp.arange(1, M + 3, dtype=rdt)))
    poly = _polynomial_Pns(M, tau_pows, taup1_pows, factorials)
    rest = _rest_RMN(z, tau, M, N, tau_pows, taup1_pows, Nm_tau_pows, factorials, poly)
    return res + (z ** 3) * rest


def bdg_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _log_barnesdoublegamma_mid(z, tau, prec_bits)


def bdg_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return jnp.exp(bdg_log_barnesdoublegamma(z, tau, prec_bits))


def bdg_log_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    cdt = el.complex_dtype_from(w, beta)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    beta = jnp.asarray(beta, dtype=cdt)
    w = jnp.asarray(w, dtype=cdt)
    one = jnp.asarray(1.0, dtype=rdt)
    half = jnp.asarray(0.5, dtype=rdt)
    beta = jnp.where(jnp.real(beta - one / beta) < 0.0, one / beta, beta)
    invb = one / beta
    tau = invb * invb
    logb = jnp.log(beta)
    c1 = half * invb * jnp.log(jnp.asarray(2.0 * el.PI, dtype=rdt))
    c2 = -beta - invb
    l = bdg_log_barnesdoublegamma(w / beta, tau, prec_bits)
    return w * c1 + (half * w * (w + c2) + one) * logb - l


def bdg_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return jnp.exp(bdg_log_barnesgamma2(w, beta, prec_bits))


def bdg_log_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    cdt = el.complex_dtype_from(w, beta)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    one = jnp.asarray(1.0, dtype=rdt)
    half = jnp.asarray(0.5, dtype=rdt)
    return bdg_log_barnesgamma2(w, beta, prec_bits) - bdg_log_barnesgamma2((beta + one / beta) * half, beta, prec_bits)


def bdg_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return jnp.exp(bdg_log_normalizeddoublegamma(w, beta, prec_bits))


def bdg_double_sine(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    cdt = el.complex_dtype_from(z, b)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    b = jnp.asarray(b, dtype=cdt)
    z = jnp.asarray(z, dtype=cdt)
    b2 = b * b
    Nmax = max(4, int(prec_bits // 4))
    one = jnp.asarray(1.0, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    six = jnp.asarray(6.0, dtype=rdt)
    Q = b + one / b
    minus_ipiover2 = -jnp.asarray(1.0j * el.PI / 2.0, dtype=cdt)
    sixth = (Q * Q + one) / six
    twopii_binv = jnp.asarray(2.0 * el.PI * 1.0j, dtype=cdt) / b
    twopii_b = jnp.asarray(2.0 * el.PI * 1.0j, dtype=cdt) * b
    twopii_m_bsq_inv = twopii_binv / b
    twopii_m_b2 = twopii_b * b

    res = jnp.exp(minus_ipiover2 * (z * z - Q * z + sixth))
    res = res / (one - jnp.exp(twopii_b * z))

    def body(m, acc):
        term_num = one - jnp.exp(twopii_binv * z - jnp.asarray(m + 1, dtype=rdt) * twopii_m_bsq_inv)
        term_den = one - jnp.exp(twopii_b * z + jnp.asarray(m + 1, dtype=rdt) * twopii_m_b2)
        return acc * (term_num / term_den)

    return lax.fori_loop(0, Nmax, body, res)


def _pad_point_batch_last(args, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)


def _vectorize_complex_point2(fn, x: jax.Array, y: jax.Array, *, prec_bits: int):
    xx = el.as_complex(x)
    yy = el.as_complex(y)
    bx, by = jnp.broadcast_arrays(xx, yy)
    flat_x = jnp.ravel(bx)
    flat_y = jnp.ravel(by)
    out = jax.vmap(lambda a, b: fn(a, b, prec_bits=prec_bits))(flat_x, flat_y)
    return out.reshape(bx.shape)


def bdg_log_barnesdoublegamma_batch_fixed_point(z: jax.Array, tau: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _vectorize_complex_point2(bdg_log_barnesdoublegamma, z, tau, prec_bits=prec_bits)


def bdg_log_barnesdoublegamma_batch_padded_point(
    z: jax.Array, tau: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((z, tau), pad_to)
    return bdg_log_barnesdoublegamma_batch_fixed_point(*call_args, prec_bits=prec_bits)


def bdg_barnesdoublegamma_batch_fixed_point(z: jax.Array, tau: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _vectorize_complex_point2(bdg_barnesdoublegamma, z, tau, prec_bits=prec_bits)


def bdg_barnesdoublegamma_batch_padded_point(
    z: jax.Array, tau: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((z, tau), pad_to)
    return bdg_barnesdoublegamma_batch_fixed_point(*call_args, prec_bits=prec_bits)


def bdg_log_barnesgamma2_batch_fixed_point(w: jax.Array, beta: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _vectorize_complex_point2(bdg_log_barnesgamma2, w, beta, prec_bits=prec_bits)


def bdg_log_barnesgamma2_batch_padded_point(
    w: jax.Array, beta: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((w, beta), pad_to)
    return bdg_log_barnesgamma2_batch_fixed_point(*call_args, prec_bits=prec_bits)


def bdg_barnesgamma2_batch_fixed_point(w: jax.Array, beta: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _vectorize_complex_point2(bdg_barnesgamma2, w, beta, prec_bits=prec_bits)


def bdg_barnesgamma2_batch_padded_point(
    w: jax.Array, beta: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((w, beta), pad_to)
    return bdg_barnesgamma2_batch_fixed_point(*call_args, prec_bits=prec_bits)


def bdg_log_normalizeddoublegamma_batch_fixed_point(
    w: jax.Array, beta: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return _vectorize_complex_point2(bdg_log_normalizeddoublegamma, w, beta, prec_bits=prec_bits)


def bdg_log_normalizeddoublegamma_batch_padded_point(
    w: jax.Array, beta: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((w, beta), pad_to)
    return bdg_log_normalizeddoublegamma_batch_fixed_point(*call_args, prec_bits=prec_bits)


def bdg_normalizeddoublegamma_batch_fixed_point(
    w: jax.Array, beta: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return _vectorize_complex_point2(bdg_normalizeddoublegamma, w, beta, prec_bits=prec_bits)


def bdg_normalizeddoublegamma_batch_padded_point(
    w: jax.Array, beta: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((w, beta), pad_to)
    return bdg_normalizeddoublegamma_batch_fixed_point(*call_args, prec_bits=prec_bits)


def bdg_double_sine_batch_fixed_point(z: jax.Array, b: jax.Array, *, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _vectorize_complex_point2(bdg_double_sine, z, b, prec_bits=prec_bits)


def bdg_double_sine_batch_padded_point(
    z: jax.Array, b: jax.Array, *, pad_to: int, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    call_args, _ = _pad_point_batch_last((z, b), pad_to)
    return bdg_double_sine_batch_fixed_point(*call_args, prec_bits=prec_bits)


# Interval/box wrappers and mode dispatch


def _acb_from_complex(z: jax.Array) -> jax.Array:
    return acb_core.acb_box(
        di.interval(di._below(jnp.real(z)), di._above(jnp.real(z))),
        di.interval(di._below(jnp.imag(z)), di._above(jnp.imag(z))),
    )


def _acb_mid_eval2(x: jax.Array, y: jax.Array, fn, prec_bits: int) -> jax.Array:
    xb = acb_core.as_acb_box(x)
    yb = acb_core.as_acb_box(y)
    xmid = acb_core.acb_midpoint(xb)
    ymid = acb_core.acb_midpoint(yb)
    return _acb_from_complex(fn(xmid, ymid, prec_bits))


def bdg_complex_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(z, tau, bdg_log_barnesdoublegamma, prec_bits)


def bdg_complex_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(z, tau, bdg_barnesdoublegamma, prec_bits)


def bdg_complex_log_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(w, beta, bdg_log_barnesgamma2, prec_bits)


def bdg_complex_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(w, beta, bdg_barnesgamma2, prec_bits)


def bdg_complex_log_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(w, beta, bdg_log_normalizeddoublegamma, prec_bits)


def bdg_complex_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(w, beta, bdg_normalizeddoublegamma, prec_bits)


def bdg_complex_double_sine(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_mid_eval2(z, b, bdg_double_sine, prec_bits)


def _interval_from_mid(val: jax.Array) -> jax.Array:
    return di.interval(di._below(val), di._above(val))


def _interval_real_mid_eval2(x: jax.Array, y: jax.Array, fn, prec_bits: int) -> jax.Array:
    xmid = di.midpoint(di.as_interval(x))
    ymid = di.midpoint(di.as_interval(y))
    return _interval_from_mid(jnp.real(fn(xmid, ymid, prec_bits)))


def bdg_interval_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _interval_real_mid_eval2(z, tau, bdg_log_barnesdoublegamma, prec_bits)


def bdg_interval_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _interval_real_mid_eval2(z, tau, bdg_barnesdoublegamma, prec_bits)


def bdg_interval_log_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _interval_real_mid_eval2(w, beta, bdg_log_barnesgamma2, prec_bits)


def bdg_interval_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _interval_real_mid_eval2(w, beta, bdg_barnesgamma2, prec_bits)


def bdg_interval_log_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _interval_real_mid_eval2(w, beta, bdg_log_normalizeddoublegamma, prec_bits)


def bdg_interval_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _interval_real_mid_eval2(w, beta, bdg_normalizeddoublegamma, prec_bits)


def bdg_interval_log_barnesdoublegamma_mode(
    z: jax.Array,
    tau: jax.Array,
    impl: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: bdg_interval_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_interval_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_interval_log_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        False,
        pb,
        (z, tau),
        {},
    )


def bdg_interval_barnesdoublegamma_mode(
    z: jax.Array,
    tau: jax.Array,
    impl: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: bdg_interval_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_interval_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_interval_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        False,
        pb,
        (z, tau),
        {},
    )


def bdg_interval_log_barnesgamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_interval_log_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_log_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_log_barnesgamma2_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


def bdg_interval_barnesgamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_interval_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_barnesgamma2_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


def bdg_interval_log_normalizeddoublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_interval_log_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_log_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_log_normalizeddoublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


def bdg_interval_normalizeddoublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_interval_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_interval_normalizeddoublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


def bdg_complex_log_barnesdoublegamma_mode(
    z: jax.Array,
    tau: jax.Array,
    impl: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: bdg_complex_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_complex_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_complex_log_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        True,
        pb,
        (z, tau),
        {},
    )


def bdg_complex_barnesdoublegamma_mode(z: jax.Array, tau: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: bdg_complex_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_complex_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.bdg_complex_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        True,
        pb,
        (z, tau),
        {},
    )


def bdg_complex_log_barnesgamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_complex_log_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_log_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_log_barnesgamma2_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


def bdg_complex_barnesgamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_complex_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_barnesgamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_barnesgamma2_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


def bdg_complex_log_normalizeddoublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_complex_log_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_log_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_log_normalizeddoublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


def bdg_complex_normalizeddoublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: bdg_complex_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_normalizeddoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.bdg_complex_normalizeddoublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


def bdg_complex_double_sine_mode(z: jax.Array, b: jax.Array, impl: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, bb, prec_bits: bdg_complex_double_sine(zz, bb, prec_bits=prec_bits),
        lambda zz, bb, prec_bits: ball_wrappers.bdg_complex_double_sine(zz, bb, prec_bits=prec_bits),
        lambda zz, bb, prec_bits: ball_wrappers.bdg_complex_double_sine_adaptive(zz, bb, prec_bits=prec_bits),
        True,
        pb,
        (z, b),
        {},
    )


__all__ = [
    "bdg_log_barnesdoublegamma",
    "bdg_barnesdoublegamma",
    "bdg_log_barnesgamma2",
    "bdg_barnesgamma2",
    "bdg_log_normalizeddoublegamma",
    "bdg_normalizeddoublegamma",
    "bdg_double_sine",
    "bdg_complex_log_barnesdoublegamma",
    "bdg_complex_barnesdoublegamma",
    "bdg_complex_log_barnesgamma2",
    "bdg_complex_barnesgamma2",
    "bdg_complex_log_normalizeddoublegamma",
    "bdg_complex_normalizeddoublegamma",
    "bdg_complex_double_sine",
    "bdg_interval_log_barnesdoublegamma",
    "bdg_interval_barnesdoublegamma",
    "bdg_interval_log_barnesgamma2",
    "bdg_interval_barnesgamma2",
    "bdg_interval_log_normalizeddoublegamma",
    "bdg_interval_normalizeddoublegamma",
    "bdg_interval_log_barnesdoublegamma_mode",
    "bdg_interval_barnesdoublegamma_mode",
    "bdg_interval_log_barnesgamma2_mode",
    "bdg_interval_barnesgamma2_mode",
    "bdg_interval_log_normalizeddoublegamma_mode",
    "bdg_interval_normalizeddoublegamma_mode",
    "bdg_complex_log_barnesdoublegamma_mode",
    "bdg_complex_barnesdoublegamma_mode",
    "bdg_complex_log_barnesgamma2_mode",
    "bdg_complex_barnesgamma2_mode",
    "bdg_complex_log_normalizeddoublegamma_mode",
    "bdg_complex_normalizeddoublegamma_mode",
    "bdg_complex_double_sine_mode",
]
