from __future__ import annotations

import math
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

jax.config.update("jax_enable_x64", True)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    return barnesg._complex_loggamma(z)


def _digamma(z: jax.Array) -> jax.Array:
    h = jnp.complex128(1e-6 + 0.0j)
    return (_complex_loggamma(z + h) - _complex_loggamma(z - h)) / (2.0 * h)


def _trigamma(z: jax.Array) -> jax.Array:
    h = jnp.complex128(1e-5 + 0.0j)
    return (_digamma(z + h) - _digamma(z - h)) / (2.0 * h)


def _integrate_trapz(fn, a: float, b: float, n: int) -> jax.Array:
    xs = jnp.linspace(jnp.float64(a), jnp.float64(b), n)
    vals = fn(xs)
    return jnp.trapz(vals, xs)


def _integrandC(x: jax.Array, tau: jax.Array) -> jax.Array:
    one_minus_tau = 1.0 - tau
    sinh_x = jnp.sinh(x)
    sinh_taux = jnp.sinh(tau * x)
    exp1 = jnp.exp(one_minus_tau * x)
    exp2 = jnp.exp(-2.0 * x)
    exp3 = jnp.exp(x)
    term1 = exp1 / (2.0 * sinh_x * sinh_taux)
    term2 = exp2 / (tau * x) * (exp3 / (2.0 * sinh_x) + 1.0 - tau / 2.0)
    return term1 - term2


def _integrandD(x: jax.Array, tau: jax.Array) -> jax.Array:
    return x * jnp.exp((1.0 - tau) * x) / (jnp.sinh(x) * jnp.sinh(tau * x)) - jnp.exp(-2.0 * x) / (tau * x)


def _modularC(tau: jax.Array, prec_bits: int) -> jax.Array:
    P = max(int(prec_bits // 2), 8)
    cutoff = jnp.exp2(-jnp.float64(P) / 3.0)
    upper = jnp.maximum(20.0, 5.0 * jnp.abs(tau))

    def fn(x):
        return _integrandC(x, tau)

    val = _integrate_trapz(fn, float(cutoff), float(upper), 256)
    c0 = (2.0 / tau - 1.5 + tau / 6.0) * cutoff + (5.0 / 12.0 - 1.0 / tau + tau / 12.0) * cutoff ** 2
    c0 = c0 + (4.0 / (9.0 * tau) - 2.0 / 9.0 + tau / 54.0 - (tau ** 3) / 270.0) * cutoff ** 3
    return (1.0 / (2.0 * tau)) * jnp.log(2.0 * jnp.pi) - val - c0


def _modularD(tau: jax.Array, prec_bits: int) -> jax.Array:
    upper = jnp.maximum(20.0, 5.0 * jnp.abs(tau))

    def fn(x):
        return _integrandD(x, tau)

    return _integrate_trapz(fn, 1e-8, float(upper), 256)


def _modularcoeff_a(tau: jax.Array, prec_bits: int) -> jax.Array:
    modc = _modularC(tau, prec_bits)
    return 0.5 * tau * jnp.log(2.0 * (jnp.pi * tau)) + 0.5 * jnp.log(tau) - tau * modc


def _modularcoeff_b(tau: jax.Array, prec_bits: int) -> jax.Array:
    modd = _modularD(tau, prec_bits)
    return -tau * jnp.log(tau) - tau * tau * modd


def _evalpoly(z: jax.Array, coeffs: jax.Array) -> jax.Array:
    acc = jnp.zeros((), dtype=jnp.complex128)
    for c in coeffs[::-1]:
        acc = acc * z + c
    return acc


def _polynomial_Pns(M: int, tau_pows, taup1_pows, factorials):
    coeffs = [jnp.zeros((n,), dtype=jnp.complex128) for n in range(1, M + 1)]
    tau_factors = []
    for k in range(1, M + 1):
        tf = (taup1_pows[k + 1] - 1.0 - tau_pows[k + 1]) / factorials[k + 1] / tau_pows[0]
        tau_factors.append(tf)
    for n in range(1, M + 1):
        c = coeffs[n - 1]
        c = c.at[n - 1].set(1.0 / factorials[n + 1])
        for j in range(0, n - 1):
            acc = 0.0 + 0.0j
            for k in range(1, n - j):
                acc = acc + tau_factors[k - 1] * coeffs[n - k - 1][j]
            c = c.at[j].set(-acc)
        coeffs[n - 1] = c
    return coeffs


def _rest_RMN(z: jax.Array, tau: jax.Array, M: int, N: int, tau_pows, taup1_pows, Nm_tau_pows, factorials, poly):
    coeffs_sum = [jnp.zeros((), dtype=jnp.complex128) for _ in range(M + 1)]
    coeffs_sum[1] = _evalpoly(z, poly[0])
    for k in range(2, M + 1):
        coeffs_sum[k] = factorials[k - 1] * _evalpoly(z, poly[k - 1])
    m_tau = -tau
    acc = 0.0 + 0.0j
    for i in range(M + 1):
        acc = acc + Nm_tau_pows[i] * coeffs_sum[i]
    return acc / m_tau


def _log_barnesdoublegamma_mid(z: jax.Array, tau: jax.Array, prec_bits: int) -> jax.Array:
    tau = jnp.asarray(tau, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    M = max(2, int(math.floor(0.5 / math.log(20.0) * max(prec_bits, 32))))
    N = max(20, 50 * M)

    logtau = jnp.log(tau)
    moda = _modularcoeff_a(tau, prec_bits)
    modb = _modularcoeff_b(tau, prec_bits)

    m = jnp.arange(1, N + 1, dtype=jnp.float64)
    mt = m * tau
    loggammas = _complex_loggamma(mt)
    digammas = _digamma(mt)
    trigammas = _trigamma(mt)

    zsq2 = 0.5 * z * z
    res = -_complex_loggamma(z) - logtau
    res = res + moda * (z / tau)
    res = res + modb * (zsq2 / (tau * tau))

    res = res + jnp.sum(loggammas - _complex_loggamma(z + mt) + z * digammas + zsq2 * trigammas)

    tau_pows = [(-tau) ** (k + 1) for k in range(M + 1)]
    taup1_pows = [(-tau + 1.0) ** (k + 1) for k in range(M + 1)]
    Nm_tau_pows = [jnp.ones((), dtype=jnp.complex128)]
    for i in range(1, M + 1):
        Nm_tau_pows.append(-(1.0 / (jnp.float64(N) * tau)) * Nm_tau_pows[i - 1])
    factorials = [jnp.exp(lax.lgamma(jnp.float64(n + 1))) for n in range(M + 2)]
    poly = _polynomial_Pns(M, tau_pows, taup1_pows, factorials)
    rest = _rest_RMN(z, tau, M, N, tau_pows, taup1_pows, Nm_tau_pows, factorials, poly)
    return res + (z ** 3) * rest


def log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _log_barnesdoublegamma_mid(z, tau, prec_bits)


def barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return jnp.exp(log_barnesdoublegamma(z, tau, prec_bits))


def loggamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    beta = jnp.asarray(beta, dtype=jnp.complex128)
    w = jnp.asarray(w, dtype=jnp.complex128)
    beta = jnp.where(jnp.real(beta - 1.0 / beta) < 0.0, 1.0 / beta, beta)
    invb = 1.0 / beta
    tau = invb * invb
    logb = jnp.log(beta)
    c1 = 0.5 * invb * jnp.log(2.0 * jnp.pi)
    c2 = -beta - invb
    l = log_barnesdoublegamma(w / beta, tau, prec_bits)
    return w * c1 + (0.5 * w * (w + c2) + 1.0) * logb - l


def gamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return jnp.exp(loggamma2(w, beta, prec_bits))


def logdoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return loggamma2(w, beta, prec_bits) - loggamma2((beta + 1.0 / beta) / 2.0, beta, prec_bits)


def doublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return jnp.exp(logdoublegamma(w, beta, prec_bits))


def double_sine(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    b = jnp.asarray(b, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    b2 = b * b
    Nmax = max(4, int(prec_bits // 4))
    Q = b + 1.0 / b
    minus_ipiover2 = -1.0j * jnp.pi / 2.0
    sixth = (Q * Q + 1.0) / 6.0
    twopii_binv = 2.0 * jnp.pi * 1.0j / b
    twopii_b = 2.0 * jnp.pi * 1.0j * b
    twopii_m_bsq_inv = twopii_binv / b
    twopii_m_b2 = twopii_b * b

    res = jnp.exp(minus_ipiover2 * (z * z - Q * z + sixth))
    res = res / (1.0 - jnp.exp(twopii_b * z))

    def body(m, acc):
        term_num = 1.0 - jnp.exp(twopii_binv * z - jnp.float64(m + 1) * twopii_m_bsq_inv)
        term_den = 1.0 - jnp.exp(twopii_b * z + jnp.float64(m + 1) * twopii_m_b2)
        return acc * (term_num / term_den)

    return lax.fori_loop(0, Nmax, body, res)


# Interval/box wrappers and mode dispatch


def _acb_from_complex(z: jax.Array) -> jax.Array:
    return acb_core.acb_box(
        di.interval(di._below(jnp.real(z)), di._above(jnp.real(z))),
        di.interval(di._below(jnp.imag(z)), di._above(jnp.imag(z))),
    )


def acb_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    zb = acb_core.as_acb_box(z)
    tb = acb_core.as_acb_box(tau)
    zmid = acb_core.acb_midpoint(zb)
    tmid = acb_core.acb_midpoint(tb)
    return _acb_from_complex(log_barnesdoublegamma(zmid, tmid, prec_bits))


def acb_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    zb = acb_core.as_acb_box(z)
    tb = acb_core.as_acb_box(tau)
    zmid = acb_core.acb_midpoint(zb)
    tmid = acb_core.acb_midpoint(tb)
    return _acb_from_complex(barnesdoublegamma(zmid, tmid, prec_bits))


def acb_loggamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    wb = acb_core.as_acb_box(w)
    bb = acb_core.as_acb_box(beta)
    wmid = acb_core.acb_midpoint(wb)
    bmid = acb_core.acb_midpoint(bb)
    return _acb_from_complex(loggamma2(wmid, bmid, prec_bits))


def acb_gamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_from_complex(gamma2(acb_core.acb_midpoint(w), acb_core.acb_midpoint(beta), prec_bits))


def acb_logdoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_from_complex(logdoublegamma(acb_core.acb_midpoint(w), acb_core.acb_midpoint(beta), prec_bits))


def acb_doublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_from_complex(doublegamma(acb_core.acb_midpoint(w), acb_core.acb_midpoint(beta), prec_bits))


def acb_double_sine(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_from_complex(double_sine(acb_core.acb_midpoint(z), acb_core.acb_midpoint(b), prec_bits))


def _interval_from_mid(val: jax.Array) -> jax.Array:
    return di.interval(di._below(val), di._above(val))


def arb_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    zm = di.midpoint(di.as_interval(z))
    tm = di.midpoint(di.as_interval(tau))
    val = log_barnesdoublegamma(zm, tm, prec_bits)
    return _interval_from_mid(jnp.real(val))


def arb_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    zm = di.midpoint(di.as_interval(z))
    tm = di.midpoint(di.as_interval(tau))
    val = barnesdoublegamma(zm, tm, prec_bits)
    return _interval_from_mid(jnp.real(val))


def arb_loggamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    wm = di.midpoint(di.as_interval(w))
    bm = di.midpoint(di.as_interval(beta))
    val = loggamma2(wm, bm, prec_bits)
    return _interval_from_mid(jnp.real(val))


def arb_gamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    wm = di.midpoint(di.as_interval(w))
    bm = di.midpoint(di.as_interval(beta))
    val = gamma2(wm, bm, prec_bits)
    return _interval_from_mid(jnp.real(val))


def arb_logdoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    wm = di.midpoint(di.as_interval(w))
    bm = di.midpoint(di.as_interval(beta))
    val = logdoublegamma(wm, bm, prec_bits)
    return _interval_from_mid(jnp.real(val))


def arb_doublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    wm = di.midpoint(di.as_interval(w))
    bm = di.midpoint(di.as_interval(beta))
    val = doublegamma(wm, bm, prec_bits)
    return _interval_from_mid(jnp.real(val))


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def arb_log_barnesdoublegamma_mode(
    z: jax.Array,
    tau: jax.Array,
    impl: str = "baseline",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: arb_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.arb_ball_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.arb_ball_log_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        False,
        pb,
        (z, tau),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def arb_barnesdoublegamma_mode(
    z: jax.Array,
    tau: jax.Array,
    impl: str = "baseline",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: arb_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.arb_ball_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.arb_ball_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        False,
        pb,
        (z, tau),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def arb_loggamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: arb_loggamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_loggamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_loggamma2_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def arb_gamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: arb_gamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_gamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_gamma2_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def arb_logdoublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: arb_logdoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_logdoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_logdoublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def arb_doublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: arb_doublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_doublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.arb_ball_doublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        False,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_log_barnesdoublegamma_mode(
    z: jax.Array,
    tau: jax.Array,
    impl: str = "baseline",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: acb_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.acb_ball_log_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.acb_ball_log_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        True,
        pb,
        (z, tau),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_barnesdoublegamma_mode(z: jax.Array, tau: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, tt, prec_bits: acb_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.acb_ball_barnesdoublegamma(zz, tt, prec_bits=prec_bits),
        lambda zz, tt, prec_bits: ball_wrappers.acb_ball_barnesdoublegamma_adaptive(zz, tt, prec_bits=prec_bits),
        True,
        pb,
        (z, tau),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_loggamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: acb_loggamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_loggamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_loggamma2_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_gamma2_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: acb_gamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_gamma2(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_gamma2_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_logdoublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: acb_logdoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_logdoublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_logdoublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_doublegamma_mode(w: jax.Array, beta: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda ww, bb, prec_bits: acb_doublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_doublegamma(ww, bb, prec_bits=prec_bits),
        lambda ww, bb, prec_bits: ball_wrappers.acb_ball_doublegamma_adaptive(ww, bb, prec_bits=prec_bits),
        True,
        pb,
        (w, beta),
        {},
    )


@partial(jax.jit, static_argnames=("impl", "prec_bits", "dps"))
def acb_double_sine_mode(z: jax.Array, b: jax.Array, impl: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = precision.get_prec_bits() if prec_bits is None and dps is None else wc.resolve_prec_bits(dps, prec_bits)
    from . import ball_wrappers
    return wc.dispatch_mode(
        impl,
        lambda zz, bb, prec_bits: acb_double_sine(zz, bb, prec_bits=prec_bits),
        lambda zz, bb, prec_bits: ball_wrappers.acb_ball_double_sine(zz, bb, prec_bits=prec_bits),
        lambda zz, bb, prec_bits: ball_wrappers.acb_ball_double_sine_adaptive(zz, bb, prec_bits=prec_bits),
        True,
        pb,
        (z, b),
        {},
    )


__all__ = [
    "log_barnesdoublegamma",
    "barnesdoublegamma",
    "loggamma2",
    "gamma2",
    "logdoublegamma",
    "doublegamma",
    "double_sine",
    "acb_log_barnesdoublegamma",
    "acb_barnesdoublegamma",
    "acb_loggamma2",
    "acb_gamma2",
    "acb_logdoublegamma",
    "acb_doublegamma",
    "acb_double_sine",
    "arb_log_barnesdoublegamma",
    "arb_barnesdoublegamma",
    "arb_loggamma2",
    "arb_gamma2",
    "arb_logdoublegamma",
    "arb_doublegamma",
    "arb_log_barnesdoublegamma_mode",
    "arb_barnesdoublegamma_mode",
    "arb_loggamma2_mode",
    "arb_gamma2_mode",
    "arb_logdoublegamma_mode",
    "arb_doublegamma_mode",
    "acb_log_barnesdoublegamma_mode",
    "acb_barnesdoublegamma_mode",
    "acb_loggamma2_mode",
    "acb_gamma2_mode",
    "acb_logdoublegamma_mode",
    "acb_doublegamma_mode",
    "acb_double_sine_mode",
]
