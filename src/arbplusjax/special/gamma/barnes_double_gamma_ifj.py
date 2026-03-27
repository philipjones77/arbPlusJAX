from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import jax
from jax import core as jax_core
from jax import lax
import jax.numpy as jnp

from ... import barnesg
from ... import point_wrappers


_MIN_M = 64
_RICHARDSON_LEVELS = 3
_RECURRENCE_TARGET = 6.0


@dataclass(frozen=True)
class IFJBarnesDoubleGammaDiagnostics:
    dps: int
    tau: float
    m_base: int
    m_used: int
    max_m_cap: int
    m_capped: bool
    n_shift: int
    richardson_levels: int


def _tau_real(tau: jax.Array) -> jax.Array:
    tau_arr = jnp.asarray(tau)
    if tau_arr.ndim != 0:
        raise ValueError("tau must be a scalar positive real for the IFJ Barnes provider")
    if jnp.issubdtype(tau_arr.dtype, jnp.complexfloating):
        imag_arr = jnp.imag(tau_arr)
        if not isinstance(imag_arr, jax_core.Tracer):
            imag = float(imag_arr)
            if abs(imag) > 0.0:
                raise ValueError("tau must be strictly positive real for the IFJ Barnes provider")
        elif tau_arr.dtype in {jnp.complex64, jnp.complex128}:
            raise ValueError("tau must be strictly positive real for the IFJ Barnes provider")
        tau_arr = jnp.real(tau_arr)
    tau_real = jnp.asarray(tau_arr, dtype=jnp.float64)
    if not isinstance(tau_real, jax_core.Tracer) and float(tau_real) <= 0.0:
        raise ValueError("tau must be strictly positive real for the IFJ Barnes provider")
    return tau_real


def _complex_dtype() -> jnp.dtype:
    return jnp.dtype(jnp.complex128)


def _choose_m_base(tau: float, dps: int) -> int:
    scale = 1.0 / max(float(tau), 1e-6)
    return max(_MIN_M, int(math.ceil(2.0 * float(dps) * max(1.0, scale))))


def _real_point_digamma(x: jax.Array) -> jax.Array:
    return jnp.real(point_wrappers.acb_digamma_point(jnp.asarray(x, dtype=_complex_dtype())))


def _real_point_polygamma(n: int, x: jax.Array) -> jax.Array:
    return jnp.real(point_wrappers.acb_polygamma_point(n, jnp.asarray(x, dtype=_complex_dtype())))


def _masked_real_sum(values: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.sum(jnp.where(mask, values, jnp.zeros_like(values)))


def _log_double_gamma_core_scalar(z: jax.Array, tau: jax.Array, m_used: int) -> jax.Array:
    cdt = _complex_dtype()
    rdt = jnp.float64
    m_used = max(8, int(m_used))

    idx = jnp.arange(1, m_used, dtype=rdt)
    x = tau * idx
    mt = tau * jnp.asarray(float(m_used), dtype=rdt)

    sum_c = jnp.sum(_real_point_digamma(x))
    sum_d = jnp.sum(_real_point_polygamma(1, x))

    c = (
        sum_c
        + jnp.asarray(0.5, dtype=rdt) * _real_point_digamma(mt)
        - (jnp.real(barnesg._complex_loggamma(jnp.asarray(mt, dtype=cdt))) - jnp.asarray(0.5 * jnp.log(2.0 * jnp.pi), dtype=rdt)) / tau
        - (tau / jnp.asarray(12.0, dtype=rdt)) * _real_point_polygamma(1, mt)
        + (tau**3 / jnp.asarray(720.0, dtype=rdt)) * _real_point_polygamma(3, mt)
        - (tau**5 / jnp.asarray(30240.0, dtype=rdt)) * _real_point_polygamma(5, mt)
        + (tau**7 / jnp.asarray(1209600.0, dtype=rdt)) * _real_point_polygamma(7, mt)
    )
    d = (
        sum_d
        + jnp.asarray(0.5, dtype=rdt) * _real_point_polygamma(1, mt)
        - _real_point_digamma(mt) / tau
        - (tau / jnp.asarray(12.0, dtype=rdt)) * _real_point_polygamma(2, mt)
        + (tau**3 / jnp.asarray(720.0, dtype=rdt)) * _real_point_polygamma(4, mt)
        - (tau**5 / jnp.asarray(30240.0, dtype=rdt)) * _real_point_polygamma(6, mt)
        + (tau**7 / jnp.asarray(1209600.0, dtype=rdt)) * _real_point_polygamma(8, mt)
    )

    zc = jnp.asarray(z, dtype=cdt)
    tc = jnp.asarray(tau, dtype=cdt)
    a_tilde = jnp.asarray(0.5, dtype=rdt) * tc * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=cdt) * tc) + jnp.asarray(0.5, dtype=rdt) * jnp.log(tc) - tc * jnp.asarray(c, dtype=cdt)
    b_tilde = -tc * jnp.log(tc) - (tc**2) * jnp.asarray(d, dtype=cdt)

    full_idx = jnp.arange(1, m_used + 1, dtype=rdt)
    x_full = tau * full_idx
    xc = jnp.asarray(x_full, dtype=cdt)
    sum_terms = (
        barnesg._complex_loggamma(xc)
        - barnesg._complex_loggamma(zc + xc)
        + zc * jnp.asarray(_real_point_digamma(x_full), dtype=cdt)
        + jnp.asarray(0.5, dtype=rdt) * (zc**2) * jnp.asarray(_real_point_polygamma(1, x_full), dtype=cdt)
    )
    s = jnp.sum(sum_terms)
    value = -jnp.log(tc) - barnesg._complex_loggamma(zc) + a_tilde * (zc / tc) + b_tilde * (zc**2 / (jnp.asarray(2.0, dtype=rdt) * tc**2)) + s
    return value


def _log_double_gamma_scalar_fixed(z: jax.Array, tau: jax.Array, m1: int, m2: int, m3: int) -> tuple[jax.Array, jax.Array]:
    cdt = _complex_dtype()
    rdt = jnp.float64
    zc = jnp.asarray(z, dtype=cdt)
    tc = jnp.asarray(tau, dtype=cdt)
    n_shift = jnp.maximum(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.ceil(jnp.asarray(_RECURRENCE_TARGET, dtype=rdt) - jnp.real(zc / tc)).astype(jnp.int32),
    )

    def recurrence_body(k, acc):
        kk = jnp.asarray(k, dtype=rdt)
        return acc + barnesg._complex_loggamma((zc + jnp.asarray(kk, dtype=cdt)) / tc)

    rec_sum = lax.fori_loop(0, n_shift, recurrence_body, jnp.asarray(0.0 + 0.0j, dtype=cdt))
    z_eval = zc + n_shift.astype(rdt).astype(cdt)

    l1 = _log_double_gamma_core_scalar(z_eval, tau, m1)
    l2 = _log_double_gamma_core_scalar(z_eval, tau, m2)
    l3 = _log_double_gamma_core_scalar(z_eval, tau, m3)
    r1 = jnp.asarray(2.0, dtype=rdt) * l2 - l1
    r2 = jnp.asarray(2.0, dtype=rdt) * l3 - l2
    value = (jnp.asarray(4.0, dtype=rdt) * r2 - r1) / jnp.asarray(3.0, dtype=rdt) - rec_sum
    return value, n_shift


@lru_cache(maxsize=32)
def _normalization_log_at_one_cached(tau: float, m1: int, m2: int, m3: int) -> complex:
    one = jnp.asarray(1.0 + 0.0j, dtype=_complex_dtype())
    value, _ = _log_double_gamma_scalar_fixed(one, jnp.asarray(tau, dtype=jnp.float64), m1, m2, m3)
    return complex(jax.device_get(value))


def log_barnesdoublegamma_ifj(
    z: jax.Array,
    tau: jax.Array,
    *,
    dps: int = 50,
    max_m_cap: int = 1024,
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, IFJBarnesDoubleGammaDiagnostics]:
    tau_real = _tau_real(tau)
    if isinstance(tau_real, jax_core.Tracer):
        m_base_i = _choose_m_base(1.0, int(dps))
    else:
        m_base_i = _choose_m_base(float(tau_real), int(dps))
    max_cap_i = int(max_m_cap)
    m1 = min(m_base_i, max_cap_i)
    m2 = min(2 * m_base_i, max_cap_i)
    m3 = min(4 * m_base_i, max_cap_i)
    m_capped = (m1 != m_base_i) or (m2 != 2 * m_base_i) or (m3 != 4 * m_base_i)
    z_arr = jnp.asarray(z, dtype=_complex_dtype())
    traced_context = isinstance(tau_real, jax_core.Tracer) or isinstance(z_arr, jax_core.Tracer)
    if traced_context:
        norm_log = _log_double_gamma_scalar_fixed(jnp.asarray(1.0 + 0.0j, dtype=_complex_dtype()), tau_real, m1, m2, m3)[0]
    else:
        norm_log = jnp.asarray(_normalization_log_at_one_cached(float(tau_real), m1, m2, m3), dtype=_complex_dtype())
    if z_arr.ndim == 0:
        value, n_shift = _log_double_gamma_scalar_fixed(z_arr, tau_real, m1, m2, m3)
        value = value - norm_log
        if not return_diagnostics:
            return value
        diag = IFJBarnesDoubleGammaDiagnostics(
            dps=int(dps),
            tau=float(tau_real),
            m_base=m_base_i,
            m_used=m1,
            max_m_cap=max_cap_i,
            m_capped=bool(m_capped),
            n_shift=int(n_shift),
            richardson_levels=_RICHARDSON_LEVELS,
        )
        return value, diag

    if return_diagnostics:
        raise ValueError("return_diagnostics is only supported for scalar z")
    flat = z_arr.reshape(-1)
    values = jax.vmap(lambda zz: _log_double_gamma_scalar_fixed(zz, tau_real, m1, m2, m3)[0] - norm_log)(flat)
    return values.reshape(z_arr.shape)


def barnesdoublegamma_ifj(
    z: jax.Array,
    tau: jax.Array,
    *,
    dps: int = 50,
    max_m_cap: int = 1024,
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, IFJBarnesDoubleGammaDiagnostics]:
    out = log_barnesdoublegamma_ifj(
        z,
        tau,
        dps=dps,
        max_m_cap=max_m_cap,
        return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
        return jnp.exp(out)
    value, diagnostics = out
    return jnp.exp(value), diagnostics


def barnesdoublegamma_ifj_diagnostics(
    z: jax.Array,
    tau: jax.Array,
    *,
    dps: int = 50,
    max_m_cap: int = 1024,
) -> IFJBarnesDoubleGammaDiagnostics:
    _, diagnostics = log_barnesdoublegamma_ifj(
        z,
        tau,
        dps=dps,
        max_m_cap=max_m_cap,
        return_diagnostics=True,
    )
    return diagnostics


__all__ = [
    "IFJBarnesDoubleGammaDiagnostics",
    "log_barnesdoublegamma_ifj",
    "barnesdoublegamma_ifj",
    "barnesdoublegamma_ifj_diagnostics",
]
