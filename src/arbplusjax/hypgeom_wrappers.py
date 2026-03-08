from __future__ import annotations

import inspect
from functools import partial
from typing import Callable

import jax

from . import ball_wrappers
from . import hypgeom
from . import wrappers_common as wc

jax.config.update("jax_enable_x64", True)


def _wrap_ball(fn):
    def wrapped(*args, prec_bits: int, **kwargs):
        return fn(*args, prec_bits)

    return wrapped


def _rig_bessel_i_integration(nu, z, *, prec_bits: int, scaled: bool = False, **kwargs):
    return ball_wrappers.arb_ball_bessel_i_scaled(nu, z, prec_bits) if scaled else ball_wrappers.arb_ball_bessel_i(nu, z, prec_bits)


def _rig_bessel_k_integration(nu, z, *, prec_bits: int, scaled: bool = False, **kwargs):
    return ball_wrappers.arb_ball_bessel_k_scaled(nu, z, prec_bits) if scaled else ball_wrappers.arb_ball_bessel_k(nu, z, prec_bits)


def _adapt_bessel_i_integration(nu, z, *, prec_bits: int, scaled: bool = False, **kwargs):
    return (
        ball_wrappers.arb_ball_bessel_i_scaled_adaptive(nu, z, prec_bits)
        if scaled
        else ball_wrappers.arb_ball_bessel_i_adaptive(nu, z, prec_bits)
    )


def _adapt_bessel_k_integration(nu, z, *, prec_bits: int, scaled: bool = False, **kwargs):
    return (
        ball_wrappers.arb_ball_bessel_k_scaled_adaptive(nu, z, prec_bits)
        if scaled
        else ball_wrappers.arb_ball_bessel_k_adaptive(nu, z, prec_bits)
    )


def _arb_unit_interval(dtype):
    one = jax.numpy.asarray(1.0, dtype=dtype)
    return hypgeom.di.interval(one, one)


def _acb_unit_box(dtype):
    return hypgeom.acb_box(_arb_unit_interval(dtype), hypgeom.di.interval(jax.numpy.asarray(0.0, dtype=dtype), jax.numpy.asarray(0.0, dtype=dtype)))


def _arb_incomplete_gamma_special(s, z, *, prec_bits: int, regularized: bool, upper: bool, adaptive: bool):
    kernel = wc.adaptive_interval_kernel if adaptive else wc.rigorous_interval_kernel
    gamma_kernel = ball_wrappers.arb_ball_gamma_adaptive if adaptive else ball_wrappers.arb_ball_gamma
    direct_fn = hypgeom.arb_hypgeom_gamma_upper if upper else hypgeom.arb_hypgeom_gamma_lower
    comp_fn = hypgeom.arb_hypgeom_gamma_lower if upper else hypgeom.arb_hypgeom_gamma_upper
    direct = kernel(lambda ss, zz: direct_fn(ss, zz, regularized=regularized), (s, z), prec_bits)
    comp_other = kernel(lambda ss, zz: comp_fn(ss, zz, regularized=regularized), (s, z), prec_bits)
    if regularized:
        comp = hypgeom.di.fast_sub(_arb_unit_interval(direct.dtype), comp_other)
    else:
        gamma_box = gamma_kernel(s, prec_bits)
        comp = hypgeom.di.fast_sub(gamma_box, comp_other)
    return hypgeom._select_tighter_interval(direct, comp)


def _arb_incomplete_gamma_special_batch(s, z, *, prec_bits: int, regularized: bool, upper: bool, adaptive: bool):
    return jax.vmap(
        lambda ss, zz: _arb_incomplete_gamma_special(ss, zz, prec_bits=prec_bits, regularized=regularized, upper=upper, adaptive=adaptive)
    )(s, z)


def _acb_incomplete_gamma_special(s, z, *, prec_bits: int, regularized: bool, upper: bool, adaptive: bool):
    kernel = wc.adaptive_acb_kernel if adaptive else wc.rigorous_acb_kernel
    gamma_kernel = ball_wrappers.acb_ball_gamma_adaptive if adaptive else ball_wrappers.acb_ball_gamma
    direct_fn = hypgeom.acb_hypgeom_gamma_upper if upper else hypgeom.acb_hypgeom_gamma_lower
    comp_fn = hypgeom.acb_hypgeom_gamma_lower if upper else hypgeom.acb_hypgeom_gamma_upper
    direct = kernel(lambda ss, zz: direct_fn(ss, zz, regularized=regularized), (s, z), prec_bits)
    comp_other = kernel(lambda ss, zz: comp_fn(ss, zz, regularized=regularized), (s, z), prec_bits)
    if regularized:
        comp = hypgeom.acb_box_sub(_acb_unit_box(direct.dtype), comp_other)
    else:
        gamma_box = gamma_kernel(s, prec_bits)
        comp = hypgeom.acb_box_sub(gamma_box, comp_other)
    return hypgeom._select_tighter_acb(direct, comp)


def _acb_incomplete_gamma_special_batch(s, z, *, prec_bits: int, regularized: bool, upper: bool, adaptive: bool):
    return jax.vmap(
        lambda ss, zz: _acb_incomplete_gamma_special(ss, zz, prec_bits=prec_bits, regularized=regularized, upper=upper, adaptive=adaptive)
    )(s, z)


def _wrap_series_rig(fn):
    accepts_prec = "prec_bits" in inspect.signature(fn).parameters

    def wrapped(*args, prec_bits: int, **kwargs):
        if accepts_prec:
            return fn(*args, prec_bits=prec_bits, **kwargs)
        return fn(*args, **kwargs)

    return wrapped


def _wrap_series_rig_batch(fn, in_axes):
    accepts_prec = "prec_bits" in inspect.signature(fn).parameters

    def wrapped(*args, prec_bits: int, **kwargs):
        if accepts_prec:
            return jax.vmap(lambda *aa: fn(*aa, prec_bits=prec_bits, **kwargs), in_axes=in_axes)(*args)
        return jax.vmap(lambda *aa: fn(*aa, **kwargs), in_axes=in_axes)(*args)

    return wrapped


def _pick_base_fn(name: str) -> Callable[..., jax.Array]:
    jit_name = f"{name}_jit"
    fn = getattr(hypgeom, jit_name, None)
    if fn is not None:
        return fn
    return getattr(hypgeom, name)


def _kernel_name(name: str) -> str:
    if name.endswith("_batch_prec"):
        return name[:-11] + "_batch"
    if name.endswith("_prec"):
        return name[:-5]
    return name


_SPECIAL_RIG: dict[str, Callable[..., jax.Array]] = {
    "arb_hypgeom_gamma_prec": ball_wrappers.arb_ball_gamma,
    "arb_hypgeom_gamma_batch_prec": ball_wrappers.arb_ball_gamma,
    "acb_hypgeom_gamma_prec": ball_wrappers.acb_ball_gamma,
    "acb_hypgeom_gamma_batch_prec": ball_wrappers.acb_ball_gamma,
    "arb_hypgeom_erf_prec": ball_wrappers.arb_ball_erf,
    "arb_hypgeom_erf_batch_prec": ball_wrappers.arb_ball_erf,
    "arb_hypgeom_erfc_prec": ball_wrappers.arb_ball_erfc,
    "arb_hypgeom_erfc_batch_prec": ball_wrappers.arb_ball_erfc,
    "arb_hypgeom_erfi_prec": ball_wrappers.arb_ball_erfi,
    "arb_hypgeom_erfi_batch_prec": ball_wrappers.arb_ball_erfi,
    "arb_hypgeom_erfinv_prec": ball_wrappers.arb_ball_erfinv,
    "arb_hypgeom_erfinv_batch_prec": ball_wrappers.arb_ball_erfinv,
    "arb_hypgeom_erfcinv_prec": ball_wrappers.arb_ball_erfcinv,
    "arb_hypgeom_erfcinv_batch_prec": ball_wrappers.arb_ball_erfcinv,
    "acb_hypgeom_erf_prec": ball_wrappers.acb_ball_erf,
    "acb_hypgeom_erf_batch_prec": ball_wrappers.acb_ball_erf,
    "acb_hypgeom_erfc_prec": ball_wrappers.acb_ball_erfc,
    "acb_hypgeom_erfc_batch_prec": ball_wrappers.acb_ball_erfc,
    "acb_hypgeom_erfi_prec": ball_wrappers.acb_ball_erfi,
    "acb_hypgeom_erfi_batch_prec": ball_wrappers.acb_ball_erfi,
    "arb_hypgeom_ei_prec": ball_wrappers.arb_ball_ei,
    "arb_hypgeom_ei_batch_prec": ball_wrappers.arb_ball_ei,
    "arb_hypgeom_si_prec": ball_wrappers.arb_ball_si,
    "arb_hypgeom_si_batch_prec": ball_wrappers.arb_ball_si,
    "arb_hypgeom_si_1f2_prec": _wrap_series_rig(hypgeom.arb_hypgeom_si_1f2_prec),
    "arb_hypgeom_ci_prec": ball_wrappers.arb_ball_ci,
    "arb_hypgeom_ci_batch_prec": ball_wrappers.arb_ball_ci,
    "arb_hypgeom_shi_prec": ball_wrappers.arb_ball_shi,
    "arb_hypgeom_shi_batch_prec": ball_wrappers.arb_ball_shi,
    "arb_hypgeom_chi_prec": ball_wrappers.arb_ball_chi,
    "arb_hypgeom_chi_batch_prec": ball_wrappers.arb_ball_chi,
    "arb_hypgeom_li_prec": ball_wrappers.arb_ball_li,
    "arb_hypgeom_li_batch_prec": ball_wrappers.arb_ball_li,
    "arb_hypgeom_dilog_prec": ball_wrappers.arb_ball_dilog,
    "arb_hypgeom_dilog_batch_prec": ball_wrappers.arb_ball_dilog,
    "arb_hypgeom_fresnel_prec": ball_wrappers.arb_ball_fresnel,
    "arb_hypgeom_fresnel_batch_prec": ball_wrappers.arb_ball_fresnel,
    "arb_hypgeom_airy_prec": ball_wrappers.arb_ball_airy,
    "arb_hypgeom_airy_batch_prec": ball_wrappers.arb_ball_airy,
    "arb_hypgeom_bessel_j_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_j),
    "arb_hypgeom_bessel_y_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_y),
    "arb_hypgeom_bessel_i_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_i),
    "arb_hypgeom_bessel_k_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_k),
    "arb_hypgeom_bessel_i_scaled_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_i_scaled),
    "arb_hypgeom_bessel_k_scaled_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_k_scaled),
    "arb_hypgeom_bessel_i_integration_batch_prec": _rig_bessel_i_integration,
    "arb_hypgeom_bessel_k_integration_batch_prec": _rig_bessel_k_integration,
    "acb_hypgeom_bessel_j_batch_prec": ball_wrappers.acb_ball_bessel_j,
    "acb_hypgeom_bessel_y_batch_prec": ball_wrappers.acb_ball_bessel_y,
    "acb_hypgeom_bessel_i_batch_prec": ball_wrappers.acb_ball_bessel_i,
    "acb_hypgeom_bessel_k_batch_prec": ball_wrappers.acb_ball_bessel_k,
    "acb_hypgeom_bessel_i_scaled_batch_prec": ball_wrappers.acb_ball_bessel_i_scaled,
    "acb_hypgeom_bessel_k_scaled_batch_prec": ball_wrappers.acb_ball_bessel_k_scaled,
    "arb_hypgeom_0f1_prec": _wrap_series_rig(hypgeom.arb_hypgeom_0f1_rigorous),
    "arb_hypgeom_1f1_prec": _wrap_series_rig(hypgeom.arb_hypgeom_1f1_rigorous),
    "arb_hypgeom_2f1_prec": _wrap_series_rig(hypgeom.arb_hypgeom_2f1_rigorous),
    "arb_hypgeom_u_prec": _wrap_series_rig(hypgeom.arb_hypgeom_u_rigorous),
    "arb_hypgeom_0f1_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_0f1_rigorous, (0, 0)),
    "arb_hypgeom_1f1_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_1f1_rigorous, (0, 0, 0)),
    "arb_hypgeom_2f1_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_2f1_rigorous, (0, 0, 0, 0)),
    "arb_hypgeom_u_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_u_rigorous, (0, 0, 0)),
    "arb_hypgeom_legendre_p_prec": _wrap_series_rig(hypgeom.arb_hypgeom_legendre_p_rigorous),
    "arb_hypgeom_legendre_q_prec": _wrap_series_rig(hypgeom.arb_hypgeom_legendre_q_rigorous),
    "arb_hypgeom_jacobi_p_prec": _wrap_series_rig(hypgeom.arb_hypgeom_jacobi_p_rigorous),
    "arb_hypgeom_gegenbauer_c_prec": _wrap_series_rig(hypgeom.arb_hypgeom_gegenbauer_c_rigorous),
    "arb_hypgeom_legendre_p_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_legendre_p_rigorous, (None, 0, 0)),
    "arb_hypgeom_legendre_q_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_legendre_q_rigorous, (None, 0, 0)),
    "arb_hypgeom_jacobi_p_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_jacobi_p_rigorous, (None, 0, 0, 0)),
    "arb_hypgeom_gegenbauer_c_batch_prec": _wrap_series_rig_batch(hypgeom.arb_hypgeom_gegenbauer_c_rigorous, (None, 0, 0)),
    "acb_hypgeom_0f1_prec": _wrap_series_rig(hypgeom.acb_hypgeom_0f1_rigorous),
    "acb_hypgeom_1f1_prec": _wrap_series_rig(hypgeom.acb_hypgeom_1f1_rigorous),
    "acb_hypgeom_2f1_prec": _wrap_series_rig(hypgeom.acb_hypgeom_2f1_rigorous),
    "acb_hypgeom_u_prec": _wrap_series_rig(hypgeom.acb_hypgeom_u_rigorous),
    "acb_hypgeom_0f1_batch_prec": _wrap_series_rig_batch(hypgeom.acb_hypgeom_0f1_rigorous, (0, 0)),
    "acb_hypgeom_1f1_batch_prec": _wrap_series_rig_batch(hypgeom.acb_hypgeom_1f1_rigorous, (0, 0, 0)),
    "acb_hypgeom_2f1_batch_prec": _wrap_series_rig_batch(hypgeom.acb_hypgeom_2f1_rigorous, (0, 0, 0, 0)),
    "acb_hypgeom_u_batch_prec": _wrap_series_rig_batch(hypgeom.acb_hypgeom_u_rigorous, (0, 0, 0)),
    "arb_hypgeom_gamma_lower_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=False
    ),
    "arb_hypgeom_gamma_upper_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=False
    ),
    "arb_hypgeom_gamma_lower_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=False
    ),
    "arb_hypgeom_gamma_upper_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=False
    ),
    "acb_hypgeom_gamma_lower_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=False
    ),
    "acb_hypgeom_gamma_upper_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=False
    ),
    "acb_hypgeom_gamma_lower_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=False
    ),
    "acb_hypgeom_gamma_upper_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=False
    ),
    "arb_hypgeom_rising_coeffs_1_prec": hypgeom.arb_hypgeom_rising_coeffs_1_prec,
    "arb_hypgeom_rising_coeffs_2_prec": hypgeom.arb_hypgeom_rising_coeffs_2_prec,
    "arb_hypgeom_rising_coeffs_fmpz_prec": hypgeom.arb_hypgeom_rising_coeffs_fmpz_prec,
    "arb_hypgeom_gamma_coeff_shallow_prec": hypgeom.arb_hypgeom_gamma_coeff_shallow_prec,
    "arb_hypgeom_gamma_stirling_term_bounds_prec": hypgeom.arb_hypgeom_gamma_stirling_term_bounds_prec,
    "arb_hypgeom_gamma_lower_fmpq_0_choose_N_prec": hypgeom.arb_hypgeom_gamma_lower_fmpq_0_choose_N_prec,
    "arb_hypgeom_gamma_upper_fmpq_inf_choose_N_prec": hypgeom.arb_hypgeom_gamma_upper_fmpq_inf_choose_N_prec,
    "arb_hypgeom_gamma_upper_singular_si_choose_N_prec": hypgeom.arb_hypgeom_gamma_upper_singular_si_choose_N_prec,
    "acb_hypgeom_legendre_p_uiui_rec_prec": _wrap_series_rig(hypgeom.acb_hypgeom_legendre_p_uiui_rec_prec),
}

_SPECIAL_ADAPT: dict[str, Callable[..., jax.Array]] = {
    "arb_hypgeom_gamma_prec": ball_wrappers.arb_ball_gamma_adaptive,
    "arb_hypgeom_gamma_batch_prec": ball_wrappers.arb_ball_gamma_adaptive,
    "acb_hypgeom_gamma_prec": ball_wrappers.acb_ball_gamma_adaptive,
    "acb_hypgeom_gamma_batch_prec": ball_wrappers.acb_ball_gamma_adaptive,
    "arb_hypgeom_erf_prec": ball_wrappers.arb_ball_erf_adaptive,
    "arb_hypgeom_erf_batch_prec": ball_wrappers.arb_ball_erf_adaptive,
    "arb_hypgeom_erfc_prec": ball_wrappers.arb_ball_erfc_adaptive,
    "arb_hypgeom_erfc_batch_prec": ball_wrappers.arb_ball_erfc_adaptive,
    "arb_hypgeom_erfi_prec": ball_wrappers.arb_ball_erfi_adaptive,
    "arb_hypgeom_erfi_batch_prec": ball_wrappers.arb_ball_erfi_adaptive,
    "arb_hypgeom_erfinv_prec": ball_wrappers.arb_ball_erfinv_adaptive,
    "arb_hypgeom_erfinv_batch_prec": ball_wrappers.arb_ball_erfinv_adaptive,
    "arb_hypgeom_erfcinv_prec": ball_wrappers.arb_ball_erfcinv_adaptive,
    "arb_hypgeom_erfcinv_batch_prec": ball_wrappers.arb_ball_erfcinv_adaptive,
    "acb_hypgeom_erf_prec": ball_wrappers.acb_ball_erf_adaptive,
    "acb_hypgeom_erf_batch_prec": ball_wrappers.acb_ball_erf_adaptive,
    "acb_hypgeom_erfc_prec": ball_wrappers.acb_ball_erfc_adaptive,
    "acb_hypgeom_erfc_batch_prec": ball_wrappers.acb_ball_erfc_adaptive,
    "acb_hypgeom_erfi_prec": ball_wrappers.acb_ball_erfi_adaptive,
    "acb_hypgeom_erfi_batch_prec": ball_wrappers.acb_ball_erfi_adaptive,
    "arb_hypgeom_ei_prec": ball_wrappers.arb_ball_ei_adaptive,
    "arb_hypgeom_ei_batch_prec": ball_wrappers.arb_ball_ei_adaptive,
    "arb_hypgeom_si_prec": ball_wrappers.arb_ball_si_adaptive,
    "arb_hypgeom_si_batch_prec": ball_wrappers.arb_ball_si_adaptive,
    "arb_hypgeom_ci_prec": ball_wrappers.arb_ball_ci_adaptive,
    "arb_hypgeom_ci_batch_prec": ball_wrappers.arb_ball_ci_adaptive,
    "arb_hypgeom_shi_prec": ball_wrappers.arb_ball_shi_adaptive,
    "arb_hypgeom_shi_batch_prec": ball_wrappers.arb_ball_shi_adaptive,
    "arb_hypgeom_chi_prec": ball_wrappers.arb_ball_chi_adaptive,
    "arb_hypgeom_chi_batch_prec": ball_wrappers.arb_ball_chi_adaptive,
    "arb_hypgeom_li_prec": ball_wrappers.arb_ball_li_adaptive,
    "arb_hypgeom_li_batch_prec": ball_wrappers.arb_ball_li_adaptive,
    "arb_hypgeom_dilog_prec": ball_wrappers.arb_ball_dilog_adaptive,
    "arb_hypgeom_dilog_batch_prec": ball_wrappers.arb_ball_dilog_adaptive,
    "arb_hypgeom_fresnel_prec": ball_wrappers.arb_ball_fresnel_adaptive,
    "arb_hypgeom_fresnel_batch_prec": ball_wrappers.arb_ball_fresnel_adaptive,
    "arb_hypgeom_airy_prec": ball_wrappers.arb_ball_airy_adaptive,
    "arb_hypgeom_airy_batch_prec": ball_wrappers.arb_ball_airy_adaptive,
    "arb_hypgeom_bessel_j_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_j_adaptive),
    "arb_hypgeom_bessel_y_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_y_adaptive),
    "arb_hypgeom_bessel_i_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_i_adaptive),
    "arb_hypgeom_bessel_k_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_k_adaptive),
    "arb_hypgeom_bessel_i_scaled_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_i_scaled_adaptive),
    "arb_hypgeom_bessel_k_scaled_batch_prec": _wrap_ball(ball_wrappers.arb_ball_bessel_k_scaled_adaptive),
    "arb_hypgeom_bessel_i_integration_batch_prec": _adapt_bessel_i_integration,
    "arb_hypgeom_bessel_k_integration_batch_prec": _adapt_bessel_k_integration,
    "acb_hypgeom_bessel_j_batch_prec": ball_wrappers.acb_ball_bessel_j_adaptive,
    "acb_hypgeom_bessel_y_batch_prec": ball_wrappers.acb_ball_bessel_y_adaptive,
    "acb_hypgeom_bessel_i_batch_prec": ball_wrappers.acb_ball_bessel_i_adaptive,
    "acb_hypgeom_bessel_k_batch_prec": ball_wrappers.acb_ball_bessel_k_adaptive,
    "acb_hypgeom_bessel_i_scaled_batch_prec": ball_wrappers.acb_ball_bessel_i_scaled_adaptive,
    "acb_hypgeom_bessel_k_scaled_batch_prec": ball_wrappers.acb_ball_bessel_k_scaled_adaptive,
    "arb_hypgeom_gamma_lower_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=True
    ),
    "arb_hypgeom_gamma_upper_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=True
    ),
    "arb_hypgeom_gamma_lower_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=True
    ),
    "arb_hypgeom_gamma_upper_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _arb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=True
    ),
    "acb_hypgeom_gamma_lower_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=True
    ),
    "acb_hypgeom_gamma_upper_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=True
    ),
    "acb_hypgeom_gamma_lower_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=True
    ),
    "acb_hypgeom_gamma_upper_batch_prec": lambda s, z, *, prec_bits, regularized=False, **kwargs: _acb_incomplete_gamma_special_batch(
        s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=True
    ),
    "arb_hypgeom_rising_coeffs_1_prec": hypgeom.arb_hypgeom_rising_coeffs_1_prec,
    "arb_hypgeom_rising_coeffs_2_prec": hypgeom.arb_hypgeom_rising_coeffs_2_prec,
    "arb_hypgeom_rising_coeffs_fmpz_prec": hypgeom.arb_hypgeom_rising_coeffs_fmpz_prec,
    "arb_hypgeom_gamma_coeff_shallow_prec": hypgeom.arb_hypgeom_gamma_coeff_shallow_prec,
    "arb_hypgeom_gamma_stirling_term_bounds_prec": hypgeom.arb_hypgeom_gamma_stirling_term_bounds_prec,
    "arb_hypgeom_gamma_lower_fmpq_0_choose_N_prec": hypgeom.arb_hypgeom_gamma_lower_fmpq_0_choose_N_prec,
    "arb_hypgeom_gamma_upper_fmpq_inf_choose_N_prec": hypgeom.arb_hypgeom_gamma_upper_fmpq_inf_choose_N_prec,
    "arb_hypgeom_gamma_upper_singular_si_choose_N_prec": hypgeom.arb_hypgeom_gamma_upper_singular_si_choose_N_prec,
    "acb_hypgeom_legendre_p_uiui_rec_prec": _wrap_series_rig(hypgeom.acb_hypgeom_legendre_p_uiui_rec_prec),
}

_TUPLE_PRECS = {
    "arb_hypgeom_fresnel_prec",
    "arb_hypgeom_fresnel_batch_prec",
    "arb_hypgeom_airy_prec",
    "arb_hypgeom_airy_batch_prec",
    "acb_hypgeom_airy_prec",
    "acb_hypgeom_fresnel_prec",
    "acb_hypgeom_coulomb_prec",
    "acb_hypgeom_pfq_sum_prec",
    "acb_hypgeom_pfq_sum_rs_prec",
    "acb_hypgeom_pfq_sum_bs_prec",
    "acb_hypgeom_pfq_sum_forward_prec",
    "acb_hypgeom_pfq_sum_fme_prec",
    "acb_hypgeom_pfq_sum_invz_prec",
    "acb_hypgeom_pfq_sum_bs_invz_prec",
    "acb_hypgeom_legendre_p_uiui_rec_prec",
}


def _rigorous_tuple_interval(kernel_fn, args: tuple, prec_bits: int, **kwargs):
    out = kernel_fn(*args, **kwargs)
    if not isinstance(out, tuple):
        return wc.rigorous_interval_kernel(kernel_fn, args, prec_bits, **kwargs)
    items = []
    for idx in range(len(out)):
        def sel_fn(*a, **k):
            return kernel_fn(*a, **k)[idx]
        items.append(wc.rigorous_interval_kernel(sel_fn, args, prec_bits, **kwargs))
    return tuple(items)


def _adaptive_tuple_interval(kernel_fn, args: tuple, prec_bits: int, **kwargs):
    out = kernel_fn(*args, **kwargs)
    if not isinstance(out, tuple):
        return wc.adaptive_interval_kernel(kernel_fn, args, prec_bits, **kwargs)
    items = []
    for idx in range(len(out)):
        def sel_fn(*a, **k):
            return kernel_fn(*a, **k)[idx]
        items.append(wc.adaptive_interval_kernel(sel_fn, args, prec_bits, **kwargs))
    return tuple(items)


def _rigorous_tuple_acb(kernel_fn, args: tuple, prec_bits: int, **kwargs):
    out = kernel_fn(*args, **kwargs)
    if not isinstance(out, tuple):
        return wc.rigorous_acb_kernel(kernel_fn, args, prec_bits, **kwargs)
    items = []
    for idx in range(len(out)):
        def sel_fn(*a, **k):
            return kernel_fn(*a, **k)[idx]
        items.append(wc.rigorous_acb_kernel(sel_fn, args, prec_bits, **kwargs))
    return tuple(items)


def _adaptive_tuple_acb(kernel_fn, args: tuple, prec_bits: int, **kwargs):
    out = kernel_fn(*args, **kwargs)
    if not isinstance(out, tuple):
        return wc.adaptive_acb_kernel(kernel_fn, args, prec_bits, **kwargs)
    items = []
    for idx in range(len(out)):
        def sel_fn(*a, **k):
            return kernel_fn(*a, **k)[idx]
        items.append(wc.adaptive_acb_kernel(sel_fn, args, prec_bits, **kwargs))
    return tuple(items)


def _make_wrapper(name: str) -> Callable[..., jax.Array]:
    base_fn = _pick_base_fn(name)
    kernel_fn = getattr(hypgeom, _kernel_name(name), base_fn)
    is_acb = name.startswith("acb_")
    is_tuple = name in _TUPLE_PRECS

    def wrapper(*args, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = wc.resolve_prec_bits(dps, prec_bits)
        static_kwargs = {}
        dynamic_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (bool, int, float, str)):
                static_kwargs[k] = v
            else:
                dynamic_kwargs[k] = v

        base_bound = partial(base_fn, **static_kwargs) if static_kwargs else base_fn
        kernel_bound = partial(kernel_fn, **static_kwargs) if static_kwargs else kernel_fn

        def rig_fn(*r_args, prec_bits: int, **r_kwargs):
            special = _SPECIAL_RIG.get(name)
            if special is not None:
                special_bound = partial(special, **static_kwargs) if static_kwargs else special
                return special_bound(*r_args, prec_bits=prec_bits, **r_kwargs)
            if is_tuple:
                if is_acb:
                    return _rigorous_tuple_acb(kernel_bound, r_args, prec_bits, **r_kwargs)
                return _rigorous_tuple_interval(kernel_bound, r_args, prec_bits, **r_kwargs)
            try:
                if is_acb:
                    return wc.rigorous_acb_kernel(kernel_bound, r_args, prec_bits, **r_kwargs)
                return wc.rigorous_interval_kernel(kernel_bound, r_args, prec_bits, **r_kwargs)
            except Exception:
                return base_bound(*r_args, prec_bits=prec_bits, **r_kwargs)

        def adapt_fn(*a_args, prec_bits: int, **a_kwargs):
            special = _SPECIAL_ADAPT.get(name)
            if special is not None:
                special_bound = partial(special, **static_kwargs) if static_kwargs else special
                return special_bound(*a_args, prec_bits=prec_bits, **a_kwargs)
            if is_tuple:
                if is_acb:
                    return _adaptive_tuple_acb(kernel_bound, a_args, prec_bits, **a_kwargs)
                return _adaptive_tuple_interval(kernel_bound, a_args, prec_bits, **a_kwargs)
            try:
                if is_acb:
                    return wc.adaptive_acb_kernel(kernel_bound, a_args, prec_bits, **a_kwargs)
                return wc.adaptive_interval_kernel(kernel_bound, a_args, prec_bits, **a_kwargs)
            except Exception:
                return base_bound(*a_args, prec_bits=prec_bits, **a_kwargs)

        return wc.dispatch_mode(impl, base_bound, rig_fn, adapt_fn, is_acb, pb, args, dynamic_kwargs)

    wrapper.__name__ = name.replace("_prec", "_mode")
    wrapper.__doc__ = f"Mode-dispatched wrapper around {name}. impl: basic|rigorous|adaptive."
    return wrapper


__all__: list[str] = []

for _name in dir(hypgeom):
    if _name.startswith("_"):
        continue
    if "hypgeom" not in _name:
        continue
    if not (_name.endswith("_prec") or _name.endswith("_batch_prec")):
        continue
    _fn = getattr(hypgeom, _name, None)
    if not callable(_fn):
        continue
    try:
        sig = inspect.signature(_fn)
    except (TypeError, ValueError):
        continue
    if "prec_bits" not in sig.parameters:
        continue
    _wrapper = _make_wrapper(_name)
    globals()[_wrapper.__name__] = _wrapper
    __all__.append(_wrapper.__name__)


def _pad_trim_call(core_fn, args: tuple[jax.Array, ...], pad_to: int):
    n = int(args[0].shape[0])
    padded = tuple(hypgeom._pad_rows_to(arg, pad_to) for arg in args)
    out = core_fn(*padded)
    return hypgeom._trim_rows(out, n)


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _arb_hypgeom_0f1_batch_mode_core(a, z, *, prec_bits: int, impl: str, regularized: bool = False):
    return jax.vmap(lambda aa, zz: arb_hypgeom_0f1_mode(aa, zz, impl=impl, prec_bits=prec_bits, regularized=regularized))(a, z)


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _arb_hypgeom_1f1_batch_mode_core(a, b, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.arb_hypgeom_1f1_batch_prec(a, b, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _SPECIAL_RIG["arb_hypgeom_1f1_batch_prec"](a, b, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "adaptive":
        return jax.vmap(lambda aa, bb, zz: wc.adaptive_interval_kernel(hypgeom.arb_hypgeom_1f1, (aa, bb, zz), prec_bits, regularized=regularized))(a, b, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _arb_hypgeom_2f1_batch_mode_core(a, b, c, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.arb_hypgeom_2f1_batch_prec(a, b, c, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _SPECIAL_RIG["arb_hypgeom_2f1_batch_prec"](a, b, c, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "adaptive":
        return jax.vmap(lambda aa, bb, cc, zz: wc.adaptive_interval_kernel(hypgeom.arb_hypgeom_2f1, (aa, bb, cc, zz), prec_bits, regularized=regularized))(a, b, c, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl"))
def _arb_hypgeom_u_batch_mode_core(a, b, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return hypgeom.arb_hypgeom_u_batch_prec(a, b, z, prec_bits=prec_bits)
    if impl == "rigorous":
        return _SPECIAL_RIG["arb_hypgeom_u_batch_prec"](a, b, z, prec_bits=prec_bits)
    if impl == "adaptive":
        return jax.vmap(lambda aa, bb, zz: wc.adaptive_interval_kernel(hypgeom.arb_hypgeom_u, (aa, bb, zz), prec_bits))(a, b, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _arb_hypgeom_gamma_lower_batch_mode_core(s, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.arb_hypgeom_gamma_lower_batch_prec(s, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _arb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=False)
    if impl == "adaptive":
        return _arb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=True)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _arb_hypgeom_gamma_upper_batch_mode_core(s, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.arb_hypgeom_gamma_upper_batch_prec(s, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _arb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=False)
    if impl == "adaptive":
        return _arb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=True)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "type", "prec_bits", "impl"))
def _arb_hypgeom_legendre_p_batch_mode_core(n, m, z, *, type: int, prec_bits: int, impl: str):
    return jax.vmap(lambda mm, zz: arb_hypgeom_legendre_p_mode(n, mm, zz, type=type, impl=impl, prec_bits=prec_bits))(m, z)


@partial(jax.jit, static_argnames=("n", "type", "prec_bits", "impl"))
def _arb_hypgeom_legendre_q_batch_mode_core(n, m, z, *, type: int, prec_bits: int, impl: str):
    return jax.vmap(lambda mm, zz: arb_hypgeom_legendre_q_mode(n, mm, zz, type=type, impl=impl, prec_bits=prec_bits))(m, z)


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _arb_hypgeom_jacobi_p_batch_mode_core(n, a, b, z, *, prec_bits: int, impl: str):
    return jax.vmap(lambda aa, bb, zz: arb_hypgeom_jacobi_p_mode(n, aa, bb, zz, impl=impl, prec_bits=prec_bits))(a, b, z)


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _arb_hypgeom_gegenbauer_c_batch_mode_core(n, m, z, *, prec_bits: int, impl: str):
    return jax.vmap(lambda mm, zz: arb_hypgeom_gegenbauer_c_mode(n, mm, zz, impl=impl, prec_bits=prec_bits))(m, z)


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _acb_hypgeom_0f1_batch_mode_core(a, z, *, prec_bits: int, impl: str, regularized: bool = False):
    return jax.vmap(lambda aa, zz: acb_hypgeom_0f1_mode(aa, zz, impl=impl, prec_bits=prec_bits, regularized=regularized))(a, z)


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _acb_hypgeom_1f1_batch_mode_core(a, b, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.acb_hypgeom_1f1_batch_prec(a, b, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _SPECIAL_RIG["acb_hypgeom_1f1_batch_prec"](a, b, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "adaptive":
        return jax.vmap(lambda aa, bb, zz: wc.adaptive_acb_kernel(hypgeom.acb_hypgeom_1f1, (aa, bb, zz), prec_bits, regularized=regularized))(a, b, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _acb_hypgeom_2f1_batch_mode_core(a, b, c, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.acb_hypgeom_2f1_batch_prec(a, b, c, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _SPECIAL_RIG["acb_hypgeom_2f1_batch_prec"](a, b, c, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "adaptive":
        return jax.vmap(lambda aa, bb, cc, zz: wc.adaptive_acb_kernel(hypgeom.acb_hypgeom_2f1, (aa, bb, cc, zz), prec_bits, regularized=regularized))(a, b, c, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl"))
def _acb_hypgeom_u_batch_mode_core(a, b, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return hypgeom.acb_hypgeom_u_batch_prec(a, b, z, prec_bits=prec_bits)
    if impl == "rigorous":
        return _SPECIAL_RIG["acb_hypgeom_u_batch_prec"](a, b, z, prec_bits=prec_bits)
    if impl == "adaptive":
        return jax.vmap(lambda aa, bb, zz: wc.adaptive_acb_kernel(hypgeom.acb_hypgeom_u, (aa, bb, zz), prec_bits))(a, b, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _acb_hypgeom_gamma_lower_batch_mode_core(s, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.acb_hypgeom_gamma_lower_batch_prec(s, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _acb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=False)
    if impl == "adaptive":
        return _acb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=False, adaptive=True)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "regularized"))
def _acb_hypgeom_gamma_upper_batch_mode_core(s, z, *, prec_bits: int, impl: str, regularized: bool = False):
    if impl == "basic":
        return hypgeom.acb_hypgeom_gamma_upper_batch_prec(s, z, prec_bits=prec_bits, regularized=regularized)
    if impl == "rigorous":
        return _acb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=False)
    if impl == "adaptive":
        return _acb_incomplete_gamma_special_batch(s, z, prec_bits=prec_bits, regularized=regularized, upper=True, adaptive=True)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _arb_hypgeom_chebyshev_t_batch_mode_core(n, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda zz: hypgeom.arb_hypgeom_chebyshev_t_prec(n, zz, prec_bits=prec_bits))(z)
    if impl == "rigorous":
        return jax.vmap(lambda zz: wc.rigorous_interval_kernel(partial(hypgeom.arb_hypgeom_chebyshev_t, n), (zz,), prec_bits))(z)
    if impl == "adaptive":
        return jax.vmap(lambda zz: wc.adaptive_interval_kernel(partial(hypgeom.arb_hypgeom_chebyshev_t, n), (zz,), prec_bits))(z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _arb_hypgeom_chebyshev_u_batch_mode_core(n, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda zz: hypgeom.arb_hypgeom_chebyshev_u_prec(n, zz, prec_bits=prec_bits))(z)
    if impl == "rigorous":
        return jax.vmap(lambda zz: wc.rigorous_interval_kernel(partial(hypgeom.arb_hypgeom_chebyshev_u, n), (zz,), prec_bits))(z)
    if impl == "adaptive":
        return jax.vmap(lambda zz: wc.adaptive_interval_kernel(partial(hypgeom.arb_hypgeom_chebyshev_u, n), (zz,), prec_bits))(z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _arb_hypgeom_laguerre_l_batch_mode_core(n, m, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda mm, zz: hypgeom.arb_hypgeom_laguerre_l_prec(n, mm, zz, prec_bits=prec_bits))(m, z)
    if impl == "rigorous":
        return jax.vmap(lambda mm, zz: wc.rigorous_interval_kernel(partial(hypgeom.arb_hypgeom_laguerre_l, n), (mm, zz), prec_bits))(m, z)
    if impl == "adaptive":
        return jax.vmap(lambda mm, zz: wc.adaptive_interval_kernel(partial(hypgeom.arb_hypgeom_laguerre_l, n), (mm, zz), prec_bits))(m, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _arb_hypgeom_hermite_h_batch_mode_core(n, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda zz: hypgeom.arb_hypgeom_hermite_h_prec(n, zz, prec_bits=prec_bits))(z)
    if impl == "rigorous":
        return jax.vmap(lambda zz: wc.rigorous_interval_kernel(partial(hypgeom.arb_hypgeom_hermite_h, n), (zz,), prec_bits))(z)
    if impl == "adaptive":
        return jax.vmap(lambda zz: wc.adaptive_interval_kernel(partial(hypgeom.arb_hypgeom_hermite_h, n), (zz,), prec_bits))(z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "reciprocal", "n_terms"))
def _arb_hypgeom_pfq_batch_mode_core(a, b, z, *, prec_bits: int, impl: str, reciprocal: bool = False, n_terms: int = 32):
    if impl == "basic":
        return hypgeom.arb_hypgeom_pfq_batch_prec(a, b, z, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms)
    if impl == "rigorous":
        return jax.vmap(
            lambda aa, bb, zz: wc.rigorous_interval_kernel(
                hypgeom.arb_hypgeom_pfq,
                (aa, bb, zz),
                prec_bits,
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
        )(a, b, z)
    if impl == "adaptive":
        return jax.vmap(
            lambda aa, bb, zz: wc.adaptive_interval_kernel(
                hypgeom.arb_hypgeom_pfq,
                (aa, bb, zz),
                prec_bits,
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
        )(a, b, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _acb_hypgeom_chebyshev_t_batch_mode_core(n, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda zz: hypgeom.acb_hypgeom_chebyshev_t_prec(n, zz, prec_bits=prec_bits))(z)
    if impl == "rigorous":
        return jax.vmap(lambda zz: wc.rigorous_acb_kernel(partial(hypgeom.acb_hypgeom_chebyshev_t, n), (zz,), prec_bits))(z)
    if impl == "adaptive":
        return jax.vmap(lambda zz: wc.adaptive_acb_kernel(partial(hypgeom.acb_hypgeom_chebyshev_t, n), (zz,), prec_bits))(z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _acb_hypgeom_chebyshev_u_batch_mode_core(n, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda zz: hypgeom.acb_hypgeom_chebyshev_u_prec(n, zz, prec_bits=prec_bits))(z)
    if impl == "rigorous":
        return jax.vmap(lambda zz: wc.rigorous_acb_kernel(partial(hypgeom.acb_hypgeom_chebyshev_u, n), (zz,), prec_bits))(z)
    if impl == "adaptive":
        return jax.vmap(lambda zz: wc.adaptive_acb_kernel(partial(hypgeom.acb_hypgeom_chebyshev_u, n), (zz,), prec_bits))(z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _acb_hypgeom_laguerre_l_batch_mode_core(n, a, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda aa, zz: hypgeom.acb_hypgeom_laguerre_l_prec(n, aa, zz, prec_bits=prec_bits))(a, z)
    if impl == "rigorous":
        return jax.vmap(lambda aa, zz: wc.rigorous_acb_kernel(partial(hypgeom.acb_hypgeom_laguerre_l, n), (aa, zz), prec_bits))(a, z)
    if impl == "adaptive":
        return jax.vmap(lambda aa, zz: wc.adaptive_acb_kernel(partial(hypgeom.acb_hypgeom_laguerre_l, n), (aa, zz), prec_bits))(a, z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("n", "prec_bits", "impl"))
def _acb_hypgeom_hermite_h_batch_mode_core(n, z, *, prec_bits: int, impl: str):
    if impl == "basic":
        return jax.vmap(lambda zz: hypgeom.acb_hypgeom_hermite_h_prec(n, zz, prec_bits=prec_bits))(z)
    if impl == "rigorous":
        return jax.vmap(lambda zz: wc.rigorous_acb_kernel(partial(hypgeom.acb_hypgeom_hermite_h, n), (zz,), prec_bits))(z)
    if impl == "adaptive":
        return jax.vmap(lambda zz: wc.adaptive_acb_kernel(partial(hypgeom.acb_hypgeom_hermite_h, n), (zz,), prec_bits))(z)
    raise ValueError(f"Unsupported impl: {impl}")


@partial(jax.jit, static_argnames=("prec_bits", "impl", "reciprocal", "n_terms"))
def _acb_hypgeom_pfq_batch_mode_core(a, b, z, *, prec_bits: int, impl: str, reciprocal: bool = False, n_terms: int = 32):
    if impl == "basic":
        return hypgeom.acb_hypgeom_pfq_batch_prec(a, b, z, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms)
    if impl == "rigorous":
        return jax.vmap(
            lambda aa, bb, zz: wc.rigorous_acb_kernel(
                hypgeom.acb_hypgeom_pfq,
                (aa, bb, zz),
                prec_bits,
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
        )(a, b, z)
    if impl == "adaptive":
        return jax.vmap(
            lambda aa, bb, zz: wc.adaptive_acb_kernel(
                hypgeom.acb_hypgeom_pfq,
                (aa, bb, zz),
                prec_bits,
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
        )(a, b, z)
    raise ValueError(f"Unsupported impl: {impl}")


def arb_hypgeom_0f1_batch_mode_padded(a, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda aa, zz: _arb_hypgeom_0f1_batch_mode_core(aa, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (a, z), pad_to)


def arb_hypgeom_0f1_batch_mode_fixed(a, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _arb_hypgeom_0f1_batch_mode_core(a, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def arb_hypgeom_1f1_batch_mode_padded(a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda aa, bb, zz: _arb_hypgeom_1f1_batch_mode_core(aa, bb, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (a, b, z), pad_to)


def arb_hypgeom_1f1_batch_mode_fixed(a, b, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _arb_hypgeom_1f1_batch_mode_core(a, b, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def arb_hypgeom_2f1_batch_mode_padded(a, b, c, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(
        lambda aa, bb, cc, zz: _arb_hypgeom_2f1_batch_mode_core(aa, bb, cc, zz, prec_bits=prec_bits, impl=impl, regularized=regularized),
        (a, b, c, z),
        pad_to,
    )


def arb_hypgeom_2f1_batch_mode_fixed(a, b, c, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _arb_hypgeom_2f1_batch_mode_core(a, b, c, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def arb_hypgeom_u_batch_mode_padded(a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda aa, bb, zz: _arb_hypgeom_u_batch_mode_core(aa, bb, zz, prec_bits=prec_bits, impl=impl), (a, b, z), pad_to)


def arb_hypgeom_u_batch_mode_fixed(a, b, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_u_batch_mode_core(a, b, z, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_gamma_lower_batch_mode_padded(s, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda ss, zz: _arb_hypgeom_gamma_lower_batch_mode_core(ss, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (s, z), pad_to)


def arb_hypgeom_gamma_lower_batch_mode_fixed(s, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _arb_hypgeom_gamma_lower_batch_mode_core(s, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def arb_hypgeom_gamma_upper_batch_mode_padded(s, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda ss, zz: _arb_hypgeom_gamma_upper_batch_mode_core(ss, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (s, z), pad_to)


def arb_hypgeom_gamma_upper_batch_mode_fixed(s, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _arb_hypgeom_gamma_upper_batch_mode_core(s, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def arb_hypgeom_legendre_p_batch_mode_padded(n, m, z, *, pad_to: int, impl: str, prec_bits: int = 53, type: int = 0):
    return _pad_trim_call(lambda mm, zz: _arb_hypgeom_legendre_p_batch_mode_core(n, mm, zz, type=type, prec_bits=prec_bits, impl=impl), (m, z), pad_to)


def arb_hypgeom_legendre_p_batch_mode_fixed(n, m, z, *, impl: str, prec_bits: int = 53, type: int = 0):
    return _arb_hypgeom_legendre_p_batch_mode_core(n, m, z, type=type, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_legendre_q_batch_mode_padded(n, m, z, *, pad_to: int, impl: str, prec_bits: int = 53, type: int = 0):
    return _pad_trim_call(lambda mm, zz: _arb_hypgeom_legendre_q_batch_mode_core(n, mm, zz, type=type, prec_bits=prec_bits, impl=impl), (m, z), pad_to)


def arb_hypgeom_legendre_q_batch_mode_fixed(n, m, z, *, impl: str, prec_bits: int = 53, type: int = 0):
    return _arb_hypgeom_legendre_q_batch_mode_core(n, m, z, type=type, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_jacobi_p_batch_mode_padded(n, a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda aa, bb, zz: _arb_hypgeom_jacobi_p_batch_mode_core(n, aa, bb, zz, prec_bits=prec_bits, impl=impl), (a, b, z), pad_to)


def arb_hypgeom_jacobi_p_batch_mode_fixed(n, a, b, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_jacobi_p_batch_mode_core(n, a, b, z, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_gegenbauer_c_batch_mode_padded(n, m, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda mm, zz: _arb_hypgeom_gegenbauer_c_batch_mode_core(n, mm, zz, prec_bits=prec_bits, impl=impl), (m, z), pad_to)


def arb_hypgeom_gegenbauer_c_batch_mode_fixed(n, m, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_gegenbauer_c_batch_mode_core(n, m, z, prec_bits=prec_bits, impl=impl)


def acb_hypgeom_0f1_batch_mode_padded(a, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda aa, zz: _acb_hypgeom_0f1_batch_mode_core(aa, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (a, z), pad_to)


def acb_hypgeom_0f1_batch_mode_fixed(a, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _acb_hypgeom_0f1_batch_mode_core(a, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def acb_hypgeom_1f1_batch_mode_padded(a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda aa, bb, zz: _acb_hypgeom_1f1_batch_mode_core(aa, bb, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (a, b, z), pad_to)


def acb_hypgeom_1f1_batch_mode_fixed(a, b, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _acb_hypgeom_1f1_batch_mode_core(a, b, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def acb_hypgeom_2f1_batch_mode_padded(a, b, c, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(
        lambda aa, bb, cc, zz: _acb_hypgeom_2f1_batch_mode_core(aa, bb, cc, zz, prec_bits=prec_bits, impl=impl, regularized=regularized),
        (a, b, c, z),
        pad_to,
    )


def acb_hypgeom_2f1_batch_mode_fixed(a, b, c, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _acb_hypgeom_2f1_batch_mode_core(a, b, c, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def acb_hypgeom_u_batch_mode_padded(a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda aa, bb, zz: _acb_hypgeom_u_batch_mode_core(aa, bb, zz, prec_bits=prec_bits, impl=impl), (a, b, z), pad_to)


def acb_hypgeom_u_batch_mode_fixed(a, b, z, *, impl: str, prec_bits: int = 53):
    return _acb_hypgeom_u_batch_mode_core(a, b, z, prec_bits=prec_bits, impl=impl)


def acb_hypgeom_gamma_lower_batch_mode_padded(s, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda ss, zz: _acb_hypgeom_gamma_lower_batch_mode_core(ss, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (s, z), pad_to)


def acb_hypgeom_gamma_lower_batch_mode_fixed(s, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _acb_hypgeom_gamma_lower_batch_mode_core(s, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def acb_hypgeom_gamma_upper_batch_mode_padded(s, z, *, pad_to: int, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _pad_trim_call(lambda ss, zz: _acb_hypgeom_gamma_upper_batch_mode_core(ss, zz, prec_bits=prec_bits, impl=impl, regularized=regularized), (s, z), pad_to)


def acb_hypgeom_gamma_upper_batch_mode_fixed(s, z, *, impl: str, prec_bits: int = 53, regularized: bool = False):
    return _acb_hypgeom_gamma_upper_batch_mode_core(s, z, prec_bits=prec_bits, impl=impl, regularized=regularized)


def arb_hypgeom_chebyshev_t_batch_mode_padded(n, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda zz: _arb_hypgeom_chebyshev_t_batch_mode_core(n, zz, prec_bits=prec_bits, impl=impl), (z,), pad_to)


def arb_hypgeom_chebyshev_t_batch_mode_fixed(n, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_chebyshev_t_batch_mode_core(n, z, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_chebyshev_u_batch_mode_padded(n, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda zz: _arb_hypgeom_chebyshev_u_batch_mode_core(n, zz, prec_bits=prec_bits, impl=impl), (z,), pad_to)


def arb_hypgeom_chebyshev_u_batch_mode_fixed(n, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_chebyshev_u_batch_mode_core(n, z, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_laguerre_l_batch_mode_padded(n, m, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda mm, zz: _arb_hypgeom_laguerre_l_batch_mode_core(n, mm, zz, prec_bits=prec_bits, impl=impl), (m, z), pad_to)


def arb_hypgeom_laguerre_l_batch_mode_fixed(n, m, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_laguerre_l_batch_mode_core(n, m, z, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_hermite_h_batch_mode_padded(n, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda zz: _arb_hypgeom_hermite_h_batch_mode_core(n, zz, prec_bits=prec_bits, impl=impl), (z,), pad_to)


def arb_hypgeom_hermite_h_batch_mode_fixed(n, z, *, impl: str, prec_bits: int = 53):
    return _arb_hypgeom_hermite_h_batch_mode_core(n, z, prec_bits=prec_bits, impl=impl)


def arb_hypgeom_pfq_batch_mode_padded(a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53, reciprocal: bool = False, n_terms: int = 32):
    return _pad_trim_call(
        lambda aa, bb, zz: _arb_hypgeom_pfq_batch_mode_core(
            aa, bb, zz, prec_bits=prec_bits, impl=impl, reciprocal=reciprocal, n_terms=n_terms
        ),
        (a, b, z),
        pad_to,
    )


def arb_hypgeom_pfq_batch_mode_fixed(a, b, z, *, impl: str, prec_bits: int = 53, reciprocal: bool = False, n_terms: int = 32):
    return _arb_hypgeom_pfq_batch_mode_core(a, b, z, prec_bits=prec_bits, impl=impl, reciprocal=reciprocal, n_terms=n_terms)


def acb_hypgeom_chebyshev_t_batch_mode_padded(n, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda zz: _acb_hypgeom_chebyshev_t_batch_mode_core(n, zz, prec_bits=prec_bits, impl=impl), (z,), pad_to)


def acb_hypgeom_chebyshev_t_batch_mode_fixed(n, z, *, impl: str, prec_bits: int = 53):
    return _acb_hypgeom_chebyshev_t_batch_mode_core(n, z, prec_bits=prec_bits, impl=impl)


def acb_hypgeom_chebyshev_u_batch_mode_padded(n, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda zz: _acb_hypgeom_chebyshev_u_batch_mode_core(n, zz, prec_bits=prec_bits, impl=impl), (z,), pad_to)


def acb_hypgeom_chebyshev_u_batch_mode_fixed(n, z, *, impl: str, prec_bits: int = 53):
    return _acb_hypgeom_chebyshev_u_batch_mode_core(n, z, prec_bits=prec_bits, impl=impl)


def acb_hypgeom_laguerre_l_batch_mode_padded(n, a, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda aa, zz: _acb_hypgeom_laguerre_l_batch_mode_core(n, aa, zz, prec_bits=prec_bits, impl=impl), (a, z), pad_to)


def acb_hypgeom_laguerre_l_batch_mode_fixed(n, a, z, *, impl: str, prec_bits: int = 53):
    return _acb_hypgeom_laguerre_l_batch_mode_core(n, a, z, prec_bits=prec_bits, impl=impl)


def acb_hypgeom_hermite_h_batch_mode_padded(n, z, *, pad_to: int, impl: str, prec_bits: int = 53):
    return _pad_trim_call(lambda zz: _acb_hypgeom_hermite_h_batch_mode_core(n, zz, prec_bits=prec_bits, impl=impl), (z,), pad_to)


def acb_hypgeom_hermite_h_batch_mode_fixed(n, z, *, impl: str, prec_bits: int = 53):
    return _acb_hypgeom_hermite_h_batch_mode_core(n, z, prec_bits=prec_bits, impl=impl)


def acb_hypgeom_pfq_batch_mode_padded(a, b, z, *, pad_to: int, impl: str, prec_bits: int = 53, reciprocal: bool = False, n_terms: int = 32):
    return _pad_trim_call(
        lambda aa, bb, zz: _acb_hypgeom_pfq_batch_mode_core(
            aa, bb, zz, prec_bits=prec_bits, impl=impl, reciprocal=reciprocal, n_terms=n_terms
        ),
        (a, b, z),
        pad_to,
    )


def acb_hypgeom_pfq_batch_mode_fixed(a, b, z, *, impl: str, prec_bits: int = 53, reciprocal: bool = False, n_terms: int = 32):
    return _acb_hypgeom_pfq_batch_mode_core(a, b, z, prec_bits=prec_bits, impl=impl, reciprocal=reciprocal, n_terms=n_terms)


__all__.extend(
    [
        "arb_hypgeom_0f1_batch_mode_padded",
        "arb_hypgeom_0f1_batch_mode_fixed",
        "arb_hypgeom_1f1_batch_mode_padded",
        "arb_hypgeom_1f1_batch_mode_fixed",
        "arb_hypgeom_2f1_batch_mode_padded",
        "arb_hypgeom_2f1_batch_mode_fixed",
        "arb_hypgeom_u_batch_mode_padded",
        "arb_hypgeom_u_batch_mode_fixed",
        "arb_hypgeom_gamma_lower_batch_mode_padded",
        "arb_hypgeom_gamma_lower_batch_mode_fixed",
        "arb_hypgeom_gamma_upper_batch_mode_padded",
        "arb_hypgeom_gamma_upper_batch_mode_fixed",
        "arb_hypgeom_legendre_p_batch_mode_padded",
        "arb_hypgeom_legendre_p_batch_mode_fixed",
        "arb_hypgeom_legendre_q_batch_mode_padded",
        "arb_hypgeom_legendre_q_batch_mode_fixed",
        "arb_hypgeom_jacobi_p_batch_mode_padded",
        "arb_hypgeom_jacobi_p_batch_mode_fixed",
        "arb_hypgeom_gegenbauer_c_batch_mode_padded",
        "arb_hypgeom_gegenbauer_c_batch_mode_fixed",
        "arb_hypgeom_chebyshev_t_batch_mode_padded",
        "arb_hypgeom_chebyshev_t_batch_mode_fixed",
        "arb_hypgeom_chebyshev_u_batch_mode_padded",
        "arb_hypgeom_chebyshev_u_batch_mode_fixed",
        "arb_hypgeom_laguerre_l_batch_mode_padded",
        "arb_hypgeom_laguerre_l_batch_mode_fixed",
        "arb_hypgeom_hermite_h_batch_mode_padded",
        "arb_hypgeom_hermite_h_batch_mode_fixed",
        "arb_hypgeom_pfq_batch_mode_padded",
        "arb_hypgeom_pfq_batch_mode_fixed",
        "acb_hypgeom_0f1_batch_mode_padded",
        "acb_hypgeom_0f1_batch_mode_fixed",
        "acb_hypgeom_1f1_batch_mode_padded",
        "acb_hypgeom_1f1_batch_mode_fixed",
        "acb_hypgeom_2f1_batch_mode_padded",
        "acb_hypgeom_2f1_batch_mode_fixed",
        "acb_hypgeom_u_batch_mode_padded",
        "acb_hypgeom_u_batch_mode_fixed",
        "acb_hypgeom_gamma_lower_batch_mode_padded",
        "acb_hypgeom_gamma_lower_batch_mode_fixed",
        "acb_hypgeom_gamma_upper_batch_mode_padded",
        "acb_hypgeom_gamma_upper_batch_mode_fixed",
        "acb_hypgeom_chebyshev_t_batch_mode_padded",
        "acb_hypgeom_chebyshev_t_batch_mode_fixed",
        "acb_hypgeom_chebyshev_u_batch_mode_padded",
        "acb_hypgeom_chebyshev_u_batch_mode_fixed",
        "acb_hypgeom_laguerre_l_batch_mode_padded",
        "acb_hypgeom_laguerre_l_batch_mode_fixed",
        "acb_hypgeom_hermite_h_batch_mode_padded",
        "acb_hypgeom_hermite_h_batch_mode_fixed",
        "acb_hypgeom_pfq_batch_mode_padded",
        "acb_hypgeom_pfq_batch_mode_fixed",
    ]
)
