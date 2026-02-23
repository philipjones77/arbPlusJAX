from __future__ import annotations

import inspect
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


def _wrap_series_rig(fn):
    def wrapped(*args, prec_bits: int, **kwargs):
        return fn(*args, prec_bits=prec_bits, **kwargs)

    return wrapped


def _wrap_series_rig_batch(fn, in_axes):
    def wrapped(*args, prec_bits: int, **kwargs):
        return jax.vmap(lambda *aa: fn(*aa, prec_bits=prec_bits, **kwargs), in_axes=in_axes)(*args)

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

    def wrapper(*args, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = wc.resolve_prec_bits(dps, prec_bits)

        def rig_fn(*r_args, prec_bits: int, **r_kwargs):
            special = _SPECIAL_RIG.get(name)
            if special is not None:
                return special(*r_args, prec_bits=prec_bits, **r_kwargs)
            if is_tuple:
                if is_acb:
                    return _rigorous_tuple_acb(kernel_fn, r_args, prec_bits, **r_kwargs)
                return _rigorous_tuple_interval(kernel_fn, r_args, prec_bits, **r_kwargs)
            if is_acb:
                return wc.rigorous_acb_kernel(kernel_fn, r_args, prec_bits, **r_kwargs)
            return wc.rigorous_interval_kernel(kernel_fn, r_args, prec_bits, **r_kwargs)

        def adapt_fn(*a_args, prec_bits: int, **a_kwargs):
            special = _SPECIAL_ADAPT.get(name)
            if special is not None:
                return special(*a_args, prec_bits=prec_bits, **a_kwargs)
            if is_tuple:
                if is_acb:
                    return _adaptive_tuple_acb(kernel_fn, a_args, prec_bits, **a_kwargs)
                return _adaptive_tuple_interval(kernel_fn, a_args, prec_bits, **a_kwargs)
            if is_acb:
                return wc.adaptive_acb_kernel(kernel_fn, a_args, prec_bits, **a_kwargs)
            return wc.adaptive_interval_kernel(kernel_fn, a_args, prec_bits, **a_kwargs)

        return wc.dispatch_mode(impl, base_fn, rig_fn, adapt_fn, is_acb, pb, args, kwargs)

    wrapper.__name__ = name.replace("_prec", "_mode")
    wrapper.__doc__ = f"Mode-dispatched wrapper around {name}. impl: baseline|rigorous|adaptive."
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
