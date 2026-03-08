from __future__ import annotations

import importlib
import inspect
from functools import lru_cache, partial
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp

from . import acb_core
from . import baseline_wrappers
from . import boost_hypgeom
from . import cubesselk
from . import double_gamma
from . import double_interval as di
from . import hypgeom
from . import hypgeom_wrappers
from .kernel_helpers import (
    mixed_batch_size_or_none,
    pad_batch_args,
    pad_mixed_batch_args_repeat_last,
    trim_batch_out,
)
from . import point_wrappers

# Public API for optimized calls.
# - eval_point: point-only kernels (fastest, no bounds)
# - eval_interval: interval kernels (basic/adaptive/rigorous) with optional batching

_COMPLEX_BY_FLOAT_DTYPE = {
    jnp.dtype(jnp.float32): jnp.dtype(jnp.complex64),
    jnp.dtype(jnp.float64): jnp.dtype(jnp.complex128),
}

_HYPGEOM_BASIC_BATCH_FASTPATHS = {
    "arb_hypgeom_0f1": ("arb_hypgeom_0f1_batch_fixed_prec", "arb_hypgeom_0f1_batch_padded_prec"),
    "arb_hypgeom_1f1": ("arb_hypgeom_1f1_batch_fixed_prec", "arb_hypgeom_1f1_batch_padded_prec"),
    "arb_hypgeom_m": ("arb_hypgeom_m_batch_fixed_prec", "arb_hypgeom_m_batch_padded_prec"),
    "arb_hypgeom_2f1": ("arb_hypgeom_2f1_batch_fixed_prec", "arb_hypgeom_2f1_batch_padded_prec"),
    "arb_hypgeom_u": ("arb_hypgeom_u_batch_fixed_prec", "arb_hypgeom_u_batch_padded_prec"),
    "arb_hypgeom_gamma_lower": ("arb_hypgeom_gamma_lower_batch_fixed_prec", "arb_hypgeom_gamma_lower_batch_padded_prec"),
    "arb_hypgeom_gamma_upper": ("arb_hypgeom_gamma_upper_batch_fixed_prec", "arb_hypgeom_gamma_upper_batch_padded_prec"),
    "arb_hypgeom_legendre_p": ("arb_hypgeom_legendre_p_batch_fixed_prec", "arb_hypgeom_legendre_p_batch_padded_prec"),
    "arb_hypgeom_legendre_q": ("arb_hypgeom_legendre_q_batch_fixed_prec", "arb_hypgeom_legendre_q_batch_padded_prec"),
    "arb_hypgeom_jacobi_p": ("arb_hypgeom_jacobi_p_batch_fixed_prec", "arb_hypgeom_jacobi_p_batch_padded_prec"),
    "arb_hypgeom_gegenbauer_c": ("arb_hypgeom_gegenbauer_c_batch_fixed_prec", "arb_hypgeom_gegenbauer_c_batch_padded_prec"),
    "arb_hypgeom_chebyshev_t": ("arb_hypgeom_chebyshev_t_batch_fixed_prec", "arb_hypgeom_chebyshev_t_batch_padded_prec"),
    "arb_hypgeom_chebyshev_u": ("arb_hypgeom_chebyshev_u_batch_fixed_prec", "arb_hypgeom_chebyshev_u_batch_padded_prec"),
    "arb_hypgeom_laguerre_l": ("arb_hypgeom_laguerre_l_batch_fixed_prec", "arb_hypgeom_laguerre_l_batch_padded_prec"),
    "arb_hypgeom_hermite_h": ("arb_hypgeom_hermite_h_batch_fixed_prec", "arb_hypgeom_hermite_h_batch_padded_prec"),
    "arb_hypgeom_pfq": ("arb_hypgeom_pfq_batch_fixed_prec", "arb_hypgeom_pfq_batch_padded_prec"),
    "acb_hypgeom_0f1": ("acb_hypgeom_0f1_batch_fixed_prec", "acb_hypgeom_0f1_batch_padded_prec"),
    "acb_hypgeom_1f1": ("acb_hypgeom_1f1_batch_fixed_prec", "acb_hypgeom_1f1_batch_padded_prec"),
    "acb_hypgeom_m": ("acb_hypgeom_m_batch_fixed_prec", "acb_hypgeom_m_batch_padded_prec"),
    "acb_hypgeom_2f1": ("acb_hypgeom_2f1_batch_fixed_prec", "acb_hypgeom_2f1_batch_padded_prec"),
    "acb_hypgeom_u": ("acb_hypgeom_u_batch_fixed_prec", "acb_hypgeom_u_batch_padded_prec"),
    "acb_hypgeom_gamma_lower": ("acb_hypgeom_gamma_lower_batch_fixed_prec", "acb_hypgeom_gamma_lower_batch_padded_prec"),
    "acb_hypgeom_gamma_upper": ("acb_hypgeom_gamma_upper_batch_fixed_prec", "acb_hypgeom_gamma_upper_batch_padded_prec"),
    "acb_hypgeom_chebyshev_t": ("acb_hypgeom_chebyshev_t_batch_fixed_prec", "acb_hypgeom_chebyshev_t_batch_padded_prec"),
    "acb_hypgeom_chebyshev_u": ("acb_hypgeom_chebyshev_u_batch_fixed_prec", "acb_hypgeom_chebyshev_u_batch_padded_prec"),
    "acb_hypgeom_laguerre_l": ("acb_hypgeom_laguerre_l_batch_fixed_prec", "acb_hypgeom_laguerre_l_batch_padded_prec"),
    "acb_hypgeom_hermite_h": ("acb_hypgeom_hermite_h_batch_fixed_prec", "acb_hypgeom_hermite_h_batch_padded_prec"),
    "acb_hypgeom_pfq": ("acb_hypgeom_pfq_batch_fixed_prec", "acb_hypgeom_pfq_batch_padded_prec"),
}

_BOOST_HYPGEOM_BASIC_BATCH_FASTPATHS = {
    "boost_hypergeometric_0f1": ("boost_hypergeometric_0f1_batch_fixed_prec", "boost_hypergeometric_0f1_batch_padded_prec"),
    "boost_hypergeometric_1f1": ("boost_hypergeometric_1f1_batch_fixed_prec", "boost_hypergeometric_1f1_batch_padded_prec"),
    "boost_hyp2f1_series": ("boost_hyp2f1_series_batch_fixed_prec", "boost_hyp2f1_series_batch_padded_prec"),
    "boost_hyp2f1_cf": ("boost_hyp2f1_series_batch_fixed_prec", "boost_hyp2f1_series_batch_padded_prec"),
    "boost_hyp2f1_pade": ("boost_hyp2f1_series_batch_fixed_prec", "boost_hyp2f1_series_batch_padded_prec"),
    "boost_hyp2f1_rational": ("boost_hyp2f1_series_batch_fixed_prec", "boost_hyp2f1_series_batch_padded_prec"),
    "boost_hypergeometric_pfq": ("boost_hypergeometric_pfq_batch_fixed_prec", "boost_hypergeometric_pfq_batch_padded_prec"),
}

_HYPGEOM_MODE_BATCH_FASTPATHS = {
    "arb_hypgeom_0f1": ("arb_hypgeom_0f1_batch_mode_fixed", "arb_hypgeom_0f1_batch_mode_padded"),
    "arb_hypgeom_1f1": ("arb_hypgeom_1f1_batch_mode_fixed", "arb_hypgeom_1f1_batch_mode_padded"),
    "arb_hypgeom_2f1": ("arb_hypgeom_2f1_batch_mode_fixed", "arb_hypgeom_2f1_batch_mode_padded"),
    "arb_hypgeom_u": ("arb_hypgeom_u_batch_mode_fixed", "arb_hypgeom_u_batch_mode_padded"),
    "arb_hypgeom_gamma_lower": ("arb_hypgeom_gamma_lower_batch_mode_fixed", "arb_hypgeom_gamma_lower_batch_mode_padded"),
    "arb_hypgeom_gamma_upper": ("arb_hypgeom_gamma_upper_batch_mode_fixed", "arb_hypgeom_gamma_upper_batch_mode_padded"),
    "arb_hypgeom_legendre_p": ("arb_hypgeom_legendre_p_batch_mode_fixed", "arb_hypgeom_legendre_p_batch_mode_padded"),
    "arb_hypgeom_legendre_q": ("arb_hypgeom_legendre_q_batch_mode_fixed", "arb_hypgeom_legendre_q_batch_mode_padded"),
    "arb_hypgeom_jacobi_p": ("arb_hypgeom_jacobi_p_batch_mode_fixed", "arb_hypgeom_jacobi_p_batch_mode_padded"),
    "arb_hypgeom_gegenbauer_c": ("arb_hypgeom_gegenbauer_c_batch_mode_fixed", "arb_hypgeom_gegenbauer_c_batch_mode_padded"),
    "arb_hypgeom_chebyshev_t": ("arb_hypgeom_chebyshev_t_batch_mode_fixed", "arb_hypgeom_chebyshev_t_batch_mode_padded"),
    "arb_hypgeom_chebyshev_u": ("arb_hypgeom_chebyshev_u_batch_mode_fixed", "arb_hypgeom_chebyshev_u_batch_mode_padded"),
    "arb_hypgeom_laguerre_l": ("arb_hypgeom_laguerre_l_batch_mode_fixed", "arb_hypgeom_laguerre_l_batch_mode_padded"),
    "arb_hypgeom_hermite_h": ("arb_hypgeom_hermite_h_batch_mode_fixed", "arb_hypgeom_hermite_h_batch_mode_padded"),
    "arb_hypgeom_pfq": ("arb_hypgeom_pfq_batch_mode_fixed", "arb_hypgeom_pfq_batch_mode_padded"),
    "acb_hypgeom_0f1": ("acb_hypgeom_0f1_batch_mode_fixed", "acb_hypgeom_0f1_batch_mode_padded"),
    "acb_hypgeom_1f1": ("acb_hypgeom_1f1_batch_mode_fixed", "acb_hypgeom_1f1_batch_mode_padded"),
    "acb_hypgeom_2f1": ("acb_hypgeom_2f1_batch_mode_fixed", "acb_hypgeom_2f1_batch_mode_padded"),
    "acb_hypgeom_u": ("acb_hypgeom_u_batch_mode_fixed", "acb_hypgeom_u_batch_mode_padded"),
    "acb_hypgeom_gamma_lower": ("acb_hypgeom_gamma_lower_batch_mode_fixed", "acb_hypgeom_gamma_lower_batch_mode_padded"),
    "acb_hypgeom_gamma_upper": ("acb_hypgeom_gamma_upper_batch_mode_fixed", "acb_hypgeom_gamma_upper_batch_mode_padded"),
    "acb_hypgeom_chebyshev_t": ("acb_hypgeom_chebyshev_t_batch_mode_fixed", "acb_hypgeom_chebyshev_t_batch_mode_padded"),
    "acb_hypgeom_chebyshev_u": ("acb_hypgeom_chebyshev_u_batch_mode_fixed", "acb_hypgeom_chebyshev_u_batch_mode_padded"),
    "acb_hypgeom_laguerre_l": ("acb_hypgeom_laguerre_l_batch_mode_fixed", "acb_hypgeom_laguerre_l_batch_mode_padded"),
    "acb_hypgeom_hermite_h": ("acb_hypgeom_hermite_h_batch_mode_fixed", "acb_hypgeom_hermite_h_batch_mode_padded"),
    "acb_hypgeom_pfq": ("acb_hypgeom_pfq_batch_mode_fixed", "acb_hypgeom_pfq_batch_mode_padded"),
}

_BOOST_HYPGEOM_MODE_BATCH_FASTPATHS = {
    "boost_hypergeometric_0f1": ("boost_hypergeometric_0f1_batch_mode_fixed", "boost_hypergeometric_0f1_batch_mode_padded"),
    "boost_hypergeometric_1f1": ("boost_hypergeometric_1f1_batch_mode_fixed", "boost_hypergeometric_1f1_batch_mode_padded"),
    "boost_hyp2f1_series": ("boost_hyp2f1_series_batch_mode_fixed", "boost_hyp2f1_series_batch_mode_padded"),
    "boost_hyp2f1_cf": ("boost_hyp2f1_series_batch_mode_fixed", "boost_hyp2f1_series_batch_mode_padded"),
    "boost_hyp2f1_pade": ("boost_hyp2f1_series_batch_mode_fixed", "boost_hyp2f1_series_batch_mode_padded"),
    "boost_hyp2f1_rational": ("boost_hyp2f1_series_batch_mode_fixed", "boost_hyp2f1_series_batch_mode_padded"),
    "boost_hypergeometric_pfq": ("boost_hypergeometric_pfq_batch_mode_fixed", "boost_hypergeometric_pfq_batch_mode_padded"),
}

_HYPGEOM_PREFER_PADDED_FASTPATHS = {
    "arb_hypgeom_1f1",
    "arb_hypgeom_2f1",
    "arb_hypgeom_u",
    "arb_hypgeom_gamma_lower",
    "arb_hypgeom_gamma_upper",
    "acb_hypgeom_1f1",
    "acb_hypgeom_2f1",
    "acb_hypgeom_u",
    "acb_hypgeom_gamma_lower",
    "acb_hypgeom_gamma_upper",
    "boost_hypergeometric_0f1",
    "boost_hypergeometric_1f1",
    "boost_hyp2f1_series",
    "boost_hyp2f1_cf",
    "boost_hyp2f1_pade",
    "boost_hyp2f1_rational",
    "boost_hypergeometric_pfq",
}

_DIRECT_INTERVAL_BASIC_BATCH_FASTPATHS = {
    "hypgeom.arb_hypgeom_1f1": (hypgeom.arb_hypgeom_1f1_batch_fixed_prec, hypgeom.arb_hypgeom_1f1_batch_padded_prec),
    "hypgeom.arb_hypgeom_2f1": (hypgeom.arb_hypgeom_2f1_batch_fixed_prec, hypgeom.arb_hypgeom_2f1_batch_padded_prec),
    "hypgeom.arb_hypgeom_u": (hypgeom.arb_hypgeom_u_batch_fixed_prec, hypgeom.arb_hypgeom_u_batch_padded_prec),
    "hypgeom.arb_hypgeom_gamma_lower": (hypgeom.arb_hypgeom_gamma_lower_batch_fixed_prec, hypgeom.arb_hypgeom_gamma_lower_batch_padded_prec),
    "hypgeom.arb_hypgeom_gamma_upper": (hypgeom.arb_hypgeom_gamma_upper_batch_fixed_prec, hypgeom.arb_hypgeom_gamma_upper_batch_padded_prec),
    "hypgeom.acb_hypgeom_1f1": (hypgeom.acb_hypgeom_1f1_batch_fixed_prec, hypgeom.acb_hypgeom_1f1_batch_padded_prec),
    "hypgeom.acb_hypgeom_2f1": (hypgeom.acb_hypgeom_2f1_batch_fixed_prec, hypgeom.acb_hypgeom_2f1_batch_padded_prec),
    "hypgeom.acb_hypgeom_u": (hypgeom.acb_hypgeom_u_batch_fixed_prec, hypgeom.acb_hypgeom_u_batch_padded_prec),
    "hypgeom.acb_hypgeom_gamma_lower": (hypgeom.acb_hypgeom_gamma_lower_batch_fixed_prec, hypgeom.acb_hypgeom_gamma_lower_batch_padded_prec),
    "hypgeom.acb_hypgeom_gamma_upper": (hypgeom.acb_hypgeom_gamma_upper_batch_fixed_prec, hypgeom.acb_hypgeom_gamma_upper_batch_padded_prec),
    "boost_hypergeometric_0f1": (boost_hypgeom.boost_hypergeometric_0f1_batch_fixed_prec, boost_hypgeom.boost_hypergeometric_0f1_batch_padded_prec),
    "boost_hypergeometric_1f1": (boost_hypgeom.boost_hypergeometric_1f1_batch_fixed_prec, boost_hypgeom.boost_hypergeometric_1f1_batch_padded_prec),
    "boost_hyp2f1_series": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_prec, boost_hypgeom.boost_hyp2f1_series_batch_padded_prec),
    "boost_hyp2f1_cf": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_prec, boost_hypgeom.boost_hyp2f1_series_batch_padded_prec),
    "boost_hyp2f1_pade": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_prec, boost_hypgeom.boost_hyp2f1_series_batch_padded_prec),
    "boost_hyp2f1_rational": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_prec, boost_hypgeom.boost_hyp2f1_series_batch_padded_prec),
    "boost_hypergeometric_pfq": (boost_hypgeom.boost_hypergeometric_pfq_batch_fixed_prec, boost_hypgeom.boost_hypergeometric_pfq_batch_padded_prec),
}

_DIRECT_INTERVAL_MODE_BATCH_FASTPATHS = {
    "hypgeom.arb_hypgeom_1f1": (hypgeom_wrappers.arb_hypgeom_1f1_batch_mode_fixed, hypgeom_wrappers.arb_hypgeom_1f1_batch_mode_padded),
    "hypgeom.arb_hypgeom_2f1": (hypgeom_wrappers.arb_hypgeom_2f1_batch_mode_fixed, hypgeom_wrappers.arb_hypgeom_2f1_batch_mode_padded),
    "hypgeom.arb_hypgeom_u": (hypgeom_wrappers.arb_hypgeom_u_batch_mode_fixed, hypgeom_wrappers.arb_hypgeom_u_batch_mode_padded),
    "hypgeom.arb_hypgeom_gamma_lower": (hypgeom_wrappers.arb_hypgeom_gamma_lower_batch_mode_fixed, hypgeom_wrappers.arb_hypgeom_gamma_lower_batch_mode_padded),
    "hypgeom.arb_hypgeom_gamma_upper": (hypgeom_wrappers.arb_hypgeom_gamma_upper_batch_mode_fixed, hypgeom_wrappers.arb_hypgeom_gamma_upper_batch_mode_padded),
    "hypgeom.acb_hypgeom_1f1": (hypgeom_wrappers.acb_hypgeom_1f1_batch_mode_fixed, hypgeom_wrappers.acb_hypgeom_1f1_batch_mode_padded),
    "hypgeom.acb_hypgeom_2f1": (hypgeom_wrappers.acb_hypgeom_2f1_batch_mode_fixed, hypgeom_wrappers.acb_hypgeom_2f1_batch_mode_padded),
    "hypgeom.acb_hypgeom_u": (hypgeom_wrappers.acb_hypgeom_u_batch_mode_fixed, hypgeom_wrappers.acb_hypgeom_u_batch_mode_padded),
    "hypgeom.acb_hypgeom_gamma_lower": (hypgeom_wrappers.acb_hypgeom_gamma_lower_batch_mode_fixed, hypgeom_wrappers.acb_hypgeom_gamma_lower_batch_mode_padded),
    "hypgeom.acb_hypgeom_gamma_upper": (hypgeom_wrappers.acb_hypgeom_gamma_upper_batch_mode_fixed, hypgeom_wrappers.acb_hypgeom_gamma_upper_batch_mode_padded),
    "boost_hypergeometric_0f1": (boost_hypgeom.boost_hypergeometric_0f1_batch_mode_fixed, boost_hypgeom.boost_hypergeometric_0f1_batch_mode_padded),
    "boost_hypergeometric_1f1": (boost_hypgeom.boost_hypergeometric_1f1_batch_mode_fixed, boost_hypgeom.boost_hypergeometric_1f1_batch_mode_padded),
    "boost_hyp2f1_series": (boost_hypgeom.boost_hyp2f1_series_batch_mode_fixed, boost_hypgeom.boost_hyp2f1_series_batch_mode_padded),
    "boost_hyp2f1_cf": (boost_hypgeom.boost_hyp2f1_series_batch_mode_fixed, boost_hypgeom.boost_hyp2f1_series_batch_mode_padded),
    "boost_hyp2f1_pade": (boost_hypgeom.boost_hyp2f1_series_batch_mode_fixed, boost_hypgeom.boost_hyp2f1_series_batch_mode_padded),
    "boost_hyp2f1_rational": (boost_hypgeom.boost_hyp2f1_series_batch_mode_fixed, boost_hypgeom.boost_hyp2f1_series_batch_mode_padded),
    "boost_hypergeometric_pfq": (boost_hypgeom.boost_hypergeometric_pfq_batch_mode_fixed, boost_hypgeom.boost_hypergeometric_pfq_batch_mode_padded),
}

def _mid_arb_batch(fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        return di.midpoint(fn(*args, **kwargs))
    return wrapped


def _mid_acb_batch(fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        return acb_core.acb_midpoint(fn(*args, **kwargs))
    return wrapped


_DIRECT_POINT_BATCH_FASTPATHS = {
    "hypgeom.arb_hypgeom_gamma": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_gamma_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_gamma_point)),
    "hypgeom.arb_hypgeom_erf": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_erf_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_erf_point)),
    "hypgeom.arb_hypgeom_erfc": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_erfc_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_erfc_point)),
    "hypgeom.arb_hypgeom_erfi": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_erfi_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_erfi_point)),
    "hypgeom.arb_hypgeom_erfinv": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_erfinv_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_erfinv_point)),
    "hypgeom.arb_hypgeom_erfcinv": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_erfcinv_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_erfcinv_point)),
    "hypgeom.arb_hypgeom_ei": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_ei_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_ei_point)),
    "hypgeom.arb_hypgeom_si": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_si_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_si_point)),
    "hypgeom.arb_hypgeom_ci": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_ci_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_ci_point)),
    "hypgeom.arb_hypgeom_shi": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_shi_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_shi_point)),
    "hypgeom.arb_hypgeom_chi": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_chi_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_chi_point)),
    "hypgeom.arb_hypgeom_li": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_li_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_li_point)),
    "hypgeom.arb_hypgeom_dilog": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_dilog_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_dilog_point)),
    "hypgeom.arb_hypgeom_fresnel": (partial(point_wrappers._fixed_unary_point, point_wrappers.arb_hypgeom_fresnel_point), partial(point_wrappers._padded_unary_point, point_wrappers.arb_hypgeom_fresnel_point)),
    "hypgeom.arb_hypgeom_0f1": (point_wrappers.arb_hypgeom_0f1_batch_fixed_point, point_wrappers.arb_hypgeom_0f1_batch_padded_point),
    "hypgeom.arb_hypgeom_1f1": (point_wrappers.arb_hypgeom_1f1_batch_fixed_point, point_wrappers.arb_hypgeom_1f1_batch_padded_point),
    "hypgeom.arb_hypgeom_m": (point_wrappers.arb_hypgeom_m_batch_fixed_point, point_wrappers.arb_hypgeom_m_batch_padded_point),
    "hypgeom.arb_hypgeom_2f1": (point_wrappers.arb_hypgeom_2f1_batch_fixed_point, point_wrappers.arb_hypgeom_2f1_batch_padded_point),
    "hypgeom.arb_hypgeom_u": (point_wrappers.arb_hypgeom_u_batch_fixed_point, point_wrappers.arb_hypgeom_u_batch_padded_point),
    "hypgeom.arb_hypgeom_gamma_lower": (point_wrappers.arb_hypgeom_gamma_lower_batch_fixed_point, point_wrappers.arb_hypgeom_gamma_lower_batch_padded_point),
    "hypgeom.arb_hypgeom_gamma_upper": (point_wrappers.arb_hypgeom_gamma_upper_batch_fixed_point, point_wrappers.arb_hypgeom_gamma_upper_batch_padded_point),
    "hypgeom.arb_hypgeom_chebyshev_t": point_wrappers.arb_hypgeom_chebyshev_t_point,
    "hypgeom.arb_hypgeom_chebyshev_u": point_wrappers.arb_hypgeom_chebyshev_u_point,
    "hypgeom.arb_hypgeom_laguerre_l": point_wrappers.arb_hypgeom_laguerre_l_point,
    "hypgeom.arb_hypgeom_hermite_h": point_wrappers.arb_hypgeom_hermite_h_point,
    "hypgeom.arb_hypgeom_legendre_p": point_wrappers.arb_hypgeom_legendre_p_point,
    "hypgeom.arb_hypgeom_legendre_q": point_wrappers.arb_hypgeom_legendre_q_point,
    "hypgeom.arb_hypgeom_jacobi_p": point_wrappers.arb_hypgeom_jacobi_p_point,
    "hypgeom.arb_hypgeom_gegenbauer_c": point_wrappers.arb_hypgeom_gegenbauer_c_point,
    "hypgeom.arb_hypgeom_pfq": (point_wrappers.arb_hypgeom_pfq_batch_fixed_point, point_wrappers.arb_hypgeom_pfq_batch_padded_point),
    "hypgeom.acb_hypgeom_0f1": (point_wrappers.acb_hypgeom_0f1_batch_fixed_point, point_wrappers.acb_hypgeom_0f1_batch_padded_point),
    "hypgeom.acb_hypgeom_1f1": (point_wrappers.acb_hypgeom_1f1_batch_fixed_point, point_wrappers.acb_hypgeom_1f1_batch_padded_point),
    "hypgeom.acb_hypgeom_m": (point_wrappers.acb_hypgeom_m_batch_fixed_point, point_wrappers.acb_hypgeom_m_batch_padded_point),
    "hypgeom.acb_hypgeom_2f1": (point_wrappers.acb_hypgeom_2f1_batch_fixed_point, point_wrappers.acb_hypgeom_2f1_batch_padded_point),
    "hypgeom.acb_hypgeom_u": (point_wrappers.acb_hypgeom_u_batch_fixed_point, point_wrappers.acb_hypgeom_u_batch_padded_point),
    "hypgeom.acb_hypgeom_gamma_lower": (point_wrappers.acb_hypgeom_gamma_lower_batch_fixed_point, point_wrappers.acb_hypgeom_gamma_lower_batch_padded_point),
    "hypgeom.acb_hypgeom_gamma_upper": (point_wrappers.acb_hypgeom_gamma_upper_batch_fixed_point, point_wrappers.acb_hypgeom_gamma_upper_batch_padded_point),
    "hypgeom.acb_hypgeom_gamma": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_gamma_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_gamma_point)),
    "hypgeom.acb_hypgeom_erf": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_erf_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_erf_point)),
    "hypgeom.acb_hypgeom_erfc": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_erfc_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_erfc_point)),
    "hypgeom.acb_hypgeom_erfi": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_erfi_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_erfi_point)),
    "hypgeom.acb_hypgeom_ei": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_ei_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_ei_point)),
    "hypgeom.acb_hypgeom_si": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_si_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_si_point)),
    "hypgeom.acb_hypgeom_ci": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_ci_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_ci_point)),
    "hypgeom.acb_hypgeom_shi": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_shi_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_shi_point)),
    "hypgeom.acb_hypgeom_chi": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_chi_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_chi_point)),
    "hypgeom.acb_hypgeom_li": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_li_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_li_point)),
    "hypgeom.acb_hypgeom_dilog": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_dilog_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_dilog_point)),
    "hypgeom.acb_hypgeom_fresnel": (partial(point_wrappers._fixed_unary_point, point_wrappers.acb_hypgeom_fresnel_point), partial(point_wrappers._padded_unary_point, point_wrappers.acb_hypgeom_fresnel_point)),
    "hypgeom.acb_hypgeom_chebyshev_t": point_wrappers.acb_hypgeom_chebyshev_t_point,
    "hypgeom.acb_hypgeom_chebyshev_u": point_wrappers.acb_hypgeom_chebyshev_u_point,
    "hypgeom.acb_hypgeom_laguerre_l": point_wrappers.acb_hypgeom_laguerre_l_point,
    "hypgeom.acb_hypgeom_hermite_h": point_wrappers.acb_hypgeom_hermite_h_point,
    "hypgeom.acb_hypgeom_pfq": (point_wrappers.acb_hypgeom_pfq_batch_fixed_point, point_wrappers.acb_hypgeom_pfq_batch_padded_point),
    "boost_hypergeometric_0f1": (boost_hypgeom.boost_hypergeometric_0f1_batch_fixed_point, boost_hypgeom.boost_hypergeometric_0f1_batch_padded_point),
    "boost_hypergeometric_1f1": (boost_hypgeom.boost_hypergeometric_1f1_batch_fixed_point, boost_hypgeom.boost_hypergeometric_1f1_batch_padded_point),
    "boost_hyp2f1_series": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_point, boost_hypgeom.boost_hyp2f1_series_batch_padded_point),
    "boost_hyp2f1_cf": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_point, boost_hypgeom.boost_hyp2f1_series_batch_padded_point),
    "boost_hyp2f1_pade": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_point, boost_hypgeom.boost_hyp2f1_series_batch_padded_point),
    "boost_hyp2f1_rational": (boost_hypgeom.boost_hyp2f1_series_batch_fixed_point, boost_hypgeom.boost_hyp2f1_series_batch_padded_point),
    "boost_hypergeometric_pfq": (boost_hypgeom.boost_hypergeometric_pfq_batch_fixed_point, boost_hypgeom.boost_hypergeometric_pfq_batch_padded_point),
    "acb_dirichlet_zeta": (point_wrappers.acb_dirichlet_zeta_batch_fixed_point, point_wrappers.acb_dirichlet_zeta_batch_padded_point),
    "acb_dirichlet_eta": (point_wrappers.acb_dirichlet_eta_batch_fixed_point, point_wrappers.acb_dirichlet_eta_batch_padded_point),
    "acb_modular_j": (point_wrappers.acb_modular_j_batch_fixed_point, point_wrappers.acb_modular_j_batch_padded_point),
    "acb_elliptic_k": (point_wrappers.acb_elliptic_k_batch_fixed_point, point_wrappers.acb_elliptic_k_batch_padded_point),
    "acb_elliptic_e": (point_wrappers.acb_elliptic_e_batch_fixed_point, point_wrappers.acb_elliptic_e_batch_padded_point),
    "bdg_log_barnesdoublegamma": (double_gamma.bdg_log_barnesdoublegamma_batch_fixed_point, double_gamma.bdg_log_barnesdoublegamma_batch_padded_point),
    "bdg_barnesdoublegamma": (double_gamma.bdg_barnesdoublegamma_batch_fixed_point, double_gamma.bdg_barnesdoublegamma_batch_padded_point),
    "bdg_log_barnesgamma2": (double_gamma.bdg_log_barnesgamma2_batch_fixed_point, double_gamma.bdg_log_barnesgamma2_batch_padded_point),
    "bdg_barnesgamma2": (double_gamma.bdg_barnesgamma2_batch_fixed_point, double_gamma.bdg_barnesgamma2_batch_padded_point),
    "bdg_log_normalizeddoublegamma": (double_gamma.bdg_log_normalizeddoublegamma_batch_fixed_point, double_gamma.bdg_log_normalizeddoublegamma_batch_padded_point),
    "bdg_normalizeddoublegamma": (double_gamma.bdg_normalizeddoublegamma_batch_fixed_point, double_gamma.bdg_normalizeddoublegamma_batch_padded_point),
    "bdg_double_sine": (double_gamma.bdg_double_sine_batch_fixed_point, double_gamma.bdg_double_sine_batch_padded_point),
    "shahen_log_barnesdoublegamma": (double_gamma.bdg_log_barnesdoublegamma_batch_fixed_point, double_gamma.bdg_log_barnesdoublegamma_batch_padded_point),
    "shahen_barnesdoublegamma": (double_gamma.bdg_barnesdoublegamma_batch_fixed_point, double_gamma.bdg_barnesdoublegamma_batch_padded_point),
    "shahen_log_barnesgamma2": (double_gamma.bdg_log_barnesgamma2_batch_fixed_point, double_gamma.bdg_log_barnesgamma2_batch_padded_point),
    "shahen_barnesgamma2": (double_gamma.bdg_barnesgamma2_batch_fixed_point, double_gamma.bdg_barnesgamma2_batch_padded_point),
    "shahen_log_normalizeddoublegamma": (double_gamma.bdg_log_normalizeddoublegamma_batch_fixed_point, double_gamma.bdg_log_normalizeddoublegamma_batch_padded_point),
    "shahen_normalizeddoublegamma": (double_gamma.bdg_normalizeddoublegamma_batch_fixed_point, double_gamma.bdg_normalizeddoublegamma_batch_padded_point),
    "shahen_double_sine": (double_gamma.bdg_double_sine_batch_fixed_point, double_gamma.bdg_double_sine_batch_padded_point),
}


def _normalize_dtype(dtype: str | jnp.dtype | None) -> jnp.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in ("float32", "f32", "fp32"):
            return jnp.dtype(jnp.float32)
        if key in ("float64", "f64", "fp64"):
            return jnp.dtype(jnp.float64)
        raise ValueError("dtype must be one of: float32, float64")
    norm = jnp.dtype(dtype)
    if norm == jnp.dtype(jnp.float32):
        return norm
    if norm == jnp.dtype(jnp.float64):
        return norm
    raise ValueError("dtype must be one of: float32, float64")


def _float_dtype_for_arg(arg: object) -> jnp.dtype | None:
    if hasattr(arg, "dtype"):
        dt = jnp.dtype(getattr(arg, "dtype"))
    elif isinstance(arg, float):
        dt = jnp.asarray(arg).dtype
    elif isinstance(arg, complex):
        dt = jnp.asarray(arg).dtype
    else:
        return None
    if jnp.issubdtype(dt, jnp.floating):
        return jnp.dtype(dt)
    if jnp.issubdtype(dt, jnp.complexfloating):
        if dt == jnp.dtype(jnp.complex64):
            return jnp.dtype(jnp.float32)
        if dt == jnp.dtype(jnp.complex128):
            return jnp.dtype(jnp.float64)
    return None


def _resolve_dtype_for_args(args: tuple[object, ...], dtype: str | jnp.dtype | None) -> jnp.dtype:
    target = _normalize_dtype(dtype)
    seen: set[jnp.dtype] = set()
    for arg in args:
        dt = _float_dtype_for_arg(arg)
        if dt is not None:
            seen.add(dt)
    if target is not None:
        return target
    if len(seen) <= 1:
        return next(iter(seen)) if seen else jnp.dtype(jnp.float64)
    seen_str = ", ".join(sorted(str(x) for x in seen))
    raise ValueError(f"Mixed floating dtypes in one call are not supported: {seen_str}. Pass dtype='float32' or dtype='float64'.")


def _cast_arg_to_dtype(arg: object, float_dtype: jnp.dtype) -> object:
    if hasattr(arg, "dtype"):
        dt = jnp.dtype(getattr(arg, "dtype"))
        if jnp.issubdtype(dt, jnp.floating):
            if dt == float_dtype:
                return arg
            return lax.convert_element_type(jnp.asarray(arg), float_dtype)
        if jnp.issubdtype(dt, jnp.complexfloating):
            target = _COMPLEX_BY_FLOAT_DTYPE[float_dtype]
            if dt == target:
                return arg
            return lax.convert_element_type(jnp.asarray(arg), target)
        return arg
    if isinstance(arg, float):
        return jnp.asarray(arg, dtype=float_dtype)
    if isinstance(arg, complex):
        return jnp.asarray(arg, dtype=_COMPLEX_BY_FLOAT_DTYPE[float_dtype])
    return arg


def _cast_out_to_dtype(out: object, float_dtype: jnp.dtype):
    if isinstance(out, tuple):
        return tuple(_cast_out_to_dtype(item, float_dtype) for item in out)
    if hasattr(out, "dtype"):
        dt = jnp.dtype(getattr(out, "dtype"))
        if jnp.issubdtype(dt, jnp.floating):
            if dt == float_dtype:
                return out
            return lax.convert_element_type(jnp.asarray(out), float_dtype)
        if jnp.issubdtype(dt, jnp.complexfloating):
            target = _COMPLEX_BY_FLOAT_DTYPE[float_dtype]
            if dt == target:
                return out
            return lax.convert_element_type(jnp.asarray(out), target)
    return out


def _midpoint_from_interval_like(out: object):
    if isinstance(out, tuple):
        return tuple(_midpoint_from_interval_like(item) for item in out)
    arr = jnp.asarray(out)
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return di.midpoint(arr)
    if arr.ndim >= 1 and arr.shape[-1] == 4:
        return acb_core.acb_midpoint(arr)
    return out
def _normalize_hypgeom_name(name: str) -> str:
    if name.startswith("hypgeom."):
        return name.split(".", 1)[1]
    return name


def _maybe_hypgeom_basic_batch_fastpath(
    name: str,
    args: tuple[object, ...],
    *,
    mode: str,
    prec_bits: int | None,
    dps: int | None,
    pad_to: int | None,
    extra_kwargs: dict | None = None,
):
    if mode != "basic" or dps is not None:
        return None
    short = _normalize_hypgeom_name(name)
    if short in _HYPGEOM_BASIC_BATCH_FASTPATHS:
        fixed_name, padded_name = _HYPGEOM_BASIC_BATCH_FASTPATHS[short]
        mod = importlib.import_module(".hypgeom", package=__package__)
    elif short in _BOOST_HYPGEOM_BASIC_BATCH_FASTPATHS:
        fixed_name, padded_name = _BOOST_HYPGEOM_BASIC_BATCH_FASTPATHS[short]
        mod = importlib.import_module(".boost_hypgeom", package=__package__)
    else:
        return None
    if pad_to is not None and short in _HYPGEOM_PREFER_PADDED_FASTPATHS:
        batch_n = mixed_batch_size_or_none(args)
        if batch_n is not None and int(pad_to) == batch_n:
            fn = getattr(mod, fixed_name, None)
            if fn is None:
                return None
            kwargs = {"prec_bits": prec_bits if prec_bits is not None else None}
            if kwargs["prec_bits"] is None:
                del kwargs["prec_bits"]
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            return fn(*args, **kwargs)
        fn = getattr(mod, padded_name, None)
        if fn is None:
            return None
        kwargs = {"pad_to": pad_to, "prec_bits": prec_bits if prec_bits is not None else None}
        if kwargs["prec_bits"] is None:
            del kwargs["prec_bits"]
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return fn(*args, **kwargs)
    fn = getattr(mod, fixed_name, None)
    if fn is None:
        return None
    call_args, trim_n = pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)
    kwargs = {"prec_bits": prec_bits if prec_bits is not None else None}
    if kwargs["prec_bits"] is None:
        del kwargs["prec_bits"]
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    out = fn(*call_args, **kwargs)
    return trim_batch_out(out, trim_n)


def _maybe_direct_interval_batch_fastpath(
    name: str,
    args: tuple[object, ...],
    *,
    mode: str,
    prec_bits: int | None,
    dps: int | None,
    pad_to: int | None,
    extra_kwargs: dict | None = None,
):
    if dps is not None:
        return None
    if mode == "basic":
        mapping = _DIRECT_INTERVAL_BASIC_BATCH_FASTPATHS
        kwargs = {"prec_bits": prec_bits} if prec_bits is not None else {}
    elif mode in ("adaptive", "rigorous"):
        mapping = _DIRECT_INTERVAL_MODE_BATCH_FASTPATHS
        kwargs = {"impl": mode}
        if prec_bits is not None:
            kwargs["prec_bits"] = prec_bits
    else:
        return None
    pair = mapping.get(name)
    if pair is None:
        return None
    fixed_fn, padded_fn = pair
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    batch_n = mixed_batch_size_or_none(args)
    if pad_to is not None and name in _DIRECT_INTERVAL_BASIC_BATCH_FASTPATHS | _DIRECT_INTERVAL_MODE_BATCH_FASTPATHS:
        if batch_n is not None and int(pad_to) == batch_n:
            return fixed_fn(*args, **kwargs)
        return padded_fn(*args, pad_to=pad_to, **kwargs)
    call_args, trim_n = pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)
    out = fixed_fn(*call_args, **kwargs)
    return trim_batch_out(out, trim_n)


def _maybe_direct_point_batch_fastpath(
    name: str,
    args: tuple[object, ...],
    *,
    pad_to: int | None,
    extra_kwargs: dict | None = None,
):
    entry = _DIRECT_POINT_BATCH_FASTPATHS.get(name)
    if entry is None:
        return None
    kwargs = dict(extra_kwargs or {})
    if isinstance(entry, tuple):
        fixed_fn, padded_fn = entry
        batch_n = mixed_batch_size_or_none(args)
        if pad_to is not None:
            if batch_n is not None and int(pad_to) == batch_n:
                return fixed_fn(*args, **kwargs)
            out = padded_fn(*args, pad_to=pad_to, **kwargs)
            return trim_batch_out(out, batch_n if batch_n is not None else 0)
        call_args, trim_n = pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)
        out = fixed_fn(*call_args, **kwargs)
        return trim_batch_out(out, trim_n)
    fn = entry
    call_args, trim_n = pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)
    out = fn(*call_args, **kwargs)
    return trim_batch_out(out, trim_n)


def _maybe_hypgeom_mode_batch_fastpath(
    name: str,
    args: tuple[object, ...],
    *,
    mode: str,
    prec_bits: int | None,
    dps: int | None,
    pad_to: int | None,
    extra_kwargs: dict | None = None,
):
    if mode not in ("adaptive", "rigorous") or dps is not None:
        return None
    short = _normalize_hypgeom_name(name)
    if short in _HYPGEOM_MODE_BATCH_FASTPATHS:
        fixed_name, padded_name = _HYPGEOM_MODE_BATCH_FASTPATHS[short]
        mod = hypgeom_wrappers
    elif short in _BOOST_HYPGEOM_MODE_BATCH_FASTPATHS:
        fixed_name, padded_name = _BOOST_HYPGEOM_MODE_BATCH_FASTPATHS[short]
        mod = importlib.import_module(".boost_hypgeom", package=__package__)
    else:
        return None
    if pad_to is not None and short in _HYPGEOM_PREFER_PADDED_FASTPATHS:
        batch_n = mixed_batch_size_or_none(args)
        if batch_n is not None and int(pad_to) == batch_n:
            fn = getattr(mod, fixed_name, None)
            if fn is None:
                return None
            kwargs = {"impl": mode}
            if prec_bits is not None:
                kwargs["prec_bits"] = prec_bits
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            return fn(*args, **kwargs)
        fn = getattr(mod, padded_name, None)
        if fn is None:
            return None
        kwargs = {"pad_to": pad_to, "impl": mode}
        if prec_bits is not None:
            kwargs["prec_bits"] = prec_bits
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return fn(*args, **kwargs)
    fn = getattr(mod, fixed_name, None)
    if fn is None:
        return None
    call_args, trim_n = pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)
    kwargs = {"impl": mode}
    if prec_bits is not None:
        kwargs["prec_bits"] = prec_bits
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    out = fn(*call_args, **kwargs)
    return trim_batch_out(out, trim_n)


_POINT_FUNCS = {
    "exp": point_wrappers.arb_exp_point,
    "log": point_wrappers.arb_log_point,
    "sqrt": point_wrappers.arb_sqrt_point,
    "sin": point_wrappers.arb_sin_point,
    "cos": point_wrappers.arb_cos_point,
    "tan": point_wrappers.arb_tan_point,
    "sinh": point_wrappers.arb_sinh_point,
    "cosh": point_wrappers.arb_cosh_point,
    "tanh": point_wrappers.arb_tanh_point,
    "abs": point_wrappers.arb_abs_point,
    "add": point_wrappers.arb_add_point,
    "sub": point_wrappers.arb_sub_point,
    "mul": point_wrappers.arb_mul_point,
    "div": point_wrappers.arb_div_point,
    "inv": point_wrappers.arb_inv_point,
    "fma": point_wrappers.arb_fma_point,
    "log1p": point_wrappers.arb_log1p_point,
    "expm1": point_wrappers.arb_expm1_point,
    "sin_cos": point_wrappers.arb_sin_cos_point,
    "sinh_cosh": point_wrappers.arb_sinh_cosh_point,
    "sin_pi": point_wrappers.arb_sin_pi_point,
    "cos_pi": point_wrappers.arb_cos_pi_point,
    "tan_pi": point_wrappers.arb_tan_pi_point,
    "sinc": point_wrappers.arb_sinc_point,
    "sinc_pi": point_wrappers.arb_sinc_pi_point,
    "asin": point_wrappers.arb_asin_point,
    "acos": point_wrappers.arb_acos_point,
    "atan": point_wrappers.arb_atan_point,
    "asinh": point_wrappers.arb_asinh_point,
    "acosh": point_wrappers.arb_acosh_point,
    "atanh": point_wrappers.arb_atanh_point,
    "sign": point_wrappers.arb_sign_point,
    "pow": point_wrappers.arb_pow_point,
    "pow_ui": point_wrappers.arb_pow_ui_point,
    "root_ui": point_wrappers.arb_root_ui_point,
    "cbrt": point_wrappers.arb_cbrt_point,
    "gamma": point_wrappers.arb_gamma_point,
    "erf": point_wrappers.arb_erf_point,
    "erfc": point_wrappers.arb_erfc_point,
    "besselj": point_wrappers.arb_bessel_j_point,
    "bessely": point_wrappers.arb_bessel_y_point,
    "besseli": point_wrappers.arb_bessel_i_point,
    "besselk": point_wrappers.arb_bessel_k_point,
    "cuda_besselk": cubesselk.cuda_besselk_point,
}

for _name, _fn in _POINT_FUNCS.items():
    _DIRECT_POINT_BATCH_FASTPATHS.setdefault(_name, _fn)
    if _name.startswith("arb_"):
        _DIRECT_POINT_BATCH_FASTPATHS.setdefault(f"arb_core.{_name}", _fn)
    if _name.startswith("acb_"):
        _DIRECT_POINT_BATCH_FASTPATHS.setdefault(f"acb_core.{_name}", _fn)

for _name in dir(point_wrappers):
    if _name.startswith("_") or not _name.endswith("_point"):
        continue
    _fn = getattr(point_wrappers, _name)
    if not callable(_fn):
        continue
    _public = _name[:-6]
    _POINT_FUNCS.setdefault(_public, _fn)
    _DIRECT_POINT_BATCH_FASTPATHS.setdefault(_public, _fn)
    if _public.startswith("arb_"):
        _DIRECT_POINT_BATCH_FASTPATHS.setdefault(f"arb_core.{_public}", _fn)
    if _public.startswith("acb_"):
        _DIRECT_POINT_BATCH_FASTPATHS.setdefault(f"acb_core.{_public}", _fn)
    if _public.startswith(("arb_hypgeom_", "acb_hypgeom_")):
        _DIRECT_POINT_BATCH_FASTPATHS.setdefault(f"hypgeom.{_public}", _fn)
        _POINT_FUNCS.setdefault(f"hypgeom.{_public}", _fn)


_HYPGEOM_POINT_BATCH_ALIASES = {
    "arb_hypgeom_gamma": (point_wrappers.arb_hypgeom_gamma_batch_fixed_point, point_wrappers.arb_hypgeom_gamma_batch_padded_point),
    "arb_hypgeom_erf": (point_wrappers.arb_hypgeom_erf_batch_fixed_point, point_wrappers.arb_hypgeom_erf_batch_padded_point),
    "arb_hypgeom_erfc": (point_wrappers.arb_hypgeom_erfc_batch_fixed_point, point_wrappers.arb_hypgeom_erfc_batch_padded_point),
    "arb_hypgeom_erfi": (point_wrappers.arb_hypgeom_erfi_batch_fixed_point, point_wrappers.arb_hypgeom_erfi_batch_padded_point),
    "arb_hypgeom_erfinv": (point_wrappers.arb_hypgeom_erfinv_batch_fixed_point, point_wrappers.arb_hypgeom_erfinv_batch_padded_point),
    "arb_hypgeom_erfcinv": (point_wrappers.arb_hypgeom_erfcinv_batch_fixed_point, point_wrappers.arb_hypgeom_erfcinv_batch_padded_point),
    "arb_hypgeom_ei": (point_wrappers.arb_hypgeom_ei_batch_fixed_point, point_wrappers.arb_hypgeom_ei_batch_padded_point),
    "arb_hypgeom_si": (point_wrappers.arb_hypgeom_si_batch_fixed_point, point_wrappers.arb_hypgeom_si_batch_padded_point),
    "arb_hypgeom_ci": (point_wrappers.arb_hypgeom_ci_batch_fixed_point, point_wrappers.arb_hypgeom_ci_batch_padded_point),
    "arb_hypgeom_shi": (point_wrappers.arb_hypgeom_shi_batch_fixed_point, point_wrappers.arb_hypgeom_shi_batch_padded_point),
    "arb_hypgeom_chi": (point_wrappers.arb_hypgeom_chi_batch_fixed_point, point_wrappers.arb_hypgeom_chi_batch_padded_point),
    "arb_hypgeom_li": (point_wrappers.arb_hypgeom_li_batch_fixed_point, point_wrappers.arb_hypgeom_li_batch_padded_point),
    "arb_hypgeom_dilog": (point_wrappers.arb_hypgeom_dilog_batch_fixed_point, point_wrappers.arb_hypgeom_dilog_batch_padded_point),
    "arb_hypgeom_fresnel": (point_wrappers.arb_hypgeom_fresnel_batch_fixed_point, point_wrappers.arb_hypgeom_fresnel_batch_padded_point),
    "arb_hypgeom_legendre_p": (point_wrappers.arb_hypgeom_legendre_p_batch_fixed_point, point_wrappers.arb_hypgeom_legendre_p_batch_padded_point),
    "arb_hypgeom_legendre_q": (point_wrappers.arb_hypgeom_legendre_q_batch_fixed_point, point_wrappers.arb_hypgeom_legendre_q_batch_padded_point),
    "arb_hypgeom_jacobi_p": (point_wrappers.arb_hypgeom_jacobi_p_batch_fixed_point, point_wrappers.arb_hypgeom_jacobi_p_batch_padded_point),
    "arb_hypgeom_gegenbauer_c": (point_wrappers.arb_hypgeom_gegenbauer_c_batch_fixed_point, point_wrappers.arb_hypgeom_gegenbauer_c_batch_padded_point),
    "arb_hypgeom_chebyshev_t": (point_wrappers.arb_hypgeom_chebyshev_t_batch_fixed_point, point_wrappers.arb_hypgeom_chebyshev_t_batch_padded_point),
    "arb_hypgeom_chebyshev_u": (point_wrappers.arb_hypgeom_chebyshev_u_batch_fixed_point, point_wrappers.arb_hypgeom_chebyshev_u_batch_padded_point),
    "arb_hypgeom_laguerre_l": (point_wrappers.arb_hypgeom_laguerre_l_batch_fixed_point, point_wrappers.arb_hypgeom_laguerre_l_batch_padded_point),
    "arb_hypgeom_hermite_h": (point_wrappers.arb_hypgeom_hermite_h_batch_fixed_point, point_wrappers.arb_hypgeom_hermite_h_batch_padded_point),
    "acb_hypgeom_gamma": (point_wrappers.acb_hypgeom_gamma_batch_fixed_point, point_wrappers.acb_hypgeom_gamma_batch_padded_point),
    "acb_hypgeom_erf": (point_wrappers.acb_hypgeom_erf_batch_fixed_point, point_wrappers.acb_hypgeom_erf_batch_padded_point),
    "acb_hypgeom_erfc": (point_wrappers.acb_hypgeom_erfc_batch_fixed_point, point_wrappers.acb_hypgeom_erfc_batch_padded_point),
    "acb_hypgeom_erfi": (point_wrappers.acb_hypgeom_erfi_batch_fixed_point, point_wrappers.acb_hypgeom_erfi_batch_padded_point),
    "acb_hypgeom_ei": (point_wrappers.acb_hypgeom_ei_batch_fixed_point, point_wrappers.acb_hypgeom_ei_batch_padded_point),
    "acb_hypgeom_si": (point_wrappers.acb_hypgeom_si_batch_fixed_point, point_wrappers.acb_hypgeom_si_batch_padded_point),
    "acb_hypgeom_ci": (point_wrappers.acb_hypgeom_ci_batch_fixed_point, point_wrappers.acb_hypgeom_ci_batch_padded_point),
    "acb_hypgeom_shi": (point_wrappers.acb_hypgeom_shi_batch_fixed_point, point_wrappers.acb_hypgeom_shi_batch_padded_point),
    "acb_hypgeom_chi": (point_wrappers.acb_hypgeom_chi_batch_fixed_point, point_wrappers.acb_hypgeom_chi_batch_padded_point),
    "acb_hypgeom_li": (point_wrappers.acb_hypgeom_li_batch_fixed_point, point_wrappers.acb_hypgeom_li_batch_padded_point),
    "acb_hypgeom_dilog": (point_wrappers.acb_hypgeom_dilog_batch_fixed_point, point_wrappers.acb_hypgeom_dilog_batch_padded_point),
    "acb_hypgeom_fresnel": (point_wrappers.acb_hypgeom_fresnel_batch_fixed_point, point_wrappers.acb_hypgeom_fresnel_batch_padded_point),
    "acb_hypgeom_legendre_p": (point_wrappers.acb_hypgeom_legendre_p_batch_fixed_point, point_wrappers.acb_hypgeom_legendre_p_batch_padded_point),
    "acb_hypgeom_legendre_q": (point_wrappers.acb_hypgeom_legendre_q_batch_fixed_point, point_wrappers.acb_hypgeom_legendre_q_batch_padded_point),
    "acb_hypgeom_jacobi_p": (point_wrappers.acb_hypgeom_jacobi_p_batch_fixed_point, point_wrappers.acb_hypgeom_jacobi_p_batch_padded_point),
    "acb_hypgeom_gegenbauer_c": (point_wrappers.acb_hypgeom_gegenbauer_c_batch_fixed_point, point_wrappers.acb_hypgeom_gegenbauer_c_batch_padded_point),
    "acb_hypgeom_chebyshev_t": (point_wrappers.acb_hypgeom_chebyshev_t_batch_fixed_point, point_wrappers.acb_hypgeom_chebyshev_t_batch_padded_point),
    "acb_hypgeom_chebyshev_u": (point_wrappers.acb_hypgeom_chebyshev_u_batch_fixed_point, point_wrappers.acb_hypgeom_chebyshev_u_batch_padded_point),
    "acb_hypgeom_laguerre_l": (point_wrappers.acb_hypgeom_laguerre_l_batch_fixed_point, point_wrappers.acb_hypgeom_laguerre_l_batch_padded_point),
    "acb_hypgeom_hermite_h": (point_wrappers.acb_hypgeom_hermite_h_batch_fixed_point, point_wrappers.acb_hypgeom_hermite_h_batch_padded_point),
}

for _name, _entry in _HYPGEOM_POINT_BATCH_ALIASES.items():
    _DIRECT_POINT_BATCH_FASTPATHS[_name] = _entry
    _DIRECT_POINT_BATCH_FASTPATHS[f"hypgeom.{_name}"] = _entry


_INTERVAL_FUNCS = {
    "exp": baseline_wrappers.arb_exp_mp,
    "log": baseline_wrappers.arb_log_mp,
    "sqrt": baseline_wrappers.arb_sqrt_mp,
    "sin": baseline_wrappers.arb_sin_mp,
    "cos": baseline_wrappers.arb_cos_mp,
    "tan": baseline_wrappers.arb_tan_mp,
    "sinh": baseline_wrappers.arb_sinh_mp,
    "cosh": baseline_wrappers.arb_cosh_mp,
    "tanh": baseline_wrappers.arb_tanh_mp,
    "abs": baseline_wrappers.arb_abs_mp,
    "add": baseline_wrappers.arb_add_mp,
    "sub": baseline_wrappers.arb_sub_mp,
    "mul": baseline_wrappers.arb_mul_mp,
    "div": baseline_wrappers.arb_div_mp,
    "inv": baseline_wrappers.arb_inv_mp,
    "fma": baseline_wrappers.arb_fma_mp,
    "log1p": baseline_wrappers.arb_log1p_mp,
    "expm1": baseline_wrappers.arb_expm1_mp,
    "sin_cos": baseline_wrappers.arb_sin_cos_mp,
    "gamma": baseline_wrappers.arb_gamma_mp,
    "erf": baseline_wrappers.arb_erf_mp,
    "erfc": baseline_wrappers.arb_erfc_mp,
    "erfi": baseline_wrappers.arb_erfi_mp,
    "barnesg": baseline_wrappers.arb_barnesg_mp,
    "besselj": baseline_wrappers.arb_bessel_j_mp,
    "bessely": baseline_wrappers.arb_bessel_y_mp,
    "besseli": baseline_wrappers.arb_bessel_i_mp,
    "besselk": baseline_wrappers.arb_bessel_k_mp,
    "cuda_besselk": cubesselk.cuda_besselk,
}


_MODULE_NAMES = (
    "acb_calc",
    "acb_core",
    "acb_dirichlet",
    "acb_elliptic",
    "acb_mat",
    "acb_modular",
    "acb_poly",
    "acf",
    "arb_calc",
    "arb_core",
    "arb_fmpz_poly",
    "arb_fpwrap",
    "arb_mat",
    "arb_poly",
    "arf",
    "bernoulli",
    "bool_mat",
    "dft",
    "dlog",
    "boost_hypgeom",
    "cusf_compat",
    "double_interval",
    "double_gamma",
    "fmpz_extras",
    "fmpzi",
    "fmpr",
    "hypgeom",
    "mag",
    "partitions",
    "shahen_double_gamma",
)


@lru_cache(maxsize=1)
def _public_registry() -> dict[str, Callable]:
    mapping: dict[str, Callable] = {}
    seen: dict[str, str] = {}
    for mod_name in _MODULE_NAMES:
        mod = importlib.import_module(f".{mod_name}", package=__package__)
        mod_name = mod.__name__.rsplit(".", 1)[-1]
        for name, value in vars(mod).items():
            if name.startswith("_"):
                continue
            if not callable(value):
                continue
            full = f"{mod_name}.{name}"
            mapping[full] = value
            if name not in seen:
                mapping[name] = value
                seen[name] = full
            else:
                # disambiguate by requiring module prefix
                if name in mapping:
                    del mapping[name]
    return mapping


def _require(mapping: dict[str, Callable], name: str) -> Callable:
    if name not in mapping:
        raise KeyError(f"Unknown function '{name}'.")
    return mapping[name]


@lru_cache(maxsize=None)
def _resolve_point_fn(name: str) -> Callable:
    if name in _POINT_FUNCS:
        return _POINT_FUNCS[name]
    return _require(_public_registry(), name)


@lru_cache(maxsize=None)
def _resolve_interval_fn(name: str) -> Callable:
    if name in _INTERVAL_FUNCS:
        return _INTERVAL_FUNCS[name]
    return _require(_public_registry(), name)


@lru_cache(maxsize=None)
def _optional_kwarg_support(fn: Callable) -> tuple[bool, bool, bool]:
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return False, False, False
    return "mode" in params, "prec_bits" in params, "dps" in params


def _call_with_optional_args(fn: Callable, args: tuple, mode: str, prec_bits: int | None, dps: int | None):
    has_mode, has_prec_bits, has_dps = _optional_kwarg_support(fn)
    if not (has_mode or has_prec_bits or has_dps):
        return fn(*args)
    kwargs = {}
    if has_mode:
        kwargs["mode"] = mode
    if has_prec_bits:
        kwargs["prec_bits"] = prec_bits
    if has_dps:
        kwargs["dps"] = dps
    return fn(*args, **kwargs)


def _is_host_point_backend(name: str) -> bool:
    return False


@lru_cache(maxsize=None)
def _resolve_hypgeom_mode_fn(name: str) -> Callable | None:
    fn_name = name
    if "." in name:
        mod, short = name.split(".", 1)
        if mod != "hypgeom":
            return None
        fn_name = short
    if not (fn_name.startswith("arb_hypgeom_") or fn_name.startswith("acb_hypgeom_")):
        return None
    mode_name = f"{fn_name}_mode"
    fn = getattr(hypgeom_wrappers, mode_name, None)
    return fn if callable(fn) else None


@lru_cache(maxsize=None)
def _bound_interval_fn(name: str, mode: str, prec_bits: int | None, dps: int | None) -> Callable:
    mode_fn = _resolve_hypgeom_mode_fn(name)
    if mode_fn is not None:
        if prec_bits is None and dps is None:
            return lambda *args: mode_fn(*args, impl=mode)
        if prec_bits is None:
            return lambda *args: mode_fn(*args, impl=mode, dps=dps)
        if dps is None:
            return lambda *args: mode_fn(*args, impl=mode, prec_bits=prec_bits)
        return lambda *args: mode_fn(*args, impl=mode, prec_bits=prec_bits, dps=dps)

    fn = _resolve_interval_fn(name)
    has_mode, has_prec_bits, has_dps = _optional_kwarg_support(fn)
    if not (has_mode or has_prec_bits or has_dps):
        return fn
    if has_mode and has_prec_bits and has_dps:
        return lambda *args: fn(*args, mode=mode, prec_bits=prec_bits, dps=dps)
    if has_mode and has_prec_bits:
        return lambda *args: fn(*args, mode=mode, prec_bits=prec_bits)
    if has_mode and has_dps:
        return lambda *args: fn(*args, mode=mode, dps=dps)
    if has_prec_bits and has_dps:
        return lambda *args: fn(*args, prec_bits=prec_bits, dps=dps)
    if has_mode:
        return lambda *args: fn(*args, mode=mode)
    if has_prec_bits:
        return lambda *args: fn(*args, prec_bits=prec_bits)
    if has_dps:
        return lambda *args: fn(*args, dps=dps)
    return fn


def eval_point(
    name: str,
    *args: jax.Array,
    jit: bool = False,
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    fn = _point_jit_fn(name) if jit else _resolve_point_fn(name)
    target = _resolve_dtype_for_args(args, dtype)
    out = fn(*tuple(_cast_arg_to_dtype(arg, target) for arg in args), **kwargs)
    return _cast_out_to_dtype(out, target)


def eval_point_batch(
    name: str,
    *args: jax.Array,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    pad_value: float | complex = 0.0,
    **kwargs,
) -> jax.Array:
    target = _resolve_dtype_for_args(args, dtype)
    cast_args = tuple(_cast_arg_to_dtype(arg, target) for arg in args)
    direct_fast = _maybe_direct_point_batch_fastpath(name, cast_args, pad_to=pad_to, extra_kwargs=kwargs if kwargs else None)
    if direct_fast is not None:
        return _cast_out_to_dtype(direct_fast, target)
    batch_args, n = pad_batch_args(cast_args, pad_to=pad_to, pad_value=pad_value)
    batched = _point_batch_fn(name)
    out = batched(*batch_args, **kwargs)
    return trim_batch_out(_cast_out_to_dtype(out, target), n)


def eval_interval(
    name: str,
    *args: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    jit: bool = False,
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    target = _resolve_dtype_for_args(args, dtype)
    cast_args = tuple(_cast_arg_to_dtype(arg, target) for arg in args)
    if kwargs:
        mode_fn = _resolve_hypgeom_mode_fn(name)
        if mode_fn is not None:
            call_kwargs = {"impl": mode, **kwargs}
            if prec_bits is not None:
                call_kwargs["prec_bits"] = prec_bits
            if dps is not None:
                call_kwargs["dps"] = dps
            out = mode_fn(*cast_args, **call_kwargs)
            return _cast_out_to_dtype(out, target)
    fn = _interval_jit_fn(name, mode, prec_bits, dps) if jit else _bound_interval_fn(name, mode, prec_bits, dps)
    out = fn(*cast_args)
    return _cast_out_to_dtype(out, target)


def eval_interval_batch(
    name: str,
    *args: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    pad_value: float | complex = 0.0,
    **kwargs,
) -> jax.Array:
    target = _resolve_dtype_for_args(args, dtype)
    cast_args = tuple(_cast_arg_to_dtype(arg, target) for arg in args)
    direct_fast = _maybe_direct_interval_batch_fastpath(
        name,
        cast_args,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        pad_to=pad_to,
        extra_kwargs=kwargs if kwargs else None,
    )
    if direct_fast is not None:
        return _cast_out_to_dtype(direct_fast, target)
    fast = _maybe_hypgeom_basic_batch_fastpath(
        name,
        cast_args,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        pad_to=pad_to,
        extra_kwargs=kwargs if kwargs else None,
    )
    if fast is not None:
        return _cast_out_to_dtype(fast, target)
    mode_fast = _maybe_hypgeom_mode_batch_fastpath(
        name,
        cast_args,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        pad_to=pad_to,
        extra_kwargs=kwargs if kwargs else None,
    )
    if mode_fast is not None:
        return _cast_out_to_dtype(mode_fast, target)
    if kwargs:
        batch_args, n = pad_batch_args(cast_args, pad_to=pad_to, pad_value=pad_value)
        out = jax.vmap(
            lambda *aa: eval_interval(name, *aa, mode=mode, prec_bits=prec_bits, dps=dps, dtype=target, **kwargs)
        )(*batch_args)
        return trim_batch_out(_cast_out_to_dtype(out, target), n)
    batched = _interval_batch_fn(name, mode, prec_bits, dps)
    batch_args, n = pad_batch_args(cast_args, pad_to=pad_to, pad_value=pad_value)
    out = batched(*batch_args)
    return trim_batch_out(_cast_out_to_dtype(out, target), n)


def _chunked_apply(fn: Callable, args: tuple[object, ...], chunk_size: int) -> jax.Array:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    n = _batch_size(args)
    if n <= chunk_size:
        return fn(*args)

    outs = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk_args = tuple(arr[start:stop] for arr in args)
        outs.append(fn(*chunk_args))
    return jnp.concatenate(outs, axis=0)


def eval_point_batch_chunked(
    name: str,
    *args: jax.Array,
    chunk_size: int = 1024,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    pad_value: float | complex = 0.0,
) -> jax.Array:
    batched = _point_batch_fn(name)
    target = _resolve_dtype_for_args(args, dtype)
    batch_args, n = pad_batch_args(
        tuple(_cast_arg_to_dtype(arg, target) for arg in args),
        pad_to=pad_to,
        pad_value=pad_value,
    )
    out = _chunked_apply(batched, batch_args, chunk_size)
    return trim_batch_out(_cast_out_to_dtype(out, target), n)


def eval_interval_batch_chunked(
    name: str,
    *args: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    chunk_size: int = 1024,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    pad_value: float | complex = 0.0,
) -> jax.Array:
    batched = _interval_batch_fn(name, mode, prec_bits, dps)
    target = _resolve_dtype_for_args(args, dtype)
    batch_args, n = pad_batch_args(
        tuple(_cast_arg_to_dtype(arg, target) for arg in args),
        pad_to=pad_to,
        pad_value=pad_value,
    )
    out = _chunked_apply(batched, batch_args, chunk_size)
    return trim_batch_out(_cast_out_to_dtype(out, target), n)


def bind_point(name: str, dtype: str | jnp.dtype | None = None) -> Callable:
    fn = _resolve_point_fn(name)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        out = fn(*tuple(_cast_arg_to_dtype(arg, target) for arg in args))
        return _cast_out_to_dtype(out, target)

    return wrapped


def bind_interval(
    name: str,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    dtype: str | jnp.dtype | None = None,
) -> Callable:
    fn = _bound_interval_fn(name, mode, prec_bits, dps)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        out = fn(*tuple(_cast_arg_to_dtype(arg, target) for arg in args))
        return _cast_out_to_dtype(out, target)

    return wrapped


def bind_point_jit(name: str, dtype: str | jnp.dtype | None = None) -> Callable:
    fn = _point_jit_fn(name)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        out = fn(*tuple(_cast_arg_to_dtype(arg, target) for arg in args))
        return _cast_out_to_dtype(out, target)

    return wrapped


def bind_interval_jit(
    name: str,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    dtype: str | jnp.dtype | None = None,
) -> Callable:
    fn = _interval_jit_fn(name, mode, prec_bits, dps)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        out = fn(*tuple(_cast_arg_to_dtype(arg, target) for arg in args))
        return _cast_out_to_dtype(out, target)

    return wrapped


def bind_point_ad(
    name: str,
    kind: str = "grad",
    argnums: int | tuple[int, ...] = 0,
    dtype: str | jnp.dtype | None = None,
) -> Callable:
    fn = _point_ad_fn(name, kind, argnums)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        out = fn(*tuple(_cast_arg_to_dtype(arg, target) for arg in args))
        return _cast_out_to_dtype(out, target)

    return wrapped


def bind_point_batch(
    name: str,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    pad_value: float | complex = 0.0,
) -> Callable:
    fn = _point_batch_fn(name)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        batch_args, n = pad_batch_args(
            tuple(_cast_arg_to_dtype(arg, target) for arg in args),
            pad_to=pad_to,
            pad_value=pad_value,
        )
        out = fn(*batch_args)
        return trim_batch_out(_cast_out_to_dtype(out, target), n)

    return wrapped


def bind_interval_batch(
    name: str,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    pad_value: float | complex = 0.0,
) -> Callable:
    fn = _interval_batch_fn(name, mode, prec_bits, dps)

    def wrapped(*args):
        target = _resolve_dtype_for_args(args, dtype)
        batch_args, n = pad_batch_args(
            tuple(_cast_arg_to_dtype(arg, target) for arg in args),
            pad_to=pad_to,
            pad_value=pad_value,
        )
        out = fn(*batch_args)
        return trim_batch_out(_cast_out_to_dtype(out, target), n)

    return wrapped


@lru_cache(maxsize=None)
def _point_batch_fn(name: str):
    fn = _resolve_point_fn(name)
    if _is_host_point_backend(name):
        return fn

    @jax.jit
    def _batched(*vals):
        return jax.vmap(fn)(*vals)

    return _batched


@lru_cache(maxsize=None)
def _point_jit_fn(name: str):
    fn = _resolve_point_fn(name)
    if _is_host_point_backend(name):
        return fn
    return jax.jit(fn)


@lru_cache(maxsize=None)
def _interval_jit_fn(name: str, mode: str, prec_bits: int | None, dps: int | None):
    fn = _bound_interval_fn(name, mode, prec_bits, dps)
    if _is_host_point_backend(name):
        return fn
    return jax.jit(fn)


@lru_cache(maxsize=None)
def _interval_batch_fn(name: str, mode: str, prec_bits: int | None, dps: int | None):
    fn = _bound_interval_fn(name, mode, prec_bits, dps)
    if _is_host_point_backend(name):
        return fn

    @jax.jit
    def _batched(*vals):
        return jax.vmap(fn)(*vals)

    return _batched


@lru_cache(maxsize=None)
def _point_ad_fn(name: str, kind: str, argnums: int | tuple[int, ...]):
    fn = _point_jit_fn(name)
    if kind == "grad":
        ad_fn = jax.grad(fn, argnums=argnums)
    elif kind == "jacfwd":
        ad_fn = jax.jacfwd(fn, argnums=argnums)
    elif kind == "jacrev":
        ad_fn = jax.jacrev(fn, argnums=argnums)
    else:
        raise ValueError("kind must be one of: grad, jacfwd, jacrev")
    return jax.jit(ad_fn)


__all__ = [
    "eval_point",
    "eval_point_batch",
    "eval_point_batch_chunked",
    "eval_interval",
    "eval_interval_batch",
    "eval_interval_batch_chunked",
    "bind_point",
    "bind_point_batch",
    "bind_interval",
    "bind_point_jit",
    "bind_interval_jit",
    "bind_point_ad",
    "bind_interval_batch",
    "list_public_functions",
    "list_point_functions",
    "list_interval_functions",
]


def list_point_functions() -> list[str]:
    return list(_POINT_FUNC_NAMES)


def list_interval_functions() -> list[str]:
    return list(_INTERVAL_FUNC_NAMES)


def list_public_functions() -> list[str]:
    return list(_PUBLIC_FUNC_NAMES())


_POINT_FUNC_NAMES = tuple(sorted(_POINT_FUNCS.keys()))
_INTERVAL_FUNC_NAMES = tuple(sorted(_INTERVAL_FUNCS.keys()))


@lru_cache(maxsize=1)
def _PUBLIC_FUNC_NAMES() -> tuple[str, ...]:
    return tuple(sorted(_public_registry().keys()))
