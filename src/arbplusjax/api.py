from __future__ import annotations

import importlib
import inspect
from functools import lru_cache, partial
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp

from . import acb_core
from . import acb_calc
from . import arb_calc
from . import arb_mat
from . import acb_mat
from . import baseline_wrappers
from . import boost_hypgeom
from . import cubesselk
from . import double_gamma
from . import double_interval as di
from . import hypgeom
from . import hypgeom_wrappers
from . import mat_wrappers
from .kernel_helpers import (
    mixed_batch_size_or_none,
    pad_batch_args,
    pad_mixed_batch_args_repeat_last,
    trim_batch_out,
)
from . import point_wrappers
from .public_metadata import PublicFunctionMetadata, build_public_metadata_registry
from . import scb_block_mat
from . import scb_mat
from . import scb_vblock_mat
from .special.tail_acceleration import (
    TailDerivativeMetadata,
    TailEvaluationDiagnostics,
    TailIntegralProblem,
    TailRatioRecurrence,
    TailRegimeMetadata,
    evaluate_tail_integral,
)
from .special.bessel import (
    hankel1 as _hankel1_impl,
    hankel1_derivative,
    hankel1_point as _hankel1_point_impl,
    hankel2 as _hankel2_impl,
    hankel2_derivative,
    hankel2_point as _hankel2_point_impl,
    incomplete_bessel_i as _incomplete_bessel_i_impl,
    incomplete_bessel_i_argument_derivative,
    incomplete_bessel_i_derivative,
    incomplete_bessel_i_point as _incomplete_bessel_i_point_impl,
    incomplete_bessel_i_upper_limit_derivative,
    incomplete_bessel_k as _incomplete_bessel_k_impl,
    incomplete_bessel_k_argument_derivative,
    incomplete_bessel_k_derivative,
    incomplete_bessel_k_lower_limit_derivative,
    incomplete_bessel_k_point as _incomplete_bessel_k_point_impl,
    scaled_hankel1 as _scaled_hankel1_impl,
    scaled_hankel1_derivative,
    scaled_hankel1_point as _scaled_hankel1_point_impl,
    scaled_hankel2 as _scaled_hankel2_impl,
    scaled_hankel2_derivative,
    scaled_hankel2_point as _scaled_hankel2_point_impl,
    spherical_bessel_j as _spherical_bessel_j_impl,
    spherical_bessel_j_derivative,
    spherical_bessel_j_point as _spherical_bessel_j_point_impl,
    spherical_bessel_y as _spherical_bessel_y_impl,
    spherical_bessel_y_derivative,
    spherical_bessel_y_point as _spherical_bessel_y_point_impl,
    modified_spherical_bessel_i as _modified_spherical_bessel_i_impl,
    modified_spherical_bessel_i_derivative,
    modified_spherical_bessel_i_point as _modified_spherical_bessel_i_point_impl,
    modified_spherical_bessel_k as _modified_spherical_bessel_k_impl,
    modified_spherical_bessel_k_derivative,
    modified_spherical_bessel_k_point as _modified_spherical_bessel_k_point_impl,
)
from .special.gamma import (
    incomplete_gamma_lower as _incomplete_gamma_lower_impl,
    incomplete_gamma_lower_argument_derivative as _incomplete_gamma_lower_argument_derivative_impl,
    incomplete_gamma_lower_derivative as _incomplete_gamma_lower_derivative_impl,
    incomplete_gamma_lower_parameter_derivative as _incomplete_gamma_lower_parameter_derivative_impl,
    incomplete_gamma_lower_point as _incomplete_gamma_lower_point_impl,
    incomplete_gamma_upper as _incomplete_gamma_upper_impl,
    incomplete_gamma_upper_argument_derivative as _incomplete_gamma_upper_argument_derivative_impl,
    incomplete_gamma_upper_derivative as _incomplete_gamma_upper_derivative_impl,
    incomplete_gamma_upper_parameter_derivative as _incomplete_gamma_upper_parameter_derivative_impl,
    incomplete_gamma_upper_point as _incomplete_gamma_upper_point_impl,
)
from .special.laplace_bessel import (
    laplace_bessel_k_tail as _laplace_bessel_k_tail_impl,
    laplace_bessel_k_tail_derivative as _laplace_bessel_k_tail_derivative_impl,
    laplace_bessel_k_tail_lambda_derivative as _laplace_bessel_k_tail_lambda_derivative_impl,
    laplace_bessel_k_tail_lower_limit_derivative as _laplace_bessel_k_tail_lower_limit_derivative_impl,
    laplace_bessel_k_tail_point as _laplace_bessel_k_tail_point_impl,
)
from . import srb_mat
from . import srb_block_mat
from . import srb_vblock_mat

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
    "arb_calc_integrate_line": (arb_calc.arb_calc_integrate_line_batch_fixed_prec, arb_calc.arb_calc_integrate_line_batch_padded_prec),
    "arb_calc.arb_calc_integrate_line": (arb_calc.arb_calc_integrate_line_batch_fixed_prec, arb_calc.arb_calc_integrate_line_batch_padded_prec),
    "acb_calc_integrate_line": (acb_calc.acb_calc_integrate_line_batch_fixed_prec, acb_calc.acb_calc_integrate_line_batch_padded_prec),
    "acb_calc.acb_calc_integrate_line": (acb_calc.acb_calc_integrate_line_batch_fixed_prec, acb_calc.acb_calc_integrate_line_batch_padded_prec),
    "arb_mat_matmul": (arb_mat.arb_mat_matmul_batch_fixed_prec, arb_mat.arb_mat_matmul_batch_padded_prec),
    "arb_mat_permutation_matrix": (arb_mat.arb_mat_permutation_matrix_batch_fixed_prec, arb_mat.arb_mat_permutation_matrix_batch_padded_prec),
    "arb_mat_transpose": (arb_mat.arb_mat_transpose_batch_fixed_prec, arb_mat.arb_mat_transpose_batch_padded_prec),
    "arb_mat_diag": (arb_mat.arb_mat_diag_batch_fixed_prec, arb_mat.arb_mat_diag_batch_padded_prec),
    "arb_mat_diag_matrix": (arb_mat.arb_mat_diag_matrix_batch_fixed_prec, arb_mat.arb_mat_diag_matrix_batch_padded_prec),
    "arb_mat_matvec": (arb_mat.arb_mat_matvec_batch_fixed_prec, arb_mat.arb_mat_matvec_batch_padded_prec),
    "arb_mat_banded_matvec": (arb_mat.arb_mat_banded_matvec_batch_fixed_prec, arb_mat.arb_mat_banded_matvec_batch_padded_prec),
    "arb_mat_matvec_cached_prepare": (arb_mat.arb_mat_matvec_cached_prepare_batch_fixed_prec, arb_mat.arb_mat_matvec_cached_prepare_batch_padded_prec),
    "arb_mat_matvec_cached_apply": (arb_mat.arb_mat_matvec_cached_apply_batch_fixed_prec, arb_mat.arb_mat_matvec_cached_apply_batch_padded_prec),
    "arb_mat_solve": (arb_mat.arb_mat_solve_batch_fixed_prec, arb_mat.arb_mat_solve_batch_padded_prec),
    "arb_mat_inv": (arb_mat.arb_mat_inv_batch_fixed_prec, arb_mat.arb_mat_inv_batch_padded_prec),
    "arb_mat_triangular_solve": (arb_mat.arb_mat_triangular_solve_batch_fixed_prec, arb_mat.arb_mat_triangular_solve_batch_padded_prec),
    "arb_mat_lu": (arb_mat.arb_mat_lu_batch_fixed_prec, arb_mat.arb_mat_lu_batch_padded_prec),
    "arb_mat_lu_solve": (arb_mat.arb_mat_lu_solve_batch_fixed_prec, arb_mat.arb_mat_lu_solve_batch_padded_prec),
    "arb_mat_qr": (arb_mat.arb_mat_qr_batch_fixed_prec, arb_mat.arb_mat_qr_batch_padded_prec),
    "arb_mat_det": (arb_mat.arb_mat_det_batch_fixed_prec, arb_mat.arb_mat_det_batch_padded_prec),
    "arb_mat_trace": (arb_mat.arb_mat_trace_batch_fixed_prec, arb_mat.arb_mat_trace_batch_padded_prec),
    "arb_mat_sqr": (arb_mat.arb_mat_sqr_batch_fixed_prec, arb_mat.arb_mat_sqr_batch_padded_prec),
    "arb_mat_norm_fro": (arb_mat.arb_mat_norm_fro_batch_fixed_prec, arb_mat.arb_mat_norm_fro_batch_padded_prec),
    "arb_mat_norm_1": (arb_mat.arb_mat_norm_1_batch_fixed_prec, arb_mat.arb_mat_norm_1_batch_padded_prec),
    "arb_mat_norm_inf": (arb_mat.arb_mat_norm_inf_batch_fixed_prec, arb_mat.arb_mat_norm_inf_batch_padded_prec),
    "acb_mat_matmul": (acb_mat.acb_mat_matmul_batch_fixed_prec, acb_mat.acb_mat_matmul_batch_padded_prec),
    "acb_mat_permutation_matrix": (acb_mat.acb_mat_permutation_matrix_batch_fixed_prec, acb_mat.acb_mat_permutation_matrix_batch_padded_prec),
    "acb_mat_transpose": (acb_mat.acb_mat_transpose_batch_fixed_prec, acb_mat.acb_mat_transpose_batch_padded_prec),
    "acb_mat_conjugate_transpose": (acb_mat.acb_mat_conjugate_transpose_batch_fixed_prec, acb_mat.acb_mat_conjugate_transpose_batch_padded_prec),
    "acb_mat_diag": (acb_mat.acb_mat_diag_batch_fixed_prec, acb_mat.acb_mat_diag_batch_padded_prec),
    "acb_mat_diag_matrix": (acb_mat.acb_mat_diag_matrix_batch_fixed_prec, acb_mat.acb_mat_diag_matrix_batch_padded_prec),
    "acb_mat_matvec": (acb_mat.acb_mat_matvec_batch_fixed_prec, acb_mat.acb_mat_matvec_batch_padded_prec),
    "acb_mat_banded_matvec": (acb_mat.acb_mat_banded_matvec_batch_fixed_prec, acb_mat.acb_mat_banded_matvec_batch_padded_prec),
    "acb_mat_matvec_cached_prepare": (acb_mat.acb_mat_matvec_cached_prepare_batch_fixed_prec, acb_mat.acb_mat_matvec_cached_prepare_batch_padded_prec),
    "acb_mat_matvec_cached_apply": (acb_mat.acb_mat_matvec_cached_apply_batch_fixed_prec, acb_mat.acb_mat_matvec_cached_apply_batch_padded_prec),
    "acb_mat_solve": (acb_mat.acb_mat_solve_batch_fixed_prec, acb_mat.acb_mat_solve_batch_padded_prec),
    "acb_mat_inv": (acb_mat.acb_mat_inv_batch_fixed_prec, acb_mat.acb_mat_inv_batch_padded_prec),
    "acb_mat_triangular_solve": (acb_mat.acb_mat_triangular_solve_batch_fixed_prec, acb_mat.acb_mat_triangular_solve_batch_padded_prec),
    "acb_mat_lu": (acb_mat.acb_mat_lu_batch_fixed_prec, acb_mat.acb_mat_lu_batch_padded_prec),
    "acb_mat_lu_solve": (acb_mat.acb_mat_lu_solve_batch_fixed_prec, acb_mat.acb_mat_lu_solve_batch_padded_prec),
    "acb_mat_qr": (acb_mat.acb_mat_qr_batch_fixed_prec, acb_mat.acb_mat_qr_batch_padded_prec),
    "acb_mat_det": (acb_mat.acb_mat_det_batch_fixed_prec, acb_mat.acb_mat_det_batch_padded_prec),
    "acb_mat_trace": (acb_mat.acb_mat_trace_batch_fixed_prec, acb_mat.acb_mat_trace_batch_padded_prec),
    "acb_mat_sqr": (acb_mat.acb_mat_sqr_batch_fixed_prec, acb_mat.acb_mat_sqr_batch_padded_prec),
    "acb_mat_norm_fro": (acb_mat.acb_mat_norm_fro_batch_fixed_prec, acb_mat.acb_mat_norm_fro_batch_padded_prec),
    "acb_mat_norm_1": (acb_mat.acb_mat_norm_1_batch_fixed_prec, acb_mat.acb_mat_norm_1_batch_padded_prec),
    "acb_mat_norm_inf": (acb_mat.acb_mat_norm_inf_batch_fixed_prec, acb_mat.acb_mat_norm_inf_batch_padded_prec),
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
    "arb_calc_integrate_line": (arb_calc.arb_calc_integrate_line_batch_fixed_rigorous, arb_calc.arb_calc_integrate_line_batch_padded_rigorous),
    "arb_calc.arb_calc_integrate_line": (arb_calc.arb_calc_integrate_line_batch_fixed_rigorous, arb_calc.arb_calc_integrate_line_batch_padded_rigorous),
    "acb_calc_integrate_line": (acb_calc.acb_calc_integrate_line_batch_fixed_rigorous, acb_calc.acb_calc_integrate_line_batch_padded_rigorous),
    "acb_calc.acb_calc_integrate_line": (acb_calc.acb_calc_integrate_line_batch_fixed_rigorous, acb_calc.acb_calc_integrate_line_batch_padded_rigorous),
    "arb_mat_matmul": (mat_wrappers.arb_mat_matmul_batch_fixed_mode, mat_wrappers.arb_mat_matmul_batch_padded_mode),
    "arb_mat_permutation_matrix": (mat_wrappers.arb_mat_permutation_matrix_batch_fixed_mode, mat_wrappers.arb_mat_permutation_matrix_batch_padded_mode),
    "arb_mat_transpose": (mat_wrappers.arb_mat_transpose_batch_fixed_mode, mat_wrappers.arb_mat_transpose_batch_padded_mode),
    "arb_mat_diag": (mat_wrappers.arb_mat_diag_batch_fixed_mode, mat_wrappers.arb_mat_diag_batch_padded_mode),
    "arb_mat_diag_matrix": (mat_wrappers.arb_mat_diag_matrix_batch_fixed_mode, mat_wrappers.arb_mat_diag_matrix_batch_padded_mode),
    "arb_mat_matvec": (mat_wrappers.arb_mat_matvec_batch_fixed_mode, mat_wrappers.arb_mat_matvec_batch_padded_mode),
    "arb_mat_banded_matvec": (mat_wrappers.arb_mat_banded_matvec_batch_fixed_mode, mat_wrappers.arb_mat_banded_matvec_batch_padded_mode),
    "arb_mat_matvec_cached_prepare": (mat_wrappers.arb_mat_matvec_cached_prepare_batch_fixed_mode, mat_wrappers.arb_mat_matvec_cached_prepare_batch_padded_mode),
    "arb_mat_matvec_cached_apply": (mat_wrappers.arb_mat_matvec_cached_apply_batch_fixed_mode, mat_wrappers.arb_mat_matvec_cached_apply_batch_padded_mode),
    "arb_mat_solve": (mat_wrappers.arb_mat_solve_batch_fixed_mode, mat_wrappers.arb_mat_solve_batch_padded_mode),
    "arb_mat_inv": (mat_wrappers.arb_mat_inv_batch_fixed_mode, mat_wrappers.arb_mat_inv_batch_padded_mode),
    "arb_mat_triangular_solve": (mat_wrappers.arb_mat_triangular_solve_batch_fixed_mode, mat_wrappers.arb_mat_triangular_solve_batch_padded_mode),
    "arb_mat_lu": (mat_wrappers.arb_mat_lu_batch_fixed_mode, mat_wrappers.arb_mat_lu_batch_padded_mode),
    "arb_mat_lu_solve": (mat_wrappers.arb_mat_lu_solve_batch_fixed_mode, mat_wrappers.arb_mat_lu_solve_batch_padded_mode),
    "arb_mat_qr": (mat_wrappers.arb_mat_qr_batch_fixed_mode, mat_wrappers.arb_mat_qr_batch_padded_mode),
    "arb_mat_det": (mat_wrappers.arb_mat_det_batch_fixed_mode, mat_wrappers.arb_mat_det_batch_padded_mode),
    "arb_mat_trace": (mat_wrappers.arb_mat_trace_batch_fixed_mode, mat_wrappers.arb_mat_trace_batch_padded_mode),
    "arb_mat_sqr": (mat_wrappers.arb_mat_sqr_batch_fixed_mode, mat_wrappers.arb_mat_sqr_batch_padded_mode),
    "arb_mat_norm_fro": (mat_wrappers.arb_mat_norm_fro_batch_fixed_mode, mat_wrappers.arb_mat_norm_fro_batch_padded_mode),
    "arb_mat_norm_1": (mat_wrappers.arb_mat_norm_1_batch_fixed_mode, mat_wrappers.arb_mat_norm_1_batch_padded_mode),
    "arb_mat_norm_inf": (mat_wrappers.arb_mat_norm_inf_batch_fixed_mode, mat_wrappers.arb_mat_norm_inf_batch_padded_mode),
    "acb_mat_matmul": (mat_wrappers.acb_mat_matmul_batch_fixed_mode, mat_wrappers.acb_mat_matmul_batch_padded_mode),
    "acb_mat_permutation_matrix": (mat_wrappers.acb_mat_permutation_matrix_batch_fixed_mode, mat_wrappers.acb_mat_permutation_matrix_batch_padded_mode),
    "acb_mat_transpose": (mat_wrappers.acb_mat_transpose_batch_fixed_mode, mat_wrappers.acb_mat_transpose_batch_padded_mode),
    "acb_mat_conjugate_transpose": (mat_wrappers.acb_mat_conjugate_transpose_batch_fixed_mode, mat_wrappers.acb_mat_conjugate_transpose_batch_padded_mode),
    "acb_mat_diag": (mat_wrappers.acb_mat_diag_batch_fixed_mode, mat_wrappers.acb_mat_diag_batch_padded_mode),
    "acb_mat_diag_matrix": (mat_wrappers.acb_mat_diag_matrix_batch_fixed_mode, mat_wrappers.acb_mat_diag_matrix_batch_padded_mode),
    "acb_mat_matvec": (mat_wrappers.acb_mat_matvec_batch_fixed_mode, mat_wrappers.acb_mat_matvec_batch_padded_mode),
    "acb_mat_banded_matvec": (mat_wrappers.acb_mat_banded_matvec_batch_fixed_mode, mat_wrappers.acb_mat_banded_matvec_batch_padded_mode),
    "acb_mat_matvec_cached_prepare": (mat_wrappers.acb_mat_matvec_cached_prepare_batch_fixed_mode, mat_wrappers.acb_mat_matvec_cached_prepare_batch_padded_mode),
    "acb_mat_matvec_cached_apply": (mat_wrappers.acb_mat_matvec_cached_apply_batch_fixed_mode, mat_wrappers.acb_mat_matvec_cached_apply_batch_padded_mode),
    "acb_mat_solve": (mat_wrappers.acb_mat_solve_batch_fixed_mode, mat_wrappers.acb_mat_solve_batch_padded_mode),
    "acb_mat_inv": (mat_wrappers.acb_mat_inv_batch_fixed_mode, mat_wrappers.acb_mat_inv_batch_padded_mode),
    "acb_mat_triangular_solve": (mat_wrappers.acb_mat_triangular_solve_batch_fixed_mode, mat_wrappers.acb_mat_triangular_solve_batch_padded_mode),
    "acb_mat_lu": (mat_wrappers.acb_mat_lu_batch_fixed_mode, mat_wrappers.acb_mat_lu_batch_padded_mode),
    "acb_mat_lu_solve": (mat_wrappers.acb_mat_lu_solve_batch_fixed_mode, mat_wrappers.acb_mat_lu_solve_batch_padded_mode),
    "acb_mat_qr": (mat_wrappers.acb_mat_qr_batch_fixed_mode, mat_wrappers.acb_mat_qr_batch_padded_mode),
    "acb_mat_det": (mat_wrappers.acb_mat_det_batch_fixed_mode, mat_wrappers.acb_mat_det_batch_padded_mode),
    "acb_mat_trace": (mat_wrappers.acb_mat_trace_batch_fixed_mode, mat_wrappers.acb_mat_trace_batch_padded_mode),
    "acb_mat_sqr": (mat_wrappers.acb_mat_sqr_batch_fixed_mode, mat_wrappers.acb_mat_sqr_batch_padded_mode),
    "acb_mat_norm_fro": (mat_wrappers.acb_mat_norm_fro_batch_fixed_mode, mat_wrappers.acb_mat_norm_fro_batch_padded_mode),
    "acb_mat_norm_1": (mat_wrappers.acb_mat_norm_1_batch_fixed_mode, mat_wrappers.acb_mat_norm_1_batch_padded_mode),
    "acb_mat_norm_inf": (mat_wrappers.acb_mat_norm_inf_batch_fixed_mode, mat_wrappers.acb_mat_norm_inf_batch_padded_mode),
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
    "arb_calc_integrate_line": (arb_calc.arb_calc_integrate_line_batch_fixed_point, arb_calc.arb_calc_integrate_line_batch_padded_point),
    "arb_calc.arb_calc_integrate_line": (arb_calc.arb_calc_integrate_line_batch_fixed_point, arb_calc.arb_calc_integrate_line_batch_padded_point),
    "acb_calc_integrate_line": (acb_calc.acb_calc_integrate_line_batch_fixed_point, acb_calc.acb_calc_integrate_line_batch_padded_point),
    "acb_calc.acb_calc_integrate_line": (acb_calc.acb_calc_integrate_line_batch_fixed_point, acb_calc.acb_calc_integrate_line_batch_padded_point),
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
        kwargs = {}
        if prec_bits is not None:
            kwargs["prec_bits"] = prec_bits
    else:
        return None
    pair = mapping.get(name)
    if pair is None:
        return None
    fixed_fn, padded_fn = pair
    if mode in ("adaptive", "rigorous"):
        try:
            params = inspect.signature(fixed_fn).parameters
        except Exception:
            params = {}
        if "impl" in params:
            kwargs["impl"] = mode
        elif mode == "adaptive":
            return None
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
    "hankel1": _hankel1_point_impl,
    "spherical_bessel_j": _spherical_bessel_j_point_impl,
    "spherical_bessel_y": _spherical_bessel_y_point_impl,
    "modified_spherical_bessel_i": _modified_spherical_bessel_i_point_impl,
    "incomplete_bessel_i": _incomplete_bessel_i_point_impl,
    "besselk": point_wrappers.arb_bessel_k_point,
    "hankel2": _hankel2_point_impl,
    "modified_spherical_bessel_k": _modified_spherical_bessel_k_point_impl,
    "incomplete_bessel_k": _incomplete_bessel_k_point_impl,
    "scaled_hankel1": _scaled_hankel1_point_impl,
    "scaled_hankel2": _scaled_hankel2_point_impl,
    "incomplete_gamma_lower": _incomplete_gamma_lower_point_impl,
    "incomplete_gamma_upper": _incomplete_gamma_upper_point_impl,
    "laplace_bessel_k_tail": _laplace_bessel_k_tail_point_impl,
    "cuda_besselk": cubesselk.cuda_besselk_point,
    "arb_calc_integrate_line": arb_calc.arb_calc_integrate_line_point,
    "acb_calc_integrate_line": acb_calc.acb_calc_integrate_line_point,
    "arb_calc.arb_calc_integrate_line": arb_calc.arb_calc_integrate_line_point,
    "acb_calc.acb_calc_integrate_line": acb_calc.acb_calc_integrate_line_point,
}

_POINT_FUNCS.update(
    {
        "srb_mat_shape": srb_mat.srb_mat_shape,
        "srb_mat_nnz": srb_mat.srb_mat_nnz,
        "srb_mat_zero": srb_mat.srb_mat_zero,
        "srb_mat_identity": srb_mat.srb_mat_identity,
        "srb_mat_permutation_matrix": srb_mat.srb_mat_permutation_matrix,
        "srb_mat_diag": srb_mat.srb_mat_diag,
        "srb_mat_diag_matrix": srb_mat.srb_mat_diag_matrix,
        "srb_mat_trace": srb_mat.srb_mat_trace,
        "srb_mat_norm_fro": srb_mat.srb_mat_norm_fro,
        "srb_mat_norm_1": srb_mat.srb_mat_norm_1,
        "srb_mat_norm_inf": srb_mat.srb_mat_norm_inf,
        "srb_mat_submatrix": srb_mat.srb_mat_submatrix,
        "srb_mat_to_dense": srb_mat.srb_mat_to_dense,
        "srb_mat_transpose": srb_mat.srb_mat_transpose,
        "srb_mat_scale": srb_mat.srb_mat_scale,
        "srb_mat_add": srb_mat.srb_mat_add,
        "srb_mat_sub": srb_mat.srb_mat_sub,
        "srb_mat_matvec": srb_mat.srb_mat_matvec,
        "srb_mat_matvec_cached_prepare": srb_mat.srb_mat_matvec_cached_prepare,
        "srb_mat_matvec_cached_apply": srb_mat.srb_mat_matvec_cached_apply,
        "srb_mat_matmul_dense_rhs": srb_mat.srb_mat_matmul_dense_rhs,
        "srb_mat_matmul_sparse": srb_mat.srb_mat_matmul_sparse,
        "srb_mat_triangular_solve": srb_mat.srb_mat_triangular_solve,
        "srb_mat_charpoly": srb_mat.srb_mat_charpoly,
        "srb_mat_pow_ui": srb_mat.srb_mat_pow_ui,
        "srb_mat_exp": srb_mat.srb_mat_exp,
        "srb_mat_eigvalsh": srb_mat.srb_mat_eigvalsh,
        "srb_mat_eigh": srb_mat.srb_mat_eigh,
        "srb_mat_lu": srb_mat.srb_mat_lu,
        "srb_mat_lu_solve": srb_mat.srb_mat_lu_solve,
        "srb_mat_qr": srb_mat.srb_mat_qr,
        "srb_mat_qr_r": srb_mat.srb_mat_qr_r,
        "srb_mat_qr_apply_q": srb_mat.srb_mat_qr_apply_q,
        "srb_mat_qr_explicit_q": srb_mat.srb_mat_qr_explicit_q,
        "srb_mat_qr_solve": srb_mat.srb_mat_qr_solve,
        "srb_mat_solve": srb_mat.srb_mat_solve,
        "srb_block_mat_shape": srb_block_mat.srb_block_mat_shape,
        "srb_block_mat_block_shape": srb_block_mat.srb_block_mat_block_shape,
        "srb_block_mat_nnzb": srb_block_mat.srb_block_mat_nnzb,
        "srb_block_mat_coo": srb_block_mat.srb_block_mat_coo,
        "srb_block_mat_csr": srb_block_mat.srb_block_mat_csr,
        "srb_block_mat_from_dense_coo": srb_block_mat.srb_block_mat_from_dense_coo,
        "srb_block_mat_from_dense_csr": srb_block_mat.srb_block_mat_from_dense_csr,
        "srb_block_mat_coo_to_csr": srb_block_mat.srb_block_mat_coo_to_csr,
        "srb_block_mat_csr_to_coo": srb_block_mat.srb_block_mat_csr_to_coo,
        "srb_block_mat_to_dense": srb_block_mat.srb_block_mat_to_dense,
        "srb_block_mat_transpose": srb_block_mat.srb_block_mat_transpose,
        "srb_block_mat_matvec": srb_block_mat.srb_block_mat_matvec,
        "srb_block_mat_matvec_cached_prepare": srb_block_mat.srb_block_mat_matvec_cached_prepare,
        "srb_block_mat_matvec_cached_apply": srb_block_mat.srb_block_mat_matvec_cached_apply,
        "srb_block_mat_matvec_with_diagnostics": srb_block_mat.srb_block_mat_matvec_with_diagnostics,
        "srb_block_mat_matvec_cached_apply_with_diagnostics": srb_block_mat.srb_block_mat_matvec_cached_apply_with_diagnostics,
        "srb_block_mat_matmul_dense_rhs": srb_block_mat.srb_block_mat_matmul_dense_rhs,
        "srb_block_mat_triangular_solve": srb_block_mat.srb_block_mat_triangular_solve,
        "srb_block_mat_solve": srb_block_mat.srb_block_mat_solve,
        "srb_block_mat_solve_with_diagnostics": srb_block_mat.srb_block_mat_solve_with_diagnostics,
        "srb_vblock_mat_shape": srb_vblock_mat.srb_vblock_mat_shape,
        "srb_vblock_mat_block_sizes": srb_vblock_mat.srb_vblock_mat_block_sizes,
        "srb_vblock_mat_nnzb": srb_vblock_mat.srb_vblock_mat_nnzb,
        "srb_vblock_mat_coo": srb_vblock_mat.srb_vblock_mat_coo,
        "srb_vblock_mat_csr": srb_vblock_mat.srb_vblock_mat_csr,
        "srb_vblock_mat_from_dense_coo": srb_vblock_mat.srb_vblock_mat_from_dense_coo,
        "srb_vblock_mat_from_dense_csr": srb_vblock_mat.srb_vblock_mat_from_dense_csr,
        "srb_vblock_mat_coo_to_csr": srb_vblock_mat.srb_vblock_mat_coo_to_csr,
        "srb_vblock_mat_csr_to_coo": srb_vblock_mat.srb_vblock_mat_csr_to_coo,
        "srb_vblock_mat_to_dense": srb_vblock_mat.srb_vblock_mat_to_dense,
        "srb_vblock_mat_matvec": srb_vblock_mat.srb_vblock_mat_matvec,
        "srb_vblock_mat_matvec_cached_prepare": srb_vblock_mat.srb_vblock_mat_matvec_cached_prepare,
        "srb_vblock_mat_matvec_cached_apply": srb_vblock_mat.srb_vblock_mat_matvec_cached_apply,
        "srb_vblock_mat_matvec_with_diagnostics": srb_vblock_mat.srb_vblock_mat_matvec_with_diagnostics,
        "srb_vblock_mat_matvec_cached_apply_with_diagnostics": srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_with_diagnostics,
        "srb_vblock_mat_matmul_dense_rhs": srb_vblock_mat.srb_vblock_mat_matmul_dense_rhs,
        "srb_vblock_mat_triangular_solve": srb_vblock_mat.srb_vblock_mat_triangular_solve,
        "srb_vblock_mat_lu": srb_vblock_mat.srb_vblock_mat_lu,
        "srb_vblock_mat_lu_solve": srb_vblock_mat.srb_vblock_mat_lu_solve,
        "srb_vblock_mat_lu_with_diagnostics": srb_vblock_mat.srb_vblock_mat_lu_with_diagnostics,
        "srb_vblock_mat_qr": srb_vblock_mat.srb_vblock_mat_qr,
        "srb_vblock_mat_qr_solve": srb_vblock_mat.srb_vblock_mat_qr_solve,
        "srb_vblock_mat_qr_with_diagnostics": srb_vblock_mat.srb_vblock_mat_qr_with_diagnostics,
        "srb_vblock_mat_solve": srb_vblock_mat.srb_vblock_mat_solve,
        "srb_vblock_mat_solve_with_diagnostics": srb_vblock_mat.srb_vblock_mat_solve_with_diagnostics,
        "scb_mat_shape": scb_mat.scb_mat_shape,
        "scb_mat_nnz": scb_mat.scb_mat_nnz,
        "scb_mat_zero": scb_mat.scb_mat_zero,
        "scb_mat_identity": scb_mat.scb_mat_identity,
        "scb_mat_permutation_matrix": scb_mat.scb_mat_permutation_matrix,
        "scb_mat_diag": scb_mat.scb_mat_diag,
        "scb_mat_diag_matrix": scb_mat.scb_mat_diag_matrix,
        "scb_mat_trace": scb_mat.scb_mat_trace,
        "scb_mat_norm_fro": scb_mat.scb_mat_norm_fro,
        "scb_mat_norm_1": scb_mat.scb_mat_norm_1,
        "scb_mat_norm_inf": scb_mat.scb_mat_norm_inf,
        "scb_mat_submatrix": scb_mat.scb_mat_submatrix,
        "scb_mat_to_dense": scb_mat.scb_mat_to_dense,
        "scb_mat_transpose": scb_mat.scb_mat_transpose,
        "scb_mat_conjugate_transpose": scb_mat.scb_mat_conjugate_transpose,
        "scb_mat_scale": scb_mat.scb_mat_scale,
        "scb_mat_add": scb_mat.scb_mat_add,
        "scb_mat_sub": scb_mat.scb_mat_sub,
        "scb_mat_matvec": scb_mat.scb_mat_matvec,
        "scb_mat_matvec_cached_prepare": scb_mat.scb_mat_matvec_cached_prepare,
        "scb_mat_matvec_cached_apply": scb_mat.scb_mat_matvec_cached_apply,
        "scb_mat_matmul_dense_rhs": scb_mat.scb_mat_matmul_dense_rhs,
        "scb_mat_matmul_sparse": scb_mat.scb_mat_matmul_sparse,
        "scb_mat_triangular_solve": scb_mat.scb_mat_triangular_solve,
        "scb_mat_charpoly": scb_mat.scb_mat_charpoly,
        "scb_mat_pow_ui": scb_mat.scb_mat_pow_ui,
        "scb_mat_exp": scb_mat.scb_mat_exp,
        "scb_mat_eigvalsh": scb_mat.scb_mat_eigvalsh,
        "scb_mat_eigh": scb_mat.scb_mat_eigh,
        "scb_mat_lu": scb_mat.scb_mat_lu,
        "scb_mat_lu_solve": scb_mat.scb_mat_lu_solve,
        "scb_mat_qr": scb_mat.scb_mat_qr,
        "scb_mat_qr_r": scb_mat.scb_mat_qr_r,
        "scb_mat_qr_apply_q": scb_mat.scb_mat_qr_apply_q,
        "scb_mat_qr_explicit_q": scb_mat.scb_mat_qr_explicit_q,
        "scb_mat_qr_solve": scb_mat.scb_mat_qr_solve,
        "scb_mat_solve": scb_mat.scb_mat_solve,
        "scb_block_mat_shape": scb_block_mat.scb_block_mat_shape,
        "scb_block_mat_block_shape": scb_block_mat.scb_block_mat_block_shape,
        "scb_block_mat_nnzb": scb_block_mat.scb_block_mat_nnzb,
        "scb_block_mat_coo": scb_block_mat.scb_block_mat_coo,
        "scb_block_mat_csr": scb_block_mat.scb_block_mat_csr,
        "scb_block_mat_from_dense_coo": scb_block_mat.scb_block_mat_from_dense_coo,
        "scb_block_mat_from_dense_csr": scb_block_mat.scb_block_mat_from_dense_csr,
        "scb_block_mat_coo_to_csr": scb_block_mat.scb_block_mat_coo_to_csr,
        "scb_block_mat_csr_to_coo": scb_block_mat.scb_block_mat_csr_to_coo,
        "scb_block_mat_to_dense": scb_block_mat.scb_block_mat_to_dense,
        "scb_block_mat_transpose": scb_block_mat.scb_block_mat_transpose,
        "scb_block_mat_matvec": scb_block_mat.scb_block_mat_matvec,
        "scb_block_mat_matvec_cached_prepare": scb_block_mat.scb_block_mat_matvec_cached_prepare,
        "scb_block_mat_matvec_cached_apply": scb_block_mat.scb_block_mat_matvec_cached_apply,
        "scb_block_mat_matvec_with_diagnostics": scb_block_mat.scb_block_mat_matvec_with_diagnostics,
        "scb_block_mat_matvec_cached_apply_with_diagnostics": scb_block_mat.scb_block_mat_matvec_cached_apply_with_diagnostics,
        "scb_block_mat_matmul_dense_rhs": scb_block_mat.scb_block_mat_matmul_dense_rhs,
        "scb_block_mat_triangular_solve": scb_block_mat.scb_block_mat_triangular_solve,
        "scb_block_mat_solve": scb_block_mat.scb_block_mat_solve,
        "scb_block_mat_solve_with_diagnostics": scb_block_mat.scb_block_mat_solve_with_diagnostics,
        "scb_vblock_mat_shape": scb_vblock_mat.scb_vblock_mat_shape,
        "scb_vblock_mat_block_sizes": scb_vblock_mat.scb_vblock_mat_block_sizes,
        "scb_vblock_mat_nnzb": scb_vblock_mat.scb_vblock_mat_nnzb,
        "scb_vblock_mat_coo": scb_vblock_mat.scb_vblock_mat_coo,
        "scb_vblock_mat_csr": scb_vblock_mat.scb_vblock_mat_csr,
        "scb_vblock_mat_from_dense_coo": scb_vblock_mat.scb_vblock_mat_from_dense_coo,
        "scb_vblock_mat_from_dense_csr": scb_vblock_mat.scb_vblock_mat_from_dense_csr,
        "scb_vblock_mat_coo_to_csr": scb_vblock_mat.scb_vblock_mat_coo_to_csr,
        "scb_vblock_mat_csr_to_coo": scb_vblock_mat.scb_vblock_mat_csr_to_coo,
        "scb_vblock_mat_to_dense": scb_vblock_mat.scb_vblock_mat_to_dense,
        "scb_vblock_mat_matvec": scb_vblock_mat.scb_vblock_mat_matvec,
        "scb_vblock_mat_matvec_cached_prepare": scb_vblock_mat.scb_vblock_mat_matvec_cached_prepare,
        "scb_vblock_mat_matvec_cached_apply": scb_vblock_mat.scb_vblock_mat_matvec_cached_apply,
        "scb_vblock_mat_matvec_with_diagnostics": scb_vblock_mat.scb_vblock_mat_matvec_with_diagnostics,
        "scb_vblock_mat_matvec_cached_apply_with_diagnostics": scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_with_diagnostics,
        "scb_vblock_mat_matmul_dense_rhs": scb_vblock_mat.scb_vblock_mat_matmul_dense_rhs,
        "scb_vblock_mat_triangular_solve": scb_vblock_mat.scb_vblock_mat_triangular_solve,
        "scb_vblock_mat_lu": scb_vblock_mat.scb_vblock_mat_lu,
        "scb_vblock_mat_lu_solve": scb_vblock_mat.scb_vblock_mat_lu_solve,
        "scb_vblock_mat_lu_with_diagnostics": scb_vblock_mat.scb_vblock_mat_lu_with_diagnostics,
        "scb_vblock_mat_qr": scb_vblock_mat.scb_vblock_mat_qr,
        "scb_vblock_mat_qr_solve": scb_vblock_mat.scb_vblock_mat_qr_solve,
        "scb_vblock_mat_qr_with_diagnostics": scb_vblock_mat.scb_vblock_mat_qr_with_diagnostics,
        "scb_vblock_mat_solve": scb_vblock_mat.scb_vblock_mat_solve,
        "scb_vblock_mat_solve_with_diagnostics": scb_vblock_mat.scb_vblock_mat_solve_with_diagnostics,
        "arb_mat_matmul": point_wrappers.arb_mat_matmul_point,
        "arb_mat_zero": point_wrappers.arb_mat_zero_point,
        "arb_mat_identity": point_wrappers.arb_mat_identity_point,
        "arb_mat_block_assemble": point_wrappers.arb_mat_block_assemble_point,
        "arb_mat_block_diag": point_wrappers.arb_mat_block_diag_point,
        "arb_mat_block_extract": point_wrappers.arb_mat_block_extract_point,
        "arb_mat_block_row": point_wrappers.arb_mat_block_row_point,
        "arb_mat_block_col": point_wrappers.arb_mat_block_col_point,
        "arb_mat_block_matmul": point_wrappers.arb_mat_block_matmul_point,
        "arb_mat_matvec": point_wrappers.arb_mat_matvec_point,
        "arb_mat_matvec_cached_prepare": point_wrappers.arb_mat_matvec_cached_prepare_point,
        "arb_mat_matvec_cached_apply": point_wrappers.arb_mat_matvec_cached_apply_point,
        "arb_mat_solve": point_wrappers.arb_mat_solve_point,
        "arb_mat_inv": point_wrappers.arb_mat_inv_point,
        "arb_mat_sqr": point_wrappers.arb_mat_sqr_point,
        "arb_mat_det": point_wrappers.arb_mat_det_point,
        "arb_mat_trace": point_wrappers.arb_mat_trace_point,
        "arb_mat_norm_fro": point_wrappers.arb_mat_norm_fro_point,
        "arb_mat_norm_1": point_wrappers.arb_mat_norm_1_point,
        "arb_mat_norm_inf": point_wrappers.arb_mat_norm_inf_point,
        "arb_mat_triangular_solve": point_wrappers.arb_mat_triangular_solve_point,
        "arb_mat_lu": point_wrappers.arb_mat_lu_point,
        "arb_mat_qr": point_wrappers.arb_mat_qr_point,
        "arb_mat_2x2_det": point_wrappers.arb_mat_2x2_det_point,
        "arb_mat_2x2_trace": point_wrappers.arb_mat_2x2_trace_point,
        "arb_mat_2x2_det_batch": point_wrappers.arb_mat_2x2_det_batch_point,
        "arb_mat_2x2_trace_batch": point_wrappers.arb_mat_2x2_trace_batch_point,
        "acb_mat_matmul": point_wrappers.acb_mat_matmul_point,
        "acb_mat_zero": point_wrappers.acb_mat_zero_point,
        "acb_mat_identity": point_wrappers.acb_mat_identity_point,
        "acb_mat_block_assemble": point_wrappers.acb_mat_block_assemble_point,
        "acb_mat_block_diag": point_wrappers.acb_mat_block_diag_point,
        "acb_mat_block_extract": point_wrappers.acb_mat_block_extract_point,
        "acb_mat_block_row": point_wrappers.acb_mat_block_row_point,
        "acb_mat_block_col": point_wrappers.acb_mat_block_col_point,
        "acb_mat_block_matmul": point_wrappers.acb_mat_block_matmul_point,
        "acb_mat_matvec": point_wrappers.acb_mat_matvec_point,
        "acb_mat_matvec_cached_prepare": point_wrappers.acb_mat_matvec_cached_prepare_point,
        "acb_mat_matvec_cached_apply": point_wrappers.acb_mat_matvec_cached_apply_point,
        "acb_mat_solve": point_wrappers.acb_mat_solve_point,
        "acb_mat_inv": point_wrappers.acb_mat_inv_point,
        "acb_mat_sqr": point_wrappers.acb_mat_sqr_point,
        "acb_mat_det": point_wrappers.acb_mat_det_point,
        "acb_mat_trace": point_wrappers.acb_mat_trace_point,
        "acb_mat_norm_fro": point_wrappers.acb_mat_norm_fro_point,
        "acb_mat_norm_1": point_wrappers.acb_mat_norm_1_point,
        "acb_mat_norm_inf": point_wrappers.acb_mat_norm_inf_point,
        "acb_mat_triangular_solve": point_wrappers.acb_mat_triangular_solve_point,
        "acb_mat_lu": point_wrappers.acb_mat_lu_point,
        "acb_mat_qr": point_wrappers.acb_mat_qr_point,
        "acb_mat_2x2_det": point_wrappers.acb_mat_2x2_det_point,
        "acb_mat_2x2_trace": point_wrappers.acb_mat_2x2_trace_point,
        "acb_mat_2x2_det_batch": point_wrappers.acb_mat_2x2_det_batch_point,
        "acb_mat_2x2_trace_batch": point_wrappers.acb_mat_2x2_trace_batch_point,
    }
)

_DIRECT_POINT_BATCH_FASTPATHS.update(
    {
        "srb_mat_matvec": (srb_mat.srb_mat_matvec_batch_fixed, srb_mat.srb_mat_matvec_batch_padded),
        "srb_mat_matvec_cached_apply": (srb_mat.srb_mat_matvec_cached_apply_batch_fixed, srb_mat.srb_mat_matvec_cached_apply_batch_padded),
        "srb_mat_solve": (srb_mat.srb_mat_solve_batch_fixed, srb_mat.srb_mat_solve_batch_padded),
        "srb_mat_triangular_solve": (srb_mat.srb_mat_triangular_solve_batch_fixed, srb_mat.srb_mat_triangular_solve_batch_padded),
        "srb_block_mat_matvec": (srb_block_mat.srb_block_mat_matvec_batch_fixed, srb_block_mat.srb_block_mat_matvec_batch_padded),
        "srb_block_mat_matvec_cached_apply": (srb_block_mat.srb_block_mat_matvec_cached_apply_batch_fixed, srb_block_mat.srb_block_mat_matvec_cached_apply_batch_padded),
        "srb_block_mat_solve": (srb_block_mat.srb_block_mat_solve_batch_fixed, srb_block_mat.srb_block_mat_solve_batch_padded),
        "srb_vblock_mat_matvec": (srb_vblock_mat.srb_vblock_mat_matvec_batch_fixed, srb_vblock_mat.srb_vblock_mat_matvec_batch_padded),
        "srb_vblock_mat_matvec_cached_apply": (srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_batch_fixed, srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_batch_padded),
        "srb_vblock_mat_solve": (srb_vblock_mat.srb_vblock_mat_solve_batch_fixed, srb_vblock_mat.srb_vblock_mat_solve_batch_padded),
        "scb_mat_matvec": (scb_mat.scb_mat_matvec_batch_fixed, scb_mat.scb_mat_matvec_batch_padded),
        "scb_mat_matvec_cached_apply": (scb_mat.scb_mat_matvec_cached_apply_batch_fixed, scb_mat.scb_mat_matvec_cached_apply_batch_padded),
        "scb_mat_solve": (scb_mat.scb_mat_solve_batch_fixed, scb_mat.scb_mat_solve_batch_padded),
        "scb_mat_triangular_solve": (scb_mat.scb_mat_triangular_solve_batch_fixed, scb_mat.scb_mat_triangular_solve_batch_padded),
        "scb_block_mat_matvec": (scb_block_mat.scb_block_mat_matvec_batch_fixed, scb_block_mat.scb_block_mat_matvec_batch_padded),
        "scb_block_mat_matvec_cached_apply": (scb_block_mat.scb_block_mat_matvec_cached_apply_batch_fixed, scb_block_mat.scb_block_mat_matvec_cached_apply_batch_padded),
        "scb_block_mat_solve": (scb_block_mat.scb_block_mat_solve_batch_fixed, scb_block_mat.scb_block_mat_solve_batch_padded),
        "scb_vblock_mat_matvec": (scb_vblock_mat.scb_vblock_mat_matvec_batch_fixed, scb_vblock_mat.scb_vblock_mat_matvec_batch_padded),
        "scb_vblock_mat_matvec_cached_apply": (scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_batch_fixed, scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_batch_padded),
        "scb_vblock_mat_solve": (scb_vblock_mat.scb_vblock_mat_solve_batch_fixed, scb_vblock_mat.scb_vblock_mat_solve_batch_padded),
        "arb_mat_matmul": (point_wrappers.arb_mat_matmul_batch_fixed_point, point_wrappers.arb_mat_matmul_batch_padded_point),
        "arb_mat_matvec": (point_wrappers.arb_mat_matvec_batch_fixed_point, point_wrappers.arb_mat_matvec_batch_padded_point),
        "arb_mat_matvec_cached_apply": (
            point_wrappers.arb_mat_matvec_cached_apply_batch_fixed_point,
            point_wrappers.arb_mat_matvec_cached_apply_batch_padded_point,
        ),
        "arb_mat_det": (point_wrappers.arb_mat_det_batch_fixed_point, point_wrappers.arb_mat_det_batch_padded_point),
        "arb_mat_trace": (point_wrappers.arb_mat_trace_batch_fixed_point, point_wrappers.arb_mat_trace_batch_padded_point),
        "arb_mat_sqr": (point_wrappers.arb_mat_sqr_batch_fixed_point, point_wrappers.arb_mat_sqr_batch_padded_point),
        "arb_mat_norm_fro": (point_wrappers.arb_mat_norm_fro_batch_fixed_point, point_wrappers.arb_mat_norm_fro_batch_padded_point),
        "arb_mat_norm_1": (point_wrappers.arb_mat_norm_1_batch_fixed_point, point_wrappers.arb_mat_norm_1_batch_padded_point),
        "arb_mat_norm_inf": (point_wrappers.arb_mat_norm_inf_batch_fixed_point, point_wrappers.arb_mat_norm_inf_batch_padded_point),
        "acb_mat_matmul": (point_wrappers.acb_mat_matmul_batch_fixed_point, point_wrappers.acb_mat_matmul_batch_padded_point),
        "acb_mat_matvec": (point_wrappers.acb_mat_matvec_batch_fixed_point, point_wrappers.acb_mat_matvec_batch_padded_point),
        "acb_mat_matvec_cached_apply": (
            point_wrappers.acb_mat_matvec_cached_apply_batch_fixed_point,
            point_wrappers.acb_mat_matvec_cached_apply_batch_padded_point,
        ),
        "acb_mat_det": (point_wrappers.acb_mat_det_batch_fixed_point, point_wrappers.acb_mat_det_batch_padded_point),
        "acb_mat_trace": (point_wrappers.acb_mat_trace_batch_fixed_point, point_wrappers.acb_mat_trace_batch_padded_point),
        "acb_mat_sqr": (point_wrappers.acb_mat_sqr_batch_fixed_point, point_wrappers.acb_mat_sqr_batch_padded_point),
        "acb_mat_norm_fro": (point_wrappers.acb_mat_norm_fro_batch_fixed_point, point_wrappers.acb_mat_norm_fro_batch_padded_point),
        "acb_mat_norm_1": (point_wrappers.acb_mat_norm_1_batch_fixed_point, point_wrappers.acb_mat_norm_1_batch_padded_point),
        "acb_mat_norm_inf": (point_wrappers.acb_mat_norm_inf_batch_fixed_point, point_wrappers.acb_mat_norm_inf_batch_padded_point),
    }
)

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


# Use the generic vmapped batch path for scalar incomplete-tail kernels.
_DIRECT_POINT_BATCH_FASTPATHS.pop("incomplete_bessel_k", None)
_DIRECT_POINT_BATCH_FASTPATHS.pop("incomplete_bessel_i", None)
_DIRECT_POINT_BATCH_FASTPATHS.pop("incomplete_gamma_lower", None)
_DIRECT_POINT_BATCH_FASTPATHS.pop("incomplete_gamma_upper", None)
_DIRECT_POINT_BATCH_FASTPATHS.pop("laplace_bessel_k_tail", None)


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
    "hankel1": _hankel1_impl,
    "spherical_bessel_j": _spherical_bessel_j_impl,
    "spherical_bessel_y": _spherical_bessel_y_impl,
    "modified_spherical_bessel_i": _modified_spherical_bessel_i_impl,
    "incomplete_bessel_i": _incomplete_bessel_i_impl,
    "besselk": baseline_wrappers.arb_bessel_k_mp,
    "hankel2": _hankel2_impl,
    "modified_spherical_bessel_k": _modified_spherical_bessel_k_impl,
    "incomplete_bessel_k": _incomplete_bessel_k_impl,
    "scaled_hankel1": _scaled_hankel1_impl,
    "scaled_hankel2": _scaled_hankel2_impl,
    "incomplete_gamma_lower": _incomplete_gamma_lower_impl,
    "incomplete_gamma_upper": _incomplete_gamma_upper_impl,
    "laplace_bessel_k_tail": _laplace_bessel_k_tail_impl,
    "cuda_besselk": cubesselk.cuda_besselk,
}

for _name in mat_wrappers.SPARSE_MODE_BASES:
    _INTERVAL_FUNCS.setdefault(_name, getattr(mat_wrappers, f"{_name}_mode"))

for _name in mat_wrappers.SPARSE_BATCH_MODE_BASES:
    _DIRECT_INTERVAL_BASIC_BATCH_FASTPATHS.setdefault(
        _name,
        (
            getattr(mat_wrappers, f"{_name}_batch_mode_fixed"),
            getattr(mat_wrappers, f"{_name}_batch_mode_padded"),
        ),
    )
    _DIRECT_INTERVAL_MODE_BATCH_FASTPATHS.setdefault(
        _name,
        (
            getattr(mat_wrappers, f"{_name}_batch_mode_fixed"),
            getattr(mat_wrappers, f"{_name}_batch_mode_padded"),
        ),
    )


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
    "scb_block_mat",
    "scb_vblock_mat",
    "special.bessel",
    "srb_block_mat",
    "srb_vblock_mat",
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
    "tail_integral",
    "tail_integral_batch",
    "tail_integral_accelerated",
    "tail_integral_accelerated_batch",
    "incomplete_gamma_lower",
    "incomplete_gamma_lower_argument_derivative",
    "incomplete_gamma_lower_batch",
    "incomplete_gamma_lower_derivative",
    "incomplete_gamma_lower_parameter_derivative",
    "incomplete_gamma_upper",
    "incomplete_gamma_upper_argument_derivative",
    "incomplete_gamma_upper_batch",
    "incomplete_gamma_upper_derivative",
    "incomplete_gamma_upper_parameter_derivative",
    "laplace_bessel_k_tail",
    "laplace_bessel_k_tail_batch",
    "laplace_bessel_k_tail_derivative",
    "laplace_bessel_k_tail_lambda_derivative",
    "laplace_bessel_k_tail_lower_limit_derivative",
    "incomplete_bessel_i",
    "incomplete_bessel_i_batch",
    "incomplete_bessel_i_derivative",
    "incomplete_bessel_i_argument_derivative",
    "incomplete_bessel_i_upper_limit_derivative",
    "incomplete_bessel_k",
    "incomplete_bessel_k_batch",
    "incomplete_bessel_k_derivative",
    "incomplete_bessel_k_argument_derivative",
    "incomplete_bessel_k_lower_limit_derivative",
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
    "get_public_function_metadata",
    "list_public_function_metadata",
    "PublicFunctionMetadata",
    "TailDerivativeMetadata",
    "TailEvaluationDiagnostics",
    "TailIntegralProblem",
    "TailRatioRecurrence",
    "TailRegimeMetadata",
]


def list_point_functions() -> list[str]:
    return list(_POINT_FUNC_NAMES)


def list_interval_functions() -> list[str]:
    return list(_INTERVAL_FUNC_NAMES)


def list_public_functions() -> list[str]:
    return list(_PUBLIC_FUNC_NAMES())


def tail_integral(
    integrand_or_problem,
    lower_limit: jax.Array | float | None = None,
    *,
    panel_width: float = 0.25,
    max_panels: int = 128,
    samples_per_panel: int = 32,
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, TailEvaluationDiagnostics]:
    problem = _coerce_tail_problem(
        integrand_or_problem,
        lower_limit=lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return evaluate_tail_integral(problem, method="quadrature", return_diagnostics=return_diagnostics)


def tail_integral_batch(
    integrand_or_problem,
    lower_limit,
    *,
    panel_width: float = 0.25,
    max_panels: int = 128,
    samples_per_panel: int = 32,
):
    fn = lambda a: tail_integral(
        integrand_or_problem,
        a,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(lower_limit)


def tail_integral_accelerated(
    integrand_or_problem,
    lower_limit: jax.Array | float | None = None,
    *,
    method: str = "auto",
    panel_width: float = 0.25,
    max_panels: int = 128,
    samples_per_panel: int = 32,
    recurrence: TailRatioRecurrence | None = None,
    derivative_metadata: TailDerivativeMetadata | None = None,
    regime_metadata: TailRegimeMetadata | None = None,
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, TailEvaluationDiagnostics]:
    problem = _coerce_tail_problem(
        integrand_or_problem,
        lower_limit=lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        recurrence=recurrence,
        derivative_metadata=derivative_metadata,
        regime_metadata=regime_metadata,
    )
    return evaluate_tail_integral(problem, method=method, return_diagnostics=return_diagnostics)


def tail_integral_accelerated_batch(
    integrand_or_problem,
    lower_limit,
    *,
    method: str = "auto",
    panel_width: float = 0.25,
    max_panels: int = 128,
    samples_per_panel: int = 32,
    recurrence: TailRatioRecurrence | None = None,
    derivative_metadata: TailDerivativeMetadata | None = None,
    regime_metadata: TailRegimeMetadata | None = None,
):
    fn = lambda a: tail_integral_accelerated(
        integrand_or_problem,
        a,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        recurrence=recurrence,
        derivative_metadata=derivative_metadata,
        regime_metadata=regime_metadata,
    )
    return jax.vmap(fn)(lower_limit)


def incomplete_bessel_i(
    nu,
    z,
    upper_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_count: int = 128,
    samples_per_panel: int = 16,
    return_diagnostics: bool = False,
):
    return _incomplete_bessel_i_impl(
        nu,
        z,
        upper_limit,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_count=panel_count,
        samples_per_panel=samples_per_panel,
        return_diagnostics=return_diagnostics,
    )


def incomplete_gamma_upper(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    return _incomplete_gamma_upper_impl(
        s,
        z,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=return_diagnostics,
    )


def incomplete_gamma_upper_batch(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b: incomplete_gamma_upper(
        a,
        b,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(s, z)


def incomplete_gamma_lower(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    return _incomplete_gamma_lower_impl(
        s,
        z,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=return_diagnostics,
    )


def incomplete_gamma_lower_batch(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b: incomplete_gamma_lower(
        a,
        b,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(s, z)


def incomplete_gamma_upper_argument_derivative(
    s,
    z,
    *,
    regularized: bool = False,
):
    return _incomplete_gamma_upper_argument_derivative_impl(s, z, regularized=regularized)


def incomplete_gamma_upper_parameter_derivative(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _incomplete_gamma_upper_parameter_derivative_impl(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def incomplete_gamma_upper_derivative(
    s,
    z,
    *,
    respect_to: str = "z",
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _incomplete_gamma_upper_derivative_impl(
        s,
        z,
        respect_to=respect_to,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def incomplete_gamma_lower_argument_derivative(
    s,
    z,
    *,
    regularized: bool = False,
):
    return _incomplete_gamma_lower_argument_derivative_impl(s, z, regularized=regularized)


def incomplete_gamma_lower_parameter_derivative(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _incomplete_gamma_lower_parameter_derivative_impl(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def incomplete_gamma_lower_derivative(
    s,
    z,
    *,
    respect_to: str = "z",
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _incomplete_gamma_lower_derivative_impl(
        s,
        z,
        respect_to=respect_to,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def laplace_bessel_k_tail(
    nu,
    z,
    lam,
    lower_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    return _laplace_bessel_k_tail_impl(
        nu,
        z,
        lam,
        lower_limit,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=return_diagnostics,
    )


def laplace_bessel_k_tail_batch(
    nu,
    z,
    lam,
    lower_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b, c, d: laplace_bessel_k_tail(
        a,
        b,
        c,
        d,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(nu, z, lam, lower_limit)


def laplace_bessel_k_tail_lower_limit_derivative(nu, z, lam, lower_limit):
    return _laplace_bessel_k_tail_lower_limit_derivative_impl(nu, z, lam, lower_limit)


def laplace_bessel_k_tail_lambda_derivative(
    nu,
    z,
    lam,
    lower_limit,
    *,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _laplace_bessel_k_tail_lambda_derivative_impl(
        nu,
        z,
        lam,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def laplace_bessel_k_tail_derivative(
    nu,
    z,
    lam,
    lower_limit,
    *,
    respect_to: str = "lambda",
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _laplace_bessel_k_tail_derivative_impl(
        nu,
        z,
        lam,
        lower_limit,
        respect_to=respect_to,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def incomplete_bessel_i_batch(
    nu,
    z,
    upper_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_count: int = 128,
    samples_per_panel: int = 16,
):
    fn = lambda a, b, c: incomplete_bessel_i(
        a,
        b,
        c,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_count=panel_count,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(nu, z, upper_limit)


def incomplete_bessel_k(
    nu,
    z,
    lower_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    return _incomplete_bessel_k_impl(
        nu,
        z,
        lower_limit,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=return_diagnostics,
    )


def incomplete_bessel_k_batch(
    nu,
    z,
    lower_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b, c: incomplete_bessel_k(
        a,
        b,
        c,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(nu, z, lower_limit)


def hankel1(nu, z, *, method: str = "auto"):
    return _hankel1_impl(nu, z, method=method)


def hankel2(nu, z, *, method: str = "auto"):
    return _hankel2_impl(nu, z, method=method)


def scaled_hankel1(nu, z, *, method: str = "auto"):
    return _scaled_hankel1_impl(nu, z, method=method)


def scaled_hankel2(nu, z, *, method: str = "auto"):
    return _scaled_hankel2_impl(nu, z, method=method)


def hankel1_batch(nu, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: hankel1(a, b, method=method))(nu, z)


def hankel2_batch(nu, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: hankel2(a, b, method=method))(nu, z)


def scaled_hankel1_batch(nu, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: scaled_hankel1(a, b, method=method))(nu, z)


def scaled_hankel2_batch(nu, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: scaled_hankel2(a, b, method=method))(nu, z)


def spherical_bessel_j(n, z, *, method: str = "auto"):
    return _spherical_bessel_j_impl(n, z, method=method)


def spherical_bessel_y(n, z, *, method: str = "auto"):
    return _spherical_bessel_y_impl(n, z, method=method)


def modified_spherical_bessel_i(n, z, *, method: str = "auto"):
    return _modified_spherical_bessel_i_impl(n, z, method=method)


def modified_spherical_bessel_k(n, z, *, method: str = "auto"):
    return _modified_spherical_bessel_k_impl(n, z, method=method)


def spherical_bessel_j_batch(n, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: spherical_bessel_j(a, b, method=method))(n, z)


def spherical_bessel_y_batch(n, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: spherical_bessel_y(a, b, method=method))(n, z)


def modified_spherical_bessel_i_batch(n, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: modified_spherical_bessel_i(a, b, method=method))(n, z)


def modified_spherical_bessel_k_batch(n, z, *, method: str = "auto"):
    return jax.vmap(lambda a, b: modified_spherical_bessel_k(a, b, method=method))(n, z)


def get_public_function_metadata(name: str) -> PublicFunctionMetadata:
    registry = _PUBLIC_METADATA_REGISTRY()
    return _require(registry, name)


def list_public_function_metadata(
    *,
    family: str | None = None,
    stability: str | None = None,
) -> list[PublicFunctionMetadata]:
    entries = list(_PUBLIC_METADATA_REGISTRY().values())
    if family is not None:
        entries = [entry for entry in entries if entry.family == family]
    if stability is not None:
        entries = [entry for entry in entries if entry.stability == stability]
    return sorted(entries, key=lambda entry: entry.name)


_POINT_FUNC_NAMES = tuple(sorted(_POINT_FUNCS.keys()))
_INTERVAL_FUNC_NAMES = tuple(sorted(_INTERVAL_FUNCS.keys()))


@lru_cache(maxsize=1)
def _PUBLIC_FUNC_NAMES() -> tuple[str, ...]:
    return tuple(sorted(_public_registry().keys()))


@lru_cache(maxsize=1)
def _PUBLIC_METADATA_REGISTRY() -> dict[str, PublicFunctionMetadata]:
    combined_registry = dict(_public_registry())
    combined_registry.update(_INTERVAL_FUNCS)
    combined_registry.update(_POINT_FUNCS)
    combined_registry["tail_integral"] = tail_integral
    combined_registry["tail_integral_batch"] = tail_integral_batch
    combined_registry["tail_integral_accelerated"] = tail_integral_accelerated
    combined_registry["tail_integral_accelerated_batch"] = tail_integral_accelerated_batch
    combined_registry["incomplete_gamma_lower"] = incomplete_gamma_lower
    combined_registry["incomplete_gamma_lower_argument_derivative"] = incomplete_gamma_lower_argument_derivative
    combined_registry["incomplete_gamma_lower_batch"] = incomplete_gamma_lower_batch
    combined_registry["incomplete_gamma_lower_derivative"] = incomplete_gamma_lower_derivative
    combined_registry["incomplete_gamma_lower_parameter_derivative"] = incomplete_gamma_lower_parameter_derivative
    combined_registry["incomplete_gamma_upper"] = incomplete_gamma_upper
    combined_registry["incomplete_gamma_upper_argument_derivative"] = incomplete_gamma_upper_argument_derivative
    combined_registry["incomplete_gamma_upper_batch"] = incomplete_gamma_upper_batch
    combined_registry["incomplete_gamma_upper_derivative"] = incomplete_gamma_upper_derivative
    combined_registry["incomplete_gamma_upper_parameter_derivative"] = incomplete_gamma_upper_parameter_derivative
    combined_registry["laplace_bessel_k_tail"] = laplace_bessel_k_tail
    combined_registry["laplace_bessel_k_tail_batch"] = laplace_bessel_k_tail_batch
    combined_registry["laplace_bessel_k_tail_derivative"] = laplace_bessel_k_tail_derivative
    combined_registry["laplace_bessel_k_tail_lambda_derivative"] = laplace_bessel_k_tail_lambda_derivative
    combined_registry["laplace_bessel_k_tail_lower_limit_derivative"] = laplace_bessel_k_tail_lower_limit_derivative
    combined_registry["incomplete_bessel_i"] = incomplete_bessel_i
    combined_registry["incomplete_bessel_i_batch"] = incomplete_bessel_i_batch
    combined_registry["incomplete_bessel_i_derivative"] = incomplete_bessel_i_derivative
    combined_registry["incomplete_bessel_i_argument_derivative"] = incomplete_bessel_i_argument_derivative
    combined_registry["incomplete_bessel_i_upper_limit_derivative"] = incomplete_bessel_i_upper_limit_derivative
    combined_registry["incomplete_bessel_k"] = incomplete_bessel_k
    combined_registry["incomplete_bessel_k_batch"] = incomplete_bessel_k_batch
    combined_registry["incomplete_bessel_k_derivative"] = incomplete_bessel_k_derivative
    combined_registry["incomplete_bessel_k_argument_derivative"] = incomplete_bessel_k_argument_derivative
    combined_registry["incomplete_bessel_k_lower_limit_derivative"] = incomplete_bessel_k_lower_limit_derivative
    return build_public_metadata_registry(
        combined_registry,
        point_names=set(_POINT_FUNC_NAMES),
        interval_names=set(_INTERVAL_FUNC_NAMES),
    )


def _coerce_tail_problem(
    integrand_or_problem,
    *,
    lower_limit: jax.Array | float | None,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    recurrence: TailRatioRecurrence | None = None,
    derivative_metadata: TailDerivativeMetadata | None = None,
    regime_metadata: TailRegimeMetadata | None = None,
) -> TailIntegralProblem:
    if isinstance(integrand_or_problem, TailIntegralProblem):
        return integrand_or_problem
    if lower_limit is None:
        raise ValueError("lower_limit is required when passing a raw integrand.")
    return TailIntegralProblem(
        integrand=integrand_or_problem,
        lower_limit=lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        recurrence=recurrence,
        derivative_metadata=derivative_metadata or TailDerivativeMetadata(),
        regime_metadata=regime_metadata or TailRegimeMetadata(),
    )
