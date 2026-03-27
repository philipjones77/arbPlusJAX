from __future__ import annotations

"""Explicit import-tier policy for runtime startup boundaries.

This module is intentionally lightweight so tests and runtime code can depend on
it without widening the package cold path.
"""

from typing import Final


IMPORT_TIER_POLICY_VERSION: Final[str] = "2026-03-26"
PACKAGE_COLD_MODULE_BUDGET: Final[int] = 1
API_COLD_MODULE_BUDGET: Final[int] = 11
POINT_CORE_FIRST_USE_MODULE_BUDGET: Final[int] = 12
POINT_MATRIX_FIRST_USE_MODULE_BUDGET: Final[int] = 13
TAIL_FIRST_USE_MODULE_BUDGET: Final[int] = 20
POINT_MATRIX_DENSE_FIRST_USE_MODULE_BUDGET: Final[int] = 13
POINT_MATRIX_PLAN_PREPARE_FIRST_USE_MODULE_BUDGET: Final[int] = 14
POINT_MATRIX_PLAN_APPLY_FIRST_USE_MODULE_BUDGET: Final[int] = 14
MATRIX_FREE_OPERATOR_CREATE_FIRST_USE_MODULE_BUDGET: Final[int] = 2
MATRIX_FREE_OPERATOR_APPLY_FIRST_USE_MODULE_BUDGET: Final[int] = 2
MATRIX_FREE_KRYLOV_SOLVE_FIRST_USE_MODULE_BUDGET: Final[int] = 5
MATRIX_FREE_IMPLICIT_ADJOINT_FIRST_USE_MODULE_BUDGET: Final[int] = 5


# Package import should stay close to empty. The root package may initialize
# precision policy, but it must not import numeric/provider families.
PACKAGE_COLD_ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "arbplusjax",
        "arbplusjax.precision",
    }
)


# These modules are allowed to exist after `from arbplusjax import api`.
# They cover routing, dtype/core helpers, public metadata stubs, and point-safe
# special surfaces currently exposed directly by api.py.
API_COLD_ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "arbplusjax",
        "arbplusjax.acb_core",
        "arbplusjax.api",
        "arbplusjax.arb_core",
        "arbplusjax.checks",
        "arbplusjax.double_interval",
        "arbplusjax.elementary",
        "arbplusjax.jax_precision",
        "arbplusjax.kernel_helpers",
        "arbplusjax.lazy_jit",
        "arbplusjax.precision",
        "arbplusjax.public_metadata",
        "arbplusjax.special",
        "arbplusjax.special.bessel",
        "arbplusjax.special.gamma",
        "arbplusjax.special.laplace_bessel",
        "arbplusjax.special.tail_acceleration",
    }
)


# Modules that must stay off the package and api cold paths. They are grouped by
# intended tier so tests can enforce the startup boundary explicitly.
POINT_ON_DEMAND_MODULES: Final[frozenset[str]] = frozenset(
    {
        "arbplusjax.acb_dirichlet",
        "arbplusjax.acb_elliptic",
        "arbplusjax.acb_modular",
        "arbplusjax.barnesg",
        "arbplusjax.bessel_kernels",
        "arbplusjax.coeffs",
        "arbplusjax.core_wrappers",
        "arbplusjax.double_gamma",
        "arbplusjax.hypgeom",
        "arbplusjax.mat_common",
        "arbplusjax.point_wrappers_barnes",
        "arbplusjax.point_wrappers_core",
        "arbplusjax.point_wrappers_hypgeom",
        "arbplusjax.point_wrappers_hypgeom_complex",
        "arbplusjax.point_wrappers_hypgeom_real",
        "arbplusjax.point_wrappers_matrix",
        "arbplusjax.point_wrappers_matrix_dense",
        "arbplusjax.point_wrappers_matrix_plans",
        "arbplusjax.point_wrappers",
        "arbplusjax.sampling_helpers",
        "arbplusjax.series_missing_impl",
        "arbplusjax.series_utils",
        "arbplusjax.special.gamma.barnes_double_gamma_ifj",
        "arbplusjax.wrappers_common",
    }
)


INTERVAL_MODE_ON_DEMAND_MODULES: Final[frozenset[str]] = frozenset(
    {
        "arbplusjax.acb_calc",
        "arbplusjax.acb_mat",
        "arbplusjax.arb_calc",
        "arbplusjax.arb_mat",
        "arbplusjax.baseline_wrappers",
        "arbplusjax.hypgeom_wrappers",
        "arbplusjax.mat_wrappers",
        "arbplusjax.mat_wrappers_dense",
        "arbplusjax.mat_wrappers_plans",
        "arbplusjax.scb_block_mat",
        "arbplusjax.scb_mat",
        "arbplusjax.scb_vblock_mat",
        "arbplusjax.srb_block_mat",
        "arbplusjax.srb_mat",
        "arbplusjax.srb_vblock_mat",
    }
)


PROVIDER_BACKEND_ON_DEMAND_MODULES: Final[frozenset[str]] = frozenset(
    {
        "arbplusjax.boost_hypgeom",
        "arbplusjax.cubesselk",
        "arbplusjax.cusf_compat",
        "arbplusjax.shahen_double_gamma",
    }
)


BENCHMARK_DOCS_ONLY_MODULES: Final[frozenset[str]] = frozenset(
    {
        "arbplusjax.function_provenance",
        "arbplusjax.runtime",
    }
)


API_COLD_FORBIDDEN: Final[frozenset[str]] = frozenset().union(
    POINT_ON_DEMAND_MODULES,
    INTERVAL_MODE_ON_DEMAND_MODULES,
    PROVIDER_BACKEND_ON_DEMAND_MODULES,
    BENCHMARK_DOCS_ONLY_MODULES,
)


def classify_import_tier(module_name: str) -> str | None:
    if module_name in PACKAGE_COLD_ALLOWED:
        return "package_cold_allowed"
    if module_name in API_COLD_ALLOWED:
        return "api_cold_allowed"
    if module_name in POINT_ON_DEMAND_MODULES:
        return "point_on_demand"
    if module_name in INTERVAL_MODE_ON_DEMAND_MODULES:
        return "interval_mode_on_demand"
    if module_name in PROVIDER_BACKEND_ON_DEMAND_MODULES:
        return "provider_backend_on_demand"
    if module_name in BENCHMARK_DOCS_ONLY_MODULES:
        return "benchmark_docs_only"
    return None


__all__ = [
    "API_COLD_ALLOWED",
    "API_COLD_MODULE_BUDGET",
    "API_COLD_FORBIDDEN",
    "BENCHMARK_DOCS_ONLY_MODULES",
    "IMPORT_TIER_POLICY_VERSION",
    "INTERVAL_MODE_ON_DEMAND_MODULES",
    "MATRIX_FREE_IMPLICIT_ADJOINT_FIRST_USE_MODULE_BUDGET",
    "MATRIX_FREE_KRYLOV_SOLVE_FIRST_USE_MODULE_BUDGET",
    "MATRIX_FREE_OPERATOR_APPLY_FIRST_USE_MODULE_BUDGET",
    "MATRIX_FREE_OPERATOR_CREATE_FIRST_USE_MODULE_BUDGET",
    "PACKAGE_COLD_ALLOWED",
    "PACKAGE_COLD_MODULE_BUDGET",
    "POINT_CORE_FIRST_USE_MODULE_BUDGET",
    "POINT_MATRIX_DENSE_FIRST_USE_MODULE_BUDGET",
    "POINT_MATRIX_PLAN_APPLY_FIRST_USE_MODULE_BUDGET",
    "POINT_MATRIX_PLAN_PREPARE_FIRST_USE_MODULE_BUDGET",
    "POINT_MATRIX_FIRST_USE_MODULE_BUDGET",
    "POINT_ON_DEMAND_MODULES",
    "PROVIDER_BACKEND_ON_DEMAND_MODULES",
    "TAIL_FIRST_USE_MODULE_BUDGET",
    "classify_import_tier",
]
