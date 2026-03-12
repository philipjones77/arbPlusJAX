from __future__ import annotations

import importlib

__all__ = [
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
    "ball_wrappers",
    "baseline_wrappers",
    "boost_hypgeom",
    "bool_mat",
    "dft",
    "dft_wrappers",
    "dirichlet",
    "dirichlet_wrappers",
    "dlog",
    "cusf_compat",
    "double_interval",
    "double_interval_wrappers",
    "elementary",
    "cubesselk",
    "fmpr",
    "fmpz_extras",
    "fmpzi",
    "function_provenance",
    "hypgeom",
    "double_gamma",
    "hypgeom_wrappers",
    "jcb_mat",
    "jrb_mat",
    "core_wrappers",
    "calc_wrappers",
    "mat_wrappers",
    "poly_wrappers",
    "modular_elliptic_wrappers",
    "wrappers_common",
    "coeffs",
    "mag",
    "mp_mode",
    "point_wrappers",
    "api",
    "partitions",
    "precision",
    "public_metadata",
    "runtime",
    "special",
    "shahen_double_gamma",
    "validation",
]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
