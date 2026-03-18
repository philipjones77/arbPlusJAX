from __future__ import annotations

import importlib

from . import precision as _precision

_precision.enable_jax_x64()

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
    "backends",
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
    "iterative_solvers",
    "jcb_mat",
    "jrb_mat",
    "mesh_spectral",
    "matfree_adjoints",
    "scb_block_mat",
    "scb_mat",
    "scb_vblock_mat",
    "srb_block_mat",
    "srb_mat",
    "srb_vblock_mat",
    "sparse_common",
    "core_wrappers",
    "calc_wrappers",
    "capability_registry",
    "mat_wrappers",
    "poly_wrappers",
    "modular_elliptic_wrappers",
    "wrappers_common",
    "coeffs",
    "mag",
    "mp_mode",
    "nufft",
    "point_wrappers",
    "api",
    "partitions",
    "precision",
    "public_metadata",
    "runtime",
    "special",
    "stable_kernels",
    "shahen_double_gamma",
    "transform_common",
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
