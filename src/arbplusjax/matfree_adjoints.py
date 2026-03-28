"""Compatibility facade for matrix-free adjoint helper modules."""

from __future__ import annotations

import importlib


_LAZY_ATTRS = {
    "lanczos_tridiag": (".matfree_adjoints_decompositions", "lanczos_tridiag"),
    "arnoldi_hessenberg": (".matfree_adjoints_decompositions", "arnoldi_hessenberg"),
    "hutchinson_trace_estimator": (".matfree_adjoints_estimators", "hutchinson_trace_estimator"),
    "lanczos_quadrature_spd": (".matfree_adjoints_estimators", "lanczos_quadrature_spd"),
    "cg_fixed_iterations": (".matfree_adjoints_estimators", "cg_fixed_iterations"),
    "low_rank_preconditioner": (".matfree_adjoints_estimators", "low_rank_preconditioner"),
    "partial_cholesky": (".matfree_adjoints_estimators", "partial_cholesky"),
    "partial_cholesky_pivoted": (".matfree_adjoints_estimators", "partial_cholesky_pivoted"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name, __package__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_ATTRS)
