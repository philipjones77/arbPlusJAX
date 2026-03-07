from __future__ import annotations

"""Jones real matrix-function subsystem scaffold.

This module is a separate Jones-labeled subsystem for new matrix-function work.
It does not replace `arb_mat`; `arb_mat` remains the canonical Arb/FLINT-style
JAX extension surface for real interval matrices.

Planned scope for `jrb_mat`:
- dense and batched real matrix substrate for matrix functions
- contour-integral matrix logarithm / roots
- rational-Krylov matrix-function actions
- AD-aware matrix-function kernels with repo-standard engineering constraints

Provenance:
- classification: new
- base_names: jrb_mat
- module lineage: Jones matrix-function subsystem for real interval matrices
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

import jax
import jax.numpy as jnp

from . import checks
from . import double_interval as di

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "new",
    "base_names": ("jrb_mat",),
    "module_lineage": "Jones matrix-function subsystem for real interval matrices",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}


def jrb_mat_as_interval_matrix(a: jax.Array) -> jax.Array:
    """Canonical Jones real-matrix layout: (..., n, n, 2)."""
    arr = di.as_interval(a)
    checks.check(arr.ndim >= 3, "jrb_mat.as_interval_matrix.ndim")
    checks.check_equal(arr.shape[-1], 2, "jrb_mat.as_interval_matrix.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], "jrb_mat.as_interval_matrix.square")
    return arr


def jrb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = jrb_mat_as_interval_matrix(a)
    return tuple(int(x) for x in arr.shape)


def jrb_mat_logm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_logm is planned but not implemented yet.")


def jrb_mat_sqrtm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_sqrtm is planned but not implemented yet.")


def jrb_mat_rootm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_rootm is planned but not implemented yet.")


def jrb_mat_signm(*args, **kwargs):
    raise NotImplementedError("jrb_mat_signm is planned but not implemented yet.")


__all__ = [
    "PROVENANCE",
    "jrb_mat_as_interval_matrix",
    "jrb_mat_shape",
    "jrb_mat_logm",
    "jrb_mat_sqrtm",
    "jrb_mat_rootm",
    "jrb_mat_signm",
]
