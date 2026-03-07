from __future__ import annotations

"""Jones complex matrix-function subsystem scaffold.

This module is a separate Jones-labeled subsystem for new complex matrix-function
work. It does not replace `acb_mat`; `acb_mat` remains the canonical
Arb/FLINT-style JAX extension surface for complex box matrices.

Planned scope for `jcb_mat`:
- dense and batched complex-box matrix substrate for matrix functions
- contour-integral matrix logarithm / roots
- rational-Krylov matrix-function actions
- AD-aware matrix-function kernels with repo-standard engineering constraints

Provenance:
- classification: new
- base_names: jcb_mat
- module lineage: Jones matrix-function subsystem for complex box matrices
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "new",
    "base_names": ("jcb_mat",),
    "module_lineage": "Jones matrix-function subsystem for complex box matrices",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}


def jcb_mat_as_box_matrix(a: jax.Array) -> jax.Array:
    """Canonical Jones complex-matrix layout: (..., n, n, 4)."""
    arr = acb_core.as_acb_box(a)
    checks.check(arr.ndim >= 3, "jcb_mat.as_box_matrix.ndim")
    checks.check_equal(arr.shape[-1], 4, "jcb_mat.as_box_matrix.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], "jcb_mat.as_box_matrix.square")
    return arr


def jcb_mat_shape(a: jax.Array) -> tuple[int, ...]:
    arr = jcb_mat_as_box_matrix(a)
    return tuple(int(x) for x in arr.shape)


def jcb_mat_logm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_logm is planned but not implemented yet.")


def jcb_mat_sqrtm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_sqrtm is planned but not implemented yet.")


def jcb_mat_rootm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_rootm is planned but not implemented yet.")


def jcb_mat_signm(*args, **kwargs):
    raise NotImplementedError("jcb_mat_signm is planned but not implemented yet.")


__all__ = [
    "PROVENANCE",
    "jcb_mat_as_box_matrix",
    "jcb_mat_shape",
    "jcb_mat_logm",
    "jcb_mat_sqrtm",
    "jcb_mat_rootm",
    "jcb_mat_signm",
]
