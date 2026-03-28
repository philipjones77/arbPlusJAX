from __future__ import annotations

import jax.numpy as jnp

from . import matrix_free_core
from . import sparse_common
from . import scb_mat


_SPARSE_TYPES = (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)


def is_sparse_input(value) -> bool:
    return isinstance(value, _SPARSE_TYPES)


def operator_plan_prepare(value):
    return scb_mat.scb_mat_operator_plan_prepare(value)


def diagonal_preconditioner_plan_prepare(value):
    diag = jnp.asarray(scb_mat.scb_mat_diag(value), dtype=jnp.complex128)
    eps = jnp.asarray(1e-12, dtype=jnp.float64)
    scale = jnp.where(jnp.abs(diag) > eps, diag, jnp.asarray(eps, dtype=jnp.complex128))
    inv_diag = 1.0 / scale
    return matrix_free_core.diagonal_preconditioner_plan(inv_diag, algebra="jcb")


def _normalize_sparse_operator(value):
    if isinstance(value, matrix_free_core.OperatorPlan):
        if value.kind == "sparse_bcoo":
            return sparse_common.SparseBCOO(
                data=value.payload.data,
                indices=value.payload.indices,
                rows=value.payload.rows,
                cols=value.payload.cols,
                algebra="scb",
            )
        if value.kind == "shell" and is_sparse_input(value.payload.context):
            return value.payload.context
        raise ValueError(f"unsupported operator plan kind for sparse bridge: {value.kind}")
    if is_sparse_input(value):
        return value
    raise ValueError("sparse bridge expects sparse operator input.")


def lu_preconditioner_plan_prepare(value):
    return matrix_free_core.sparse_lu_preconditioner_plan(
        scb_mat.scb_mat_lu_solve_plan_prepare(_normalize_sparse_operator(value)),
        algebra="jcb",
    )


def structured_preconditioner_plan_prepare(value, *, hermitian: bool = True):
    if not hermitian:
        return lu_preconditioner_plan_prepare(value)
    return matrix_free_core.sparse_cholesky_preconditioner_plan(
        scb_mat.scb_mat_hpd_solve_plan_prepare(_normalize_sparse_operator(value)),
        algebra="jcb",
    )


def sparse_operator_or_passthrough(value):
    return operator_plan_prepare(value) if is_sparse_input(value) else value
