from __future__ import annotations

import jax.numpy as jnp

from . import matrix_free_core
from . import sparse_common
from . import srb_mat


_SPARSE_TYPES = (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)


def is_sparse_input(value) -> bool:
    return isinstance(value, _SPARSE_TYPES)


def operator_plan_prepare(value):
    return srb_mat.srb_mat_operator_plan_prepare(value)


def diagonal_preconditioner_plan_prepare(value):
    diag = jnp.asarray(srb_mat.srb_mat_diag(value), dtype=jnp.float64)
    eps = jnp.asarray(1e-12, dtype=jnp.float64)
    inv_diag = 1.0 / jnp.where(jnp.abs(diag) > eps, diag, jnp.sign(diag) * eps + (diag == 0.0) * eps)
    return matrix_free_core.diagonal_preconditioner_plan(inv_diag, algebra="jrb")


def _normalize_sparse_operator(value):
    if isinstance(value, matrix_free_core.OperatorPlan):
        if value.kind == "sparse_bcoo":
            return sparse_common.SparseBCOO(
                data=value.payload.data,
                indices=value.payload.indices,
                rows=value.payload.rows,
                cols=value.payload.cols,
                algebra="srb",
            )
        if value.kind == "shell" and is_sparse_input(value.payload.context):
            return value.payload.context
        raise ValueError(f"unsupported operator plan kind for sparse bridge: {value.kind}")
    if is_sparse_input(value):
        return value
    raise ValueError("sparse bridge expects sparse operator input.")


def lu_preconditioner_plan_prepare(value):
    return matrix_free_core.sparse_lu_preconditioner_plan(
        srb_mat.srb_mat_lu_solve_plan_prepare(_normalize_sparse_operator(value)),
        algebra="jrb",
    )


def structured_preconditioner_plan_prepare(value, *, symmetric: bool = True):
    if not symmetric:
        return lu_preconditioner_plan_prepare(value)
    return matrix_free_core.sparse_cholesky_preconditioner_plan(
        srb_mat.srb_mat_spd_solve_plan_prepare(_normalize_sparse_operator(value)),
        algebra="jrb",
    )


def sparse_operator_or_passthrough(value):
    return operator_plan_prepare(value) if is_sparse_input(value) else value
