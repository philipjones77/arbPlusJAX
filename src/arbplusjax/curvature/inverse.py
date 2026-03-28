from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import CurvatureOperator
from .. import matrix_free_estimators as _estimators
from .. import sparse_common


def _canonical_basis(dim: int, dtype) -> jnp.ndarray:
    return jnp.eye(dim, dtype=dtype)


def _selected_entries_from_solves(curv_op: CurvatureOperator, index_set) -> jnp.ndarray:
    idx = jnp.asarray(index_set, dtype=jnp.int32)
    cols = idx[:, 1]
    basis = _canonical_basis(curv_op.shape[1], curv_op.dtype)
    solved = jnp.stack([jnp.asarray(curv_op.solve(basis[:, col])) for col in cols], axis=0)
    rows = idx[:, 0]
    return solved[jnp.arange(idx.shape[0]), rows]


def inverse_diagonal_estimate(curv_op: CurvatureOperator, **kwargs):
    return curv_op.inverse_diagonal(**kwargs)


def colored_inverse_diagonal_with_diagnostics(
    curv_op: CurvatureOperator,
    *,
    overlap: int = 0,
    block_size: int = 1,
    correction_probes: int = 0,
    key: jax.Array | None = None,
    distance_k: int | None = None,
):
    sparse_bcoo = curv_op.metadata.get("sparse_bcoo")
    if sparse_bcoo is None:
        value = inverse_diagonal_estimate(curv_op)
        diagnostics = _estimators.ColoredInverseDiagonalDiagnostics(
            distance_k=0,
            color_count=curv_op.shape[0],
            colors=jnp.arange(curv_op.shape[0], dtype=jnp.int32),
            color_sizes=jnp.ones((curv_op.shape[0],), dtype=jnp.int32),
            block_size=int(block_size),
            correction_probes=int(correction_probes),
            stderr=jnp.zeros((curv_op.shape[0],), dtype=jnp.float64),
        )
        return value, diagnostics
    bcoo = sparse_common.as_sparse_bcoo(
        sparse_bcoo,
        algebra=str(curv_op.metadata.get("algebra", getattr(sparse_bcoo, "algebra", ""))),
        label="curvature.colored_inverse_diagonal",
    )
    k = max(1, int(overlap) + 1) if distance_k is None else int(distance_k)
    return _estimators.colored_inverse_diagonal_estimate_from_solve(
        curv_op.solve,
        size=curv_op.shape[0],
        rows=bcoo.indices[:, 0],
        cols=bcoo.indices[:, 1],
        dtype=curv_op.dtype,
        distance_k=k,
        block_size=block_size,
        correction_probes=correction_probes,
        key=key,
    )


def colored_inverse_diagonal_estimate(curv_op: CurvatureOperator, **kwargs):
    value, _ = colored_inverse_diagonal_with_diagnostics(curv_op, **kwargs)
    return value


def selected_inverse(curv_op: CurvatureOperator, *, index_set=None, pattern=None, **kwargs):
    use_colored_diag = curv_op.solve_fn is not None and any(
        name in kwargs for name in ("overlap", "block_size", "correction_probes", "key", "distance_k")
    )
    if index_set is None and pattern is None:
        if use_colored_diag and curv_op.metadata.get("sparse_bcoo") is not None:
            return colored_inverse_diagonal_estimate(curv_op, **kwargs)
        if curv_op.inverse_diagonal_fn is not None:
            return curv_op.inverse_diagonal(**kwargs)
        if curv_op.to_dense_fn is not None:
            return jnp.diag(jnp.linalg.inv(curv_op.to_dense()))
        idx = jnp.arange(curv_op.shape[0], dtype=jnp.int32)
        diag_index = jnp.stack([idx, idx], axis=1)
        return _selected_entries_from_solves(curv_op, diag_index)

    if index_set is not None:
        idx = jnp.asarray(index_set, dtype=jnp.int32)
        if use_colored_diag and curv_op.metadata.get("sparse_bcoo") is not None:
            if bool(jnp.all(idx[:, 0] == idx[:, 1])):
                diag = colored_inverse_diagonal_estimate(curv_op, **kwargs)
                return diag[idx[:, 0]]
        if curv_op.solve_fn is not None:
            return _selected_entries_from_solves(curv_op, idx)
        dense = curv_op.to_dense()
        inv = jnp.linalg.inv(dense)
        return inv[idx[:, 0], idx[:, 1]]

    dense = curv_op.to_dense()
    inv = jnp.linalg.inv(dense)
    return inv if pattern is None else inv


def posterior_marginal_variances(curv_op: CurvatureOperator, **kwargs):
    if (
        curv_op.solve_fn is not None
        and any(name in kwargs for name in ("overlap", "block_size", "correction_probes", "key", "distance_k"))
        and curv_op.metadata.get("sparse_bcoo") is not None
    ):
        return colored_inverse_diagonal_estimate(curv_op, **kwargs)
    if curv_op.inverse_diagonal_fn is not None:
        return inverse_diagonal_estimate(curv_op, **kwargs)
    idx = jnp.arange(curv_op.shape[0], dtype=jnp.int32)
    diag_index = jnp.stack([idx, idx], axis=1)
    return selected_inverse(curv_op, index_set=diag_index, **kwargs)


def covariance_pushforward(curv_op: CurvatureOperator, linear_map, *, output_dim: int | None = None):
    if curv_op.to_dense_fn is not None and not callable(linear_map):
        arr = jnp.asarray(linear_map)
        cov = jnp.linalg.inv(curv_op.to_dense())
        return arr @ cov @ jnp.swapaxes(jnp.conjugate(arr), -1, -2)

    if callable(linear_map):
        if output_dim is None:
            raise ValueError("output_dim is required when linear_map is callable")
        basis = _canonical_basis(curv_op.shape[1], curv_op.dtype)
        arr = jnp.stack([jnp.asarray(linear_map(basis[:, j])) for j in range(curv_op.shape[1])], axis=1)
        pushed_rows = jnp.stack([jnp.asarray(curv_op.solve(jnp.conjugate(arr[i]))) for i in range(arr.shape[0])], axis=0)
        return arr @ jnp.swapaxes(jnp.conjugate(pushed_rows), -1, -2)

    arr = jnp.asarray(linear_map)
    if arr.ndim != 2:
        raise ValueError("linear_map must be rank-2 or callable")
    pushed_cols = jnp.stack([jnp.asarray(curv_op.solve(jnp.conjugate(arr[i]))) for i in range(arr.shape[0])], axis=0)
    return arr @ jnp.swapaxes(jnp.conjugate(pushed_cols), -1, -2)
