from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
from jax import ops
import jax.numpy as jnp

from . import checks
from . import iterative_solvers
from . import sparse_common as sc



class SrbBlockMatPointDiagnostics(NamedTuple):
    storage: str
    rows: int
    cols: int
    nnzb: int
    block_rows: int
    block_cols: int
    batch_size: int
    method: str
    cached: bool
    direct: bool
    rhs_rank: int


def _diagnostics(
    x: sc.BlockSparseCOO | sc.BlockSparseCSR | sc.BlockSparseMatvecPlan,
    *,
    method: str,
    batch_size: int = 1,
    cached: bool = False,
    direct: bool = False,
    rhs_rank: int = 1,
) -> SrbBlockMatPointDiagnostics:
    if isinstance(x, sc.BlockSparseMatvecPlan):
        x = sc.as_block_sparse_matvec_plan(x, algebra="srb", label="srb_block_mat.diagnostics")
        nnzb = int(x.payload[0].shape[0])
        rows = x.rows
        cols = x.cols
        block_rows = x.block_rows
        block_cols = x.block_cols
        storage = x.storage
    elif isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.diagnostics")
        nnzb = int(x.data.shape[0])
        rows = x.rows
        cols = x.cols
        block_rows = x.block_rows
        block_cols = x.block_cols
        storage = "bcoo"
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.diagnostics")
        nnzb = int(x.data.shape[0])
        rows = x.rows
        cols = x.cols
        block_rows = x.block_rows
        block_cols = x.block_cols
        storage = "bcsr"
    return SrbBlockMatPointDiagnostics(
        storage=storage,
        rows=rows,
        cols=cols,
        nnzb=nnzb,
        block_rows=block_rows,
        block_cols=block_cols,
        batch_size=batch_size,
        method=method,
        cached=cached,
        direct=direct,
        rhs_rank=rhs_rank,
    )


def _as_real_vector(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check_equal(arr.ndim, 1, f"{label}.ndim")
    return arr


def _as_real_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check_equal(arr.ndim, 2, f"{label}.ndim")
    return arr


def srb_block_mat_shape(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> tuple[int, int]:
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.shape")
        return x.rows, x.cols
    x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.shape")
    return x.rows, x.cols


def srb_block_mat_block_shape(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> tuple[int, int]:
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.block_shape")
        return x.block_rows, x.block_cols
    x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.block_shape")
    return x.block_rows, x.block_cols


def srb_block_mat_nnzb(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> int:
    if isinstance(x, sc.BlockSparseCOO):
        return int(sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.nnzb").data.shape[0])
    return int(sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.nnzb").data.shape[0])


def srb_block_mat_coo(data: jax.Array, row: jax.Array, col: jax.Array, *, shape: tuple[int, int], block_shape: tuple[int, int]) -> sc.BlockSparseCOO:
    return sc.BlockSparseCOO(
        data=jnp.asarray(data, dtype=jnp.float64),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        block_rows=int(block_shape[0]),
        block_cols=int(block_shape[1]),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_block_mat_csr(data: jax.Array, indices: jax.Array, indptr: jax.Array, *, shape: tuple[int, int], block_shape: tuple[int, int]) -> sc.BlockSparseCSR:
    return sc.BlockSparseCSR(
        data=jnp.asarray(data, dtype=jnp.float64),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        block_rows=int(block_shape[0]),
        block_cols=int(block_shape[1]),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_block_mat_from_dense_coo(a: jax.Array, *, block_shape: tuple[int, int], tol: float = 0.0) -> sc.BlockSparseCOO:
    a = _as_real_matrix(a, "srb_block_mat.from_dense_coo")
    br, bc = int(block_shape[0]), int(block_shape[1])
    checks.check(a.shape[0] % br == 0, "srb_block_mat.from_dense_coo.row_multiple")
    checks.check(a.shape[1] % bc == 0, "srb_block_mat.from_dense_coo.col_multiple")
    nrb = a.shape[0] // br
    ncb = a.shape[1] // bc
    data = []
    rows = []
    cols = []
    for i in range(nrb):
        for j in range(ncb):
            block = a[i * br : (i + 1) * br, j * bc : (j + 1) * bc]
            if bool(jnp.any(jnp.abs(block) > tol)):
                data.append(block)
                rows.append(i)
                cols.append(j)
    if data:
        data_arr = jnp.stack(data, axis=0)
        row_arr = jnp.asarray(rows, dtype=jnp.int32)
        col_arr = jnp.asarray(cols, dtype=jnp.int32)
    else:
        data_arr = jnp.zeros((0, br, bc), dtype=jnp.float64)
        row_arr = jnp.zeros((0,), dtype=jnp.int32)
        col_arr = jnp.zeros((0,), dtype=jnp.int32)
    return srb_block_mat_coo(data_arr, row_arr, col_arr, shape=a.shape, block_shape=block_shape)


def srb_block_mat_from_dense_csr(a: jax.Array, *, block_shape: tuple[int, int], tol: float = 0.0) -> sc.BlockSparseCSR:
    return srb_block_mat_coo_to_csr(srb_block_mat_from_dense_coo(a, block_shape=block_shape, tol=tol))


def srb_block_mat_coo_to_csr(x: sc.BlockSparseCOO) -> sc.BlockSparseCSR:
    x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.coo_to_csr")
    ncb = x.cols // x.block_cols
    key = x.row * ncb + x.col
    order = jnp.argsort(key)
    row = x.row[order]
    col = x.col[order]
    data = x.data[order]
    counts = jnp.bincount(row, length=x.rows // x.block_rows)
    indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
    return srb_block_mat_csr(data, col, indptr, shape=(x.rows, x.cols), block_shape=(x.block_rows, x.block_cols))


def srb_block_mat_csr_to_coo(x: sc.BlockSparseCSR) -> sc.BlockSparseCOO:
    x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.csr_to_coo")
    row = sc.csr_row_ids(x.indptr, rows=x.rows // x.block_rows, nnz=x.data.shape[0])
    return srb_block_mat_coo(x.data, row, x.indices, shape=(x.rows, x.cols), block_shape=(x.block_rows, x.block_cols))


def srb_block_mat_to_dense(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> jax.Array:
    if isinstance(x, sc.BlockSparseCSR):
        x = srb_block_mat_csr_to_coo(x)
    x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.to_dense")
    out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    br, bc = x.block_rows, x.block_cols
    for k in range(x.data.shape[0]):
        r0 = int(x.row[k]) * br
        c0 = int(x.col[k]) * bc
        out = out.at[r0 : r0 + br, c0 : c0 + bc].add(x.data[k])
    return out


def srb_block_mat_transpose(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> sc.BlockSparseCOO:
    if isinstance(x, sc.BlockSparseCSR):
        x = srb_block_mat_csr_to_coo(x)
    x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.transpose")
    return srb_block_mat_coo(jnp.swapaxes(x.data, -1, -2), x.col, x.row, shape=(x.cols, x.rows), block_shape=(x.block_cols, x.block_rows))


def srb_block_mat_rmatvec(x: sc.BlockSparseCOO | sc.BlockSparseCSR, v: jax.Array) -> jax.Array:
    return srb_block_mat_matvec(srb_block_mat_transpose(x), v)


def srb_block_mat_rmatvec_cached_prepare(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> sc.BlockSparseMatvecPlan:
    return srb_block_mat_matvec_cached_prepare(srb_block_mat_transpose(x))


def srb_block_mat_rmatvec_cached_apply(plan: sc.BlockSparseMatvecPlan, v: jax.Array) -> jax.Array:
    return srb_block_mat_matvec_cached_apply(plan, v)


def srb_block_mat_matvec(x: sc.BlockSparseCOO | sc.BlockSparseCSR, v: jax.Array) -> jax.Array:
    if isinstance(x, sc.BlockSparseCSR):
        x = srb_block_mat_csr_to_coo(x)
    x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.matvec")
    v = _as_real_vector(v, "srb_block_mat.matvec")
    checks.check_equal(x.cols, v.shape[0], "srb_block_mat.matvec.inner")
    br, bc = x.block_rows, x.block_cols
    block_vecs = v.reshape(x.cols // bc, bc)
    contrib = jnp.einsum("nbc,nc->nb", x.data, block_vecs[x.col])
    block_out = ops.segment_sum(contrib, x.row, num_segments=x.rows // br)
    return block_out.reshape(x.rows)


def srb_block_mat_matvec_cached_prepare(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> sc.BlockSparseMatvecPlan:
    csr = x if isinstance(x, sc.BlockSparseCSR) else srb_block_mat_coo_to_csr(x)
    csr = sc.as_block_sparse_csr(csr, algebra="srb", label="srb_block_mat.matvec_cached_prepare")
    row_ids = sc.csr_row_ids(csr.indptr, rows=csr.rows // csr.block_rows, nnz=csr.data.shape[0])
    return sc.BlockSparseMatvecPlan(
        storage="bcsr",
        payload=(csr.data, csr.indices, row_ids),
        block_rows=csr.block_rows,
        block_cols=csr.block_cols,
        rows=csr.rows,
        cols=csr.cols,
        algebra="srb",
    )


def srb_block_mat_matvec_cached_apply(plan: sc.BlockSparseMatvecPlan, v: jax.Array) -> jax.Array:
    plan = sc.as_block_sparse_matvec_plan(plan, algebra="srb", label="srb_block_mat.matvec_cached_apply")
    v = _as_real_vector(v, "srb_block_mat.matvec_cached_apply")
    checks.check_equal(plan.cols, v.shape[0], "srb_block_mat.matvec_cached_apply.inner")
    data, indices, row_ids = plan.payload
    bc = plan.block_cols
    br = plan.block_rows
    block_vecs = v.reshape(plan.cols // bc, bc)
    contrib = jnp.einsum("nbc,nc->nb", data, block_vecs[indices])
    block_out = ops.segment_sum(contrib, row_ids, num_segments=plan.rows // br)
    return block_out.reshape(plan.rows)


def srb_block_mat_matmul_dense_rhs(x: sc.BlockSparseCOO | sc.BlockSparseCSR, b: jax.Array) -> jax.Array:
    if isinstance(x, sc.BlockSparseCSR):
        x = srb_block_mat_csr_to_coo(x)
    x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.matmul_dense_rhs")
    b = _as_real_matrix(b, "srb_block_mat.matmul_dense_rhs")
    checks.check_equal(x.cols, b.shape[0], "srb_block_mat.matmul_dense_rhs.inner")
    br, bc = x.block_rows, x.block_cols
    rhs_cols = b.shape[1]
    block_rhs = b.reshape(x.cols // bc, bc, rhs_cols)
    contrib = jnp.einsum("nbc,nck->nbk", x.data, block_rhs[x.col])
    block_out = ops.segment_sum(contrib, x.row, num_segments=x.rows // br)
    return block_out.reshape(x.rows, rhs_cols)


def srb_block_mat_matvec_batch_fixed(x: sc.BlockSparseCOO | sc.BlockSparseCSR, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.matvec_batch_fixed")
    return jax.vmap(lambda v: srb_block_mat_matvec(x, v))(vs)


def srb_block_mat_matvec_batch_padded(x: sc.BlockSparseCOO | sc.BlockSparseCSR, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.matvec_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_block_mat.matvec_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_block_mat_matvec_batch_fixed(x, padded)


def srb_block_mat_matvec_cached_apply_batch_fixed(plan: sc.BlockSparseMatvecPlan, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.matvec_cached_apply_batch_fixed")
    return jax.vmap(lambda v: srb_block_mat_matvec_cached_apply(plan, v))(vs)


def srb_block_mat_matvec_cached_apply_batch_padded(plan: sc.BlockSparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.matvec_cached_apply_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_block_mat.matvec_cached_apply_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_block_mat_matvec_cached_apply_batch_fixed(plan, padded)


def srb_block_mat_rmatvec_batch_fixed(x: sc.BlockSparseCOO | sc.BlockSparseCSR, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.rmatvec_batch_fixed")
    return jax.vmap(lambda v: srb_block_mat_rmatvec(x, v))(vs)


def srb_block_mat_rmatvec_batch_padded(x: sc.BlockSparseCOO | sc.BlockSparseCSR, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.rmatvec_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_block_mat.rmatvec_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_block_mat_rmatvec_batch_fixed(x, padded)


def srb_block_mat_rmatvec_cached_apply_batch_fixed(plan: sc.BlockSparseMatvecPlan, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.rmatvec_cached_apply_batch_fixed")
    return jax.vmap(lambda v: srb_block_mat_rmatvec_cached_apply(plan, v))(vs)


def srb_block_mat_rmatvec_cached_apply_batch_padded(plan: sc.BlockSparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_block_mat.rmatvec_cached_apply_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_block_mat.rmatvec_cached_apply_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_block_mat_rmatvec_cached_apply_batch_fixed(plan, padded)


def _bcsr_triangular_solve_vector(x: sc.BlockSparseCSR, b: jax.Array, *, lower: bool, unit_diagonal: bool) -> jax.Array:
    br = x.block_rows
    bc = x.block_cols
    checks.check_equal(br, bc, "srb_block_mat.triangular_solve.square_blocks")
    nrb = x.rows // br
    row_ids = sc.csr_row_ids(x.indptr, rows=nrb, nnz=x.data.shape[0])
    order = jnp.arange(nrb, dtype=jnp.int32) if lower else jnp.arange(nrb - 1, -1, -1, dtype=jnp.int32)

    def body(state, i):
        _, out = state
        row_mask = row_ids == i
        cols = jnp.where(row_mask, x.indices, -1)
        blocks = jnp.where(row_mask[:, None, None], x.data, 0.0)
        rhs_block = b.reshape(nrb, br)[i]
        off_mask = row_mask & (cols != i)
        valid_cols = jnp.maximum(cols, 0)
        off_blocks = jnp.where(off_mask[:, None, None], blocks, 0.0)
        accum = jnp.sum(jnp.einsum("nbc,nc->nb", off_blocks, out[valid_cols]), axis=0)
        diag_blocks = jnp.where((row_mask & (cols == i))[:, None, None], blocks, 0.0)
        diag = jnp.eye(br, dtype=x.data.dtype) if unit_diagonal else jnp.sum(diag_blocks, axis=0)
        value = jnp.linalg.solve(diag, rhs_block - accum)
        out = out.at[i].set(value)
        return (i + 1, out), value

    init = (jnp.int32(0), jnp.zeros((nrb, br), dtype=x.data.dtype))
    (_, out), _ = lax.scan(body, init, order)
    return out.reshape(x.rows)


def srb_block_mat_diag(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> jax.Array:
    """Extract diagonal of block sparse matrix."""
    return jnp.diag(srb_block_mat_to_dense(x))


def srb_block_mat_lu(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> tuple[sc.BlockSparseCOO, sc.BlockSparseCSR, sc.BlockSparseCSR]:
    """LU decomposition with partial pivoting for block sparse matrices.
    
    Args:
        x: Block sparse matrix (must be square)
        
    Returns:
        (P, L, U) where P is permutation, L is lower triangular, U is upper triangular
    """
    dense = srb_block_mat_to_dense(x)
    lu, _, perm = lax.linalg.lu(dense)
    n = dense.shape[-1]
    eye = jnp.eye(n, dtype=dense.dtype)
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    
    # Create permutation matrix as block sparse COO
    perm_data = jnp.ones((perm.shape[0],), dtype=jnp.float64)
    perm_row = jnp.arange(perm.shape[0], dtype=jnp.int32)
    
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.lu")
        br, bc = x.block_rows, x.block_cols
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.lu")
        br, bc = x.block_rows, x.block_cols
    
    # Build permutation as regular sparse, then convert
    p_coo = sc.SparseCOO(
        data=perm_data,
        row=perm_row,
        col=perm,
        rows=perm.shape[0],
        cols=perm.shape[0],
        algebra="srb",
    )
    
    # Convert L and U to block sparse
    l_sparse = srb_block_mat_from_dense_csr(jnp.tril(l), block_shape=(br, bc))
    u_sparse = srb_block_mat_from_dense_csr(jnp.triu(u), block_shape=(br, bc))
    
    return p_coo, l_sparse, u_sparse


def srb_block_mat_lu_solve(
    lu: tuple[sc.SparseCOO, sc.BlockSparseCSR, sc.BlockSparseCSR],
    b: jax.Array,
) -> jax.Array:
    p, l, u = lu
    p = sc.as_sparse_coo(p, algebra="srb", label="srb_block_mat.lu_solve.p")
    rhs = jnp.asarray(b, dtype=jnp.float64)
    checks.check(rhs.ndim in (1, 2), "srb_block_mat.lu_solve.ndim")
    pb = rhs[p.col] if rhs.ndim == 1 else rhs[p.col, :]
    y = srb_block_mat_triangular_solve(l, pb, lower=True, unit_diagonal=True)
    return srb_block_mat_triangular_solve(u, y, lower=False, unit_diagonal=False)


def srb_block_mat_qr(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> tuple[jax.Array, sc.BlockSparseCSR]:
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.qr")
        br, bc = x.block_rows, x.block_cols
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.qr")
        br, bc = x.block_rows, x.block_cols
    q, r = jnp.linalg.qr(srb_block_mat_to_dense(x))
    r_sparse = srb_block_mat_from_dense_csr(r, block_shape=(br, bc))
    return q, r_sparse


def srb_block_mat_qr_solve(qr: tuple[jax.Array, sc.BlockSparseCSR], b: jax.Array) -> jax.Array:
    q, r = qr
    rhs = jnp.asarray(b, dtype=jnp.float64)
    qt_b = q.T @ rhs if rhs.ndim == 1 else q.T @ rhs
    return srb_block_mat_triangular_solve(r, qt_b, lower=False, unit_diagonal=False)


def srb_block_mat_matmul_sparse(
    x: sc.BlockSparseCOO | sc.BlockSparseCSR,
    y: sc.BlockSparseCOO | sc.BlockSparseCSR,
) -> sc.BlockSparseCOO:
    """Multiply two block sparse matrices.
    
    Args:
        x: First block sparse matrix
        y: Second block sparse matrix
        
    Returns:
        x @ y as block sparse COO
    """
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.matmul_sparse.x")
        br, bc = x.block_rows, x.block_cols
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.matmul_sparse.x")
        br, bc = x.block_rows, x.block_cols
    
    if isinstance(y, sc.BlockSparseCOO):
        y = sc.as_block_sparse_coo(y, algebra="srb", label="srb_block_mat.matmul_sparse.y")
    else:
        y = sc.as_block_sparse_csr(y, algebra="srb", label="srb_block_mat.matmul_sparse.y")
    
    checks.check_equal(x.cols, y.rows, "srb_block_mat.matmul_sparse.inner")
    
    # Convert to dense, multiply, convert back
    x_dense = srb_block_mat_to_dense(x)
    y_dense = srb_block_mat_to_dense(y)
    result = x_dense @ y_dense
    return srb_block_mat_from_dense_coo(result, block_shape=(br, bc))


def srb_block_mat_triangular_solve(x: sc.BlockSparseCOO | sc.BlockSparseCSR, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    csr = x if isinstance(x, sc.BlockSparseCSR) else srb_block_mat_coo_to_csr(x)
    csr = sc.as_block_sparse_csr(csr, algebra="srb", label="srb_block_mat.triangular_solve")
    rhs = jnp.asarray(b, dtype=jnp.float64)
    checks.check(rhs.ndim in (1, 2), "srb_block_mat.triangular_solve.ndim")
    checks.check_equal(csr.cols, rhs.shape[0], "srb_block_mat.triangular_solve.inner")
    if rhs.ndim == 1:
        return _bcsr_triangular_solve_vector(csr, rhs, lower=lower, unit_diagonal=unit_diagonal)
    return jax.vmap(lambda col: _bcsr_triangular_solve_vector(csr, col, lower=lower, unit_diagonal=unit_diagonal), in_axes=1, out_axes=1)(rhs)


def srb_block_mat_solve(
    x: sc.BlockSparseCOO | sc.BlockSparseCSR,
    b: jax.Array,
    *,
    method: str = "gmres",
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
    x0: jax.Array | None = None,
    M=None,
) -> jax.Array:
    rhs = jnp.asarray(b, dtype=jnp.float64)
    checks.check(rhs.ndim in (1, 2), "srb_block_mat.solve.ndim")
    checks.check_equal(srb_block_mat_shape(x)[1], rhs.shape[0], "srb_block_mat.solve.inner")

    if method == "lu":
        return srb_block_mat_lu_solve(srb_block_mat_lu(x), rhs)
    if method == "qr":
        return srb_block_mat_qr_solve(srb_block_mat_qr(x), rhs)

    def matvec(v):
        return srb_block_mat_matvec(x, v)

    def solve_vec(vec, guess):
        if method == "cg":
            sol, _ = iterative_solvers.cg(matvec, vec, x0=guess, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return sol
        if method == "bicgstab":
            sol, _ = iterative_solvers.bicgstab(matvec, vec, x0=guess, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return sol
        sol, _ = iterative_solvers.gmres(matvec, vec, x0=guess, tol=tol, atol=atol, restart=restart, maxiter=maxiter, M=M)
        return sol

    if rhs.ndim == 1:
        return solve_vec(rhs, None if x0 is None else _as_real_vector(x0, "srb_block_mat.solve.x0"))
    guess = None if x0 is None else _as_real_matrix(x0, "srb_block_mat.solve.x0")
    return jax.vmap(lambda vec, g: solve_vec(vec, g), in_axes=(1, 1 if guess is not None else None), out_axes=1)(rhs, guess if guess is not None else None)


def srb_block_mat_solve_batch_fixed(x: sc.BlockSparseCOO | sc.BlockSparseCSR, bs: jax.Array, **kwargs) -> jax.Array:
    bs = _as_real_matrix(bs, "srb_block_mat.solve_batch_fixed")
    return jax.vmap(lambda b: srb_block_mat_solve(x, b, **kwargs))(bs)


def srb_block_mat_solve_batch_padded(x: sc.BlockSparseCOO | sc.BlockSparseCSR, bs: jax.Array, *, pad_to: int, **kwargs) -> jax.Array:
    bs = _as_real_matrix(bs, "srb_block_mat.solve_batch_padded")
    checks.check(pad_to >= bs.shape[0], "srb_block_mat.solve_batch_padded.pad_to")
    pad_count = int(pad_to - bs.shape[0])
    padded = jnp.concatenate([bs, jnp.repeat(bs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else bs
    return srb_block_mat_solve_batch_fixed(x, padded, **kwargs)


def srb_block_mat_matvec_with_diagnostics(
    x: sc.BlockSparseCOO | sc.BlockSparseCSR,
    v: jax.Array,
) -> tuple[jax.Array, SrbBlockMatPointDiagnostics]:
    return srb_block_mat_matvec(x, v), _diagnostics(x, method="matvec")


def srb_block_mat_matvec_cached_apply_with_diagnostics(
    plan: sc.BlockSparseMatvecPlan,
    v: jax.Array,
) -> tuple[jax.Array, SrbBlockMatPointDiagnostics]:
    return srb_block_mat_matvec_cached_apply(plan, v), _diagnostics(plan, method="matvec_cached", cached=True)


def srb_block_mat_solve_with_diagnostics(
    x: sc.BlockSparseCOO | sc.BlockSparseCSR,
    b: jax.Array,
    **kwargs,
) -> tuple[jax.Array, SrbBlockMatPointDiagnostics]:
    rhs = jnp.asarray(b)
    method = str(kwargs.get("method", "gmres"))
    return srb_block_mat_solve(x, b, **kwargs), _diagnostics(
        x,
        method=method,
        direct=method in {"lu", "qr"},
        rhs_rank=int(rhs.ndim),
    )


def srb_block_mat_det(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> jax.Array:
    """Determinant using LU decomposition - point version.
    
    Args:
        x: Block sparse real matrix (must be square)
        
    Returns:
        Determinant as real scalar
    """
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.det")
        n = x.rows
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.det")
        n = x.rows
    
    P, L, U = srb_block_mat_lu(x)
    
    # Determinant is product of U diagonal times sign from permutation
    u_diag = srb_block_mat_diag(U)
    det_u = jnp.prod(u_diag)
    
    # Compute permutation sign
    perm = P.col
    sign = jnp.where(jnp.arange(n) == perm, 1.0, -1.0).prod()
    
    return sign * det_u


def srb_block_mat_det_basic(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> jax.Array:
    """Determinant using LU decomposition - basic interval version."""
    return srb_block_mat_det(x)


def srb_block_mat_inv(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> jax.Array:
    """Matrix inverse - point version using solve.
    
    Args:
        x: Block sparse real matrix (must be square)
        
    Returns:
        Dense real inverse matrix
    """
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.inv")
        n = x.rows
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.inv")
        n = x.rows
    
    I = jnp.eye(n, dtype=jnp.float64)
    
    def solve_col(col):
        return srb_block_mat_solve(x, col, method="gmres", tol=1e-10)
    
    inv_cols = jax.vmap(solve_col, in_axes=1, out_axes=1)(I)
    return inv_cols


def srb_block_mat_inv_basic(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> jax.Array:
    """Matrix inverse - basic interval version."""
    return srb_block_mat_inv(x)


def srb_block_mat_sqr(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> sc.BlockSparseCOO:
    """Matrix square - point version.
    
    Args:
        x: Block sparse real matrix (must be square)
        
    Returns:
        x @ x as block sparse matrix
    """
    if isinstance(x, sc.BlockSparseCOO):
        x = sc.as_block_sparse_coo(x, algebra="srb", label="srb_block_mat.sqr")
        n = x.rows
    else:
        x = sc.as_block_sparse_csr(x, algebra="srb", label="srb_block_mat.sqr")
        n = x.rows
    return srb_block_mat_matmul_sparse(x, x)


def srb_block_mat_sqr_basic(x: sc.BlockSparseCOO | sc.BlockSparseCSR) -> sc.BlockSparseCOO:
    """Matrix square - basic interval version."""
    return srb_block_mat_sqr(x)


@partial(jax.jit, static_argnames=())
def srb_block_mat_matvec_jit(x, v: jax.Array) -> jax.Array:
    return srb_block_mat_matvec(x, v)


def srb_block_mat_operator_plan_prepare(x):
    from . import jrb_mat
    return jrb_mat.jrb_mat_block_sparse_operator_plan_prepare(x)


def srb_block_mat_operator_rmatvec_plan_prepare(x):
    from . import jrb_mat
    return jrb_mat.jrb_mat_block_sparse_operator_rmatvec_plan_prepare(x)


def srb_block_mat_operator_adjoint_plan_prepare(x):
    from . import jrb_mat
    return jrb_mat.jrb_mat_block_sparse_operator_adjoint_plan_prepare(x)


__all__ = [
    "srb_block_mat_shape",
    "srb_block_mat_block_shape",
    "srb_block_mat_nnzb",
    "srb_block_mat_coo",
    "srb_block_mat_csr",
    "srb_block_mat_from_dense_coo",
    "srb_block_mat_from_dense_csr",
    "srb_block_mat_coo_to_csr",
    "srb_block_mat_csr_to_coo",
    "srb_block_mat_to_dense",
    "srb_block_mat_transpose",
    "srb_block_mat_rmatvec",
    "srb_block_mat_rmatvec_cached_prepare",
    "srb_block_mat_rmatvec_cached_apply",
    "srb_block_mat_diag",
    "srb_block_mat_lu",
    "srb_block_mat_lu_solve",
    "srb_block_mat_qr",
    "srb_block_mat_qr_solve",
    "srb_block_mat_matmul_sparse",
    "srb_block_mat_matvec",
    "srb_block_mat_matvec_cached_prepare",
    "srb_block_mat_matvec_cached_apply",
    "srb_block_mat_matvec_batch_fixed",
    "srb_block_mat_matvec_batch_padded",
    "srb_block_mat_matvec_cached_apply_batch_fixed",
    "srb_block_mat_matvec_cached_apply_batch_padded",
    "srb_block_mat_rmatvec_batch_fixed",
    "srb_block_mat_rmatvec_batch_padded",
    "srb_block_mat_rmatvec_cached_apply_batch_fixed",
    "srb_block_mat_rmatvec_cached_apply_batch_padded",
    "srb_block_mat_matmul_dense_rhs",
    "srb_block_mat_triangular_solve",
    "srb_block_mat_solve",
    "srb_block_mat_solve_batch_fixed",
    "srb_block_mat_solve_batch_padded",
    "srb_block_mat_det",
    "srb_block_mat_det_basic",
    "srb_block_mat_inv",
    "srb_block_mat_inv_basic",
    "srb_block_mat_sqr",
    "srb_block_mat_sqr_basic",
    "srb_block_mat_matvec_with_diagnostics",
    "srb_block_mat_matvec_cached_apply_with_diagnostics",
    "srb_block_mat_solve_with_diagnostics",
    "srb_block_mat_matvec_jit",
    "srb_block_mat_operator_plan_prepare",
    "srb_block_mat_operator_rmatvec_plan_prepare",
    "srb_block_mat_operator_adjoint_plan_prepare",
    "SrbBlockMatPointDiagnostics",
]
