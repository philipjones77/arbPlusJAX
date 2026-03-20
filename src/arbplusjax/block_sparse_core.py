from __future__ import annotations

import jax
from jax import ops
import jax.numpy as jnp

from . import checks
from . import sparse_common as sc


def as_vector(x: jax.Array, *, dtype, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=dtype)
    checks.check_equal(arr.ndim, 1, f"{label}.ndim")
    return arr


def as_matrix(x: jax.Array, *, dtype, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=dtype)
    checks.check_equal(arr.ndim, 2, f"{label}.ndim")
    return arr


def shape(x, *, as_coo, as_csr, algebra: str, label: str) -> tuple[int, int]:
    if isinstance(x, sc.BlockSparseCOO):
        x = as_coo(x, algebra=algebra, label=label)
        return x.rows, x.cols
    x = as_csr(x, algebra=algebra, label=label)
    return x.rows, x.cols


def block_shape(x, *, as_coo, as_csr, algebra: str, label: str) -> tuple[int, int]:
    if isinstance(x, sc.BlockSparseCOO):
        x = as_coo(x, algebra=algebra, label=label)
        return x.block_rows, x.block_cols
    x = as_csr(x, algebra=algebra, label=label)
    return x.block_rows, x.block_cols


def nnzb(x, *, as_coo, as_csr, algebra: str, label: str) -> int:
    if isinstance(x, sc.BlockSparseCOO):
        return int(as_coo(x, algebra=algebra, label=label).data.shape[0])
    return int(as_csr(x, algebra=algebra, label=label).data.shape[0])


def from_dense_coo(a: jax.Array, *, dtype, block_shape: tuple[int, int], tol: float, label: str, coo_ctor):
    a = as_matrix(a, dtype=dtype, label=label)
    br, bc = int(block_shape[0]), int(block_shape[1])
    checks.check(a.shape[0] % br == 0, f"{label}.row_multiple")
    checks.check(a.shape[1] % bc == 0, f"{label}.col_multiple")
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
        data_arr = jnp.zeros((0, br, bc), dtype=dtype)
        row_arr = jnp.zeros((0,), dtype=jnp.int32)
        col_arr = jnp.zeros((0,), dtype=jnp.int32)
    return coo_ctor(data_arr, row_arr, col_arr, shape=a.shape, block_shape=block_shape)


def coo_to_csr(x: sc.BlockSparseCOO, *, as_coo, algebra: str, label: str, csr_ctor):
    x = as_coo(x, algebra=algebra, label=label)
    ncb = x.cols // x.block_cols
    key = x.row * ncb + x.col
    order = jnp.argsort(key)
    row = x.row[order]
    col = x.col[order]
    data = x.data[order]
    counts = jnp.bincount(row, length=x.rows // x.block_rows)
    indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
    return csr_ctor(data, col, indptr, shape=(x.rows, x.cols), block_shape=(x.block_rows, x.block_cols))


def csr_to_coo(x: sc.BlockSparseCSR, *, as_csr, algebra: str, label: str, coo_ctor):
    x = as_csr(x, algebra=algebra, label=label)
    row = sc.csr_row_ids(x.indptr, rows=x.rows // x.block_rows, nnz=x.data.shape[0])
    return coo_ctor(x.data, row, x.indices, shape=(x.rows, x.cols), block_shape=(x.block_rows, x.block_cols))


def to_dense(x, *, as_coo, csr_to_coo_fn, algebra: str, label: str):
    if isinstance(x, sc.BlockSparseCSR):
        x = csr_to_coo_fn(x)
    x = as_coo(x, algebra=algebra, label=label)
    out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    br, bc = x.block_rows, x.block_cols
    for k in range(x.data.shape[0]):
        r0 = int(x.row[k]) * br
        c0 = int(x.col[k]) * bc
        out = out.at[r0 : r0 + br, c0 : c0 + bc].add(x.data[k])
    return out


def transpose(x, *, as_coo, csr_to_coo_fn, algebra: str, label: str, coo_ctor):
    if isinstance(x, sc.BlockSparseCSR):
        x = csr_to_coo_fn(x)
    x = as_coo(x, algebra=algebra, label=label)
    return coo_ctor(
        jnp.swapaxes(x.data, -1, -2),
        x.col,
        x.row,
        shape=(x.cols, x.rows),
        block_shape=(x.block_cols, x.block_rows),
    )


def matvec(x, v: jax.Array, *, dtype, as_coo, csr_to_coo_fn, algebra: str, label: str):
    if isinstance(x, sc.BlockSparseCSR):
        x = csr_to_coo_fn(x)
    x = as_coo(x, algebra=algebra, label=label)
    v = as_vector(v, dtype=dtype, label=label)
    checks.check_equal(x.cols, v.shape[0], f"{label}.inner")
    br, bc = x.block_rows, x.block_cols
    block_vecs = v.reshape(x.cols // bc, bc)
    contrib = jnp.einsum("nbc,nc->nb", x.data, block_vecs[x.col])
    block_out = ops.segment_sum(contrib, x.row, num_segments=x.rows // br)
    return block_out.reshape(x.rows)


def matvec_plan_prepare(x, *, as_csr, coo_to_csr_fn, algebra: str, label: str):
    csr = x if isinstance(x, sc.BlockSparseCSR) else coo_to_csr_fn(x)
    csr = as_csr(csr, algebra=algebra, label=label)
    row_ids = sc.csr_row_ids(csr.indptr, rows=csr.rows // csr.block_rows, nnz=csr.data.shape[0])
    return sc.BlockSparseMatvecPlan(
        storage="bcsr",
        payload=(csr.data, csr.indices, row_ids),
        block_rows=csr.block_rows,
        block_cols=csr.block_cols,
        rows=csr.rows,
        cols=csr.cols,
        algebra=algebra,
    )


def matvec_plan_apply(plan: sc.BlockSparseMatvecPlan, v: jax.Array, *, dtype, algebra: str, label: str):
    plan = sc.as_block_sparse_matvec_plan(plan, algebra=algebra, label=label)
    v = as_vector(v, dtype=dtype, label=label)
    checks.check_equal(plan.cols, v.shape[0], f"{label}.inner")
    data, indices, row_ids = plan.payload
    bc = plan.block_cols
    br = plan.block_rows
    block_vecs = v.reshape(plan.cols // bc, bc)
    contrib = jnp.einsum("nbc,nc->nb", data, block_vecs[indices])
    block_out = ops.segment_sum(contrib, row_ids, num_segments=plan.rows // br)
    return block_out.reshape(plan.rows)


def matmul_dense_rhs(x, b: jax.Array, *, dtype, as_coo, csr_to_coo_fn, algebra: str, label: str):
    if isinstance(x, sc.BlockSparseCSR):
        x = csr_to_coo_fn(x)
    x = as_coo(x, algebra=algebra, label=label)
    b = as_matrix(b, dtype=dtype, label=label)
    checks.check_equal(x.cols, b.shape[0], f"{label}.inner")
    br, bc = x.block_rows, x.block_cols
    rhs_cols = b.shape[1]
    block_rhs = b.reshape(x.cols // bc, bc, rhs_cols)
    contrib = jnp.einsum("nbc,nck->nbk", x.data, block_rhs[x.col])
    block_out = ops.segment_sum(contrib, x.row, num_segments=x.rows // br)
    return block_out.reshape(x.rows, rhs_cols)


def batch_fixed(x: jax.Array, *, dtype, label: str, apply):
    xs = as_matrix(x, dtype=dtype, label=label)
    return jax.vmap(apply)(xs)


def batch_padded(x: jax.Array, *, dtype, pad_to: int, label: str, apply):
    xs = as_matrix(x, dtype=dtype, label=label)
    checks.check(pad_to >= xs.shape[0], f"{label}.pad_to")
    pad_count = int(pad_to - xs.shape[0])
    padded = jnp.concatenate([xs, jnp.repeat(xs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else xs
    return jax.vmap(apply)(padded)


def diag(x, *, to_dense_fn):
    return jnp.diag(to_dense_fn(x))


def matmul_sparse(x, y, *, to_dense_fn, from_dense_coo_fn, block_shape: tuple[int, int]):
    return from_dense_coo_fn(to_dense_fn(x) @ to_dense_fn(y), block_shape=block_shape)
