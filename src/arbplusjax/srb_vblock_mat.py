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



class SrbVBlockMatPointDiagnostics(NamedTuple):
    storage: str
    rows: int
    cols: int
    nnzb: int
    row_blocks: int
    col_blocks: int
    max_block_rows: int
    max_block_cols: int
    batch_size: int
    method: str
    cached: bool
    direct: bool
    rhs_rank: int


def _as_real_vector(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check_equal(arr.ndim, 1, f"{label}.ndim")
    return arr


def _as_real_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check_equal(arr.ndim, 2, f"{label}.ndim")
    return arr


def _offsets(sizes: jax.Array) -> jax.Array:
    return jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(jnp.asarray(sizes, dtype=jnp.int32))], axis=0)


def _max_sizes(row_block_sizes: jax.Array, col_block_sizes: jax.Array) -> tuple[int, int]:
    return int(jnp.max(row_block_sizes)), int(jnp.max(col_block_sizes))


def _diagnostics(
    x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR | sc.VariableBlockSparseMatvecPlan,
    *,
    method: str,
    batch_size: int = 1,
    cached: bool = False,
    direct: bool = False,
    rhs_rank: int = 1,
) -> SrbVBlockMatPointDiagnostics:
    if isinstance(x, sc.VariableBlockSparseMatvecPlan):
        x = sc.as_variable_block_sparse_matvec_plan(x, algebra="srb", label="srb_vblock_mat.diagnostics")
        nnzb = int(x.payload[0].shape[0])
        row_block_sizes = x.row_block_sizes
        col_block_sizes = x.col_block_sizes
        rows = x.rows
        cols = x.cols
        storage = x.storage
    elif isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.diagnostics")
        nnzb = int(x.data.shape[0])
        row_block_sizes = x.row_block_sizes
        col_block_sizes = x.col_block_sizes
        rows = x.rows
        cols = x.cols
        storage = "vcoo"
    else:
        x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.diagnostics")
        nnzb = int(x.data.shape[0])
        row_block_sizes = x.row_block_sizes
        col_block_sizes = x.col_block_sizes
        rows = x.rows
        cols = x.cols
        storage = "vcsr"
    max_br, max_bc = _max_sizes(row_block_sizes, col_block_sizes)
    return SrbVBlockMatPointDiagnostics(
        storage=storage,
        rows=rows,
        cols=cols,
        nnzb=nnzb,
        row_blocks=int(row_block_sizes.shape[0]),
        col_blocks=int(col_block_sizes.shape[0]),
        max_block_rows=max_br,
        max_block_cols=max_bc,
        batch_size=batch_size,
        method=method,
        cached=cached,
        direct=direct,
        rhs_rank=rhs_rank,
    )


def srb_vblock_mat_shape(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> tuple[int, int]:
    if isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.shape")
        return x.rows, x.cols
    x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.shape")
    return x.rows, x.cols


def srb_vblock_mat_block_sizes(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> tuple[jax.Array, jax.Array]:
    if isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.block_sizes")
        return x.row_block_sizes, x.col_block_sizes
    x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.block_sizes")
    return x.row_block_sizes, x.col_block_sizes


def srb_vblock_mat_nnzb(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> int:
    if isinstance(x, sc.VariableBlockSparseCOO):
        return int(sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.nnzb").data.shape[0])
    return int(sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.nnzb").data.shape[0])


def srb_vblock_mat_coo(
    data: jax.Array,
    row: jax.Array,
    col: jax.Array,
    *,
    row_block_sizes: jax.Array,
    col_block_sizes: jax.Array,
    shape: tuple[int, int],
) -> sc.VariableBlockSparseCOO:
    return sc.VariableBlockSparseCOO(
        data=jnp.asarray(data, dtype=jnp.float64),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        row_block_sizes=jnp.asarray(row_block_sizes, dtype=jnp.int32),
        col_block_sizes=jnp.asarray(col_block_sizes, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_vblock_mat_csr(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    *,
    row_block_sizes: jax.Array,
    col_block_sizes: jax.Array,
    shape: tuple[int, int],
) -> sc.VariableBlockSparseCSR:
    return sc.VariableBlockSparseCSR(
        data=jnp.asarray(data, dtype=jnp.float64),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        row_block_sizes=jnp.asarray(row_block_sizes, dtype=jnp.int32),
        col_block_sizes=jnp.asarray(col_block_sizes, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_vblock_mat_from_dense_coo(a: jax.Array, *, row_block_sizes: jax.Array, col_block_sizes: jax.Array, tol: float = 0.0) -> sc.VariableBlockSparseCOO:
    a = _as_real_matrix(a, "srb_vblock_mat.from_dense_coo")
    row_block_sizes = jnp.asarray(row_block_sizes, dtype=jnp.int32)
    col_block_sizes = jnp.asarray(col_block_sizes, dtype=jnp.int32)
    checks.check_equal(int(jnp.sum(row_block_sizes)), a.shape[0], "srb_vblock_mat.from_dense_coo.row_sum")
    checks.check_equal(int(jnp.sum(col_block_sizes)), a.shape[1], "srb_vblock_mat.from_dense_coo.col_sum")
    row_offsets = _offsets(row_block_sizes)
    col_offsets = _offsets(col_block_sizes)
    max_br, max_bc = _max_sizes(row_block_sizes, col_block_sizes)
    data = []
    rows = []
    cols = []
    for i in range(row_block_sizes.shape[0]):
        rs = int(row_block_sizes[i])
        r0 = int(row_offsets[i])
        for j in range(col_block_sizes.shape[0]):
            cs = int(col_block_sizes[j])
            c0 = int(col_offsets[j])
            block = a[r0 : r0 + rs, c0 : c0 + cs]
            if bool(jnp.any(jnp.abs(block) > tol)):
                padded = jnp.zeros((max_br, max_bc), dtype=a.dtype).at[:rs, :cs].set(block)
                data.append(padded)
                rows.append(i)
                cols.append(j)
    if data:
        data_arr = jnp.stack(data, axis=0)
        row_arr = jnp.asarray(rows, dtype=jnp.int32)
        col_arr = jnp.asarray(cols, dtype=jnp.int32)
    else:
        data_arr = jnp.zeros((0, max_br, max_bc), dtype=jnp.float64)
        row_arr = jnp.zeros((0,), dtype=jnp.int32)
        col_arr = jnp.zeros((0,), dtype=jnp.int32)
    return srb_vblock_mat_coo(data_arr, row_arr, col_arr, row_block_sizes=row_block_sizes, col_block_sizes=col_block_sizes, shape=a.shape)


def srb_vblock_mat_from_dense_csr(a: jax.Array, *, row_block_sizes: jax.Array, col_block_sizes: jax.Array, tol: float = 0.0) -> sc.VariableBlockSparseCSR:
    return srb_vblock_mat_coo_to_csr(srb_vblock_mat_from_dense_coo(a, row_block_sizes=row_block_sizes, col_block_sizes=col_block_sizes, tol=tol))


def srb_vblock_mat_coo_to_csr(x: sc.VariableBlockSparseCOO) -> sc.VariableBlockSparseCSR:
    x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.coo_to_csr")
    ncb = x.col_block_sizes.shape[0]
    key = x.row * ncb + x.col
    order = jnp.argsort(key)
    row = x.row[order]
    col = x.col[order]
    data = x.data[order]
    counts = jnp.bincount(row, length=x.row_block_sizes.shape[0])
    indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
    return srb_vblock_mat_csr(data, col, indptr, row_block_sizes=x.row_block_sizes, col_block_sizes=x.col_block_sizes, shape=(x.rows, x.cols))


def srb_vblock_mat_csr_to_coo(x: sc.VariableBlockSparseCSR) -> sc.VariableBlockSparseCOO:
    x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.csr_to_coo")
    row = sc.csr_row_ids(x.indptr, rows=x.row_block_sizes.shape[0], nnz=x.data.shape[0])
    return srb_vblock_mat_coo(x.data, row, x.indices, row_block_sizes=x.row_block_sizes, col_block_sizes=x.col_block_sizes, shape=(x.rows, x.cols))


def srb_vblock_mat_to_dense(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> jax.Array:
    if isinstance(x, sc.VariableBlockSparseCSR):
        x = srb_vblock_mat_csr_to_coo(x)
    x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.to_dense")
    out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    row_offsets = _offsets(x.row_block_sizes)
    col_offsets = _offsets(x.col_block_sizes)
    for k in range(x.data.shape[0]):
        i = int(x.row[k])
        j = int(x.col[k])
        rs = int(x.row_block_sizes[i])
        cs = int(x.col_block_sizes[j])
        r0 = int(row_offsets[i])
        c0 = int(col_offsets[j])
        out = out.at[r0 : r0 + rs, c0 : c0 + cs].add(x.data[k, :rs, :cs])
    return out


def srb_vblock_mat_transpose(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> sc.VariableBlockSparseCOO:
    if isinstance(x, sc.VariableBlockSparseCSR):
        x = srb_vblock_mat_csr_to_coo(x)
    x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.transpose")
    return srb_vblock_mat_coo(
        jnp.swapaxes(x.data, -1, -2),
        x.col,
        x.row,
        row_block_sizes=x.col_block_sizes,
        col_block_sizes=x.row_block_sizes,
        shape=(x.cols, x.rows),
    )


def _pack_variable_blocks(v: jax.Array, sizes: jax.Array, max_size: int) -> jax.Array:
    sizes = jnp.asarray(sizes, dtype=jnp.int32)
    offsets = _offsets(sizes)[:-1]
    local = jnp.arange(max_size, dtype=jnp.int32)[None, :]
    gather_idx = offsets[:, None] + local
    valid = local < sizes[:, None]
    clipped = jnp.clip(gather_idx, 0, max(v.shape[0] - 1, 0))
    gathered = v[clipped]
    return jnp.where(valid, gathered, jnp.zeros_like(gathered))


def _unpack_variable_blocks(blocks: jax.Array, sizes: jax.Array, total_size: int) -> jax.Array:
    sizes = jnp.asarray(sizes, dtype=jnp.int32)
    offsets = _offsets(sizes)[:-1]
    local = jnp.arange(blocks.shape[1], dtype=jnp.int32)[None, :]
    scatter_idx = offsets[:, None] + local
    valid = local < sizes[:, None]
    out = jnp.zeros((total_size,), dtype=blocks.dtype)
    clipped = jnp.clip(scatter_idx, 0, max(total_size - 1, 0))
    values = jnp.where(valid, blocks, jnp.zeros_like(blocks))
    return out.at[clipped.reshape(-1)].add(values.reshape(-1))


def srb_vblock_mat_matvec(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, v: jax.Array) -> jax.Array:
    if isinstance(x, sc.VariableBlockSparseCSR):
        x = srb_vblock_mat_csr_to_coo(x)
    x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.matvec")
    v = _as_real_vector(v, "srb_vblock_mat.matvec")
    checks.check_equal(x.cols, v.shape[0], "srb_vblock_mat.matvec.inner")
    max_br, max_bc = x.data.shape[1], x.data.shape[2]
    block_vecs = _pack_variable_blocks(v, x.col_block_sizes, max_bc)
    contrib = jnp.einsum("nbc,nc->nb", x.data, block_vecs[x.col])
    block_out = ops.segment_sum(contrib, x.row, num_segments=x.row_block_sizes.shape[0])
    return _unpack_variable_blocks(block_out, x.row_block_sizes, x.rows)


def srb_vblock_mat_rmatvec(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, v: jax.Array) -> jax.Array:
    return srb_vblock_mat_matvec(srb_vblock_mat_transpose(x), v)


def srb_vblock_mat_rmatvec_cached_prepare(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> sc.VariableBlockSparseMatvecPlan:
    return srb_vblock_mat_matvec_cached_prepare(srb_vblock_mat_transpose(x))


def srb_vblock_mat_rmatvec_cached_apply(plan: sc.VariableBlockSparseMatvecPlan, v: jax.Array) -> jax.Array:
    return srb_vblock_mat_matvec_cached_apply(plan, v)


def srb_vblock_mat_matvec_cached_prepare(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> sc.VariableBlockSparseMatvecPlan:
    csr = x if isinstance(x, sc.VariableBlockSparseCSR) else srb_vblock_mat_coo_to_csr(x)
    csr = sc.as_variable_block_sparse_csr(csr, algebra="srb", label="srb_vblock_mat.matvec_cached_prepare")
    row_ids = sc.csr_row_ids(csr.indptr, rows=csr.row_block_sizes.shape[0], nnz=csr.data.shape[0])
    return sc.VariableBlockSparseMatvecPlan(
        storage="vcsr",
        payload=(csr.data, csr.indices, row_ids),
        row_block_sizes=csr.row_block_sizes,
        col_block_sizes=csr.col_block_sizes,
        rows=csr.rows,
        cols=csr.cols,
        algebra="srb",
    )


def srb_vblock_mat_matvec_cached_apply(plan: sc.VariableBlockSparseMatvecPlan, v: jax.Array) -> jax.Array:
    plan = sc.as_variable_block_sparse_matvec_plan(plan, algebra="srb", label="srb_vblock_mat.matvec_cached_apply")
    v = _as_real_vector(v, "srb_vblock_mat.matvec_cached_apply")
    checks.check_equal(plan.cols, v.shape[0], "srb_vblock_mat.matvec_cached_apply.inner")
    data, indices, row_ids = plan.payload
    max_br = int(data.shape[1])
    max_bc = int(data.shape[2])
    block_vecs = _pack_variable_blocks(v, plan.col_block_sizes, max_bc)
    contrib = jnp.einsum("nbc,nc->nb", data, block_vecs[indices])
    block_out = ops.segment_sum(contrib, row_ids, num_segments=plan.row_block_sizes.shape[0])
    return _unpack_variable_blocks(block_out[:, :max_br], plan.row_block_sizes, plan.rows)


def srb_vblock_mat_matmul_dense_rhs(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, b: jax.Array) -> jax.Array:
    if isinstance(x, sc.VariableBlockSparseCSR):
        x = srb_vblock_mat_csr_to_coo(x)
    x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.matmul_dense_rhs")
    b = _as_real_matrix(b, "srb_vblock_mat.matmul_dense_rhs")
    checks.check_equal(x.cols, b.shape[0], "srb_vblock_mat.matmul_dense_rhs.inner")
    return jax.vmap(lambda col: srb_vblock_mat_matvec(x, col), in_axes=1, out_axes=1)(b)


def srb_vblock_mat_matvec_batch_fixed(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.matvec_batch_fixed")
    return jax.vmap(lambda v: srb_vblock_mat_matvec(x, v))(vs)


def srb_vblock_mat_matvec_batch_padded(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.matvec_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_vblock_mat.matvec_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_vblock_mat_matvec_batch_fixed(x, padded)


def srb_vblock_mat_matvec_cached_apply_batch_fixed(plan: sc.VariableBlockSparseMatvecPlan, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.matvec_cached_apply_batch_fixed")
    return jax.vmap(lambda v: srb_vblock_mat_matvec_cached_apply(plan, v))(vs)


def srb_vblock_mat_matvec_cached_apply_batch_padded(plan: sc.VariableBlockSparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.matvec_cached_apply_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_vblock_mat.matvec_cached_apply_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_vblock_mat_matvec_cached_apply_batch_fixed(plan, padded)


def srb_vblock_mat_rmatvec_batch_fixed(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.rmatvec_batch_fixed")
    return jax.vmap(lambda v: srb_vblock_mat_rmatvec(x, v))(vs)


def srb_vblock_mat_rmatvec_batch_padded(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.rmatvec_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_vblock_mat.rmatvec_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_vblock_mat_rmatvec_batch_fixed(x, padded)


def srb_vblock_mat_rmatvec_cached_apply_batch_fixed(plan: sc.VariableBlockSparseMatvecPlan, vs: jax.Array) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.rmatvec_cached_apply_batch_fixed")
    return jax.vmap(lambda v: srb_vblock_mat_rmatvec_cached_apply(plan, v))(vs)


def srb_vblock_mat_rmatvec_cached_apply_batch_padded(plan: sc.VariableBlockSparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_real_matrix(vs, "srb_vblock_mat.rmatvec_cached_apply_batch_padded")
    checks.check(pad_to >= vs.shape[0], "srb_vblock_mat.rmatvec_cached_apply_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return srb_vblock_mat_rmatvec_cached_apply_batch_fixed(plan, padded)


def _vblock_triangular_solve_vector(x: sc.VariableBlockSparseCSR, b: jax.Array, *, lower: bool, unit_diagonal: bool) -> jax.Array:
    row_ids = sc.csr_row_ids(x.indptr, rows=x.row_block_sizes.shape[0], nnz=x.data.shape[0])
    max_br, max_bc = x.data.shape[1], x.data.shape[2]
    checks.check(bool(jnp.array_equal(x.row_block_sizes, x.col_block_sizes)), "srb_vblock_mat.triangular_solve.square_partition")
    rhs_blocks = _pack_variable_blocks(b, x.row_block_sizes, max_br)
    out_blocks = jnp.zeros_like(rhs_blocks)
    order = range(x.row_block_sizes.shape[0]) if lower else range(x.row_block_sizes.shape[0] - 1, -1, -1)
    solved = out_blocks
    for i in order:
        row_mask = row_ids == i
        cols = jnp.where(row_mask, x.indices, -1)
        blocks = jnp.where(row_mask[:, None, None], x.data, 0.0)
        rs = int(x.row_block_sizes[i])
        rhs_block = rhs_blocks[i, :rs]
        accum = jnp.zeros((rs,), dtype=x.data.dtype)
        diag = jnp.eye(rs, dtype=x.data.dtype)
        for k in range(x.data.shape[0]):
            if bool(row_mask[k]):
                j = int(cols[k])
                cs = int(x.col_block_sizes[j])
                block = blocks[k, :rs, :cs]
                if j == i:
                    diag = block if not unit_diagonal else block
                else:
                    accum = accum + block @ solved[j, :cs]
        value = jnp.linalg.solve(diag, rhs_block - accum)
        solved = solved.at[i, :rs].set(value)
    return _unpack_variable_blocks(solved, x.row_block_sizes, x.rows)


def srb_vblock_mat_triangular_solve(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    csr = x if isinstance(x, sc.VariableBlockSparseCSR) else srb_vblock_mat_coo_to_csr(x)
    csr = sc.as_variable_block_sparse_csr(csr, algebra="srb", label="srb_vblock_mat.triangular_solve")
    rhs = jnp.asarray(b, dtype=jnp.float64)
    checks.check(rhs.ndim in (1, 2), "srb_vblock_mat.triangular_solve.ndim")
    checks.check_equal(csr.cols, rhs.shape[0], "srb_vblock_mat.triangular_solve.inner")
    if rhs.ndim == 1:
        return _vblock_triangular_solve_vector(csr, rhs, lower=lower, unit_diagonal=unit_diagonal)
    return jax.vmap(lambda col: _vblock_triangular_solve_vector(csr, col, lower=lower, unit_diagonal=unit_diagonal), in_axes=1, out_axes=1)(rhs)


def srb_vblock_mat_diag(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> jax.Array:
    """Extract diagonal of variable block sparse matrix."""
    return jnp.diag(srb_vblock_mat_to_dense(x))


def srb_vblock_mat_matmul_sparse(
    x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR,
    y: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR,
) -> sc.VariableBlockSparseCOO:
    """Multiply two variable block sparse real matrices."""
    if isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.matmul_sparse.x")
        row_blocks = x.row_block_sizes
        col_blocks = x.col_block_sizes
    else:
        x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.matmul_sparse.x")
        row_blocks = x.row_block_sizes
        col_blocks = x.col_block_sizes
    
    if isinstance(y, sc.VariableBlockSparseCOO):
        y = sc.as_variable_block_sparse_coo(y, algebra="srb", label="srb_vblock_mat.matmul_sparse.y")
    else:
        y = sc.as_variable_block_sparse_csr(y, algebra="srb", label="srb_vblock_mat.matmul_sparse.y")
    
    checks.check_equal(x.cols, y.rows, "srb_vblock_mat.matmul_sparse.inner")
    
    # Convert to dense, multiply, convert back
    x_dense = srb_vblock_mat_to_dense(x)
    y_dense = srb_vblock_mat_to_dense(y)
    result = x_dense @ y_dense
    return srb_vblock_mat_from_dense_coo(result, row_block_sizes=row_blocks, col_block_sizes=y.col_block_sizes)


def srb_vblock_mat_lu(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> tuple[sc.VariableBlockSparseCSR, sc.VariableBlockSparseCSR, sc.VariableBlockSparseCSR]:
    csr = x if isinstance(x, sc.VariableBlockSparseCSR) else srb_vblock_mat_coo_to_csr(x)
    csr = sc.as_variable_block_sparse_csr(csr, algebra="srb", label="srb_vblock_mat.lu")
    lu, _, perm = lax.linalg.lu(srb_vblock_mat_to_dense(csr))
    n = csr.rows
    eye = jnp.eye(n, dtype=jnp.float64)
    p = eye[perm]
    l = jnp.tril(lu, k=-1) + eye
    u = jnp.triu(lu)
    return (
        srb_vblock_mat_from_dense_csr(p, row_block_sizes=csr.row_block_sizes, col_block_sizes=csr.col_block_sizes),
        srb_vblock_mat_from_dense_csr(l, row_block_sizes=csr.row_block_sizes, col_block_sizes=csr.col_block_sizes),
        srb_vblock_mat_from_dense_csr(u, row_block_sizes=csr.row_block_sizes, col_block_sizes=csr.col_block_sizes),
    )


def srb_vblock_mat_lu_solve(lu, b: jax.Array) -> jax.Array:
    p, l, u = lu
    rhs = jnp.asarray(b, dtype=jnp.float64)
    p_dense = srb_vblock_mat_to_dense(p)
    pb = p_dense @ rhs if rhs.ndim == 1 else p_dense @ rhs
    y = srb_vblock_mat_triangular_solve(l, pb, lower=True, unit_diagonal=True)
    return srb_vblock_mat_triangular_solve(u, y, lower=False, unit_diagonal=False)


def srb_vblock_mat_qr(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> tuple[jax.Array, sc.VariableBlockSparseCSR]:
    csr = x if isinstance(x, sc.VariableBlockSparseCSR) else srb_vblock_mat_coo_to_csr(x)
    csr = sc.as_variable_block_sparse_csr(csr, algebra="srb", label="srb_vblock_mat.qr")
    q, r = jnp.linalg.qr(srb_vblock_mat_to_dense(csr))
    r_sparse = srb_vblock_mat_from_dense_csr(r, row_block_sizes=csr.row_block_sizes, col_block_sizes=csr.col_block_sizes)
    return q, r_sparse


def srb_vblock_mat_qr_solve(qr, b: jax.Array) -> jax.Array:
    q, r = qr
    rhs = jnp.asarray(b, dtype=jnp.float64)
    qt_b = q.T @ rhs if rhs.ndim == 1 else q.T @ rhs
    return srb_vblock_mat_triangular_solve(r, qt_b, lower=False, unit_diagonal=False)


def srb_vblock_mat_solve(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, b: jax.Array, *, method: str = "gmres", tol: float = 1e-5, atol: float = 0.0, maxiter: int | None = None, restart: int = 20, x0: jax.Array | None = None, M=None) -> jax.Array:
    rhs = jnp.asarray(b, dtype=jnp.float64)
    checks.check(rhs.ndim in (1, 2), "srb_vblock_mat.solve.ndim")

    if method == "lu":
        return srb_vblock_mat_lu_solve(srb_vblock_mat_lu(x), rhs)
    if method == "qr":
        return srb_vblock_mat_qr_solve(srb_vblock_mat_qr(x), rhs)

    def matvec(v):
        return srb_vblock_mat_matvec(x, v)

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
        return solve_vec(rhs, None if x0 is None else _as_real_vector(x0, "srb_vblock_mat.solve.x0"))
    guess = None if x0 is None else _as_real_matrix(x0, "srb_vblock_mat.solve.x0")
    return jax.vmap(lambda vec, g: solve_vec(vec, g), in_axes=(1, 1 if guess is not None else None), out_axes=1)(rhs, guess if guess is not None else None)


def srb_vblock_mat_solve_batch_fixed(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, bs: jax.Array, **kwargs) -> jax.Array:
    bs = _as_real_matrix(bs, "srb_vblock_mat.solve_batch_fixed")
    return jax.vmap(lambda b: srb_vblock_mat_solve(x, b, **kwargs))(bs)


def srb_vblock_mat_solve_batch_padded(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR, bs: jax.Array, *, pad_to: int, **kwargs) -> jax.Array:
    bs = _as_real_matrix(bs, "srb_vblock_mat.solve_batch_padded")
    checks.check(pad_to >= bs.shape[0], "srb_vblock_mat.solve_batch_padded.pad_to")
    pad_count = int(pad_to - bs.shape[0])
    padded = jnp.concatenate([bs, jnp.repeat(bs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else bs
    return srb_vblock_mat_solve_batch_fixed(x, padded, **kwargs)


def srb_vblock_mat_matvec_with_diagnostics(
    x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR,
    v: jax.Array,
) -> tuple[jax.Array, SrbVBlockMatPointDiagnostics]:
    return srb_vblock_mat_matvec(x, v), _diagnostics(x, method="matvec")


def srb_vblock_mat_matvec_cached_apply_with_diagnostics(
    plan: sc.VariableBlockSparseMatvecPlan,
    v: jax.Array,
) -> tuple[jax.Array, SrbVBlockMatPointDiagnostics]:
    return srb_vblock_mat_matvec_cached_apply(plan, v), _diagnostics(plan, method="matvec_cached", cached=True)


def srb_vblock_mat_solve_with_diagnostics(
    x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR,
    b: jax.Array,
    **kwargs,
) -> tuple[jax.Array, SrbVBlockMatPointDiagnostics]:
    rhs = jnp.asarray(b)
    return srb_vblock_mat_solve(x, b, **kwargs), _diagnostics(x, method=str(kwargs.get("method", "gmres")), direct=False, rhs_rank=int(rhs.ndim))


def srb_vblock_mat_lu_with_diagnostics(
    x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR,
) -> tuple[tuple[sc.VariableBlockSparseCSR, sc.VariableBlockSparseCSR, sc.VariableBlockSparseCSR], SrbVBlockMatPointDiagnostics]:
    return srb_vblock_mat_lu(x), _diagnostics(x, method="lu", direct=True)


def srb_vblock_mat_qr_with_diagnostics(
    x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR,
) -> tuple[tuple[jax.Array, sc.VariableBlockSparseCSR], SrbVBlockMatPointDiagnostics]:
    return srb_vblock_mat_qr(x), _diagnostics(x, method="qr", direct=True)


def srb_vblock_mat_det(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> jax.Array:
    """Determinant using LU decomposition - point version."""
    if isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.det")
        n = x.rows
    else:
        x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.det")
        n = x.rows
    
    P, L, U = srb_vblock_mat_lu(x)
    
    # Determinant is product of U diagonal times sign from permutation
    u_diag = srb_vblock_mat_diag(U)
    det_u = jnp.prod(u_diag)
    
    # For permutation sign, extract from P matrix
    p_dense = srb_vblock_mat_to_dense(P)
    perm = jnp.argmax(p_dense, axis=1).astype(jnp.int32)
    sign = jnp.where(jnp.arange(n) == perm, 1.0, -1.0).prod()
    
    return sign * det_u


def srb_vblock_mat_det_basic(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> jax.Array:
    """Determinant using LU decomposition - basic interval version."""
    return srb_vblock_mat_det(x)


def srb_vblock_mat_inv(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> jax.Array:
    """Matrix inverse - point version using solve."""
    if isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.inv")
        n = x.rows
    else:
        x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.inv")
        n = x.rows
    
    I = jnp.eye(n, dtype=jnp.float64)
    
    def solve_col(col):
        return srb_vblock_mat_solve(x, col, method="gmres", tol=1e-10)
    
    inv_cols = jax.vmap(solve_col, in_axes=1, out_axes=1)(I)
    return inv_cols


def srb_vblock_mat_inv_basic(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> jax.Array:
    """Matrix inverse - basic interval version."""
    return srb_vblock_mat_inv(x)


def srb_vblock_mat_sqr(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> sc.VariableBlockSparseCOO:
    """Matrix square - point version."""
    if isinstance(x, sc.VariableBlockSparseCOO):
        x = sc.as_variable_block_sparse_coo(x, algebra="srb", label="srb_vblock_mat.sqr")
        n = x.rows
    else:
        x = sc.as_variable_block_sparse_csr(x, algebra="srb", label="srb_vblock_mat.sqr")
        n = x.rows
    return srb_vblock_mat_matmul_sparse(x, x)


def srb_vblock_mat_sqr_basic(x: sc.VariableBlockSparseCOO | sc.VariableBlockSparseCSR) -> sc.VariableBlockSparseCOO:
    """Matrix square - basic interval version."""
    return srb_vblock_mat_sqr(x)


@partial(jax.jit, static_argnames=())
def srb_vblock_mat_matvec_jit(x, v: jax.Array) -> jax.Array:
    return srb_vblock_mat_matvec(x, v)


def srb_vblock_mat_operator_plan_prepare(x):
    from . import jrb_mat
    return jrb_mat.jrb_mat_vblock_sparse_operator_plan_prepare(x)


def srb_vblock_mat_operator_rmatvec_plan_prepare(x):
    from . import jrb_mat
    return jrb_mat.jrb_mat_vblock_sparse_operator_rmatvec_plan_prepare(x)


def srb_vblock_mat_operator_adjoint_plan_prepare(x):
    from . import jrb_mat
    return jrb_mat.jrb_mat_vblock_sparse_operator_adjoint_plan_prepare(x)


__all__ = [
    "srb_vblock_mat_shape",
    "srb_vblock_mat_block_sizes",
    "srb_vblock_mat_nnzb",
    "srb_vblock_mat_coo",
    "srb_vblock_mat_csr",
    "srb_vblock_mat_from_dense_coo",
    "srb_vblock_mat_from_dense_csr",
    "srb_vblock_mat_coo_to_csr",
    "srb_vblock_mat_csr_to_coo",
    "srb_vblock_mat_to_dense",
    "srb_vblock_mat_transpose",
    "srb_vblock_mat_diag",
    "srb_vblock_mat_matmul_sparse",
    "srb_vblock_mat_matvec",
    "srb_vblock_mat_rmatvec",
    "srb_vblock_mat_matvec_cached_prepare",
    "srb_vblock_mat_matvec_cached_apply",
    "srb_vblock_mat_rmatvec_cached_prepare",
    "srb_vblock_mat_rmatvec_cached_apply",
    "srb_vblock_mat_matmul_dense_rhs",
    "srb_vblock_mat_matvec_batch_fixed",
    "srb_vblock_mat_matvec_batch_padded",
    "srb_vblock_mat_matvec_cached_apply_batch_fixed",
    "srb_vblock_mat_matvec_cached_apply_batch_padded",
    "srb_vblock_mat_rmatvec_batch_fixed",
    "srb_vblock_mat_rmatvec_batch_padded",
    "srb_vblock_mat_rmatvec_cached_apply_batch_fixed",
    "srb_vblock_mat_rmatvec_cached_apply_batch_padded",
    "srb_vblock_mat_triangular_solve",
    "srb_vblock_mat_lu",
    "srb_vblock_mat_lu_solve",
    "srb_vblock_mat_qr",
    "srb_vblock_mat_qr_solve",
    "srb_vblock_mat_solve",
    "srb_vblock_mat_solve_batch_fixed",
    "srb_vblock_mat_solve_batch_padded",
    "srb_vblock_mat_det",
    "srb_vblock_mat_det_basic",
    "srb_vblock_mat_inv",
    "srb_vblock_mat_inv_basic",
    "srb_vblock_mat_sqr",
    "srb_vblock_mat_sqr_basic",
    "srb_vblock_mat_matvec_with_diagnostics",
    "srb_vblock_mat_matvec_cached_apply_with_diagnostics",
    "srb_vblock_mat_solve_with_diagnostics",
    "srb_vblock_mat_lu_with_diagnostics",
    "srb_vblock_mat_qr_with_diagnostics",
    "srb_vblock_mat_matvec_jit",
    "srb_vblock_mat_operator_plan_prepare",
    "srb_vblock_mat_operator_rmatvec_plan_prepare",
    "srb_vblock_mat_operator_adjoint_plan_prepare",
    "SrbVBlockMatPointDiagnostics",
]
