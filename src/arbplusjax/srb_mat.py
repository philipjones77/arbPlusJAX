from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax import ops

from . import arb_core
from . import arb_mat
from . import checks
from . import double_interval as di
from . import jrb_mat
from . import mat_common
from . import sparse_core
from . import sparse_common as sc
from . import iterative_solvers



class SrbMatPointDiagnostics(NamedTuple):
    storage: str
    rows: int
    cols: int
    nnz: int
    batch_size: int
    method: str
    cached: bool
    direct: bool
    rhs_rank: int


def _diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO | sc.SparseMatvecPlan,
    *,
    method: str,
    batch_size: int = 1,
    cached: bool = False,
    direct: bool = False,
    rhs_rank: int = 1,
) -> SrbMatPointDiagnostics:
    if isinstance(x, sc.SparseMatvecPlan):
        x = sc.as_sparse_matvec_plan(x, algebra="srb", label="srb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.payload.data.shape[0]) if x.storage == "bcoo" else int(x.payload[0].shape[0])
        storage = x.storage
    elif isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.data.shape[0])
        storage = "coo"
    elif isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.data.shape[0])
        storage = "csr"
    else:
        x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.data.shape[0])
        storage = "bcoo"
    return SrbMatPointDiagnostics(storage, rows, cols, nnz, batch_size, method, cached, direct, rhs_rank)


def _as_real_vector(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check_equal(arr.ndim, 1, f"{label}.ndim")
    return arr


def _as_real_matrix(a: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(a, dtype=jnp.float64)
    checks.check_equal(arr.ndim, 2, f"{label}.ndim")
    return arr


def _as_real_rhs(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check(arr.ndim in (1, 2), f"{label}.ndim")
    return arr


def _as_jrb_operator_sparse(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    bcoo = _as_bcoo(x, label="srb_mat.as_jrb_operator_sparse")
    return sc.SparseBCOO(data=bcoo.data, indices=bcoo.indices, rows=bcoo.rows, cols=bcoo.cols, algebra="jrb")


def _dense_interval_matrix(x, label: str) -> jax.Array:
    return sc.sparse_real_to_interval_matrix(x, algebra="srb", label=label)


def _interval_sparse_matrix(x, label: str):
    return sc.sparse_real_to_interval_sparse(x, algebra="srb", label=label)


def _dense_interval_vector(x: jax.Array, label: str) -> jax.Array:
    return mat_common.interval_from_point(_as_real_vector(x, label))


def _dense_interval_rhs(x: jax.Array, label: str) -> jax.Array:
    return mat_common.interval_from_point(_as_real_rhs(x, label))


def srb_mat_shape(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO | sc.SparseMatvecPlan) -> tuple[int, int]:
    if isinstance(x, sc.SparseMatvecPlan):
        plan = sc.as_sparse_matvec_plan(x, algebra="srb", label="srb_mat.shape")
        return plan.rows, plan.cols
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.shape")
        return x.rows, x.cols
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.shape")
        return x.rows, x.cols
    x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.shape")
    return x.rows, x.cols


def srb_mat_nnz(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO | sc.SparseMatvecPlan) -> int:
    if isinstance(x, sc.SparseMatvecPlan):
        plan = sc.as_sparse_matvec_plan(x, algebra="srb", label="srb_mat.nnz")
        if plan.storage == "coo":
            data, _, _ = plan.payload
            return int(data.shape[0])
        if plan.storage == "csr":
            data, _, _ = plan.payload
            return int(data.shape[0])
        return int(plan.payload.data.shape[0])
    if isinstance(x, sc.SparseCOO):
        return int(sc.as_sparse_coo(x, algebra="srb", label="srb_mat.nnz").data.shape[0])
    if isinstance(x, sc.SparseCSR):
        return int(sc.as_sparse_csr(x, algebra="srb", label="srb_mat.nnz").data.shape[0])
    return int(sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.nnz").data.shape[0])


def srb_mat_zero(shape: tuple[int, int]) -> sc.SparseCOO:
    return srb_mat_coo(jnp.zeros((0,), dtype=jnp.float64), jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=jnp.int32), shape=shape)


def srb_mat_identity(n: int, *, dtype: jnp.dtype = jnp.float64) -> sc.SparseCOO:
    idx = jnp.arange(n, dtype=jnp.int32)
    return srb_mat_coo(jnp.ones((n,), dtype=dtype), idx, idx, shape=(n, n))


def srb_mat_permutation_matrix(perm: jax.Array, *, dtype: jnp.dtype = jnp.float64) -> sc.SparseCOO:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    row = jnp.arange(perm.shape[0], dtype=jnp.int32)
    return srb_mat_coo(jnp.ones((perm.shape[0],), dtype=dtype), row, perm, shape=(perm.shape[0], perm.shape[0]))


def srb_mat_coo(data: jax.Array, row: jax.Array, col: jax.Array, *, shape: tuple[int, int]) -> sc.SparseCOO:
    return sc.SparseCOO(
        data=jnp.asarray(data, dtype=jnp.float64),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_mat_csr(data: jax.Array, indices: jax.Array, indptr: jax.Array, *, shape: tuple[int, int]) -> sc.SparseCSR:
    return sc.SparseCSR(
        data=jnp.asarray(data, dtype=jnp.float64),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_mat_bcoo(data: jax.Array, indices: jax.Array, *, shape: tuple[int, int]) -> sc.SparseBCOO:
    return sc.SparseBCOO(
        data=jnp.asarray(data, dtype=jnp.float64),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_mat_interval_coo(data: jax.Array, row: jax.Array, col: jax.Array, *, shape: tuple[int, int]) -> sc.SparseIntervalCOO:
    return sc.SparseIntervalCOO(
        data=di.as_interval(data),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_mat_interval_csr(data: jax.Array, indices: jax.Array, indptr: jax.Array, *, shape: tuple[int, int]) -> sc.SparseIntervalCSR:
    return sc.SparseIntervalCSR(
        data=di.as_interval(data),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_mat_interval_bcoo(data: jax.Array, indices: jax.Array, *, shape: tuple[int, int]) -> sc.SparseIntervalBCOO:
    return sc.SparseIntervalBCOO(
        data=di.as_interval(data),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="srb",
    )


def srb_mat_to_interval_sparse(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return _interval_sparse_matrix(x, "srb_mat.to_interval_sparse")


def srb_mat_interval_to_dense(x: sc.SparseIntervalCOO | sc.SparseIntervalCSR | sc.SparseIntervalBCOO) -> jax.Array:
    return sc.sparse_interval_to_dense(x, algebra="srb", label="srb_mat.interval_to_dense")


def srb_mat_interval_transpose(x: sc.SparseIntervalCOO | sc.SparseIntervalCSR | sc.SparseIntervalBCOO) -> sc.SparseIntervalBCOO:
    return sc.sparse_interval_transpose(x, algebra="srb", label="srb_mat.interval_transpose")


def srb_mat_interval_add(
    x: sc.SparseIntervalCOO | sc.SparseIntervalCSR | sc.SparseIntervalBCOO,
    y: sc.SparseIntervalCOO | sc.SparseIntervalCSR | sc.SparseIntervalBCOO,
) -> sc.SparseIntervalBCOO:
    return sc.sparse_interval_add(x, y, algebra="srb", label="srb_mat.interval_add")


def srb_mat_interval_scale(
    x: sc.SparseIntervalCOO | sc.SparseIntervalCSR | sc.SparseIntervalBCOO,
    alpha,
):
    return sc.sparse_interval_scale(x, alpha, algebra="srb")


def srb_mat_interval_matvec(
    x: sc.SparseIntervalCOO | sc.SparseIntervalCSR | sc.SparseIntervalBCOO,
    v: jax.Array,
) -> jax.Array:
    return sc.sparse_interval_matvec(x, di.as_interval(v), algebra="srb", label="srb_mat.interval_matvec")


def srb_mat_from_dense_coo(a: jax.Array, *, tol: float = 0.0) -> sc.SparseCOO:
    a = _as_real_matrix(a, "srb_mat.from_dense_coo")
    mask = jnp.abs(a) > tol
    row, col = jnp.nonzero(mask, size=int(mask.size), fill_value=-1)
    valid = row >= 0
    data = a[row, col]
    return srb_mat_coo(data[valid], row[valid], col[valid], shape=a.shape)


def srb_mat_from_dense_csr(a: jax.Array, *, tol: float = 0.0) -> sc.SparseCSR:
    coo = srb_mat_from_dense_coo(a, tol=tol)
    return srb_mat_coo_to_csr(coo)


def srb_mat_from_dense_bcoo(a: jax.Array, *, tol: float = 0.0) -> sc.SparseBCOO:
    return sc.dense_to_sparse_bcoo(_as_real_matrix(a, "srb_mat.from_dense_bcoo"), algebra="srb", tol=tol)


def srb_mat_diag(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.diag")
        diag = jnp.zeros((x.rows,), dtype=x.data.dtype)
        mask = x.row == x.col
        return diag.at[x.row[mask]].add(x.data[mask])
    return sparse_core.sparse_diag(x, to_coo_fn=_to_coo_any, dtype=jnp.float64)


def srb_mat_diag_matrix(d: jax.Array) -> sc.SparseCOO:
    d = _as_real_vector(d, "srb_mat.diag_matrix")
    idx = jnp.arange(d.shape[0], dtype=jnp.int32)
    return srb_mat_coo(d, idx, idx, shape=(d.shape[0], d.shape[0]))


def srb_mat_trace(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return jnp.sum(srb_mat_diag(x))


def srb_mat_norm_fro(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.norm_fro")
        return jnp.linalg.norm(x.data)
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.norm_fro")
        return jnp.linalg.norm(x.data)
    x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.norm_fro")
    return jnp.linalg.norm(x.data)


def srb_mat_norm_1(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_norm_1(x, to_coo_fn=_to_coo_any)


def srb_mat_norm_inf(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_norm_inf(x, to_coo_fn=_to_coo_any)


def srb_mat_trace_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_interval_trace(_interval_sparse_matrix(x, "srb_mat.trace_basic"), algebra="srb", label="srb_mat.trace_basic")


def srb_mat_norm_fro_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    ix = _interval_sparse_matrix(x, "srb_mat.norm_fro_basic")
    total = mat_common.interval_sum(di.fast_mul(ix.data, ix.data), axis=0)
    return arb_core.arb_sqrt(total)


def srb_mat_norm_1_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_interval_norm_1(_interval_sparse_matrix(x, "srb_mat.norm_1_basic"), algebra="srb", label="srb_mat.norm_1_basic")


def srb_mat_norm_inf_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_interval_norm_inf(_interval_sparse_matrix(x, "srb_mat.norm_inf_basic"), algebra="srb", label="srb_mat.norm_inf_basic")


def srb_mat_submatrix(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, row_start: int, row_stop: int, col_start: int, col_stop: int) -> sc.SparseCOO:
    coo = x if isinstance(x, sc.SparseCOO) else srb_mat_bcoo_to_coo(x) if isinstance(x, sc.SparseBCOO) else srb_mat_csr_to_coo(x)
    coo = sc.as_sparse_coo(coo, algebra="srb", label="srb_mat.submatrix")
    mask = (coo.row >= row_start) & (coo.row < row_stop) & (coo.col >= col_start) & (coo.col < col_stop)
    return srb_mat_coo(coo.data[mask], coo.row[mask] - row_start, coo.col[mask] - col_start, shape=(row_stop - row_start, col_stop - col_start))


def srb_mat_lu(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[sc.SparseCOO, sc.SparseCSR, sc.SparseCSR]:
    return sparse_core.sparse_lu_via_jax_dense(
        x,
        as_csr_fn=_as_csr,
        to_dense_fn=srb_mat_to_dense,
        from_dense_csr_fn=srb_mat_from_dense_csr,
        permutation_matrix_fn=srb_mat_permutation_matrix,
        complex_=False,
    )


def srb_mat_lu_solve(
    lu: tuple[sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO],
    b: jax.Array,
) -> jax.Array:
    p, l, u = lu
    pb = srb_mat_matvec(p, _as_real_vector(b, "srb_mat.lu_solve")) if jnp.asarray(b).ndim == 1 else srb_mat_matmul_dense_rhs(p, _as_real_matrix(b, "srb_mat.lu_solve"))
    y = srb_mat_triangular_solve(l, pb, lower=True, unit_diagonal=True)
    return srb_mat_triangular_solve(u, y, lower=False, unit_diagonal=False)


def srb_mat_qr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseQRFactor:
    csr = _as_csr(x, label="srb_mat.qr")
    reflectors, taus, r_dense = sparse_core.dense_householder_qr_real(srb_mat_to_dense(csr))
    r_sparse = srb_mat_from_dense_csr(r_dense)
    return sc.SparseQRFactor(reflectors=reflectors, taus=taus, r_factor=r_sparse, rows=csr.rows, cols=csr.cols, algebra="srb")


def srb_mat_qr_r(x: sc.SparseQRFactor) -> sc.SparseCSR:
    factor = sc.as_sparse_qr_factor(x, algebra="srb", label="srb_mat.qr_r")
    return factor.r_factor


def srb_mat_qr_apply_q(x: sc.SparseQRFactor, b: jax.Array, *, transpose: bool = False) -> jax.Array:
    factor = sc.as_sparse_qr_factor(x, algebra="srb", label="srb_mat.qr_apply_q")
    arr = jnp.asarray(b, dtype=jnp.float64)
    checks.check(arr.ndim in (1, 2), "srb_mat.qr_apply_q.ndim")
    if arr.ndim == 1:
        out = arr
        ks = range(factor.taus.shape[0] - 1, -1, -1) if not transpose else range(factor.taus.shape[0])
        for j in ks:
            v = factor.reflectors[:, j]
            tau = factor.taus[j]
            proj = jnp.dot(v, out)
            out = out - tau * v * proj
        return out
    cols = jax.vmap(lambda col: srb_mat_qr_apply_q(factor, col, transpose=transpose), in_axes=1, out_axes=1)(arr)
    return cols


def srb_mat_qr_explicit_q(x: sc.SparseQRFactor) -> jax.Array:
    factor = sc.as_sparse_qr_factor(x, algebra="srb", label="srb_mat.qr_explicit_q")
    return srb_mat_qr_apply_q(factor, jnp.eye(factor.rows, dtype=jnp.float64), transpose=False)


def srb_mat_qr_solve(x: sc.SparseQRFactor, b: jax.Array) -> jax.Array:
    factor = sc.as_sparse_qr_factor(x, algebra="srb", label="srb_mat.qr_solve")
    rhs = jnp.asarray(b, dtype=jnp.float64)
    qt_b = srb_mat_qr_apply_q(factor, rhs, transpose=True)
    r = factor.r_factor
    leading = qt_b[: r.rows] if rhs.ndim == 1 else qt_b[: r.rows, :]
    return srb_mat_triangular_solve(r, leading, lower=False, unit_diagonal=False)


def srb_mat_coo_to_dense(x: sc.SparseCOO) -> jax.Array:
    x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.coo_to_dense")
    out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    return out.at[(x.row, x.col)].add(x.data)


def srb_mat_csr_to_dense(x: sc.SparseCSR) -> jax.Array:
    x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.csr_to_dense")
    row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
    return srb_mat_coo_to_dense(srb_mat_coo(x.data, row, x.indices, shape=(x.rows, x.cols)))


def srb_mat_bcoo_to_dense(x: sc.SparseBCOO) -> jax.Array:
    return sc.sparse_bcoo_to_dense(x, algebra="srb", label="srb_mat.bcoo_to_dense")


def srb_mat_to_dense(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    if isinstance(x, sc.SparseCOO):
        return srb_mat_coo_to_dense(x)
    if isinstance(x, sc.SparseCSR):
        return srb_mat_csr_to_dense(x)
    if isinstance(x, sc.SparseBCOO):
        return srb_mat_bcoo_to_dense(x)
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def srb_mat_coo_to_csr(x: sc.SparseCOO) -> sc.SparseCSR:
    x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.coo_to_csr")
    key = x.row * x.cols + x.col
    order = jnp.argsort(key)
    row = x.row[order]
    col = x.col[order]
    data = x.data[order]
    counts = jnp.bincount(row, length=x.rows)
    indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
    return srb_mat_csr(data, col, indptr, shape=(x.rows, x.cols))


def srb_mat_csr_to_coo(x: sc.SparseCSR) -> sc.SparseCOO:
    x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.csr_to_coo")
    row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
    return srb_mat_coo(x.data, row, x.indices, shape=(x.rows, x.cols))


def srb_mat_coo_to_bcoo(x: sc.SparseCOO) -> sc.SparseBCOO:
    x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.coo_to_bcoo")
    return srb_mat_bcoo(x.data, jnp.stack([x.row, x.col], axis=-1), shape=(x.rows, x.cols))


def srb_mat_csr_to_bcoo(x: sc.SparseCSR) -> sc.SparseBCOO:
    return srb_mat_coo_to_bcoo(srb_mat_csr_to_coo(x))


def srb_mat_bcoo_to_coo(x: sc.SparseBCOO) -> sc.SparseCOO:
    x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.bcoo_to_coo")
    return srb_mat_coo(x.data, x.indices[:, 0], x.indices[:, 1], shape=(x.rows, x.cols))


def _as_bcoo(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, *, label: str) -> sc.SparseBCOO:
    if isinstance(x, sc.SparseBCOO):
        return sc.as_sparse_bcoo(x, algebra="srb", label=label)
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label=label)
        return sc.coo_to_bcoo(x)
    return _as_bcoo(srb_mat_csr_to_coo(x), label=label)


def _as_csr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, *, label: str) -> sc.SparseCSR:
    if isinstance(x, sc.SparseCSR):
        return sc.as_sparse_csr(x, algebra="srb", label=label)
    if isinstance(x, sc.SparseCOO):
        return srb_mat_coo_to_csr(sc.as_sparse_coo(x, algebra="srb", label=label))
    return srb_mat_coo_to_csr(srb_mat_bcoo_to_coo(sc.as_sparse_bcoo(x, algebra="srb", label=label)))


def _to_coo_any(x):
    if isinstance(x, sc.SparseCOO):
        return sc.as_sparse_coo(x, algebra="srb", label="srb_mat._to_coo_any")
    if isinstance(x, sc.SparseCSR):
        return srb_mat_csr_to_coo(x)
    return srb_mat_bcoo_to_coo(x)


def _csr_triangular_solve_vector(x: sc.SparseCSR, b: jax.Array, *, lower: bool, unit_diagonal: bool) -> jax.Array:
    n = x.rows
    nnz = x.data.shape[0]
    rows = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=nnz)
    order = jnp.arange(n, dtype=jnp.int32) if lower else jnp.arange(n - 1, -1, -1, dtype=jnp.int32)

    def body(state, i):
        _, out = state
        row_mask = rows == i
        cols = jnp.where(row_mask, x.indices, -1)
        data = jnp.where(row_mask, x.data, 0.0)
        off_mask = row_mask & (cols != i)
        diag_mask = row_mask & (cols == i)
        accum = jnp.sum(jnp.where(off_mask, data * out[jnp.maximum(cols, 0)], 0.0))
        diag = jnp.where(unit_diagonal, 1.0, jnp.sum(jnp.where(diag_mask, data, 0.0)))
        value = (b[i] - accum) / diag
        out = out.at[i].set(value)
        return (i + 1, out), value

    init = (jnp.int32(0), jnp.zeros_like(b))
    (_, out), _ = lax.scan(body, init, order)
    return out


def srb_mat_transpose(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.transpose")
        return srb_mat_coo(x.data, x.col, x.row, shape=(x.cols, x.rows))
    if isinstance(x, sc.SparseCSR):
        return srb_mat_coo_to_csr(srb_mat_transpose(srb_mat_csr_to_coo(x)))
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.transpose")
        return srb_mat_bcoo(x.data, x.indices[:, ::-1], shape=(x.cols, x.rows))
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def srb_mat_symmetric_part(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCSR:
    return srb_mat_coo_to_csr(sparse_core.sparse_structured_part_real(x, to_coo_fn=_to_coo_any, from_coo_fn=srb_mat_coo))


def srb_mat_is_symmetric(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, *, rtol: float = 1e-10, atol: float = 1e-10) -> jax.Array:
    return sparse_core.sparse_is_symmetric_structural(x, to_coo_fn=_to_coo_any, rtol=rtol, atol=atol)


def srb_mat_is_spd(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_is_pd_structural(
        x,
        to_coo_fn=_to_coo_any,
        structured_part_fn=srb_mat_symmetric_part,
        is_structured_fn=srb_mat_is_symmetric,
        dtype=jnp.float64,
        hermitian=False,
    )


def srb_mat_cho(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCSR:
    data, row, col = sparse_core.sparse_cho_structural(
        x,
        to_coo_fn=_to_coo_any,
        structured_part_fn=srb_mat_symmetric_part,
        is_structured_fn=srb_mat_is_symmetric,
        dtype=jnp.float64,
        hermitian=False,
    )
    return srb_mat_coo_to_csr(srb_mat_coo(data, row, col, shape=srb_mat_shape(x)))


def srb_mat_ldl(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[sc.SparseCSR, jax.Array]:
    data, row, col, d = sparse_core.sparse_ldl_structural(
        x,
        to_coo_fn=_to_coo_any,
        structured_part_fn=srb_mat_symmetric_part,
        is_structured_fn=srb_mat_is_symmetric,
        dtype=jnp.float64,
        hermitian=False,
    )
    return srb_mat_coo_to_csr(srb_mat_coo(data, row, col, shape=srb_mat_shape(x))), d


def srb_mat_symmetric_part_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    ix = _interval_sparse_matrix(x, "srb_mat.symmetric_part_basic")
    sym = sc.sparse_interval_scale(
        sc.sparse_interval_add(ix, sc.sparse_interval_transpose(ix, algebra="srb", label="srb_mat.symmetric_part_basic.transpose"), algebra="srb", label="srb_mat.symmetric_part_basic.add"),
        0.5,
        algebra="srb",
    )
    return sc.sparse_interval_to_dense(sym, algebra="srb", label="srb_mat.symmetric_part_basic.dense")


def srb_mat_is_symmetric_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    dense = sc.sparse_interval_to_dense(_interval_sparse_matrix(x, "srb_mat.is_symmetric_basic"), algebra="srb", label="srb_mat.is_symmetric_basic.dense")
    return mat_common.real_midpoint_is_symmetric(di.midpoint(dense))


def srb_mat_is_spd_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    dense = sc.sparse_interval_to_dense(_interval_sparse_matrix(x, "srb_mat.is_spd_basic"), algebra="srb", label="srb_mat.is_spd_basic.dense")
    mid = di.midpoint(dense)
    chol = jnp.linalg.cholesky(mat_common.real_midpoint_symmetric_part(mid))
    return mat_common.real_midpoint_is_symmetric(mid) & mat_common.lower_cholesky_finite(chol)


def srb_mat_cho_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return arb_mat.arb_mat_cho_basic(_dense_interval_matrix(x, "srb_mat.cho_basic"))


def srb_mat_ldl_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[jax.Array, jax.Array]:
    return arb_mat.arb_mat_ldl_basic(_dense_interval_matrix(x, "srb_mat.ldl_basic"))


def srb_mat_charpoly(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    coeffs = sparse_core.sparse_charpoly_from_traces(
        x,
        to_bcoo_fn=_as_bcoo,
        matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="srb", label="srb_mat.charpoly.matmul"),
        trace_fn=srb_mat_trace,
        identity_sparse_fn=lambda n, dtype: srb_mat_bcoo(
            jnp.ones((n,), dtype=dtype),
            jnp.stack([jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32)], axis=-1),
            shape=(n, n),
        ),
    )
    return jnp.real(coeffs)


def srb_mat_charpoly_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return mat_common.interval_from_point(srb_mat_charpoly(x))


def srb_mat_pow_ui(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, n: int) -> jax.Array:
    return sparse_core.sparse_dense_power_ui(
        x,
        n,
        to_bcoo_fn=_as_bcoo,
        matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="srb", label="srb_mat.pow_ui.matmul"),
        to_dense_fn=srb_mat_to_dense,
        identity_sparse_fn=lambda size, dtype: srb_mat_bcoo(
            jnp.ones((size,), dtype=dtype),
            jnp.stack([jnp.arange(size, dtype=jnp.int32), jnp.arange(size, dtype=jnp.int32)], axis=-1),
            shape=(size, size),
        ),
    )


def srb_mat_pow_ui_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, n: int) -> jax.Array:
    return mat_common.interval_from_point(srb_mat_pow_ui(x, n))


def srb_mat_exp(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_dense_exp_taylor(
        x,
        to_bcoo_fn=_as_bcoo,
        matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="srb", label="srb_mat.exp.matmul"),
        to_dense_fn=srb_mat_to_dense,
        identity_sparse_fn=lambda size, dtype: srb_mat_bcoo(
            jnp.ones((size,), dtype=dtype),
            jnp.stack([jnp.arange(size, dtype=jnp.int32), jnp.arange(size, dtype=jnp.int32)], axis=-1),
            shape=(size, size),
        ),
        terms=24,
    )


def srb_mat_exp_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return mat_common.interval_from_point(srb_mat_exp(x))


def srb_mat_eigvalsh(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return di.midpoint(arb_mat.arb_mat_eigvalsh(_dense_interval_matrix(x, "srb_mat.eigvalsh")))


def srb_mat_eigvalsh_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return arb_mat.arb_mat_eigvalsh(_dense_interval_matrix(x, "srb_mat.eigvalsh_basic"))


def srb_mat_eigh(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[jax.Array, jax.Array]:
    values, vectors = arb_mat.arb_mat_eigh(_dense_interval_matrix(x, "srb_mat.eigh"))
    return di.midpoint(values), di.midpoint(vectors)


def srb_mat_eigh_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[jax.Array, jax.Array]:
    return arb_mat.arb_mat_eigh(_dense_interval_matrix(x, "srb_mat.eigh_basic"))


def srb_mat_eigsh(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    *,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    plan = jrb_mat.jrb_mat_bcoo_operator_plan_prepare(_as_jrb_operator_sparse(x))
    return jrb_mat.jrb_mat_eigsh_point(plan, size=int(x.rows), k=k, which=which, steps=steps, v0=v0)


def srb_mat_eigsh_basic(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    *,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    values, vectors = srb_mat_eigsh(x, k=k, which=which, steps=steps, v0=v0)
    return mat_common.interval_from_point(values), mat_common.interval_from_point(vectors)


def srb_mat_operator_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return jrb_mat.jrb_mat_sparse_operator_plan_prepare(_as_jrb_operator_sparse(x))


def srb_mat_operator_rmatvec_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return jrb_mat.jrb_mat_sparse_operator_rmatvec_plan_prepare(_as_jrb_operator_sparse(x))


def srb_mat_operator_adjoint_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return jrb_mat.jrb_mat_sparse_operator_adjoint_plan_prepare(_as_jrb_operator_sparse(x))


def srb_mat_matvec(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    v = _as_real_vector(v, "srb_mat.matvec")
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.matvec.coo")
        checks.check_equal(x.cols, v.shape[0], "srb_mat.matvec.inner")
        contrib = x.data * v[x.col]
        return ops.segment_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.matvec.csr")
        checks.check_equal(x.cols, v.shape[0], "srb_mat.matvec.inner")
        row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = x.data * v[x.indices]
        return ops.segment_sum(contrib, row, num_segments=x.rows)
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.matvec.bcoo")
        checks.check_equal(x.cols, v.shape[0], "srb_mat.matvec.inner")
        return sc.sparse_bcoo_matvec(x, v, algebra="srb", label="srb_mat.matvec.bcoo")
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def srb_mat_rmatvec(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    return srb_mat_matvec(srb_mat_transpose(x), v)


def srb_mat_matmul_dense_rhs(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    b = _as_real_matrix(b, "srb_mat.matmul_dense_rhs")
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.matmul_dense_rhs.coo")
        checks.check_equal(x.cols, b.shape[0], "srb_mat.matmul_dense_rhs.inner")
        contrib = x.data[:, None] * b[x.col, :]
        return ops.segment_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.matmul_dense_rhs.csr")
        checks.check_equal(x.cols, b.shape[0], "srb_mat.matmul_dense_rhs.inner")
        row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = x.data[:, None] * b[x.indices, :]
        return ops.segment_sum(contrib, row, num_segments=x.rows)
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.matmul_dense_rhs.bcoo")
        checks.check_equal(x.cols, b.shape[0], "srb_mat.matmul_dense_rhs.inner")
        return sc.sparse_bcoo_matmul_dense_rhs(x, b, algebra="srb", label="srb_mat.matmul_dense_rhs.bcoo")
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def srb_mat_to_dense_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_interval_to_dense(_interval_sparse_matrix(x, "srb_mat.to_dense_basic"), algebra="srb", label="srb_mat.to_dense_basic.dense")


def srb_mat_transpose_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    tx = sc.sparse_interval_transpose(_interval_sparse_matrix(x, "srb_mat.transpose_basic"), algebra="srb", label="srb_mat.transpose_basic.transpose")
    return sc.sparse_interval_to_dense(tx, algebra="srb", label="srb_mat.transpose_basic.dense")


def srb_mat_matvec_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    return sc.sparse_interval_matvec(
        _interval_sparse_matrix(x, "srb_mat.matvec_basic"),
        _dense_interval_vector(v, "srb_mat.matvec_basic"),
        algebra="srb",
        label="srb_mat.matvec_basic",
    )


def srb_mat_rmatvec_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    return srb_mat_matvec_basic(srb_mat_transpose(x), v)


def srb_mat_matmul_dense_rhs_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    return sc.sparse_interval_matmul_dense_rhs(
        _interval_sparse_matrix(x, "srb_mat.matmul_dense_rhs_basic"),
        _dense_interval_rhs(b, "srb_mat.matmul_dense_rhs_basic"),
        algebra="srb",
        label="srb_mat.matmul_dense_rhs_basic",
    )


def srb_mat_scale(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, alpha: jax.Array) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="srb", label="srb_mat.scale")
        return srb_mat_coo(x.data * alpha, x.row, x.col, shape=(x.rows, x.cols))
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="srb", label="srb_mat.scale")
        return srb_mat_csr(x.data * alpha, x.indices, x.indptr, shape=(x.rows, x.cols))
    x = sc.as_sparse_bcoo(x, algebra="srb", label="srb_mat.scale")
    return srb_mat_bcoo(x.data * alpha, x.indices, shape=(x.rows, x.cols))


def srb_mat_add(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, y: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    xb = _as_bcoo(x, label="srb_mat.add.x")
    yb = _as_bcoo(y, label="srb_mat.add.y")
    return sc.sparse_bcoo_add(xb, yb, algebra="srb", label="srb_mat.add")


def srb_mat_sub(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, y: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    return srb_mat_add(x, srb_mat_scale(y, -1.0))


def srb_mat_matmul_sparse(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, y: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    return sparse_core.sparse_matmul_sparse(x, y, to_bcoo_fn=_as_bcoo, matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="srb", label="srb_mat.matmul_sparse"))


def srb_mat_triangular_solve(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    x = _as_csr(x, label="srb_mat.triangular_solve")
    b = _as_real_rhs(b, "srb_mat.triangular_solve")
    checks.check_equal(x.cols, b.shape[0], "srb_mat.triangular_solve.inner")
    if b.ndim == 1:
        return _csr_triangular_solve_vector(x, b, lower=lower, unit_diagonal=unit_diagonal)
    return jax.vmap(lambda col: _csr_triangular_solve_vector(x, col, lower=lower, unit_diagonal=unit_diagonal), in_axes=1, out_axes=1)(b)


def srb_mat_triangular_solve_basic(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return arb_mat.arb_mat_triangular_solve_basic(
        _dense_interval_matrix(x, "srb_mat.triangular_solve_basic"),
        _dense_interval_rhs(b, "srb_mat.triangular_solve_basic"),
        lower=lower,
        unit_diagonal=unit_diagonal,
    )


def srb_mat_solve(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
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
    x = _as_bcoo(x, label="srb_mat.solve")
    b = _as_real_rhs(b, "srb_mat.solve")
    checks.check_equal(x.cols, b.shape[0], "srb_mat.solve.inner")
    if x.rows <= 32:
        return srb_mat_lu_solve_plan_apply(srb_mat_lu_solve_plan_prepare(x), b)

    def matvec(v):
        return sc.sparse_bcoo_matvec(x, v, algebra="srb", label="srb_mat.solve.apply")

    def solve_vec(rhs, guess):
        if method == "cg":
            sol, _ = iterative_solvers.cg(matvec, rhs, x0=guess, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return sol
        if method == "bicgstab":
            sol, _ = iterative_solvers.bicgstab(matvec, rhs, x0=guess, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return sol
        sol, _ = iterative_solvers.gmres(matvec, rhs, x0=guess, tol=tol, atol=atol, restart=restart, maxiter=maxiter, M=M)
        return sol

    if b.ndim == 1:
        return solve_vec(b, None if x0 is None else jnp.asarray(x0, dtype=jnp.float64))
    guess = None if x0 is None else _as_real_matrix(x0, "srb_mat.solve.x0")
    return jax.vmap(lambda rhs, g: solve_vec(rhs, g), in_axes=(1, 1 if guess is not None else None), out_axes=1)(
        b,
        guess if guess is not None else None,
    )


def srb_mat_solve_basic(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    **kwargs,
) -> jax.Array:
    del kwargs
    return arb_mat.arb_mat_solve_basic(_dense_interval_matrix(x, "srb_mat.solve_basic"), _dense_interval_rhs(b, "srb_mat.solve_basic"))


def srb_mat_lu_solve_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseLUSolvePlan:
    p, l, u = srb_mat_lu(x)
    return sc.sparse_lu_solve_plan_from_factors(p, l, u, algebra="srb")


def srb_mat_lu_solve_plan_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return srb_mat_lu_solve_plan_prepare(x)


def srb_mat_lu_solve_plan_apply(plan: sc.SparseLUSolvePlan | tuple, b: jax.Array) -> jax.Array:
    plan = sc.as_sparse_lu_solve_plan(plan, algebra="srb", label="srb_mat.lu_solve_plan_apply")
    return srb_mat_lu_solve((plan.p, plan.l, plan.u), b)


def srb_mat_lu_solve_plan_apply_basic(plan, b: jax.Array) -> jax.Array:
    if isinstance(plan, (sc.SparseLUSolvePlan, tuple)):
        return mat_common.interval_from_point(srb_mat_lu_solve_plan_apply(plan, _as_real_rhs(b, "srb_mat.lu_solve_plan_apply_basic")))
    return arb_mat.arb_mat_lu_solve_basic(plan, _dense_interval_rhs(b, "srb_mat.lu_solve_plan_apply_basic"))


def srb_mat_spd_solve_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCholeskySolvePlan:
    return sc.sparse_cholesky_solve_plan_from_factor(srb_mat_cho(x), algebra="srb", structure="spd")


def srb_mat_spd_solve_plan_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return srb_mat_spd_solve_plan_prepare(x)


def srb_mat_spd_solve_plan_apply(plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    plan = sc.as_sparse_cholesky_solve_plan(plan, algebra="srb", structure="spd", label="srb_mat.spd_solve_plan_apply")
    factor = _as_csr(plan.factor, label="srb_mat.spd_solve_plan_apply.factor")
    y = srb_mat_triangular_solve(factor, b, lower=True, unit_diagonal=False)
    return srb_mat_triangular_solve(srb_mat_transpose(factor), y, lower=False, unit_diagonal=False)


def srb_mat_spd_solve_plan_apply_basic(plan, b: jax.Array) -> jax.Array:
    if isinstance(plan, (sc.SparseCholeskySolvePlan, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_spd_solve_plan_apply(plan, _as_real_rhs(b, "srb_mat.spd_solve_plan_apply_basic")))
    return arb_mat.arb_mat_spd_solve_basic(plan, _dense_interval_rhs(b, "srb_mat.spd_solve_plan_apply_basic"))


def srb_mat_spd_solve(x_or_plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, sc.SparseCholeskySolvePlan):
        return srb_mat_spd_solve_plan_apply(x_or_plan, b)
    return srb_mat_spd_solve_plan_apply(srb_mat_spd_solve_plan_prepare(x_or_plan), b)


def srb_mat_spd_solve_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_spd_solve(x_or_plan, _as_real_rhs(b, "srb_mat.spd_solve_basic")))
    return arb_mat.arb_mat_spd_solve_basic(x_or_plan, _dense_interval_rhs(b, "srb_mat.spd_solve_basic"))


def srb_mat_spd_inv(x_or_plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    rows, _ = sc.sparse_shape(x_or_plan, algebra="srb", label="srb_mat.spd_inv")
    eye = jnp.eye(rows, dtype=jnp.float64)
    return srb_mat_spd_solve(x_or_plan, eye)


def srb_mat_spd_inv_basic(x_or_plan) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_spd_inv(x_or_plan))
    return arb_mat.arb_mat_spd_inv_basic(x_or_plan)


def srb_mat_solve_lu(x_or_plan: sc.SparseLUSolvePlan | tuple | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple)):
        return srb_mat_lu_solve_plan_apply(x_or_plan, b)
    return srb_mat_lu_solve_plan_apply(srb_mat_lu_solve_plan_prepare(x_or_plan), b)


def srb_mat_solve_lu_precomp(plan: sc.SparseLUSolvePlan | tuple, b: jax.Array) -> jax.Array:
    return srb_mat_lu_solve_plan_apply(plan, b)


def srb_mat_solve_lu_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_solve_lu(x_or_plan, _as_real_rhs(b, "srb_mat.solve_lu_basic")))
    return arb_mat.arb_mat_solve_lu(x_or_plan, _dense_interval_rhs(b, "srb_mat.solve_lu_basic"))


def srb_mat_solve_lu_precomp_basic(plan, b: jax.Array) -> jax.Array:
    if isinstance(plan, (sc.SparseLUSolvePlan, tuple)):
        return mat_common.interval_from_point(srb_mat_solve_lu_precomp(plan, _as_real_rhs(b, "srb_mat.solve_lu_precomp_basic")))
    return arb_mat.arb_mat_solve_lu_precomp(plan, _dense_interval_rhs(b, "srb_mat.solve_lu_precomp_basic"))


def srb_mat_solve_transpose(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, sc.SparseCholeskySolvePlan):
        return srb_mat_spd_solve_plan_apply(x_or_plan, b)
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple)):
        plan = sc.as_sparse_lu_solve_plan(x_or_plan, algebra="srb", label="srb_mat.solve_transpose")
        pb = srb_mat_matvec(srb_mat_transpose(plan.p), _as_real_vector(b, "srb_mat.solve_transpose")) if jnp.asarray(b).ndim == 1 else srb_mat_matmul_dense_rhs(srb_mat_transpose(plan.p), _as_real_matrix(b, "srb_mat.solve_transpose"))
        y = srb_mat_triangular_solve(srb_mat_transpose(plan.u), pb, lower=True, unit_diagonal=False)
        return srb_mat_triangular_solve(srb_mat_transpose(plan.l), y, lower=False, unit_diagonal=True)
    if bool(srb_mat_is_spd(x_or_plan)):
        return srb_mat_spd_solve(x_or_plan, b)
    return srb_mat_solve(srb_mat_transpose(x_or_plan), b)


def srb_mat_solve_transpose_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_solve_transpose(x_or_plan, _as_real_rhs(b, "srb_mat.solve_transpose_basic")))
    return arb_mat.arb_mat_solve_transpose(x_or_plan, _dense_interval_rhs(b, "srb_mat.solve_transpose_basic"))


def srb_mat_solve_add(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.asarray(y, dtype=jnp.float64) + srb_mat_mat_solve(x_or_plan, b)


def srb_mat_solve_add_basic(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(
            srb_mat_solve_add(
                x_or_plan,
                _as_real_rhs(b, "srb_mat.solve_add_basic"),
                _as_real_rhs(y, "srb_mat.solve_add_basic.y"),
            )
        )
    return arb_mat.arb_mat_solve_add(
        _dense_interval_matrix(x_or_plan, "srb_mat.solve_add_basic") if isinstance(x_or_plan, (sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)) else x_or_plan,
        _dense_interval_rhs(b, "srb_mat.solve_add_basic"),
        _dense_interval_rhs(y, "srb_mat.solve_add_basic.y"),
    )


def srb_mat_solve_transpose_add(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.asarray(y, dtype=jnp.float64) + srb_mat_solve_transpose(x_or_plan, b)


def srb_mat_solve_transpose_add_basic(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(
            srb_mat_solve_transpose_add(
                x_or_plan,
                _as_real_rhs(b, "srb_mat.solve_transpose_add_basic"),
                _as_real_rhs(y, "srb_mat.solve_transpose_add_basic.y"),
            )
        )
    return arb_mat.arb_mat_solve_transpose_add(
        _dense_interval_matrix(x_or_plan, "srb_mat.solve_transpose_add_basic") if isinstance(x_or_plan, (sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)) else x_or_plan,
        _dense_interval_rhs(b, "srb_mat.solve_transpose_add_basic"),
        _dense_interval_rhs(y, "srb_mat.solve_transpose_add_basic.y"),
    )


def srb_mat_mat_solve(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, sc.SparseCholeskySolvePlan):
        return srb_mat_spd_solve_plan_apply(x_or_plan, b)
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple)):
        return srb_mat_lu_solve_plan_apply(x_or_plan, b)
    if bool(srb_mat_is_spd(x_or_plan)):
        return srb_mat_spd_solve(x_or_plan, b)
    return srb_mat_solve(x_or_plan, b)


def srb_mat_mat_solve_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_mat_solve(x_or_plan, _as_real_rhs(b, "srb_mat.mat_solve_basic")))
    return arb_mat.arb_mat_mat_solve(x_or_plan, _dense_interval_rhs(b, "srb_mat.mat_solve_basic"))


def srb_mat_mat_solve_transpose(x_or_plan, b: jax.Array) -> jax.Array:
    return srb_mat_solve_transpose(x_or_plan, b)


def srb_mat_mat_solve_transpose_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.interval_from_point(srb_mat_mat_solve_transpose(x_or_plan, _as_real_rhs(b, "srb_mat.mat_solve_transpose_basic")))
    return arb_mat.arb_mat_mat_solve_transpose(x_or_plan, _dense_interval_rhs(b, "srb_mat.mat_solve_transpose_basic"))


def srb_mat_solve_batch_fixed(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    method: str = "gmres",
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_real_matrix,
        label="srb_mat.solve_batch_fixed",
        apply=lambda b: srb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart),
    )


def srb_mat_solve_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, **kwargs) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_real_matrix, label="srb_mat.solve_basic_batch_fixed", apply=lambda b: srb_mat_solve_basic(x, b, **kwargs))


def srb_mat_solve_batch_padded(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    pad_to: int,
    method: str = "gmres",
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
) -> jax.Array:
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_real_matrix,
        label="srb_mat.solve_batch_padded",
        apply=lambda b: srb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart),
    )


def srb_mat_solve_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, *, pad_to: int, **kwargs) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.solve_basic_batch_padded", apply=lambda b: srb_mat_solve_basic(x, b, **kwargs))


def srb_mat_triangular_solve_batch_fixed(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_real_matrix,
        label="srb_mat.triangular_solve_batch_fixed",
        apply=lambda b: srb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal),
    )


def srb_mat_triangular_solve_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_real_matrix, label="srb_mat.triangular_solve_basic_batch_fixed", apply=lambda b: srb_mat_triangular_solve_basic(x, b, lower=lower, unit_diagonal=unit_diagonal))


def srb_mat_triangular_solve_batch_padded(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    pad_to: int,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_real_matrix,
        label="srb_mat.triangular_solve_batch_padded",
        apply=lambda b: srb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal),
    )


def srb_mat_triangular_solve_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, *, pad_to: int, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.triangular_solve_basic_batch_padded", apply=lambda b: srb_mat_triangular_solve_basic(x, b, lower=lower, unit_diagonal=unit_diagonal))


def srb_mat_lu_solve_plan_apply_batch_fixed(plan: sc.SparseLUSolvePlan | tuple, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_real_matrix,
        label="srb_mat.lu_solve_plan_apply_batch_fixed",
        apply=lambda b: srb_mat_lu_solve_plan_apply(plan, b),
    )


def srb_mat_lu_solve_plan_apply_basic_batch_fixed(plan, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_real_matrix, label="srb_mat.lu_solve_plan_apply_basic_batch_fixed", apply=lambda b: srb_mat_lu_solve_plan_apply_basic(plan, b))


def srb_mat_lu_solve_plan_apply_batch_padded(plan: sc.SparseLUSolvePlan | tuple, bs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_real_matrix,
        label="srb_mat.lu_solve_plan_apply_batch_padded",
        apply=lambda b: srb_mat_lu_solve_plan_apply(plan, b),
    )


def srb_mat_lu_solve_plan_apply_basic_batch_padded(plan, bs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.lu_solve_plan_apply_basic_batch_padded", apply=lambda b: srb_mat_lu_solve_plan_apply_basic(plan, b))


def srb_mat_spd_solve_plan_apply_batch_fixed(plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_real_matrix,
        label="srb_mat.spd_solve_plan_apply_batch_fixed",
        apply=lambda b: srb_mat_spd_solve_plan_apply(plan, b),
    )


def srb_mat_spd_solve_plan_apply_basic_batch_fixed(plan, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_real_matrix, label="srb_mat.spd_solve_plan_apply_basic_batch_fixed", apply=lambda b: srb_mat_spd_solve_plan_apply_basic(plan, b))


def srb_mat_spd_solve_plan_apply_batch_padded(
    plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_real_matrix,
        label="srb_mat.spd_solve_plan_apply_batch_padded",
        apply=lambda b: srb_mat_spd_solve_plan_apply(plan, b),
    )


def srb_mat_spd_solve_plan_apply_basic_batch_padded(plan, bs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.spd_solve_plan_apply_basic_batch_padded", apply=lambda b: srb_mat_spd_solve_plan_apply_basic(plan, b))


def srb_mat_matvec_cached_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseMatvecPlan:
    return sc.sparse_matvec_plan_from_sparse(x, algebra="srb", label="srb_mat.matvec_cached_prepare")


def srb_mat_rmatvec_cached_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseMatvecPlan:
    return srb_mat_matvec_cached_prepare(srb_mat_transpose(x))


def srb_mat_matvec_cached_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return sc.sparse_interval_matvec_plan_from_sparse(
        _interval_sparse_matrix(x, "srb_mat.matvec_cached_prepare_basic"),
        algebra="srb",
        label="srb_mat.matvec_cached_prepare_basic",
    )


def srb_mat_rmatvec_cached_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return srb_mat_matvec_cached_prepare_basic(srb_mat_transpose(x))


def srb_mat_matvec_cached_apply(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return sc.sparse_matvec_plan_apply(plan, _as_real_vector(v, "srb_mat.matvec_cached_apply"), algebra="srb", label="srb_mat.matvec_cached_apply")


def srb_mat_rmatvec_cached_apply(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return srb_mat_matvec_cached_apply(plan, v)


def srb_mat_matvec_cached_apply_basic(plan, v: jax.Array) -> jax.Array:
    return sc.sparse_interval_matvec_plan_apply(
        plan,
        _dense_interval_vector(v, "srb_mat.matvec_cached_apply_basic"),
        algebra="srb",
        label="srb_mat.matvec_cached_apply_basic",
    )


def srb_mat_rmatvec_cached_apply_basic(plan, v: jax.Array) -> jax.Array:
    return srb_mat_matvec_cached_apply_basic(plan, v)


def srb_mat_matvec_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        vs,
        validate=_as_real_matrix,
        label="srb_mat.matvec_batch_fixed",
        apply=lambda v: srb_mat_matvec(x, v),
    )


def srb_mat_matvec_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_real_matrix, label="srb_mat.matvec_basic_batch_fixed", apply=lambda v: srb_mat_matvec_basic(x, v))


def srb_mat_rmatvec_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_real_matrix, label="srb_mat.rmatvec_batch_fixed", apply=lambda v: srb_mat_rmatvec(x, v))


def srb_mat_rmatvec_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_real_matrix, label="srb_mat.rmatvec_basic_batch_fixed", apply=lambda v: srb_mat_rmatvec_basic(x, v))


def srb_mat_matvec_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(
        vs,
        pad_to=pad_to,
        validate=_as_real_matrix,
        label="srb_mat.matvec_batch_padded",
        apply=lambda v: srb_mat_matvec(x, v),
    )


def srb_mat_matvec_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.matvec_basic_batch_padded", apply=lambda v: srb_mat_matvec_basic(x, v))


def srb_mat_rmatvec_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.rmatvec_batch_padded", apply=lambda v: srb_mat_rmatvec(x, v))


def srb_mat_rmatvec_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.rmatvec_basic_batch_padded", apply=lambda v: srb_mat_rmatvec_basic(x, v))


def srb_mat_matvec_cached_apply_batch_fixed(plan: sc.SparseMatvecPlan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        vs,
        validate=_as_real_matrix,
        label="srb_mat.matvec_cached_apply_batch_fixed",
        apply=lambda v: srb_mat_matvec_cached_apply(plan, v),
    )


def srb_mat_matvec_cached_apply_basic_batch_fixed(plan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_real_matrix, label="srb_mat.matvec_cached_apply_basic_batch_fixed", apply=lambda v: srb_mat_matvec_cached_apply_basic(plan, v))


def srb_mat_rmatvec_cached_apply_batch_fixed(plan: sc.SparseMatvecPlan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_real_matrix, label="srb_mat.rmatvec_cached_apply_batch_fixed", apply=lambda v: srb_mat_rmatvec_cached_apply(plan, v))


def srb_mat_rmatvec_cached_apply_basic_batch_fixed(plan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_real_matrix, label="srb_mat.rmatvec_cached_apply_basic_batch_fixed", apply=lambda v: srb_mat_rmatvec_cached_apply_basic(plan, v))


def srb_mat_matvec_cached_apply_batch_padded(plan: sc.SparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(
        vs,
        pad_to=pad_to,
        validate=_as_real_matrix,
        label="srb_mat.matvec_cached_apply_batch_padded",
        apply=lambda v: srb_mat_matvec_cached_apply(plan, v),
    )


def srb_mat_matvec_cached_apply_basic_batch_padded(plan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.matvec_cached_apply_basic_batch_padded", apply=lambda v: srb_mat_matvec_cached_apply_basic(plan, v))


def srb_mat_rmatvec_cached_apply_batch_padded(plan: sc.SparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.rmatvec_cached_apply_batch_padded", apply=lambda v: srb_mat_rmatvec_cached_apply(plan, v))


def srb_mat_rmatvec_cached_apply_basic_batch_padded(plan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_real_matrix, label="srb_mat.rmatvec_cached_apply_basic_batch_padded", apply=lambda v: srb_mat_rmatvec_cached_apply_basic(plan, v))


@partial(jax.jit, static_argnames=())
def srb_mat_coo_to_dense_jit(x: sc.SparseCOO) -> jax.Array:
    return srb_mat_coo_to_dense(x)


@partial(jax.jit, static_argnames=())
def srb_mat_csr_to_dense_jit(x: sc.SparseCSR) -> jax.Array:
    return srb_mat_csr_to_dense(x)


@partial(jax.jit, static_argnames=())
def srb_mat_bcoo_to_dense_jit(x: sc.SparseBCOO) -> jax.Array:
    return srb_mat_bcoo_to_dense(x)


@partial(jax.jit, static_argnames=())
def srb_mat_matvec_jit(x, v: jax.Array) -> jax.Array:
    return srb_mat_matvec(x, v)


@partial(jax.jit, static_argnames=())
def srb_mat_rmatvec_jit(x, v: jax.Array) -> jax.Array:
    return srb_mat_rmatvec(x, v)


@partial(jax.jit, static_argnames=())
def srb_mat_matmul_dense_rhs_jit(x, b: jax.Array) -> jax.Array:
    return srb_mat_matmul_dense_rhs(x, b)


@partial(jax.jit, static_argnames=())
def srb_mat_matvec_cached_apply_jit(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return srb_mat_matvec_cached_apply(plan, v)


@partial(jax.jit, static_argnames=())
def srb_mat_rmatvec_cached_apply_jit(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return srb_mat_rmatvec_cached_apply(plan, v)


def srb_mat_matvec_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    v: jax.Array,
) -> tuple[jax.Array, SrbMatPointDiagnostics]:
    return srb_mat_matvec(x, v), _diagnostics(x, method="matvec")


def srb_mat_matvec_cached_apply_with_diagnostics(
    plan: sc.SparseMatvecPlan,
    v: jax.Array,
) -> tuple[jax.Array, SrbMatPointDiagnostics]:
    return srb_mat_matvec_cached_apply(plan, v), _diagnostics(plan, method="matvec_cached", cached=True)


def srb_mat_det(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_det_from_lu(x, lu_fn=srb_mat_lu, diag_fn=srb_mat_diag, to_dense_fn=srb_mat_to_dense)


def srb_mat_det_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return arb_mat.arb_mat_det_basic(_dense_interval_matrix(x, "srb_mat.det_basic"))


def srb_mat_inv(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_inv_via_solve(
        x,
        algebra="srb",
        label="srb_mat.inv",
        dtype=jnp.float64,
        solve_fn=lambda a, col: srb_mat_solve(a, col, method="gmres", tol=1e-10),
    )


def srb_mat_inv_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return arb_mat.arb_mat_inv_basic(_dense_interval_matrix(x, "srb_mat.inv_basic"))


def srb_mat_sqr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    return sc.sparse_square_via_matmul(x, matmul_sparse_fn=srb_mat_matmul_sparse)


def srb_mat_sqr_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    return arb_mat.arb_mat_sqr_basic(_dense_interval_matrix(x, "srb_mat.sqr_basic"))


def srb_mat_solve_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    **kwargs,
) -> tuple[jax.Array, SrbMatPointDiagnostics]:
    rhs = jnp.asarray(b)
    return srb_mat_solve(x, b, **kwargs), _diagnostics(x, method=str(kwargs.get("method", "gmres")), rhs_rank=int(rhs.ndim))


def srb_mat_lu_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
) -> tuple[tuple[sc.SparseCOO, sc.SparseCSR, sc.SparseCSR], SrbMatPointDiagnostics]:
    return srb_mat_lu(x), _diagnostics(x, method="lu", direct=True)


def srb_mat_qr_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
) -> tuple[sc.SparseQRFactor, SrbMatPointDiagnostics]:
    return srb_mat_qr(x), _diagnostics(x, method="qr", direct=True)


@partial(jax.jit, static_argnames=("lower", "unit_diagonal"))
def srb_mat_triangular_solve_jit(x, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return srb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal)


@partial(jax.jit, static_argnames=("method", "maxiter", "restart"))
def srb_mat_solve_jit(
    x,
    b: jax.Array,
    *,
    method: str = "gmres",
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
) -> jax.Array:
    return srb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart)


@partial(jax.jit, static_argnames=())
def srb_mat_trace_jit(x) -> jax.Array:
    return srb_mat_trace(x)


@partial(jax.jit, static_argnames=())
def srb_mat_norm_fro_jit(x) -> jax.Array:
    return srb_mat_norm_fro(x)


@partial(jax.jit, static_argnames=())
def srb_mat_norm_1_jit(x) -> jax.Array:
    return srb_mat_norm_1(x)


@partial(jax.jit, static_argnames=())
def srb_mat_norm_inf_jit(x) -> jax.Array:
    return srb_mat_norm_inf(x)


def srb_mat_symmetric_part_jit(x) -> sc.SparseCSR:
    return srb_mat_symmetric_part(x)


def srb_mat_is_symmetric_jit(x, *, rtol: float = 1e-10, atol: float = 1e-10) -> jax.Array:
    return srb_mat_is_symmetric(x, rtol=rtol, atol=atol)


def srb_mat_is_spd_jit(x) -> jax.Array:
    return srb_mat_is_spd(x)


def srb_mat_cho_jit(x):
    return srb_mat_cho(x)


def srb_mat_ldl_jit(x):
    return srb_mat_ldl(x)


def srb_mat_charpoly_jit(x) -> jax.Array:
    return srb_mat_charpoly(x)


@partial(jax.jit, static_argnames=("n",))
def srb_mat_pow_ui_jit(x, n: int) -> jax.Array:
    return srb_mat_pow_ui(x, n)


def srb_mat_exp_jit(x) -> jax.Array:
    return srb_mat_exp(x)


def srb_mat_eigvalsh_jit(x) -> jax.Array:
    return srb_mat_eigvalsh(x)


def srb_mat_eigh_jit(x):
    return srb_mat_eigh(x)


@partial(jax.jit, static_argnames=("k", "which", "steps"))
def srb_mat_eigsh_jit(x, *, k: int = 6, which: str = "largest", steps: int | None = None, v0: jax.Array | None = None):
    return srb_mat_eigsh(x, k=k, which=which, steps=steps, v0=v0)


def srb_mat_det_jit(x) -> jax.Array:
    return srb_mat_det(x)


def srb_mat_inv_jit(x) -> jax.Array:
    return srb_mat_inv(x)


def srb_mat_sqr_jit(x):
    return srb_mat_sqr(x)


__all__ = [
    "srb_mat_coo",
    "srb_mat_csr",
    "srb_mat_bcoo",
    "srb_mat_interval_coo",
    "srb_mat_interval_csr",
    "srb_mat_interval_bcoo",
    "srb_mat_to_interval_sparse",
    "srb_mat_interval_to_dense",
    "srb_mat_interval_transpose",
    "srb_mat_interval_add",
    "srb_mat_interval_scale",
    "srb_mat_interval_matvec",
    "srb_mat_shape",
    "srb_mat_nnz",
    "srb_mat_zero",
    "srb_mat_identity",
    "srb_mat_permutation_matrix",
    "srb_mat_from_dense_coo",
    "srb_mat_from_dense_csr",
    "srb_mat_from_dense_bcoo",
    "srb_mat_diag",
    "srb_mat_diag_matrix",
    "srb_mat_trace",
    "srb_mat_norm_fro",
    "srb_mat_norm_1",
    "srb_mat_norm_inf",
    "srb_mat_submatrix",
    "srb_mat_coo_to_dense",
    "srb_mat_csr_to_dense",
    "srb_mat_bcoo_to_dense",
    "srb_mat_to_dense",
    "srb_mat_coo_to_csr",
    "srb_mat_csr_to_coo",
    "srb_mat_coo_to_bcoo",
    "srb_mat_csr_to_bcoo",
    "srb_mat_bcoo_to_coo",
    "srb_mat_transpose",
    "srb_mat_transpose_basic",
    "srb_mat_symmetric_part",
    "srb_mat_symmetric_part_basic",
    "srb_mat_is_symmetric",
    "srb_mat_is_symmetric_basic",
    "srb_mat_is_spd",
    "srb_mat_is_spd_basic",
    "srb_mat_cho",
    "srb_mat_cho_basic",
    "srb_mat_ldl",
    "srb_mat_ldl_basic",
    "srb_mat_charpoly",
    "srb_mat_charpoly_basic",
    "srb_mat_pow_ui",
    "srb_mat_pow_ui_basic",
    "srb_mat_exp",
    "srb_mat_exp_basic",
    "srb_mat_eigvalsh",
    "srb_mat_eigvalsh_basic",
    "srb_mat_eigh",
    "srb_mat_eigh_basic",
    "srb_mat_eigsh",
    "srb_mat_eigsh_basic",
    "srb_mat_operator_plan_prepare",
    "srb_mat_operator_rmatvec_plan_prepare",
    "srb_mat_operator_adjoint_plan_prepare",
    "srb_mat_scale",
    "srb_mat_add",
    "srb_mat_sub",
    "srb_mat_matvec",
    "srb_mat_rmatvec",
    "srb_mat_matvec_basic",
    "srb_mat_rmatvec_basic",
    "srb_mat_matvec_cached_prepare",
    "srb_mat_matvec_cached_prepare_basic",
    "srb_mat_rmatvec_cached_prepare",
    "srb_mat_rmatvec_cached_prepare_basic",
    "srb_mat_matvec_cached_apply",
    "srb_mat_matvec_cached_apply_basic",
    "srb_mat_rmatvec_cached_apply",
    "srb_mat_rmatvec_cached_apply_basic",
    "srb_mat_triangular_solve",
    "srb_mat_triangular_solve_basic",
    "srb_mat_triangular_solve_batch_fixed",
    "srb_mat_triangular_solve_basic_batch_fixed",
    "srb_mat_triangular_solve_batch_padded",
    "srb_mat_triangular_solve_basic_batch_padded",
    "srb_mat_triangular_solve_jit",
    "srb_mat_lu",
    "srb_mat_lu_solve",
    "srb_mat_lu_solve_plan_prepare",
    "srb_mat_lu_solve_plan_prepare_basic",
    "srb_mat_lu_solve_plan_apply",
    "srb_mat_lu_solve_plan_apply_basic",
    "srb_mat_lu_solve_plan_apply_batch_fixed",
    "srb_mat_lu_solve_plan_apply_basic_batch_fixed",
    "srb_mat_lu_solve_plan_apply_batch_padded",
    "srb_mat_lu_solve_plan_apply_basic_batch_padded",
    "srb_mat_solve_lu",
    "srb_mat_solve_lu_basic",
    "srb_mat_solve_lu_precomp",
    "srb_mat_solve_lu_precomp_basic",
    "srb_mat_qr",
    "srb_mat_qr_r",
    "srb_mat_qr_apply_q",
    "srb_mat_qr_explicit_q",
    "srb_mat_qr_solve",
    "srb_mat_solve",
    "srb_mat_solve_basic",
    "srb_mat_solve_transpose",
    "srb_mat_solve_transpose_basic",
    "srb_mat_solve_add",
    "srb_mat_solve_add_basic",
    "srb_mat_solve_transpose_add",
    "srb_mat_solve_transpose_add_basic",
    "srb_mat_mat_solve",
    "srb_mat_mat_solve_basic",
    "srb_mat_mat_solve_transpose",
    "srb_mat_mat_solve_transpose_basic",
    "srb_mat_solve_batch_fixed",
    "srb_mat_solve_basic_batch_fixed",
    "srb_mat_solve_batch_padded",
    "srb_mat_solve_basic_batch_padded",
    "srb_mat_solve_jit",
    "srb_mat_spd_solve_plan_prepare",
    "srb_mat_spd_solve_plan_prepare_basic",
    "srb_mat_spd_solve_plan_apply",
    "srb_mat_spd_solve_plan_apply_basic",
    "srb_mat_spd_solve_plan_apply_batch_fixed",
    "srb_mat_spd_solve_plan_apply_basic_batch_fixed",
    "srb_mat_spd_solve_plan_apply_batch_padded",
    "srb_mat_spd_solve_plan_apply_basic_batch_padded",
    "srb_mat_spd_solve",
    "srb_mat_spd_solve_basic",
    "srb_mat_spd_inv",
    "srb_mat_spd_inv_basic",
    "srb_mat_matvec",
    "srb_mat_matmul_dense_rhs",
    "srb_mat_matmul_dense_rhs_basic",
    "srb_mat_matmul_sparse",
    "srb_mat_matvec_cached_prepare",
    "srb_mat_matvec_cached_apply",
    "srb_mat_matvec_batch_fixed",
    "srb_mat_matvec_basic_batch_fixed",
    "srb_mat_matvec_batch_padded",
    "srb_mat_matvec_basic_batch_padded",
    "srb_mat_matvec_cached_apply_batch_fixed",
    "srb_mat_matvec_cached_apply_basic_batch_fixed",
    "srb_mat_matvec_cached_apply_batch_padded",
    "srb_mat_matvec_cached_apply_basic_batch_padded",
    "srb_mat_rmatvec_batch_fixed",
    "srb_mat_rmatvec_basic_batch_fixed",
    "srb_mat_rmatvec_batch_padded",
    "srb_mat_rmatvec_basic_batch_padded",
    "srb_mat_rmatvec_cached_apply_batch_fixed",
    "srb_mat_rmatvec_cached_apply_basic_batch_fixed",
    "srb_mat_rmatvec_cached_apply_batch_padded",
    "srb_mat_rmatvec_cached_apply_basic_batch_padded",
    "srb_mat_coo_to_dense_jit",
    "srb_mat_csr_to_dense_jit",
    "srb_mat_bcoo_to_dense_jit",
    "srb_mat_matvec_jit",
    "srb_mat_rmatvec_jit",
    "srb_mat_matmul_dense_rhs_jit",
    "srb_mat_matvec_cached_apply_jit",
    "srb_mat_rmatvec_cached_apply_jit",
    "srb_mat_matvec_with_diagnostics",
    "srb_mat_matvec_cached_apply_with_diagnostics",
    "srb_mat_solve_with_diagnostics",
    "srb_mat_lu_with_diagnostics",
    "srb_mat_qr_with_diagnostics",
    "srb_mat_trace_jit",
    "srb_mat_norm_fro_jit",
    "srb_mat_norm_1_jit",
    "srb_mat_norm_inf_jit",
    "srb_mat_symmetric_part_jit",
    "srb_mat_is_symmetric_jit",
    "srb_mat_is_spd_jit",
    "srb_mat_cho_jit",
    "srb_mat_ldl_jit",
    "srb_mat_charpoly_jit",
    "srb_mat_pow_ui_jit",
    "srb_mat_exp_jit",
    "srb_mat_eigvalsh_jit",
    "srb_mat_eigh_jit",
    "srb_mat_eigsh_jit",
    "srb_mat_det",
    "srb_mat_det_basic",
    "srb_mat_det_jit",
    "srb_mat_inv",
    "srb_mat_inv_basic",
    "srb_mat_inv_jit",
    "srb_mat_sqr",
    "srb_mat_sqr_basic",
    "srb_mat_sqr_jit",
    "srb_mat_to_dense_basic",
    "srb_mat_trace_basic",
    "srb_mat_norm_fro_basic",
    "srb_mat_norm_1_basic",
    "srb_mat_norm_inf_basic",
    "SrbMatPointDiagnostics",
]
