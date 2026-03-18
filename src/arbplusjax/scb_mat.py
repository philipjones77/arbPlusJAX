from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax import ops

from . import checks
from . import iterative_solvers
from . import sparse_common as sc



class ScbMatPointDiagnostics(NamedTuple):
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
) -> ScbMatPointDiagnostics:
    if isinstance(x, sc.SparseMatvecPlan):
        x = sc.as_sparse_matvec_plan(x, algebra="scb", label="scb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.payload.data.shape[0]) if x.storage == "bcoo" else int(x.payload[0].shape[0])
        storage = x.storage
    elif isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.data.shape[0])
        storage = "coo"
    elif isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.data.shape[0])
        storage = "csr"
    else:
        x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.diagnostics")
        rows, cols = x.rows, x.cols
        nnz = int(x.data.shape[0])
        storage = "bcoo"
    return ScbMatPointDiagnostics(storage, rows, cols, nnz, batch_size, method, cached, direct, rhs_rank)


def _as_complex_vector(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.complex128)
    checks.check_equal(arr.ndim, 1, f"{label}.ndim")
    return arr


def _as_complex_matrix(a: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(a, dtype=jnp.complex128)
    checks.check_equal(arr.ndim, 2, f"{label}.ndim")
    return arr


def _as_complex_rhs(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.complex128)
    checks.check(arr.ndim in (1, 2), f"{label}.ndim")
    return arr


def scb_mat_shape(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO | sc.SparseMatvecPlan) -> tuple[int, int]:
    if isinstance(x, sc.SparseMatvecPlan):
        plan = sc.as_sparse_matvec_plan(x, algebra="scb", label="scb_mat.shape")
        return plan.rows, plan.cols
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.shape")
        return x.rows, x.cols
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.shape")
        return x.rows, x.cols
    x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.shape")
    return x.rows, x.cols


def scb_mat_nnz(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO | sc.SparseMatvecPlan) -> int:
    if isinstance(x, sc.SparseMatvecPlan):
        plan = sc.as_sparse_matvec_plan(x, algebra="scb", label="scb_mat.nnz")
        if plan.storage == "coo":
            data, _, _ = plan.payload
            return int(data.shape[0])
        if plan.storage == "csr":
            data, _, _ = plan.payload
            return int(data.shape[0])
        return int(plan.payload.data.shape[0])
    if isinstance(x, sc.SparseCOO):
        return int(sc.as_sparse_coo(x, algebra="scb", label="scb_mat.nnz").data.shape[0])
    if isinstance(x, sc.SparseCSR):
        return int(sc.as_sparse_csr(x, algebra="scb", label="scb_mat.nnz").data.shape[0])
    return int(sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.nnz").data.shape[0])


def scb_mat_zero(shape: tuple[int, int]) -> sc.SparseCOO:
    return scb_mat_coo(jnp.zeros((0,), dtype=jnp.complex128), jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=jnp.int32), shape=shape)


def scb_mat_identity(n: int, *, dtype: jnp.dtype = jnp.complex128) -> sc.SparseCOO:
    idx = jnp.arange(n, dtype=jnp.int32)
    return scb_mat_coo(jnp.ones((n,), dtype=dtype), idx, idx, shape=(n, n))


def scb_mat_permutation_matrix(perm: jax.Array, *, dtype: jnp.dtype = jnp.complex128) -> sc.SparseCOO:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    row = jnp.arange(perm.shape[0], dtype=jnp.int32)
    return scb_mat_coo(jnp.ones((perm.shape[0],), dtype=dtype), row, perm, shape=(perm.shape[0], perm.shape[0]))


def scb_mat_coo(data: jax.Array, row: jax.Array, col: jax.Array, *, shape: tuple[int, int]) -> sc.SparseCOO:
    return sc.SparseCOO(
        data=jnp.asarray(data, dtype=jnp.complex128),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="scb",
    )


def scb_mat_csr(data: jax.Array, indices: jax.Array, indptr: jax.Array, *, shape: tuple[int, int]) -> sc.SparseCSR:
    return sc.SparseCSR(
        data=jnp.asarray(data, dtype=jnp.complex128),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="scb",
    )


def scb_mat_bcoo(data: jax.Array, indices: jax.Array, *, shape: tuple[int, int]) -> sc.SparseBCOO:
    return sc.SparseBCOO(
        data=jnp.asarray(data, dtype=jnp.complex128),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="scb",
    )


def scb_mat_from_dense_coo(a: jax.Array, *, tol: float = 0.0) -> sc.SparseCOO:
    a = _as_complex_matrix(a, "scb_mat.from_dense_coo")
    mask = jnp.abs(a) > tol
    row, col = jnp.nonzero(mask, size=int(mask.size), fill_value=-1)
    valid = row >= 0
    data = a[row, col]
    return scb_mat_coo(data[valid], row[valid], col[valid], shape=a.shape)


def scb_mat_from_dense_csr(a: jax.Array, *, tol: float = 0.0) -> sc.SparseCSR:
    return scb_mat_coo_to_csr(scb_mat_from_dense_coo(a, tol=tol))


def scb_mat_from_dense_bcoo(a: jax.Array, *, tol: float = 0.0) -> sc.SparseBCOO:
    return sc.dense_to_sparse_bcoo(_as_complex_matrix(a, "scb_mat.from_dense_bcoo"), algebra="scb", tol=tol)


def scb_mat_diag(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.diag")
        diag = jnp.zeros((x.rows,), dtype=x.data.dtype)
        mask = x.row == x.col
        return diag.at[x.row[mask]].add(x.data[mask])
    return jnp.diag(scb_mat_to_dense(x))


def scb_mat_diag_matrix(d: jax.Array) -> sc.SparseCOO:
    d = _as_complex_vector(d, "scb_mat.diag_matrix")
    idx = jnp.arange(d.shape[0], dtype=jnp.int32)
    return scb_mat_coo(d, idx, idx, shape=(d.shape[0], d.shape[0]))


def scb_mat_trace(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return jnp.sum(scb_mat_diag(x))


def scb_mat_norm_fro(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.norm_fro")
        return jnp.linalg.norm(x.data)
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.norm_fro")
        return jnp.linalg.norm(x.data)
    x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.norm_fro")
    return jnp.linalg.norm(x.data)


def scb_mat_norm_1(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return jnp.linalg.norm(scb_mat_to_dense(x), ord=1)


def scb_mat_norm_inf(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return jnp.linalg.norm(scb_mat_to_dense(x), ord=jnp.inf)


def scb_mat_submatrix(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, row_start: int, row_stop: int, col_start: int, col_stop: int) -> sc.SparseCOO:
    coo = x if isinstance(x, sc.SparseCOO) else scb_mat_bcoo_to_coo(x) if isinstance(x, sc.SparseBCOO) else scb_mat_csr_to_coo(x)
    coo = sc.as_sparse_coo(coo, algebra="scb", label="scb_mat.submatrix")
    mask = (coo.row >= row_start) & (coo.row < row_stop) & (coo.col >= col_start) & (coo.col < col_stop)
    return scb_mat_coo(coo.data[mask], coo.row[mask] - row_start, coo.col[mask] - col_start, shape=(row_stop - row_start, col_stop - col_start))


def _swap_rows_dense(a: jax.Array, i: int, j: int) -> jax.Array:
    if i == j:
        return a
    row_i = a[i, :]
    row_j = a[j, :]
    a = a.at[i, :].set(row_j)
    return a.at[j, :].set(row_i)


def _swap_perm(perm: jax.Array, i: int, j: int) -> jax.Array:
    if i == j:
        return perm
    vi = perm[i]
    vj = perm[j]
    perm = perm.at[i].set(vj)
    return perm.at[j].set(vi)


def _dense_sparse_lu_partial_pivot(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = a.shape[0]
    u = jnp.array(a, copy=True)
    l = jnp.eye(n, dtype=a.dtype)
    perm = jnp.arange(n, dtype=jnp.int32)
    for k in range(n):
        pivot = int(k + jnp.argmax(jnp.abs(u[k:, k])))
        u = _swap_rows_dense(u, k, pivot)
        if k > 0 and pivot != k:
            l_k = l[k, :k]
            l_p = l[pivot, :k]
            l = l.at[k, :k].set(l_p)
            l = l.at[pivot, :k].set(l_k)
        perm = _swap_perm(perm, k, pivot)
        pivot_value = u[k, k]
        if k + 1 < n:
            factors = u[k + 1 :, k] / pivot_value
            l = l.at[k + 1 :, k].set(factors)
            u = u.at[k + 1 :, k:].set(u[k + 1 :, k:] - factors[:, None] * u[k, k:][None, :])
            u = u.at[k + 1 :, k].set(0.0 + 0.0j)
    return perm, l, u


def _dense_householder_qr(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    m, n = a.shape
    k = min(m, n)
    work = jnp.array(a, copy=True)
    reflectors = jnp.zeros((m, k), dtype=a.dtype)
    taus = jnp.zeros((k,), dtype=a.dtype)
    for j in range(k):
        x = work[j:, j]
        x0 = x[0]
        normx = jnp.linalg.norm(x)
        phase = jnp.where(jnp.abs(x0) > 0.0, x0 / jnp.abs(x0), 1.0 + 0.0j)
        alpha = -phase * normx
        v = x.at[0].add(-alpha)
        beta = jnp.vdot(v, v)
        tau = jnp.where(jnp.abs(beta) > 0.0, 2.0 / beta, 0.0 + 0.0j)
        trailing = work[j:, j:]
        trailing = trailing - tau * v[:, None] * (jnp.conjugate(v) @ trailing)[None, :]
        work = work.at[j:, j:].set(trailing)
        reflector = jnp.zeros((m,), dtype=a.dtype).at[j:].set(v)
        reflectors = reflectors.at[:, j].set(reflector)
        taus = taus.at[j].set(tau)
        work = work.at[j, j].set(alpha)
        if j + 1 < m:
            work = work.at[j + 1 :, j].set(v[1:])
    return reflectors, taus, jnp.triu(work[:k, :])


def scb_mat_lu(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[sc.SparseCOO, sc.SparseCSR, sc.SparseCSR]:
    csr = _as_csr(x, label="scb_mat.lu")
    a = scb_mat_to_dense(csr)
    perm, l, u = _dense_sparse_lu_partial_pivot(a)
    n = csr.rows
    p = scb_mat_permutation_matrix(perm)
    l_sparse = scb_mat_from_dense_csr(jnp.tril(l))
    u_sparse = scb_mat_from_dense_csr(jnp.triu(u))
    return p, l_sparse, u_sparse


def scb_mat_lu_solve(
    lu: tuple[sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO],
    b: jax.Array,
) -> jax.Array:
    p, l, u = lu
    pb = scb_mat_matvec(p, _as_complex_vector(b, "scb_mat.lu_solve")) if jnp.asarray(b).ndim == 1 else scb_mat_matmul_dense_rhs(p, _as_complex_matrix(b, "scb_mat.lu_solve"))
    y = scb_mat_triangular_solve(l, pb, lower=True, unit_diagonal=True)
    return scb_mat_triangular_solve(u, y, lower=False, unit_diagonal=False)


def scb_mat_qr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseQRFactor:
    csr = _as_csr(x, label="scb_mat.qr")
    reflectors, taus, r_dense = _dense_householder_qr(scb_mat_to_dense(csr))
    r_sparse = scb_mat_from_dense_csr(r_dense)
    return sc.SparseQRFactor(reflectors=reflectors, taus=taus, r_factor=r_sparse, rows=csr.rows, cols=csr.cols, algebra="scb")


def scb_mat_qr_r(x: sc.SparseQRFactor) -> sc.SparseCSR:
    factor = sc.as_sparse_qr_factor(x, algebra="scb", label="scb_mat.qr_r")
    return factor.r_factor


def scb_mat_qr_apply_q(x: sc.SparseQRFactor, b: jax.Array, *, adjoint: bool = False) -> jax.Array:
    factor = sc.as_sparse_qr_factor(x, algebra="scb", label="scb_mat.qr_apply_q")
    arr = jnp.asarray(b, dtype=jnp.complex128)
    checks.check(arr.ndim in (1, 2), "scb_mat.qr_apply_q.ndim")
    if arr.ndim == 1:
        out = arr
        ks = range(factor.taus.shape[0] - 1, -1, -1) if not adjoint else range(factor.taus.shape[0])
        for j in ks:
            v = factor.reflectors[:, j]
            tau = factor.taus[j]
            proj = jnp.vdot(v, out)
            out = out - (jnp.conjugate(tau) if adjoint else tau) * v * proj
        return out
    cols = jax.vmap(lambda col: scb_mat_qr_apply_q(factor, col, adjoint=adjoint), in_axes=1, out_axes=1)(arr)
    return cols


def scb_mat_qr_explicit_q(x: sc.SparseQRFactor) -> jax.Array:
    factor = sc.as_sparse_qr_factor(x, algebra="scb", label="scb_mat.qr_explicit_q")
    return scb_mat_qr_apply_q(factor, jnp.eye(factor.rows, dtype=jnp.complex128), adjoint=False)


def scb_mat_qr_solve(x: sc.SparseQRFactor, b: jax.Array) -> jax.Array:
    factor = sc.as_sparse_qr_factor(x, algebra="scb", label="scb_mat.qr_solve")
    rhs = jnp.asarray(b, dtype=jnp.complex128)
    qh_b = scb_mat_qr_apply_q(factor, rhs, adjoint=True)
    r = factor.r_factor
    leading = qh_b[: r.rows] if rhs.ndim == 1 else qh_b[: r.rows, :]
    return scb_mat_triangular_solve(r, leading, lower=False, unit_diagonal=False)


def scb_mat_coo_to_dense(x: sc.SparseCOO) -> jax.Array:
    x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.coo_to_dense")
    out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    return out.at[(x.row, x.col)].add(x.data)


def scb_mat_csr_to_dense(x: sc.SparseCSR) -> jax.Array:
    x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.csr_to_dense")
    row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
    return scb_mat_coo_to_dense(scb_mat_coo(x.data, row, x.indices, shape=(x.rows, x.cols)))


def scb_mat_bcoo_to_dense(x: sc.SparseBCOO) -> jax.Array:
    return sc.sparse_bcoo_to_dense(x, algebra="scb", label="scb_mat.bcoo_to_dense")


def scb_mat_to_dense(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    if isinstance(x, sc.SparseCOO):
        return scb_mat_coo_to_dense(x)
    if isinstance(x, sc.SparseCSR):
        return scb_mat_csr_to_dense(x)
    if isinstance(x, sc.SparseBCOO):
        return scb_mat_bcoo_to_dense(x)
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def scb_mat_coo_to_csr(x: sc.SparseCOO) -> sc.SparseCSR:
    x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.coo_to_csr")
    key = x.row * x.cols + x.col
    order = jnp.argsort(key)
    row = x.row[order]
    col = x.col[order]
    data = x.data[order]
    counts = jnp.bincount(row, length=x.rows)
    indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
    return scb_mat_csr(data, col, indptr, shape=(x.rows, x.cols))


def scb_mat_csr_to_coo(x: sc.SparseCSR) -> sc.SparseCOO:
    x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.csr_to_coo")
    row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
    return scb_mat_coo(x.data, row, x.indices, shape=(x.rows, x.cols))


def scb_mat_coo_to_bcoo(x: sc.SparseCOO) -> sc.SparseBCOO:
    x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.coo_to_bcoo")
    return scb_mat_bcoo(x.data, jnp.stack([x.row, x.col], axis=-1), shape=(x.rows, x.cols))


def scb_mat_csr_to_bcoo(x: sc.SparseCSR) -> sc.SparseBCOO:
    return scb_mat_coo_to_bcoo(scb_mat_csr_to_coo(x))


def scb_mat_bcoo_to_coo(x: sc.SparseBCOO) -> sc.SparseCOO:
    x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.bcoo_to_coo")
    return scb_mat_coo(x.data, x.indices[:, 0], x.indices[:, 1], shape=(x.rows, x.cols))


def _as_bcoo(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, *, label: str) -> sc.SparseBCOO:
    if isinstance(x, sc.SparseBCOO):
        return sc.as_sparse_bcoo(x, algebra="scb", label=label)
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label=label)
        return sc.coo_to_bcoo(x)
    return _as_bcoo(scb_mat_csr_to_coo(x), label=label)


def _as_csr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, *, label: str) -> sc.SparseCSR:
    if isinstance(x, sc.SparseCSR):
        return sc.as_sparse_csr(x, algebra="scb", label=label)
    if isinstance(x, sc.SparseCOO):
        return scb_mat_coo_to_csr(sc.as_sparse_coo(x, algebra="scb", label=label))
    return scb_mat_coo_to_csr(scb_mat_bcoo_to_coo(sc.as_sparse_bcoo(x, algebra="scb", label=label)))


def _csr_triangular_solve_vector(x: sc.SparseCSR, b: jax.Array, *, lower: bool, unit_diagonal: bool) -> jax.Array:
    n = x.rows
    nnz = x.data.shape[0]
    rows = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=nnz)
    order = jnp.arange(n, dtype=jnp.int32) if lower else jnp.arange(n - 1, -1, -1, dtype=jnp.int32)

    def body(state, i):
        _, out = state
        row_mask = rows == i
        cols = jnp.where(row_mask, x.indices, -1)
        data = jnp.where(row_mask, x.data, 0.0 + 0.0j)
        off_mask = row_mask & (cols != i)
        diag_mask = row_mask & (cols == i)
        accum = jnp.sum(jnp.where(off_mask, data * out[jnp.maximum(cols, 0)], 0.0 + 0.0j))
        diag = jnp.where(unit_diagonal, 1.0 + 0.0j, jnp.sum(jnp.where(diag_mask, data, 0.0 + 0.0j)))
        value = (b[i] - accum) / diag
        out = out.at[i].set(value)
        return (i + 1, out), value

    init = (jnp.int32(0), jnp.zeros_like(b))
    (_, out), _ = lax.scan(body, init, order)
    return out


def scb_mat_transpose(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.transpose")
        return scb_mat_coo(x.data, x.col, x.row, shape=(x.cols, x.rows))
    if isinstance(x, sc.SparseCSR):
        return scb_mat_coo_to_csr(scb_mat_transpose(scb_mat_csr_to_coo(x)))
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.transpose")
        return scb_mat_bcoo(x.data, x.indices[:, ::-1], shape=(x.cols, x.rows))
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def scb_mat_conjugate_transpose(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    tx = scb_mat_transpose(x)
    if isinstance(tx, sc.SparseCOO):
        return scb_mat_coo(jnp.conj(tx.data), tx.row, tx.col, shape=(tx.rows, tx.cols))
    if isinstance(tx, sc.SparseCSR):
        return scb_mat_csr(jnp.conj(tx.data), tx.indices, tx.indptr, shape=(tx.rows, tx.cols))
    return scb_mat_bcoo(jnp.conj(tx.data), tx.indices, shape=(tx.rows, tx.cols))


def scb_mat_matvec(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    v = _as_complex_vector(v, "scb_mat.matvec")
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.matvec.coo")
        checks.check_equal(x.cols, v.shape[0], "scb_mat.matvec.inner")
        contrib = x.data * v[x.col]
        return ops.segment_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.matvec.csr")
        checks.check_equal(x.cols, v.shape[0], "scb_mat.matvec.inner")
        row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = x.data * v[x.indices]
        return ops.segment_sum(contrib, row, num_segments=x.rows)
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.matvec.bcoo")
        checks.check_equal(x.cols, v.shape[0], "scb_mat.matvec.inner")
        return sc.sparse_bcoo_matvec(x, v, algebra="scb", label="scb_mat.matvec.bcoo")
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def scb_mat_matmul_dense_rhs(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    b = _as_complex_matrix(b, "scb_mat.matmul_dense_rhs")
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.matmul_dense_rhs.coo")
        checks.check_equal(x.cols, b.shape[0], "scb_mat.matmul_dense_rhs.inner")
        contrib = x.data[:, None] * b[x.col, :]
        return ops.segment_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.matmul_dense_rhs.csr")
        checks.check_equal(x.cols, b.shape[0], "scb_mat.matmul_dense_rhs.inner")
        row = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = x.data[:, None] * b[x.indices, :]
        return ops.segment_sum(contrib, row, num_segments=x.rows)
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.matmul_dense_rhs.bcoo")
        checks.check_equal(x.cols, b.shape[0], "scb_mat.matmul_dense_rhs.inner")
        return sc.sparse_bcoo_matmul_dense_rhs(x, b, algebra="scb", label="scb_mat.matmul_dense_rhs.bcoo")
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def scb_mat_scale(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, alpha: jax.Array) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    alpha = jnp.asarray(alpha, dtype=jnp.complex128)
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.scale")
        return scb_mat_coo(x.data * alpha, x.row, x.col, shape=(x.rows, x.cols))
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.scale")
        return scb_mat_csr(x.data * alpha, x.indices, x.indptr, shape=(x.rows, x.cols))
    x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.scale")
    return scb_mat_bcoo(x.data * alpha, x.indices, shape=(x.rows, x.cols))


def scb_mat_add(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, y: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    xb = _as_bcoo(x, label="scb_mat.add.x")
    yb = _as_bcoo(y, label="scb_mat.add.y")
    return sc.sparse_bcoo_add(xb, yb, algebra="scb", label="scb_mat.add")


def scb_mat_sub(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, y: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    return scb_mat_add(x, scb_mat_scale(y, -1.0 + 0.0j))


def scb_mat_matmul_sparse(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, y: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    xb = _as_bcoo(x, label="scb_mat.matmul_sparse.x")
    yb = _as_bcoo(y, label="scb_mat.matmul_sparse.y")
    return sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="scb", label="scb_mat.matmul_sparse")


def scb_mat_triangular_solve(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    x = _as_csr(x, label="scb_mat.triangular_solve")
    b = _as_complex_rhs(b, "scb_mat.triangular_solve")
    checks.check_equal(x.cols, b.shape[0], "scb_mat.triangular_solve.inner")
    if b.ndim == 1:
        return _csr_triangular_solve_vector(x, b, lower=lower, unit_diagonal=unit_diagonal)
    return jax.vmap(lambda col: _csr_triangular_solve_vector(x, col, lower=lower, unit_diagonal=unit_diagonal), in_axes=1, out_axes=1)(b)


def scb_mat_solve(
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
    x = _as_bcoo(x, label="scb_mat.solve")
    b = _as_complex_rhs(b, "scb_mat.solve")
    checks.check_equal(x.cols, b.shape[0], "scb_mat.solve.inner")
    if x.rows <= 16:
        dense = sc.sparse_bcoo_to_dense(x, algebra="scb", label="scb_mat.solve.dense")
        return jnp.linalg.solve(dense, b)

    def matvec(v):
        return sc.sparse_bcoo_matvec(x, v, algebra="scb", label="scb_mat.solve.apply")

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
        return solve_vec(b, None if x0 is None else jnp.asarray(x0, dtype=jnp.complex128))
    guess = None if x0 is None else _as_complex_matrix(x0, "scb_mat.solve.x0")
    return jax.vmap(lambda rhs, g: solve_vec(rhs, g), in_axes=(1, 1 if guess is not None else None), out_axes=1)(
        b,
        guess if guess is not None else None,
    )


def scb_mat_solve_batch_fixed(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    method: str = "gmres",
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
) -> jax.Array:
    bs = _as_complex_matrix(bs, "scb_mat.solve_batch_fixed")
    return jax.vmap(lambda b: scb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart))(bs)


def scb_mat_solve_batch_padded(
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
    bs = _as_complex_matrix(bs, "scb_mat.solve_batch_padded")
    checks.check(pad_to >= bs.shape[0], "scb_mat.solve_batch_padded.pad_to")
    pad_count = int(pad_to - bs.shape[0])
    padded = jnp.concatenate([bs, jnp.repeat(bs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else bs
    return scb_mat_solve_batch_fixed(x, padded, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart)


def scb_mat_triangular_solve_batch_fixed(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    bs = _as_complex_matrix(bs, "scb_mat.triangular_solve_batch_fixed")
    return jax.vmap(lambda b: scb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal))(bs)


def scb_mat_triangular_solve_batch_padded(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    pad_to: int,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    bs = _as_complex_matrix(bs, "scb_mat.triangular_solve_batch_padded")
    checks.check(pad_to >= bs.shape[0], "scb_mat.triangular_solve_batch_padded.pad_to")
    pad_count = int(pad_to - bs.shape[0])
    padded = jnp.concatenate([bs, jnp.repeat(bs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else bs
    return scb_mat_triangular_solve_batch_fixed(x, padded, lower=lower, unit_diagonal=unit_diagonal)


def scb_mat_matvec_cached_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseMatvecPlan:
    if isinstance(x, sc.SparseCOO):
        x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.matvec_cached_prepare")
        return sc.SparseMatvecPlan(storage="coo", payload=(x.data, x.row, x.col), rows=x.rows, cols=x.cols, algebra="scb")
    if isinstance(x, sc.SparseCSR):
        x = sc.as_sparse_csr(x, algebra="scb", label="scb_mat.matvec_cached_prepare")
        row_ids = sc.csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        return sc.SparseMatvecPlan(storage="csr", payload=(x.data, x.indices, row_ids), rows=x.rows, cols=x.cols, algebra="scb")
    if isinstance(x, sc.SparseBCOO):
        x = sc.as_sparse_bcoo(x, algebra="scb", label="scb_mat.matvec_cached_prepare")
        return sc.SparseMatvecPlan(storage="bcoo", payload=x, rows=x.rows, cols=x.cols, algebra="scb")
    raise TypeError("expected SparseCOO, SparseCSR, or SparseBCOO")


def scb_mat_matvec_cached_apply(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    plan = sc.as_sparse_matvec_plan(plan, algebra="scb", label="scb_mat.matvec_cached_apply")
    v = _as_complex_vector(v, "scb_mat.matvec_cached_apply")
    checks.check_equal(plan.cols, v.shape[0], "scb_mat.matvec_cached_apply.inner")
    if plan.storage == "coo":
        data, row, col = plan.payload
        return ops.segment_sum(data * v[col], row, num_segments=plan.rows)
    if plan.storage == "csr":
        data, indices, row_ids = plan.payload
        return ops.segment_sum(data * v[indices], row_ids, num_segments=plan.rows)
    return sc.sparse_bcoo_matvec(plan.payload, v, algebra="scb", label="scb_mat.matvec_cached_apply.bcoo")


def scb_mat_matvec_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    vs = _as_complex_matrix(vs, "scb_mat.matvec_batch_fixed")
    return jax.vmap(lambda v: scb_mat_matvec(x, v))(vs)


def scb_mat_matvec_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_complex_matrix(vs, "scb_mat.matvec_batch_padded")
    checks.check(pad_to >= vs.shape[0], "scb_mat.matvec_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return scb_mat_matvec_batch_fixed(x, padded)


def scb_mat_matvec_cached_apply_batch_fixed(plan: sc.SparseMatvecPlan, vs: jax.Array) -> jax.Array:
    vs = _as_complex_matrix(vs, "scb_mat.matvec_cached_apply_batch_fixed")
    return jax.vmap(lambda v: scb_mat_matvec_cached_apply(plan, v))(vs)


def scb_mat_matvec_cached_apply_batch_padded(plan: sc.SparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    vs = _as_complex_matrix(vs, "scb_mat.matvec_cached_apply_batch_padded")
    checks.check(pad_to >= vs.shape[0], "scb_mat.matvec_cached_apply_batch_padded.pad_to")
    pad_count = int(pad_to - vs.shape[0])
    padded = jnp.concatenate([vs, jnp.repeat(vs[-1:, :], pad_count, axis=0)], axis=0) if pad_count > 0 else vs
    return scb_mat_matvec_cached_apply_batch_fixed(plan, padded)


@partial(jax.jit, static_argnames=())
def scb_mat_coo_to_dense_jit(x: sc.SparseCOO) -> jax.Array:
    return scb_mat_coo_to_dense(x)


@partial(jax.jit, static_argnames=())
def scb_mat_csr_to_dense_jit(x: sc.SparseCSR) -> jax.Array:
    return scb_mat_csr_to_dense(x)


@partial(jax.jit, static_argnames=())
def scb_mat_bcoo_to_dense_jit(x: sc.SparseBCOO) -> jax.Array:
    return scb_mat_bcoo_to_dense(x)


@partial(jax.jit, static_argnames=())
def scb_mat_matvec_jit(x, v: jax.Array) -> jax.Array:
    return scb_mat_matvec(x, v)


@partial(jax.jit, static_argnames=())
def scb_mat_matmul_dense_rhs_jit(x, b: jax.Array) -> jax.Array:
    return scb_mat_matmul_dense_rhs(x, b)


@partial(jax.jit, static_argnames=())
def scb_mat_matvec_cached_apply_jit(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return scb_mat_matvec_cached_apply(plan, v)


def scb_mat_matvec_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    v: jax.Array,
) -> tuple[jax.Array, ScbMatPointDiagnostics]:
    return scb_mat_matvec(x, v), _diagnostics(x, method="matvec")


def scb_mat_matvec_cached_apply_with_diagnostics(
    plan: sc.SparseMatvecPlan,
    v: jax.Array,
) -> tuple[jax.Array, ScbMatPointDiagnostics]:
    return scb_mat_matvec_cached_apply(plan, v), _diagnostics(plan, method="matvec_cached", cached=True)


def scb_mat_solve_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    **kwargs,
) -> tuple[jax.Array, ScbMatPointDiagnostics]:
    rhs = jnp.asarray(b)
    return scb_mat_solve(x, b, **kwargs), _diagnostics(x, method=str(kwargs.get("method", "gmres")), rhs_rank=int(rhs.ndim))


def scb_mat_lu_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
) -> tuple[tuple[sc.SparseCOO, sc.SparseCSR, sc.SparseCSR], ScbMatPointDiagnostics]:
    return scb_mat_lu(x), _diagnostics(x, method="lu", direct=True)


def scb_mat_qr_with_diagnostics(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
) -> tuple[sc.SparseQRFactor, ScbMatPointDiagnostics]:
    return scb_mat_qr(x), _diagnostics(x, method="qr", direct=True)


@partial(jax.jit, static_argnames=("lower", "unit_diagonal"))
def scb_mat_triangular_solve_jit(x, b: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return scb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal)


@partial(jax.jit, static_argnames=("method", "maxiter", "restart"))
def scb_mat_solve_jit(
    x,
    b: jax.Array,
    *,
    method: str = "gmres",
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int = 20,
) -> jax.Array:
    return scb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart)


@partial(jax.jit, static_argnames=())
def scb_mat_trace_jit(x) -> jax.Array:
    return scb_mat_trace(x)


@partial(jax.jit, static_argnames=())
def scb_mat_norm_fro_jit(x) -> jax.Array:
    return scb_mat_norm_fro(x)


@partial(jax.jit, static_argnames=())
def scb_mat_norm_1_jit(x) -> jax.Array:
    return scb_mat_norm_1(x)


@partial(jax.jit, static_argnames=())
def scb_mat_norm_inf_jit(x) -> jax.Array:
    return scb_mat_norm_inf(x)


def scb_mat_det(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    """Determinant using LU decomposition - point version.
    
    Args:
        x: Sparse complex matrix (must be square)
        
    Returns:
        Determinant as complex scalar
    """
    x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.det")
    
    P, L, U = scb_mat_lu(x)
    
    # Determinant is product of U diagonal times sign from permutation
    u_diag = scb_mat_diag(U)
    det_u = jnp.prod(u_diag)
    
    # Compute permutation sign
    perm = P.indices[:, 0]
    n = perm.shape[0]
    sign = jnp.where(jnp.arange(n) == perm, 1.0, -1.0).prod()
    
    return sign * det_u


def scb_mat_det_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    """Determinant using LU decomposition - basic interval version."""
    return scb_mat_det(x)


def scb_mat_inv(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    """Matrix inverse - point version using solve.
    
    Args:
        x: Sparse complex matrix (must be square)
        
    Returns:
        Dense complex inverse matrix
    """
    x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.inv")
    
    n = x.rows
    I = jnp.eye(n, dtype=jnp.complex128)
    
    def solve_col(col):
        return scb_mat_solve(x, col, method="gmres", tol=1e-10)
    
    inv_cols = jax.vmap(solve_col, in_axes=1, out_axes=1)(I)
    return inv_cols


def scb_mat_inv_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    """Matrix inverse - basic interval version."""
    return scb_mat_inv(x)


def scb_mat_sqr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    """Matrix square - point version.
    
    Args:
        x: Sparse complex matrix (must be square)
        
    Returns:
        x @ x as sparse matrix
    """
    x = sc.as_sparse_coo(x, algebra="scb", label="scb_mat.sqr")
    return scb_mat_matmul_sparse(x, x)


def scb_mat_sqr_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    """Matrix square - basic interval version."""
    return scb_mat_sqr(x)


__all__ = [
    "scb_mat_coo",
    "scb_mat_csr",
    "scb_mat_bcoo",
    "scb_mat_shape",
    "scb_mat_nnz",
    "scb_mat_zero",
    "scb_mat_identity",
    "scb_mat_permutation_matrix",
    "scb_mat_from_dense_coo",
    "scb_mat_from_dense_csr",
    "scb_mat_from_dense_bcoo",
    "scb_mat_diag",
    "scb_mat_diag_matrix",
    "scb_mat_trace",
    "scb_mat_norm_fro",
    "scb_mat_norm_1",
    "scb_mat_norm_inf",
    "scb_mat_submatrix",
    "scb_mat_coo_to_dense",
    "scb_mat_csr_to_dense",
    "scb_mat_bcoo_to_dense",
    "scb_mat_to_dense",
    "scb_mat_coo_to_csr",
    "scb_mat_csr_to_coo",
    "scb_mat_coo_to_bcoo",
    "scb_mat_csr_to_bcoo",
    "scb_mat_bcoo_to_coo",
    "scb_mat_transpose",
    "scb_mat_conjugate_transpose",
    "scb_mat_scale",
    "scb_mat_add",
    "scb_mat_sub",
    "scb_mat_triangular_solve",
    "scb_mat_triangular_solve_batch_fixed",
    "scb_mat_triangular_solve_batch_padded",
    "scb_mat_triangular_solve_jit",
    "scb_mat_lu",
    "scb_mat_lu_solve",
    "scb_mat_qr",
    "scb_mat_qr_r",
    "scb_mat_qr_apply_q",
    "scb_mat_qr_explicit_q",
    "scb_mat_qr_solve",
    "scb_mat_solve",
    "scb_mat_solve_batch_fixed",
    "scb_mat_solve_batch_padded",
    "scb_mat_solve_jit",
    "scb_mat_matvec",
    "scb_mat_matmul_dense_rhs",
    "scb_mat_matmul_sparse",
    "scb_mat_matvec_cached_prepare",
    "scb_mat_matvec_cached_apply",
    "scb_mat_matvec_batch_fixed",
    "scb_mat_matvec_batch_padded",
    "scb_mat_matvec_cached_apply_batch_fixed",
    "scb_mat_matvec_cached_apply_batch_padded",
    "scb_mat_coo_to_dense_jit",
    "scb_mat_csr_to_dense_jit",
    "scb_mat_bcoo_to_dense_jit",
    "scb_mat_matvec_jit",
    "scb_mat_matmul_dense_rhs_jit",
    "scb_mat_matvec_cached_apply_jit",
    "scb_mat_matvec_with_diagnostics",
    "scb_mat_matvec_cached_apply_with_diagnostics",
    "scb_mat_solve_with_diagnostics",
    "scb_mat_lu_with_diagnostics",
    "scb_mat_qr_with_diagnostics",
    "scb_mat_trace_jit",
    "scb_mat_norm_fro_jit",
    "scb_mat_norm_1_jit",
    "scb_mat_norm_inf_jit",
    "scb_mat_det",
    "scb_mat_det_basic",
    "scb_mat_inv",
    "scb_mat_inv_basic",
    "scb_mat_sqr",
    "scb_mat_sqr_basic",
    "ScbMatPointDiagnostics",
]
