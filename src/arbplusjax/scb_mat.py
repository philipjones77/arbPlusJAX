from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax import ops

from . import acb_core
from . import acb_mat
from . import checks
from . import double_interval as di
from . import iterative_solvers
from . import jcb_mat
from . import mat_common
from . import sparse_core
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


def _as_jcb_operator_sparse(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseBCOO:
    bcoo = _as_bcoo(x, label="scb_mat.as_jcb_operator_sparse")
    return sc.SparseBCOO(data=bcoo.data, indices=bcoo.indices, rows=bcoo.rows, cols=bcoo.cols, algebra="jcb")


def _dense_box_matrix(x, label: str) -> jax.Array:
    return sc.sparse_complex_to_box_matrix(x, algebra="scb", label=label)


def _box_sparse_matrix(x, label: str):
    return sc.sparse_complex_to_box_sparse(x, algebra="scb", label=label)


def _dense_box_vector(x: jax.Array, label: str) -> jax.Array:
    return mat_common.box_from_point(_as_complex_vector(x, label))


def _dense_box_rhs(x: jax.Array, label: str) -> jax.Array:
    return mat_common.box_from_point(_as_complex_rhs(x, label))


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


def scb_mat_box_coo(data: jax.Array, row: jax.Array, col: jax.Array, *, shape: tuple[int, int]) -> sc.SparseBoxCOO:
    return sc.SparseBoxCOO(
        data=acb_core.as_acb_box(data),
        row=jnp.asarray(row, dtype=jnp.int32),
        col=jnp.asarray(col, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="scb",
    )


def scb_mat_box_csr(data: jax.Array, indices: jax.Array, indptr: jax.Array, *, shape: tuple[int, int]) -> sc.SparseBoxCSR:
    return sc.SparseBoxCSR(
        data=acb_core.as_acb_box(data),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        indptr=jnp.asarray(indptr, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="scb",
    )


def scb_mat_box_bcoo(data: jax.Array, indices: jax.Array, *, shape: tuple[int, int]) -> sc.SparseBoxBCOO:
    return sc.SparseBoxBCOO(
        data=acb_core.as_acb_box(data),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        rows=int(shape[0]),
        cols=int(shape[1]),
        algebra="scb",
    )


def scb_mat_to_box_sparse(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return _box_sparse_matrix(x, "scb_mat.to_box_sparse")


def scb_mat_box_to_dense(x: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO) -> jax.Array:
    return sc.sparse_box_to_dense(x, algebra="scb", label="scb_mat.box_to_dense")


def scb_mat_box_transpose(x: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO) -> sc.SparseBoxBCOO:
    return sc.sparse_box_transpose(x, algebra="scb", label="scb_mat.box_transpose")


def scb_mat_box_conjugate_transpose(x: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO) -> sc.SparseBoxBCOO:
    return sc.sparse_box_conjugate_transpose(x, algebra="scb", label="scb_mat.box_conjugate_transpose")


def scb_mat_box_add(
    x: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO,
    y: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO,
) -> sc.SparseBoxBCOO:
    return sc.sparse_box_add(x, y, algebra="scb", label="scb_mat.box_add")


def scb_mat_box_scale(
    x: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO,
    alpha,
):
    return sc.sparse_box_scale(x, alpha, algebra="scb")


def scb_mat_box_matvec(
    x: sc.SparseBoxCOO | sc.SparseBoxCSR | sc.SparseBoxBCOO,
    v: jax.Array,
) -> jax.Array:
    return sc.sparse_box_matvec(x, acb_core.as_acb_box(v), algebra="scb", label="scb_mat.box_matvec")


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
    return sparse_core.sparse_diag(x, to_coo_fn=_to_coo_any, dtype=jnp.complex128)


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
    return sparse_core.sparse_norm_1(x, to_coo_fn=_to_coo_any)


def scb_mat_norm_inf(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_norm_inf(x, to_coo_fn=_to_coo_any)


def scb_mat_trace_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_box_trace(_box_sparse_matrix(x, "scb_mat.trace_basic"), algebra="scb", label="scb_mat.trace_basic")


def scb_mat_norm_fro_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    bx = _box_sparse_matrix(x, "scb_mat.norm_fro_basic")
    total = mat_common.box_sum(acb_core.acb_mul(bx.data, acb_core.acb_conj(bx.data)), axis=0)
    return acb_core.acb_sqrt(total)


def scb_mat_norm_1_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_box_norm_1(_box_sparse_matrix(x, "scb_mat.norm_1_basic"), algebra="scb", label="scb_mat.norm_1_basic")


def scb_mat_norm_inf_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_box_norm_inf(_box_sparse_matrix(x, "scb_mat.norm_inf_basic"), algebra="scb", label="scb_mat.norm_inf_basic")


def scb_mat_submatrix(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, row_start: int, row_stop: int, col_start: int, col_stop: int) -> sc.SparseCOO:
    coo = x if isinstance(x, sc.SparseCOO) else scb_mat_bcoo_to_coo(x) if isinstance(x, sc.SparseBCOO) else scb_mat_csr_to_coo(x)
    coo = sc.as_sparse_coo(coo, algebra="scb", label="scb_mat.submatrix")
    mask = (coo.row >= row_start) & (coo.row < row_stop) & (coo.col >= col_start) & (coo.col < col_stop)
    return scb_mat_coo(coo.data[mask], coo.row[mask] - row_start, coo.col[mask] - col_start, shape=(row_stop - row_start, col_stop - col_start))


def scb_mat_lu(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[sc.SparseCOO, sc.SparseCSR, sc.SparseCSR]:
    return sparse_core.sparse_lu_via_jax_dense(
        x,
        as_csr_fn=_as_csr,
        to_dense_fn=scb_mat_to_dense,
        from_dense_csr_fn=scb_mat_from_dense_csr,
        permutation_matrix_fn=scb_mat_permutation_matrix,
        complex_=True,
    )


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
    reflectors, taus, r_dense = sparse_core.dense_householder_qr_complex(scb_mat_to_dense(csr))
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


def _to_coo_any(x):
    if isinstance(x, sc.SparseCOO):
        return sc.as_sparse_coo(x, algebra="scb", label="scb_mat._to_coo_any")
    if isinstance(x, sc.SparseCSR):
        return scb_mat_csr_to_coo(x)
    return scb_mat_bcoo_to_coo(x)


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


def scb_mat_hermitian_part(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCSR:
    return scb_mat_coo_to_csr(sparse_core.sparse_structured_part_hermitian(x, to_coo_fn=_to_coo_any, from_coo_fn=scb_mat_coo))


def scb_mat_is_hermitian(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, *, rtol: float = 1e-10, atol: float = 1e-10) -> jax.Array:
    return sparse_core.sparse_is_hermitian_structural(x, to_coo_fn=_to_coo_any, rtol=rtol, atol=atol)


def scb_mat_is_hpd(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_is_pd_structural(
        x,
        to_coo_fn=_to_coo_any,
        structured_part_fn=scb_mat_hermitian_part,
        is_structured_fn=scb_mat_is_hermitian,
        dtype=jnp.complex128,
        hermitian=True,
    )


def scb_mat_cho(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCSR:
    data, row, col = sparse_core.sparse_cho_structural(
        x,
        to_coo_fn=_to_coo_any,
        structured_part_fn=scb_mat_hermitian_part,
        is_structured_fn=scb_mat_is_hermitian,
        dtype=jnp.complex128,
        hermitian=True,
    )
    return scb_mat_coo_to_csr(scb_mat_coo(data, row, col, shape=scb_mat_shape(x)))


def scb_mat_ldl(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[sc.SparseCSR, jax.Array]:
    data, row, col, d = sparse_core.sparse_ldl_structural(
        x,
        to_coo_fn=_to_coo_any,
        structured_part_fn=scb_mat_hermitian_part,
        is_structured_fn=scb_mat_is_hermitian,
        dtype=jnp.complex128,
        hermitian=True,
    )
    return scb_mat_coo_to_csr(scb_mat_coo(data, row, col, shape=scb_mat_shape(x))), d


def scb_mat_hermitian_part_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    bx = _box_sparse_matrix(x, "scb_mat.hermitian_part_basic")
    herm = sc.sparse_box_scale(
        sc.sparse_box_add(bx, sc.sparse_box_conjugate_transpose(bx, algebra="scb", label="scb_mat.hermitian_part_basic.transpose"), algebra="scb", label="scb_mat.hermitian_part_basic.add"),
        0.5 + 0.0j,
        algebra="scb",
    )
    return sc.sparse_box_to_dense(herm, algebra="scb", label="scb_mat.hermitian_part_basic.dense")


def scb_mat_is_hermitian_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    dense = sc.sparse_box_to_dense(_box_sparse_matrix(x, "scb_mat.is_hermitian_basic"), algebra="scb", label="scb_mat.is_hermitian_basic.dense")
    return mat_common.complex_midpoint_is_hermitian(acb_core.acb_midpoint(dense))


def scb_mat_is_hpd_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    dense = sc.sparse_box_to_dense(_box_sparse_matrix(x, "scb_mat.is_hpd_basic"), algebra="scb", label="scb_mat.is_hpd_basic.dense")
    mid = acb_core.acb_midpoint(dense)
    chol = jnp.linalg.cholesky(mat_common.complex_midpoint_hermitian_part(mid))
    return mat_common.complex_midpoint_is_hermitian(mid) & mat_common.lower_cholesky_finite(chol)


def scb_mat_cho_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return acb_mat.acb_mat_cho_basic(_dense_box_matrix(x, "scb_mat.cho_basic"))


def scb_mat_ldl_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[jax.Array, jax.Array]:
    return acb_mat.acb_mat_ldl_basic(_dense_box_matrix(x, "scb_mat.ldl_basic"))


def scb_mat_charpoly(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_charpoly_from_traces(
        x,
        to_bcoo_fn=_as_bcoo,
        matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="scb", label="scb_mat.charpoly.matmul"),
        trace_fn=scb_mat_trace,
        identity_sparse_fn=lambda n, dtype: scb_mat_bcoo(
            jnp.ones((n,), dtype=dtype),
            jnp.stack([jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32)], axis=-1),
            shape=(n, n),
        ),
    )


def scb_mat_charpoly_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return mat_common.box_from_point(scb_mat_charpoly(x))


def scb_mat_pow_ui(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, n: int) -> jax.Array:
    return sparse_core.sparse_dense_power_ui(
        x,
        n,
        to_bcoo_fn=_as_bcoo,
        matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="scb", label="scb_mat.pow_ui.matmul"),
        to_dense_fn=scb_mat_to_dense,
        identity_sparse_fn=lambda size, dtype: scb_mat_bcoo(
            jnp.ones((size,), dtype=dtype),
            jnp.stack([jnp.arange(size, dtype=jnp.int32), jnp.arange(size, dtype=jnp.int32)], axis=-1),
            shape=(size, size),
        ),
    )


def scb_mat_pow_ui_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, n: int) -> jax.Array:
    return mat_common.box_from_point(scb_mat_pow_ui(x, n))


def scb_mat_exp(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sparse_core.sparse_dense_exp_taylor(
        x,
        to_bcoo_fn=_as_bcoo,
        matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="scb", label="scb_mat.exp.matmul"),
        to_dense_fn=scb_mat_to_dense,
        identity_sparse_fn=lambda size, dtype: scb_mat_bcoo(
            jnp.ones((size,), dtype=dtype),
            jnp.stack([jnp.arange(size, dtype=jnp.int32), jnp.arange(size, dtype=jnp.int32)], axis=-1),
            shape=(size, size),
        ),
        terms=24,
    )


def scb_mat_exp_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return mat_common.box_from_point(scb_mat_exp(x))


def scb_mat_eigvalsh(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return acb_core.acb_midpoint(acb_mat.acb_mat_eigvalsh(_dense_box_matrix(x, "scb_mat.eigvalsh")))


def scb_mat_eigvalsh_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return acb_mat.acb_mat_eigvalsh(_dense_box_matrix(x, "scb_mat.eigvalsh_basic"))


def scb_mat_eigh(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[jax.Array, jax.Array]:
    values, vectors = acb_mat.acb_mat_eigh(_dense_box_matrix(x, "scb_mat.eigh"))
    return acb_core.acb_midpoint(values), acb_core.acb_midpoint(vectors)


def scb_mat_eigh_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> tuple[jax.Array, jax.Array]:
    return acb_mat.acb_mat_eigh(_dense_box_matrix(x, "scb_mat.eigh_basic"))


def scb_mat_eigsh(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    *,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    plan = jcb_mat.jcb_mat_bcoo_operator_plan_prepare(_as_jcb_operator_sparse(x))
    return jcb_mat.jcb_mat_eigsh_point(plan, size=int(x.rows), k=k, which=which, steps=steps, v0=v0)


def scb_mat_eigsh_basic(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    *,
    k: int = 6,
    which: str = "largest",
    steps: int | None = None,
    v0: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    values, vectors = scb_mat_eigsh(x, k=k, which=which, steps=steps, v0=v0)
    return mat_common.box_from_point(values), mat_common.box_from_point(vectors)


def scb_mat_operator_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return jcb_mat.jcb_mat_sparse_operator_plan_prepare(_as_jcb_operator_sparse(x))


def scb_mat_operator_rmatvec_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return jcb_mat.jcb_mat_sparse_operator_rmatvec_plan_prepare(_as_jcb_operator_sparse(x))


def scb_mat_operator_adjoint_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return jcb_mat.jcb_mat_sparse_operator_adjoint_plan_prepare(_as_jcb_operator_sparse(x))


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


def scb_mat_rmatvec(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    return scb_mat_matvec(scb_mat_transpose(x), v)


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


def scb_mat_to_dense_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_box_to_dense(_box_sparse_matrix(x, "scb_mat.to_dense_basic"), algebra="scb", label="scb_mat.to_dense_basic.dense")


def scb_mat_transpose_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    tx = sc.sparse_box_transpose(_box_sparse_matrix(x, "scb_mat.transpose_basic"), algebra="scb", label="scb_mat.transpose_basic.transpose")
    return sc.sparse_box_to_dense(tx, algebra="scb", label="scb_mat.transpose_basic.dense")


def scb_mat_conjugate_transpose_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    tx = sc.sparse_box_conjugate_transpose(_box_sparse_matrix(x, "scb_mat.conjugate_transpose_basic"), algebra="scb", label="scb_mat.conjugate_transpose_basic.transpose")
    return sc.sparse_box_to_dense(tx, algebra="scb", label="scb_mat.conjugate_transpose_basic.dense")


def scb_mat_matvec_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    return sc.sparse_box_matvec(
        _box_sparse_matrix(x, "scb_mat.matvec_basic"),
        _dense_box_vector(v, "scb_mat.matvec_basic"),
        algebra="scb",
        label="scb_mat.matvec_basic",
    )


def scb_mat_rmatvec_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, v: jax.Array) -> jax.Array:
    return scb_mat_matvec_basic(scb_mat_transpose(x), v)


def scb_mat_matmul_dense_rhs_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    return sc.sparse_box_matmul_dense_rhs(
        _box_sparse_matrix(x, "scb_mat.matmul_dense_rhs_basic"),
        _dense_box_rhs(b, "scb_mat.matmul_dense_rhs_basic"),
        algebra="scb",
        label="scb_mat.matmul_dense_rhs_basic",
    )


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
    return sparse_core.sparse_matmul_sparse(x, y, to_bcoo_fn=_as_bcoo, matmul_sparse_fn=lambda xb, yb: sc.sparse_bcoo_matmul_sparse(xb, yb, algebra="scb", label="scb_mat.matmul_sparse"))


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


def scb_mat_triangular_solve_basic(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return acb_mat.acb_mat_triangular_solve_basic(
        _dense_box_matrix(x, "scb_mat.triangular_solve_basic"),
        _dense_box_rhs(b, "scb_mat.triangular_solve_basic"),
        lower=lower,
        unit_diagonal=unit_diagonal,
    )


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
    if x.rows <= 32:
        return scb_mat_lu_solve_plan_apply(scb_mat_lu_solve_plan_prepare(x), b)

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


def scb_mat_solve_basic(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    b: jax.Array,
    **kwargs,
) -> jax.Array:
    del kwargs
    return acb_mat.acb_mat_solve_basic(_dense_box_matrix(x, "scb_mat.solve_basic"), _dense_box_rhs(b, "scb_mat.solve_basic"))


def scb_mat_lu_solve_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseLUSolvePlan:
    p, l, u = scb_mat_lu(x)
    return sc.sparse_lu_solve_plan_from_factors(p, l, u, algebra="scb")


def scb_mat_lu_solve_plan_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return scb_mat_lu_solve_plan_prepare(x)


def scb_mat_lu_solve_plan_apply(plan: sc.SparseLUSolvePlan | tuple, b: jax.Array) -> jax.Array:
    plan = sc.as_sparse_lu_solve_plan(plan, algebra="scb", label="scb_mat.lu_solve_plan_apply")
    return scb_mat_lu_solve((plan.p, plan.l, plan.u), b)


def scb_mat_lu_solve_plan_apply_basic(plan, b: jax.Array) -> jax.Array:
    if isinstance(plan, (sc.SparseLUSolvePlan, tuple)):
        return mat_common.box_from_point(scb_mat_lu_solve_plan_apply(plan, _as_complex_rhs(b, "scb_mat.lu_solve_plan_apply_basic")))
    return acb_mat.acb_mat_lu_solve_basic(plan, _dense_box_rhs(b, "scb_mat.lu_solve_plan_apply_basic"))


def scb_mat_hpd_solve_plan_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCholeskySolvePlan:
    return sc.sparse_cholesky_solve_plan_from_factor(scb_mat_cho(x), algebra="scb", structure="hpd")


def scb_mat_hpd_solve_plan_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return scb_mat_hpd_solve_plan_prepare(x)


def scb_mat_hpd_solve_plan_apply(plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    plan = sc.as_sparse_cholesky_solve_plan(plan, algebra="scb", structure="hpd", label="scb_mat.hpd_solve_plan_apply")
    factor = _as_csr(plan.factor, label="scb_mat.hpd_solve_plan_apply.factor")
    y = scb_mat_triangular_solve(factor, b, lower=True, unit_diagonal=False)
    return scb_mat_triangular_solve(scb_mat_conjugate_transpose(factor), y, lower=False, unit_diagonal=False)


def scb_mat_hpd_solve_plan_apply_basic(plan, b: jax.Array) -> jax.Array:
    if isinstance(plan, (sc.SparseCholeskySolvePlan, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_hpd_solve_plan_apply(plan, _as_complex_rhs(b, "scb_mat.hpd_solve_plan_apply_basic")))
    return acb_mat.acb_mat_hpd_solve_basic(plan, _dense_box_rhs(b, "scb_mat.hpd_solve_plan_apply_basic"))


def scb_mat_hpd_solve(x_or_plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, sc.SparseCholeskySolvePlan):
        return scb_mat_hpd_solve_plan_apply(x_or_plan, b)
    return scb_mat_hpd_solve_plan_apply(scb_mat_hpd_solve_plan_prepare(x_or_plan), b)


def scb_mat_hpd_solve_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_hpd_solve(x_or_plan, _as_complex_rhs(b, "scb_mat.hpd_solve_basic")))
    return acb_mat.acb_mat_hpd_solve_basic(x_or_plan, _dense_box_rhs(b, "scb_mat.hpd_solve_basic"))


def scb_mat_hpd_inv(x_or_plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    rows, _ = sc.sparse_shape(x_or_plan, algebra="scb", label="scb_mat.hpd_inv")
    eye = jnp.eye(rows, dtype=jnp.complex128)
    return scb_mat_hpd_solve(x_or_plan, eye)


def scb_mat_hpd_inv_basic(x_or_plan) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_hpd_inv(x_or_plan))
    return acb_mat.acb_mat_hpd_inv_basic(x_or_plan)


def scb_mat_solve_lu(x_or_plan: sc.SparseLUSolvePlan | tuple | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple)):
        return scb_mat_lu_solve_plan_apply(x_or_plan, b)
    return scb_mat_lu_solve_plan_apply(scb_mat_lu_solve_plan_prepare(x_or_plan), b)


def scb_mat_solve_lu_precomp(plan: sc.SparseLUSolvePlan | tuple, b: jax.Array) -> jax.Array:
    return scb_mat_lu_solve_plan_apply(plan, b)


def scb_mat_solve_lu_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_solve_lu(x_or_plan, _as_complex_rhs(b, "scb_mat.solve_lu_basic")))
    return acb_mat.acb_mat_solve_lu(x_or_plan, _dense_box_rhs(b, "scb_mat.solve_lu_basic"))


def scb_mat_solve_lu_precomp_basic(plan, b: jax.Array) -> jax.Array:
    if isinstance(plan, (sc.SparseLUSolvePlan, tuple)):
        return mat_common.box_from_point(scb_mat_solve_lu_precomp(plan, _as_complex_rhs(b, "scb_mat.solve_lu_precomp_basic")))
    return acb_mat.acb_mat_solve_lu_precomp(plan, _dense_box_rhs(b, "scb_mat.solve_lu_precomp_basic"))


def scb_mat_solve_transpose(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, sc.SparseCholeskySolvePlan):
        return scb_mat_hpd_solve_plan_apply(x_or_plan, b)
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple)):
        plan = sc.as_sparse_lu_solve_plan(x_or_plan, algebra="scb", label="scb_mat.solve_transpose")
        pb = scb_mat_matvec(scb_mat_transpose(plan.p), _as_complex_vector(b, "scb_mat.solve_transpose")) if jnp.asarray(b).ndim == 1 else scb_mat_matmul_dense_rhs(scb_mat_transpose(plan.p), _as_complex_matrix(b, "scb_mat.solve_transpose"))
        y = scb_mat_triangular_solve(scb_mat_transpose(plan.u), pb, lower=True, unit_diagonal=False)
        return scb_mat_triangular_solve(scb_mat_transpose(plan.l), y, lower=False, unit_diagonal=True)
    if bool(scb_mat_is_hpd(x_or_plan)):
        return scb_mat_hpd_solve(x_or_plan, b)
    return scb_mat_solve(scb_mat_transpose(x_or_plan), b)


def scb_mat_solve_transpose_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_solve_transpose(x_or_plan, _as_complex_rhs(b, "scb_mat.solve_transpose_basic")))
    return acb_mat.acb_mat_solve_transpose(x_or_plan, _dense_box_rhs(b, "scb_mat.solve_transpose_basic"))


def scb_mat_solve_add(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.asarray(y, dtype=jnp.complex128) + scb_mat_mat_solve(x_or_plan, b)


def scb_mat_solve_add_basic(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(
            scb_mat_solve_add(
                x_or_plan,
                _as_complex_rhs(b, "scb_mat.solve_add_basic"),
                _as_complex_rhs(y, "scb_mat.solve_add_basic.y"),
            )
        )
    return acb_mat.acb_mat_solve_add(
        _dense_box_matrix(x_or_plan, "scb_mat.solve_add_basic") if isinstance(x_or_plan, (sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)) else x_or_plan,
        _dense_box_rhs(b, "scb_mat.solve_add_basic"),
        _dense_box_rhs(y, "scb_mat.solve_add_basic.y"),
    )


def scb_mat_solve_transpose_add(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.asarray(y, dtype=jnp.complex128) + scb_mat_solve_transpose(x_or_plan, b)


def scb_mat_solve_transpose_add_basic(x_or_plan, b: jax.Array, y: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(
            scb_mat_solve_transpose_add(
                x_or_plan,
                _as_complex_rhs(b, "scb_mat.solve_transpose_add_basic"),
                _as_complex_rhs(y, "scb_mat.solve_transpose_add_basic.y"),
            )
        )
    return acb_mat.acb_mat_solve_transpose_add(
        _dense_box_matrix(x_or_plan, "scb_mat.solve_transpose_add_basic") if isinstance(x_or_plan, (sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)) else x_or_plan,
        _dense_box_rhs(b, "scb_mat.solve_transpose_add_basic"),
        _dense_box_rhs(y, "scb_mat.solve_transpose_add_basic.y"),
    )


def scb_mat_mat_solve(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, sc.SparseCholeskySolvePlan):
        return scb_mat_hpd_solve_plan_apply(x_or_plan, b)
    if isinstance(x_or_plan, (sc.SparseLUSolvePlan, tuple)):
        return scb_mat_lu_solve_plan_apply(x_or_plan, b)
    if bool(scb_mat_is_hpd(x_or_plan)):
        return scb_mat_hpd_solve(x_or_plan, b)
    return scb_mat_solve(x_or_plan, b)


def scb_mat_mat_solve_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_mat_solve(x_or_plan, _as_complex_rhs(b, "scb_mat.mat_solve_basic")))
    return acb_mat.acb_mat_mat_solve(x_or_plan, _dense_box_rhs(b, "scb_mat.mat_solve_basic"))


def scb_mat_mat_solve_transpose(x_or_plan, b: jax.Array) -> jax.Array:
    return scb_mat_solve_transpose(x_or_plan, b)


def scb_mat_mat_solve_transpose_basic(x_or_plan, b: jax.Array) -> jax.Array:
    if isinstance(x_or_plan, (sc.SparseCholeskySolvePlan, sc.SparseLUSolvePlan, tuple, sc.SparseCOO, sc.SparseCSR, sc.SparseBCOO)):
        return mat_common.box_from_point(scb_mat_mat_solve_transpose(x_or_plan, _as_complex_rhs(b, "scb_mat.mat_solve_transpose_basic")))
    return acb_mat.acb_mat_mat_solve_transpose(x_or_plan, _dense_box_rhs(b, "scb_mat.mat_solve_transpose_basic"))


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
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_complex_matrix,
        label="scb_mat.solve_batch_fixed",
        apply=lambda b: scb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart),
    )


def scb_mat_solve_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, **kwargs) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_complex_matrix, label="scb_mat.solve_basic_batch_fixed", apply=lambda b: scb_mat_solve_basic(x, b, **kwargs))


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
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_complex_matrix,
        label="scb_mat.solve_batch_padded",
        apply=lambda b: scb_mat_solve(x, b, method=method, tol=tol, atol=atol, maxiter=maxiter, restart=restart),
    )


def scb_mat_solve_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, *, pad_to: int, **kwargs) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.solve_basic_batch_padded", apply=lambda b: scb_mat_solve_basic(x, b, **kwargs))


def scb_mat_triangular_solve_batch_fixed(
    x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_complex_matrix,
        label="scb_mat.triangular_solve_batch_fixed",
        apply=lambda b: scb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal),
    )


def scb_mat_triangular_solve_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, *, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_complex_matrix, label="scb_mat.triangular_solve_basic_batch_fixed", apply=lambda b: scb_mat_triangular_solve_basic(x, b, lower=lower, unit_diagonal=unit_diagonal))


def scb_mat_triangular_solve_batch_padded(
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
        validate=_as_complex_matrix,
        label="scb_mat.triangular_solve_batch_padded",
        apply=lambda b: scb_mat_triangular_solve(x, b, lower=lower, unit_diagonal=unit_diagonal),
    )


def scb_mat_triangular_solve_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array, *, pad_to: int, lower: bool, unit_diagonal: bool = False) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.triangular_solve_basic_batch_padded", apply=lambda b: scb_mat_triangular_solve_basic(x, b, lower=lower, unit_diagonal=unit_diagonal))


def scb_mat_lu_solve_plan_apply_batch_fixed(plan: sc.SparseLUSolvePlan | tuple, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_complex_matrix,
        label="scb_mat.lu_solve_plan_apply_batch_fixed",
        apply=lambda b: scb_mat_lu_solve_plan_apply(plan, b),
    )


def scb_mat_lu_solve_plan_apply_basic_batch_fixed(plan, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_complex_matrix, label="scb_mat.lu_solve_plan_apply_basic_batch_fixed", apply=lambda b: scb_mat_lu_solve_plan_apply_basic(plan, b))


def scb_mat_lu_solve_plan_apply_batch_padded(plan: sc.SparseLUSolvePlan | tuple, bs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_complex_matrix,
        label="scb_mat.lu_solve_plan_apply_batch_padded",
        apply=lambda b: scb_mat_lu_solve_plan_apply(plan, b),
    )


def scb_mat_lu_solve_plan_apply_basic_batch_padded(plan, bs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.lu_solve_plan_apply_basic_batch_padded", apply=lambda b: scb_mat_lu_solve_plan_apply_basic(plan, b))


def scb_mat_hpd_solve_plan_apply_batch_fixed(plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        bs,
        validate=_as_complex_matrix,
        label="scb_mat.hpd_solve_plan_apply_batch_fixed",
        apply=lambda b: scb_mat_hpd_solve_plan_apply(plan, b),
    )


def scb_mat_hpd_solve_plan_apply_basic_batch_fixed(plan, bs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(bs, validate=_as_complex_matrix, label="scb_mat.hpd_solve_plan_apply_basic_batch_fixed", apply=lambda b: scb_mat_hpd_solve_plan_apply_basic(plan, b))


def scb_mat_hpd_solve_plan_apply_batch_padded(
    plan: sc.SparseCholeskySolvePlan | sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO,
    bs: jax.Array,
    *,
    pad_to: int,
) -> jax.Array:
    return sc.vmapped_batch_padded(
        bs,
        pad_to=pad_to,
        validate=_as_complex_matrix,
        label="scb_mat.hpd_solve_plan_apply_batch_padded",
        apply=lambda b: scb_mat_hpd_solve_plan_apply(plan, b),
    )


def scb_mat_hpd_solve_plan_apply_basic_batch_padded(plan, bs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(bs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.hpd_solve_plan_apply_basic_batch_padded", apply=lambda b: scb_mat_hpd_solve_plan_apply_basic(plan, b))


def scb_mat_matvec_cached_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseMatvecPlan:
    return sc.sparse_matvec_plan_from_sparse(x, algebra="scb", label="scb_mat.matvec_cached_prepare")


def scb_mat_rmatvec_cached_prepare(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseMatvecPlan:
    return scb_mat_matvec_cached_prepare(scb_mat_transpose(x))


def scb_mat_matvec_cached_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return sc.sparse_box_matvec_plan_from_sparse(
        _box_sparse_matrix(x, "scb_mat.matvec_cached_prepare_basic"),
        algebra="scb",
        label="scb_mat.matvec_cached_prepare_basic",
    )


def scb_mat_rmatvec_cached_prepare_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO):
    return scb_mat_matvec_cached_prepare_basic(scb_mat_transpose(x))


def scb_mat_matvec_cached_apply(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return sc.sparse_matvec_plan_apply(plan, _as_complex_vector(v, "scb_mat.matvec_cached_apply"), algebra="scb", label="scb_mat.matvec_cached_apply")


def scb_mat_rmatvec_cached_apply(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return scb_mat_matvec_cached_apply(plan, v)


def scb_mat_matvec_cached_apply_basic(plan, v: jax.Array) -> jax.Array:
    return sc.sparse_box_matvec_plan_apply(
        plan,
        _dense_box_vector(v, "scb_mat.matvec_cached_apply_basic"),
        algebra="scb",
        label="scb_mat.matvec_cached_apply_basic",
    )


def scb_mat_rmatvec_cached_apply_basic(plan, v: jax.Array) -> jax.Array:
    return scb_mat_matvec_cached_apply_basic(plan, v)


def scb_mat_matvec_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        vs,
        validate=_as_complex_matrix,
        label="scb_mat.matvec_batch_fixed",
        apply=lambda v: scb_mat_matvec(x, v),
    )


def scb_mat_matvec_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_complex_matrix, label="scb_mat.matvec_basic_batch_fixed", apply=lambda v: scb_mat_matvec_basic(x, v))


def scb_mat_rmatvec_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_complex_matrix, label="scb_mat.rmatvec_batch_fixed", apply=lambda v: scb_mat_rmatvec(x, v))


def scb_mat_rmatvec_basic_batch_fixed(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_complex_matrix, label="scb_mat.rmatvec_basic_batch_fixed", apply=lambda v: scb_mat_rmatvec_basic(x, v))


def scb_mat_matvec_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(
        vs,
        pad_to=pad_to,
        validate=_as_complex_matrix,
        label="scb_mat.matvec_batch_padded",
        apply=lambda v: scb_mat_matvec(x, v),
    )


def scb_mat_matvec_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.matvec_basic_batch_padded", apply=lambda v: scb_mat_matvec_basic(x, v))


def scb_mat_rmatvec_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.rmatvec_batch_padded", apply=lambda v: scb_mat_rmatvec(x, v))


def scb_mat_rmatvec_basic_batch_padded(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.rmatvec_basic_batch_padded", apply=lambda v: scb_mat_rmatvec_basic(x, v))


def scb_mat_matvec_cached_apply_batch_fixed(plan: sc.SparseMatvecPlan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(
        vs,
        validate=_as_complex_matrix,
        label="scb_mat.matvec_cached_apply_batch_fixed",
        apply=lambda v: scb_mat_matvec_cached_apply(plan, v),
    )


def scb_mat_matvec_cached_apply_basic_batch_fixed(plan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_complex_matrix, label="scb_mat.matvec_cached_apply_basic_batch_fixed", apply=lambda v: scb_mat_matvec_cached_apply_basic(plan, v))


def scb_mat_rmatvec_cached_apply_batch_fixed(plan: sc.SparseMatvecPlan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_complex_matrix, label="scb_mat.rmatvec_cached_apply_batch_fixed", apply=lambda v: scb_mat_rmatvec_cached_apply(plan, v))


def scb_mat_rmatvec_cached_apply_basic_batch_fixed(plan, vs: jax.Array) -> jax.Array:
    return sc.vmapped_batch_fixed(vs, validate=_as_complex_matrix, label="scb_mat.rmatvec_cached_apply_basic_batch_fixed", apply=lambda v: scb_mat_rmatvec_cached_apply_basic(plan, v))


def scb_mat_matvec_cached_apply_batch_padded(plan: sc.SparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(
        vs,
        pad_to=pad_to,
        validate=_as_complex_matrix,
        label="scb_mat.matvec_cached_apply_batch_padded",
        apply=lambda v: scb_mat_matvec_cached_apply(plan, v),
    )


def scb_mat_matvec_cached_apply_basic_batch_padded(plan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.matvec_cached_apply_basic_batch_padded", apply=lambda v: scb_mat_matvec_cached_apply_basic(plan, v))


def scb_mat_rmatvec_cached_apply_batch_padded(plan: sc.SparseMatvecPlan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.rmatvec_cached_apply_batch_padded", apply=lambda v: scb_mat_rmatvec_cached_apply(plan, v))


def scb_mat_rmatvec_cached_apply_basic_batch_padded(plan, vs: jax.Array, *, pad_to: int) -> jax.Array:
    return sc.vmapped_batch_padded(vs, pad_to=pad_to, validate=_as_complex_matrix, label="scb_mat.rmatvec_cached_apply_basic_batch_padded", apply=lambda v: scb_mat_rmatvec_cached_apply_basic(plan, v))


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
def scb_mat_rmatvec_jit(x, v: jax.Array) -> jax.Array:
    return scb_mat_rmatvec(x, v)


@partial(jax.jit, static_argnames=())
def scb_mat_matmul_dense_rhs_jit(x, b: jax.Array) -> jax.Array:
    return scb_mat_matmul_dense_rhs(x, b)


@partial(jax.jit, static_argnames=())
def scb_mat_matvec_cached_apply_jit(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return scb_mat_matvec_cached_apply(plan, v)


@partial(jax.jit, static_argnames=())
def scb_mat_rmatvec_cached_apply_jit(plan: sc.SparseMatvecPlan, v: jax.Array) -> jax.Array:
    return scb_mat_rmatvec_cached_apply(plan, v)


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


def scb_mat_conjugate_transpose_jit(x):
    return scb_mat_conjugate_transpose(x)


def scb_mat_hermitian_part_jit(x):
    return scb_mat_hermitian_part(x)


def scb_mat_is_hermitian_jit(x, *, rtol: float = 1e-10, atol: float = 1e-10) -> jax.Array:
    return scb_mat_is_hermitian(x, rtol=rtol, atol=atol)


def scb_mat_is_hpd_jit(x) -> jax.Array:
    return scb_mat_is_hpd(x)


def scb_mat_cho_jit(x):
    return scb_mat_cho(x)


def scb_mat_ldl_jit(x):
    return scb_mat_ldl(x)


def scb_mat_charpoly_jit(x) -> jax.Array:
    return scb_mat_charpoly(x)


@partial(jax.jit, static_argnames=("n",))
def scb_mat_pow_ui_jit(x, n: int) -> jax.Array:
    return scb_mat_pow_ui(x, n)


def scb_mat_exp_jit(x) -> jax.Array:
    return scb_mat_exp(x)


def scb_mat_eigvalsh_jit(x) -> jax.Array:
    return scb_mat_eigvalsh(x)


def scb_mat_eigh_jit(x):
    return scb_mat_eigh(x)


@partial(jax.jit, static_argnames=("k", "which", "steps"))
def scb_mat_eigsh_jit(x, *, k: int = 6, which: str = "largest", steps: int | None = None, v0: jax.Array | None = None):
    return scb_mat_eigsh(x, k=k, which=which, steps=steps, v0=v0)


def scb_mat_det_jit(x) -> jax.Array:
    return scb_mat_det(x)


def scb_mat_inv_jit(x) -> jax.Array:
    return scb_mat_inv(x)


def scb_mat_sqr_jit(x):
    return scb_mat_sqr(x)


def scb_mat_det(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_det_from_lu(x, lu_fn=scb_mat_lu, diag_fn=scb_mat_diag, to_dense_fn=scb_mat_to_dense)


def scb_mat_det_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return acb_mat.acb_mat_det_basic(_dense_box_matrix(x, "scb_mat.det_basic"))


def scb_mat_inv(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return sc.sparse_inv_via_solve(
        x,
        algebra="scb",
        label="scb_mat.inv",
        dtype=jnp.complex128,
        solve_fn=lambda a, col: scb_mat_solve(a, col, method="gmres", tol=1e-10),
    )


def scb_mat_inv_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> jax.Array:
    return acb_mat.acb_mat_inv_basic(_dense_box_matrix(x, "scb_mat.inv_basic"))


def scb_mat_sqr(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    return sc.sparse_square_via_matmul(x, matmul_sparse_fn=scb_mat_matmul_sparse)


def scb_mat_sqr_basic(x: sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO) -> sc.SparseCOO | sc.SparseCSR | sc.SparseBCOO:
    return acb_mat.acb_mat_sqr_basic(_dense_box_matrix(x, "scb_mat.sqr_basic"))


__all__ = [
    "scb_mat_coo",
    "scb_mat_csr",
    "scb_mat_bcoo",
    "scb_mat_box_coo",
    "scb_mat_box_csr",
    "scb_mat_box_bcoo",
    "scb_mat_to_box_sparse",
    "scb_mat_box_to_dense",
    "scb_mat_box_transpose",
    "scb_mat_box_conjugate_transpose",
    "scb_mat_box_add",
    "scb_mat_box_scale",
    "scb_mat_box_matvec",
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
    "scb_mat_transpose_basic",
    "scb_mat_conjugate_transpose",
    "scb_mat_conjugate_transpose_basic",
    "scb_mat_hermitian_part",
    "scb_mat_hermitian_part_basic",
    "scb_mat_is_hermitian",
    "scb_mat_is_hermitian_basic",
    "scb_mat_is_hpd",
    "scb_mat_is_hpd_basic",
    "scb_mat_cho",
    "scb_mat_cho_basic",
    "scb_mat_ldl",
    "scb_mat_ldl_basic",
    "scb_mat_charpoly",
    "scb_mat_charpoly_basic",
    "scb_mat_pow_ui",
    "scb_mat_pow_ui_basic",
    "scb_mat_exp",
    "scb_mat_exp_basic",
    "scb_mat_eigvalsh",
    "scb_mat_eigvalsh_basic",
    "scb_mat_eigh",
    "scb_mat_eigh_basic",
    "scb_mat_eigsh",
    "scb_mat_eigsh_basic",
    "scb_mat_operator_plan_prepare",
    "scb_mat_operator_rmatvec_plan_prepare",
    "scb_mat_operator_adjoint_plan_prepare",
    "scb_mat_scale",
    "scb_mat_add",
    "scb_mat_sub",
    "scb_mat_matvec",
    "scb_mat_rmatvec",
    "scb_mat_matvec_basic",
    "scb_mat_rmatvec_basic",
    "scb_mat_matvec_cached_prepare",
    "scb_mat_matvec_cached_prepare_basic",
    "scb_mat_rmatvec_cached_prepare",
    "scb_mat_rmatvec_cached_prepare_basic",
    "scb_mat_matvec_cached_apply",
    "scb_mat_matvec_cached_apply_basic",
    "scb_mat_rmatvec_cached_apply",
    "scb_mat_rmatvec_cached_apply_basic",
    "scb_mat_triangular_solve",
    "scb_mat_triangular_solve_basic",
    "scb_mat_triangular_solve_batch_fixed",
    "scb_mat_triangular_solve_basic_batch_fixed",
    "scb_mat_triangular_solve_batch_padded",
    "scb_mat_triangular_solve_basic_batch_padded",
    "scb_mat_triangular_solve_jit",
    "scb_mat_lu",
    "scb_mat_lu_solve",
    "scb_mat_lu_solve_plan_prepare",
    "scb_mat_lu_solve_plan_prepare_basic",
    "scb_mat_lu_solve_plan_apply",
    "scb_mat_lu_solve_plan_apply_basic",
    "scb_mat_lu_solve_plan_apply_batch_fixed",
    "scb_mat_lu_solve_plan_apply_basic_batch_fixed",
    "scb_mat_lu_solve_plan_apply_batch_padded",
    "scb_mat_lu_solve_plan_apply_basic_batch_padded",
    "scb_mat_solve_lu",
    "scb_mat_solve_lu_basic",
    "scb_mat_solve_lu_precomp",
    "scb_mat_solve_lu_precomp_basic",
    "scb_mat_qr",
    "scb_mat_qr_r",
    "scb_mat_qr_apply_q",
    "scb_mat_qr_explicit_q",
    "scb_mat_qr_solve",
    "scb_mat_solve",
    "scb_mat_solve_basic",
    "scb_mat_solve_transpose",
    "scb_mat_solve_transpose_basic",
    "scb_mat_solve_add",
    "scb_mat_solve_add_basic",
    "scb_mat_solve_transpose_add",
    "scb_mat_solve_transpose_add_basic",
    "scb_mat_mat_solve",
    "scb_mat_mat_solve_basic",
    "scb_mat_mat_solve_transpose",
    "scb_mat_mat_solve_transpose_basic",
    "scb_mat_solve_batch_fixed",
    "scb_mat_solve_basic_batch_fixed",
    "scb_mat_solve_batch_padded",
    "scb_mat_solve_basic_batch_padded",
    "scb_mat_solve_jit",
    "scb_mat_hpd_solve_plan_prepare",
    "scb_mat_hpd_solve_plan_prepare_basic",
    "scb_mat_hpd_solve_plan_apply",
    "scb_mat_hpd_solve_plan_apply_basic",
    "scb_mat_hpd_solve_plan_apply_batch_fixed",
    "scb_mat_hpd_solve_plan_apply_basic_batch_fixed",
    "scb_mat_hpd_solve_plan_apply_batch_padded",
    "scb_mat_hpd_solve_plan_apply_basic_batch_padded",
    "scb_mat_hpd_solve",
    "scb_mat_hpd_solve_basic",
    "scb_mat_hpd_inv",
    "scb_mat_hpd_inv_basic",
    "scb_mat_matvec",
    "scb_mat_matmul_dense_rhs",
    "scb_mat_matmul_dense_rhs_basic",
    "scb_mat_matmul_sparse",
    "scb_mat_matvec_cached_prepare",
    "scb_mat_matvec_cached_apply",
    "scb_mat_matvec_batch_fixed",
    "scb_mat_matvec_basic_batch_fixed",
    "scb_mat_matvec_batch_padded",
    "scb_mat_matvec_basic_batch_padded",
    "scb_mat_matvec_cached_apply_batch_fixed",
    "scb_mat_matvec_cached_apply_basic_batch_fixed",
    "scb_mat_matvec_cached_apply_batch_padded",
    "scb_mat_matvec_cached_apply_basic_batch_padded",
    "scb_mat_rmatvec_batch_fixed",
    "scb_mat_rmatvec_basic_batch_fixed",
    "scb_mat_rmatvec_batch_padded",
    "scb_mat_rmatvec_basic_batch_padded",
    "scb_mat_rmatvec_cached_apply_batch_fixed",
    "scb_mat_rmatvec_cached_apply_basic_batch_fixed",
    "scb_mat_rmatvec_cached_apply_batch_padded",
    "scb_mat_rmatvec_cached_apply_basic_batch_padded",
    "scb_mat_coo_to_dense_jit",
    "scb_mat_csr_to_dense_jit",
    "scb_mat_bcoo_to_dense_jit",
    "scb_mat_matvec_jit",
    "scb_mat_rmatvec_jit",
    "scb_mat_matmul_dense_rhs_jit",
    "scb_mat_matvec_cached_apply_jit",
    "scb_mat_rmatvec_cached_apply_jit",
    "scb_mat_matvec_with_diagnostics",
    "scb_mat_matvec_cached_apply_with_diagnostics",
    "scb_mat_solve_with_diagnostics",
    "scb_mat_lu_with_diagnostics",
    "scb_mat_qr_with_diagnostics",
    "scb_mat_trace_jit",
    "scb_mat_norm_fro_jit",
    "scb_mat_norm_1_jit",
    "scb_mat_norm_inf_jit",
    "scb_mat_conjugate_transpose_jit",
    "scb_mat_hermitian_part_jit",
    "scb_mat_is_hermitian_jit",
    "scb_mat_is_hpd_jit",
    "scb_mat_cho_jit",
    "scb_mat_ldl_jit",
    "scb_mat_charpoly_jit",
    "scb_mat_pow_ui_jit",
    "scb_mat_exp_jit",
    "scb_mat_eigvalsh_jit",
    "scb_mat_eigh_jit",
    "scb_mat_eigsh_jit",
    "scb_mat_det",
    "scb_mat_det_basic",
    "scb_mat_det_jit",
    "scb_mat_inv",
    "scb_mat_inv_basic",
    "scb_mat_inv_jit",
    "scb_mat_sqr",
    "scb_mat_sqr_basic",
    "scb_mat_sqr_jit",
    "scb_mat_to_dense_basic",
    "scb_mat_trace_basic",
    "scb_mat_norm_fro_basic",
    "scb_mat_norm_1_basic",
    "scb_mat_norm_inf_basic",
    "ScbMatPointDiagnostics",
]
