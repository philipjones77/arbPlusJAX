from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import ops


def swap_rows_dense_lax(a: jax.Array, i: int, j: int) -> jax.Array:
    def swap_body(arr):
        row_i = arr[i, :]
        row_j = arr[j, :]
        arr = arr.at[i, :].set(row_j)
        return arr.at[j, :].set(row_i)

    return jax.lax.cond(i == j, lambda x: x, swap_body, a)


def swap_rows_dense(a: jax.Array, i: int, j: int) -> jax.Array:
    if i == j:
        return a
    row_i = a[i, :]
    row_j = a[j, :]
    a = a.at[i, :].set(row_j)
    return a.at[j, :].set(row_i)


def swap_perm_lax(perm: jax.Array, i: int, j: int) -> jax.Array:
    def swap_body(arr):
        vi = arr[i]
        vj = arr[j]
        arr = arr.at[i].set(vj)
        return arr.at[j].set(vi)

    return jax.lax.cond(i == j, lambda x: x, swap_body, perm)


def swap_perm(perm: jax.Array, i: int, j: int) -> jax.Array:
    if i == j:
        return perm
    vi = perm[i]
    vj = perm[j]
    perm = perm.at[i].set(vj)
    return perm.at[j].set(vi)


def dense_sparse_lu_partial_pivot_lax(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = a.shape[0]
    u = jnp.array(a, copy=True)
    l = jnp.eye(n, dtype=a.dtype)
    perm = jnp.arange(n, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=a.dtype)
    for k in range(n):
        pivot = int(k + jnp.argmax(jnp.abs(u[k:, k])))
        u = swap_rows_dense_lax(u, k, pivot)
        if k > 0 and pivot != k:
            l_k = l[k, :k]
            l_p = l[pivot, :k]
            l = l.at[k, :k].set(l_p)
            l = l.at[pivot, :k].set(l_k)
        perm = swap_perm_lax(perm, k, pivot)
        pivot_value = u[k, k]
        if k + 1 < n:
            factors = u[k + 1 :, k] / pivot_value
            l = l.at[k + 1 :, k].set(factors)
            u = u.at[k + 1 :, k:].set(u[k + 1 :, k:] - factors[:, None] * u[k, k:][None, :])
            u = u.at[k + 1 :, k].set(zero)
    return perm, l, u


def dense_sparse_lu_partial_pivot(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = a.shape[0]
    u = jnp.array(a, copy=True)
    l = jnp.eye(n, dtype=a.dtype)
    perm = jnp.arange(n, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=a.dtype)
    for k in range(n):
        pivot = int(k + jnp.argmax(jnp.abs(u[k:, k])))
        u = swap_rows_dense(u, k, pivot)
        if k > 0 and pivot != k:
            l_k = l[k, :k]
            l_p = l[pivot, :k]
            l = l.at[k, :k].set(l_p)
            l = l.at[pivot, :k].set(l_k)
        perm = swap_perm(perm, k, pivot)
        pivot_value = u[k, k]
        if k + 1 < n:
            factors = u[k + 1 :, k] / pivot_value
            l = l.at[k + 1 :, k].set(factors)
            u = u.at[k + 1 :, k:].set(u[k + 1 :, k:] - factors[:, None] * u[k, k:][None, :])
            u = u.at[k + 1 :, k].set(zero)
    return perm, l, u


def dense_householder_qr_real(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    m, n = a.shape
    k = min(m, n)
    work = jnp.array(a, copy=True)
    reflectors = jnp.zeros((m, k), dtype=a.dtype)
    taus = jnp.zeros((k,), dtype=a.dtype)
    for j in range(k):
        x = work[j:, j]
        x0 = x[0]
        normx = jnp.linalg.norm(x)
        sign = jnp.where(x0 >= 0.0, 1.0, -1.0)
        alpha = -sign * normx
        v = x.at[0].add(-alpha)
        beta = jnp.vdot(v, v)
        tau = jnp.where(beta > 0.0, 2.0 / beta, 0.0)
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


def dense_householder_qr_complex(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
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


def sparse_midpoint_symmetric_part(x, *, as_csr_fn, to_dense_fn, from_dense_csr_fn):
    dense = to_dense_fn(as_csr_fn(x, label="sparse_core.symmetric_part"))
    return from_dense_csr_fn(0.5 * (dense + dense.T))


def sparse_midpoint_hermitian_part(x, *, as_csr_fn, to_dense_fn, from_dense_csr_fn):
    dense = to_dense_fn(as_csr_fn(x, label="sparse_core.hermitian_part"))
    return from_dense_csr_fn(0.5 * (dense + jnp.conj(dense.T)))


def sparse_midpoint_is_symmetric(x, *, as_csr_fn, to_dense_fn, rtol: float, atol: float):
    dense = to_dense_fn(as_csr_fn(x, label="sparse_core.is_symmetric"))
    return jnp.all(jnp.isclose(dense, dense.T, rtol=rtol, atol=atol))


def sparse_midpoint_is_hermitian(x, *, as_csr_fn, to_dense_fn, rtol: float, atol: float):
    dense = to_dense_fn(as_csr_fn(x, label="sparse_core.is_hermitian"))
    return jnp.all(jnp.isclose(dense, jnp.conj(dense.T), rtol=rtol, atol=atol))


def sparse_midpoint_is_spd(x, *, as_csr_fn, to_dense_fn, is_symmetric_fn):
    dense = to_dense_fn(as_csr_fn(x, label="sparse_core.is_spd"))
    sym = 0.5 * (dense + dense.T)
    chol = jnp.linalg.cholesky(sym)
    return is_symmetric_fn(x) & jnp.all(jnp.isfinite(chol))


def sparse_midpoint_is_hpd(x, *, as_csr_fn, to_dense_fn, is_hermitian_fn):
    dense = to_dense_fn(as_csr_fn(x, label="sparse_core.is_hpd"))
    herm = 0.5 * (dense + jnp.conj(dense.T))
    chol = jnp.linalg.cholesky(herm)
    return is_hermitian_fn(x) & jnp.all(jnp.isfinite(chol))


def sparse_midpoint_cho_from_symmetric_part(x, *, structured_part_fn, to_dense_fn, from_dense_csr_fn):
    dense = to_dense_fn(structured_part_fn(x))
    return from_dense_csr_fn(jnp.tril(jnp.linalg.cholesky(dense)))


def sparse_midpoint_ldl_from_cho(x, *, cho_fn, to_dense_fn, diag_map_fn):
    chol = to_dense_fn(cho_fn(x))
    diag = jnp.diagonal(chol)
    l = chol / diag[None, :]
    d = diag_map_fn(diag)
    return l, d


def coalesce_coo(data: jax.Array, row: jax.Array, col: jax.Array, *, rows: int, cols: int):
    nnz = data.shape[0]
    if nnz == 0:
        return data, row, col
    key = row.astype(jnp.int64) * jnp.int64(cols) + col.astype(jnp.int64)
    order = jnp.argsort(key)
    data_s = data[order]
    row_s = row[order]
    col_s = col[order]
    key_s = key[order]
    head = jnp.concatenate([jnp.asarray([True]), key_s[1:] != key_s[:-1]], axis=0)
    seg = jnp.cumsum(head.astype(jnp.int32)) - 1
    out_nnz = int(jnp.sum(head))
    data_c = ops.segment_sum(data_s, seg, num_segments=out_nnz)
    row_c = row_s[head]
    col_c = col_s[head]
    return data_c, row_c, col_c


def sparse_diag(x, *, to_coo_fn, dtype):
    coo = to_coo_fn(x)
    diag = jnp.zeros((coo.rows,), dtype=dtype)
    mask = coo.row == coo.col
    return diag.at[coo.row[mask]].add(coo.data[mask])


def sparse_norm_1(x, *, to_coo_fn):
    coo = to_coo_fn(x)
    sums = ops.segment_sum(jnp.abs(coo.data), coo.col, num_segments=coo.cols)
    return jnp.max(sums) if sums.shape[0] else jnp.asarray(0.0, dtype=jnp.float64)


def sparse_norm_inf(x, *, to_coo_fn):
    coo = to_coo_fn(x)
    sums = ops.segment_sum(jnp.abs(coo.data), coo.row, num_segments=coo.rows)
    return jnp.max(sums) if sums.shape[0] else jnp.asarray(0.0, dtype=jnp.float64)


def sparse_matmul_sparse(x, y, *, to_bcoo_fn, matmul_sparse_fn):
    return matmul_sparse_fn(
        to_bcoo_fn(x, label="sparse_core.matmul_sparse.x"),
        to_bcoo_fn(y, label="sparse_core.matmul_sparse.y"),
    )


def sparse_structured_part_real(x, *, to_coo_fn, from_coo_fn):
    coo = to_coo_fn(x)
    data = jnp.concatenate([0.5 * coo.data, 0.5 * coo.data], axis=0)
    row = jnp.concatenate([coo.row, coo.col], axis=0)
    col = jnp.concatenate([coo.col, coo.row], axis=0)
    data, row, col = coalesce_coo(data, row, col, rows=coo.rows, cols=coo.cols)
    return from_coo_fn(data, row, col, shape=(coo.rows, coo.cols))


def sparse_structured_part_hermitian(x, *, to_coo_fn, from_coo_fn):
    coo = to_coo_fn(x)
    data = jnp.concatenate([0.5 * coo.data, 0.5 * jnp.conj(coo.data)], axis=0)
    row = jnp.concatenate([coo.row, coo.col], axis=0)
    col = jnp.concatenate([coo.col, coo.row], axis=0)
    data, row, col = coalesce_coo(data, row, col, rows=coo.rows, cols=coo.cols)
    return from_coo_fn(data, row, col, shape=(coo.rows, coo.cols))


def sparse_is_symmetric_structural(x, *, to_coo_fn, rtol: float, atol: float):
    coo = to_coo_fn(x)
    data = jnp.concatenate([coo.data, -coo.data], axis=0)
    row = jnp.concatenate([coo.row, coo.col], axis=0)
    col = jnp.concatenate([coo.col, coo.row], axis=0)
    data, _, _ = coalesce_coo(data, row, col, rows=coo.rows, cols=coo.cols)
    if data.shape[0] == 0:
        return jnp.asarray(True)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(coo.data))) if coo.data.shape[0] else jnp.asarray(1.0)
    tol = atol + rtol * scale
    return jnp.max(jnp.abs(data)) <= tol


def sparse_is_hermitian_structural(x, *, to_coo_fn, rtol: float, atol: float):
    coo = to_coo_fn(x)
    data = jnp.concatenate([coo.data, -jnp.conj(coo.data)], axis=0)
    row = jnp.concatenate([coo.row, coo.col], axis=0)
    col = jnp.concatenate([coo.col, coo.row], axis=0)
    data, _, _ = coalesce_coo(data, row, col, rows=coo.rows, cols=coo.cols)
    if data.shape[0] == 0:
        return jnp.asarray(True)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(coo.data))) if coo.data.shape[0] else jnp.asarray(1.0)
    tol = atol + rtol * scale
    return jnp.max(jnp.abs(data)) <= tol


def sparse_lowest_eig_real_symmetric(x, *, to_dense_fn):
    vals = jnp.linalg.eigvalsh(to_dense_fn(x))
    return vals[0]


def sparse_lowest_eig_hermitian(x, *, to_dense_fn):
    vals = jnp.linalg.eigvalsh(to_dense_fn(x))
    return jnp.real(vals[0])


def sparse_is_spd_structural(x, *, is_symmetric_fn, structured_part_fn, to_dense_fn):
    if not bool(is_symmetric_fn(x)):
        return jnp.asarray(False)
    lam_min = sparse_lowest_eig_real_symmetric(structured_part_fn(x), to_dense_fn=to_dense_fn)
    return lam_min > 0.0


def sparse_is_hpd_structural(x, *, is_hermitian_fn, structured_part_fn, to_dense_fn):
    if not bool(is_hermitian_fn(x)):
        return jnp.asarray(False)
    lam_min = sparse_lowest_eig_hermitian(structured_part_fn(x), to_dense_fn=to_dense_fn)
    return lam_min > 0.0


def sparse_lu_via_jax_dense(x, *, as_csr_fn, to_dense_fn, from_dense_csr_fn, permutation_matrix_fn, complex_: bool):
    csr = as_csr_fn(x, label="sparse_core.lu")
    a = to_dense_fn(csr)
    perm, l, u = dense_sparse_lu_partial_pivot(a) if complex_ else dense_sparse_lu_partial_pivot_lax(a)
    p = permutation_matrix_fn(perm)
    l_sparse = from_dense_csr_fn(jnp.tril(l))
    u_sparse = from_dense_csr_fn(jnp.triu(u))
    return p, l_sparse, u_sparse


def sparse_direct_solve(x, b, *, to_dense_fn):
    return jnp.linalg.solve(to_dense_fn(x), b)


__all__ = [
    "swap_rows_dense_lax",
    "swap_rows_dense",
    "swap_perm_lax",
    "swap_perm",
    "dense_sparse_lu_partial_pivot_lax",
    "dense_sparse_lu_partial_pivot",
    "dense_householder_qr_real",
    "dense_householder_qr_complex",
    "sparse_midpoint_symmetric_part",
    "sparse_midpoint_hermitian_part",
    "sparse_midpoint_is_symmetric",
    "sparse_midpoint_is_hermitian",
    "sparse_midpoint_is_spd",
    "sparse_midpoint_is_hpd",
    "sparse_midpoint_cho_from_symmetric_part",
    "sparse_midpoint_ldl_from_cho",
    "coalesce_coo",
    "sparse_diag",
    "sparse_norm_1",
    "sparse_norm_inf",
    "sparse_matmul_sparse",
    "sparse_structured_part_real",
    "sparse_structured_part_hermitian",
    "sparse_is_symmetric_structural",
    "sparse_is_hermitian_structural",
    "sparse_lowest_eig_real_symmetric",
    "sparse_lowest_eig_hermitian",
    "sparse_is_spd_structural",
    "sparse_is_hpd_structural",
    "sparse_lu_via_jax_dense",
    "sparse_direct_solve",
]
