from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import ops


class SparseNativePolicyDiagnostics(NamedTuple):
    sparse_native: jax.Array
    dense_lift_used: jax.Array
    preserves_sparse_output: jax.Array
    structured_input_required: jax.Array
    rows: jax.Array
    cols: jax.Array
    nnz: jax.Array


def sparse_native_policy_diagnostics(
    x,
    *,
    sparse_native: bool,
    dense_lift_used: bool,
    preserves_sparse_output: bool,
    structured_input_required: bool = False,
) -> SparseNativePolicyDiagnostics:
    return SparseNativePolicyDiagnostics(
        sparse_native=jnp.asarray(sparse_native),
        dense_lift_used=jnp.asarray(dense_lift_used),
        preserves_sparse_output=jnp.asarray(preserves_sparse_output),
        structured_input_required=jnp.asarray(structured_input_required),
        rows=jnp.asarray(getattr(x, "rows")),
        cols=jnp.asarray(getattr(x, "cols")),
        nnz=jnp.asarray(getattr(x, "data").shape[0], dtype=jnp.int32),
    )


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


def _sparse_entry_map(coo):
    entry_map = {}
    for idx in range(int(coo.data.shape[0])):
        key = (int(coo.row[idx]), int(coo.col[idx]))
        value = coo.data[idx]
        if key in entry_map:
            entry_map[key] = entry_map[key] + value
        else:
            entry_map[key] = value
    return entry_map


def sparse_ldl_structural(
    x,
    *,
    to_coo_fn,
    structured_part_fn,
    is_structured_fn,
    dtype,
    hermitian: bool,
):
    if not bool(is_structured_fn(x)):
        raise ValueError("structured sparse LDL requires symmetric/Hermitian input")
    coo = to_coo_fn(structured_part_fn(x))
    n = int(coo.rows)
    zero = jnp.asarray(0, dtype=dtype)
    entry_map = _sparse_entry_map(coo)
    l_cols: list[dict[int, jax.Array]] = [{i: jnp.asarray(1, dtype=dtype)} for i in range(n)]
    d_values: list[jax.Array] = []
    for k in range(n):
        diag = entry_map.get((k, k), zero)
        for s in range(k):
            lks = l_cols[s].get(k, zero)
            if hermitian:
                diag = diag - lks * jnp.conj(lks) * d_values[s]
            else:
                diag = diag - lks * lks * d_values[s]
        d_values.append(diag)
        for i in range(k + 1, n):
            accum = entry_map.get((i, k), zero)
            for s in range(k):
                lis = l_cols[s].get(i, zero)
                lks = l_cols[s].get(k, zero)
                if hermitian:
                    accum = accum - lis * jnp.conj(lks) * d_values[s]
                else:
                    accum = accum - lis * lks * d_values[s]
            lik = accum / diag
            if bool(jnp.abs(lik) > 0.0):
                l_cols[k][i] = lik
    row = []
    col = []
    data = []
    for j in range(n):
        for i, value in sorted(l_cols[j].items()):
            row.append(i)
            col.append(j)
            data.append(value)
    if data:
        data_arr = jnp.stack(data, axis=0)
        row_arr = jnp.asarray(row, dtype=jnp.int32)
        col_arr = jnp.asarray(col, dtype=jnp.int32)
    else:
        data_arr = jnp.zeros((0,), dtype=dtype)
        row_arr = jnp.zeros((0,), dtype=jnp.int32)
        col_arr = jnp.zeros((0,), dtype=jnp.int32)
    d_arr = jnp.stack(d_values, axis=0) if d_values else jnp.zeros((0,), dtype=dtype)
    return data_arr, row_arr, col_arr, d_arr


def sparse_cho_structural(
    x,
    *,
    to_coo_fn,
    structured_part_fn,
    is_structured_fn,
    dtype,
    hermitian: bool,
):
    data, row, col, d = sparse_ldl_structural(
        x,
        to_coo_fn=to_coo_fn,
        structured_part_fn=structured_part_fn,
        is_structured_fn=is_structured_fn,
        dtype=dtype,
        hermitian=hermitian,
    )
    sqrt_d = jnp.sqrt(d)
    scaled = data * sqrt_d[col]
    return scaled, row, col


def sparse_is_pd_structural(
    x,
    *,
    to_coo_fn,
    structured_part_fn,
    is_structured_fn,
    dtype,
    hermitian: bool,
):
    if not bool(is_structured_fn(x)):
        return jnp.asarray(False)
    _, _, _, d = sparse_ldl_structural(
        x,
        to_coo_fn=to_coo_fn,
        structured_part_fn=structured_part_fn,
        is_structured_fn=is_structured_fn,
        dtype=dtype,
        hermitian=hermitian,
    )
    if hermitian:
        finite = jnp.all(jnp.isfinite(jnp.real(d))) & jnp.all(jnp.isfinite(jnp.imag(d)))
        positive = jnp.all(jnp.real(d) > 0.0)
    else:
        finite = jnp.all(jnp.isfinite(d))
        positive = jnp.all(d > 0.0)
    return finite & positive


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
    del to_dense_fn, from_dense_csr_fn, complex_
    coo = csr_to_coo(csr)
    n = int(coo.rows)
    zero = jnp.asarray(0, dtype=coo.data.dtype)
    u_rows = [dict() for _ in range(n)]
    for idx in range(int(coo.data.shape[0])):
        row = int(coo.row[idx])
        col = int(coo.col[idx])
        value = coo.data[idx]
        if col in u_rows[row]:
            u_rows[row][col] = u_rows[row][col] + value
        else:
            u_rows[row][col] = value
    l_rows = [{i: jnp.asarray(1, dtype=coo.data.dtype)} for i in range(n)]
    perm = list(range(n))
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: float(jnp.abs(u_rows[i].get(k, zero))))
        if pivot != k:
            u_rows[k], u_rows[pivot] = u_rows[pivot], u_rows[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
            for j in range(k):
                vk = l_rows[k].get(j, zero)
                vp = l_rows[pivot].get(j, zero)
                if bool(jnp.abs(vk) > 0.0):
                    l_rows[pivot][j] = vk
                else:
                    l_rows[pivot].pop(j, None)
                if bool(jnp.abs(vp) > 0.0):
                    l_rows[k][j] = vp
                else:
                    l_rows[k].pop(j, None)
        pivot_value = u_rows[k].get(k, zero)
        for i in range(k + 1, n):
            entry = u_rows[i].get(k, zero)
            if not bool(jnp.abs(entry) > 0.0):
                continue
            factor = entry / pivot_value
            l_rows[i][k] = factor
            pivot_row = list(u_rows[k].items())
            for j, value in pivot_row:
                if j < k:
                    continue
                updated = u_rows[i].get(j, zero) - factor * value
                if bool(jnp.abs(updated) > 0.0):
                    u_rows[i][j] = updated
                else:
                    u_rows[i].pop(j, None)
            u_rows[i].pop(k, None)
    l_data = []
    l_row = []
    l_col = []
    u_data = []
    u_row = []
    u_col = []
    for i in range(n):
        for j, value in sorted(l_rows[i].items()):
            if j <= i and bool(jnp.abs(value) > 0.0):
                l_data.append(value)
                l_row.append(i)
                l_col.append(j)
        for j, value in sorted(u_rows[i].items()):
            if j >= i and bool(jnp.abs(value) > 0.0):
                u_data.append(value)
                u_row.append(i)
                u_col.append(j)
    l_data_arr = jnp.stack(l_data, axis=0) if l_data else jnp.zeros((0,), dtype=coo.data.dtype)
    l_row_arr = jnp.asarray(l_row, dtype=jnp.int32) if l_row else jnp.zeros((0,), dtype=jnp.int32)
    l_col_arr = jnp.asarray(l_col, dtype=jnp.int32) if l_col else jnp.zeros((0,), dtype=jnp.int32)
    u_data_arr = jnp.stack(u_data, axis=0) if u_data else jnp.zeros((0,), dtype=coo.data.dtype)
    u_row_arr = jnp.asarray(u_row, dtype=jnp.int32) if u_row else jnp.zeros((0,), dtype=jnp.int32)
    u_col_arr = jnp.asarray(u_col, dtype=jnp.int32) if u_col else jnp.zeros((0,), dtype=jnp.int32)
    p = permutation_matrix_fn(jnp.asarray(perm, dtype=jnp.int32))
    l_sparse = type(csr)(
        data=l_data_arr,
        indices=l_col_arr,
        indptr=coo_to_csr_indptr(l_row_arr, rows=n),
        rows=n,
        cols=n,
        algebra=csr.algebra,
    )
    u_sparse = type(csr)(
        data=u_data_arr,
        indices=u_col_arr,
        indptr=coo_to_csr_indptr(u_row_arr, rows=n),
        rows=n,
        cols=n,
        algebra=csr.algebra,
    )
    return p, l_sparse, u_sparse


__all__ = [
    "SparseNativePolicyDiagnostics",
    "sparse_native_policy_diagnostics",
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
    "sparse_ldl_structural",
    "sparse_cho_structural",
    "sparse_is_pd_structural",
    "sparse_lowest_eig_real_symmetric",
    "sparse_lowest_eig_hermitian",
    "sparse_is_spd_structural",
    "sparse_is_hpd_structural",
    "sparse_lu_via_jax_dense",
]


def sparse_direct_solve(x, b, *, to_dense_fn):
    return jnp.linalg.solve(to_dense_fn(x), b)


def csr_to_coo(csr):
    row = jnp.repeat(jnp.arange(csr.rows, dtype=jnp.int32), jnp.diff(csr.indptr))
    return type("SparseCOOTmp", (), {"data": csr.data, "row": row, "col": csr.indices, "rows": csr.rows, "cols": csr.cols})()


def coo_to_csr_indptr(row: jax.Array, *, rows: int) -> jax.Array:
    counts = jnp.bincount(row, length=rows)
    return jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])


def sparse_dense_power_ui(x, n: int, *, to_bcoo_fn, matmul_sparse_fn, to_dense_fn, identity_sparse_fn):
    if n < 0:
        raise ValueError("n must be >= 0")
    base = to_bcoo_fn(x, label="sparse_core.power_ui.base")
    result = identity_sparse_fn(base.rows, dtype=base.data.dtype)
    exp = int(n)
    while exp > 0:
        if exp & 1:
            result = matmul_sparse_fn(result, base)
        exp >>= 1
        if exp:
            base = matmul_sparse_fn(base, base)
    return to_dense_fn(result)


def sparse_charpoly_from_traces(x, *, to_bcoo_fn, matmul_sparse_fn, trace_fn, identity_sparse_fn):
    a = to_bcoo_fn(x, label="sparse_core.charpoly.base")
    n = int(a.rows)
    power = identity_sparse_fn(n, dtype=a.data.dtype)
    power_sums = []
    for _ in range(n):
        power = matmul_sparse_fn(power, a)
        power_sums.append(trace_fn(power))
    dtype = jnp.asarray(power_sums[0]).dtype if power_sums else a.data.dtype
    e = [jnp.asarray(1, dtype=dtype)]
    coeffs = [jnp.asarray(1, dtype=dtype)]
    for k in range(1, n + 1):
        accum = jnp.asarray(0, dtype=dtype)
        for i in range(1, k + 1):
            accum = accum + ((-1) ** (i - 1)) * e[k - i] * power_sums[i - 1]
        e_k = accum / jnp.asarray(k, dtype=dtype)
        e.append(e_k)
        coeffs.append(((-1) ** k) * e_k)
    return jnp.stack(coeffs, axis=0)


def sparse_dense_exp_taylor(x, *, to_bcoo_fn, matmul_sparse_fn, to_dense_fn, identity_sparse_fn, terms: int = 24):
    a = to_bcoo_fn(x, label="sparse_core.exp.base")
    n = int(a.rows)
    dense_out = jnp.eye(n, dtype=a.data.dtype)
    term = identity_sparse_fn(n, dtype=a.data.dtype)
    factorial_dtype = jnp.real(jnp.asarray(0, dtype=a.data.dtype)).dtype
    factorial = jnp.asarray(1.0, dtype=factorial_dtype)
    for k in range(1, int(terms) + 1):
        term = matmul_sparse_fn(term, a)
        factorial = factorial * jnp.asarray(k, dtype=factorial.dtype)
        dense_out = dense_out + to_dense_fn(term) / factorial
    return dense_out


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
    "sparse_ldl_structural",
    "sparse_cho_structural",
    "sparse_is_pd_structural",
    "sparse_lowest_eig_real_symmetric",
    "sparse_lowest_eig_hermitian",
    "sparse_is_spd_structural",
    "sparse_is_hpd_structural",
    "sparse_lu_via_jax_dense",
    "sparse_direct_solve",
    "sparse_dense_power_ui",
    "sparse_charpoly_from_traces",
    "sparse_dense_exp_taylor",
]
