from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import acb_core
from . import arb_core
from . import checks
from . import double_interval as di
from . import kernel_helpers as kh
from . import mat_common



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseCOO:
    data: jax.Array
    row: jax.Array
    col: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.row, self.col), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, row, col = children
        return cls(data=data, row=row, col=col, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseCSR:
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices, indptr = children
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseBCOO:
    data: jax.Array
    indices: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices = children
        return cls(data=data, indices=indices, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseIntervalCOO:
    data: jax.Array
    row: jax.Array
    col: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.row, self.col), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, row, col = children
        return cls(data=data, row=row, col=col, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseIntervalCSR:
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices, indptr = children
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseIntervalBCOO:
    data: jax.Array
    indices: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices = children
        return cls(data=data, indices=indices, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseBoxCOO:
    data: jax.Array
    row: jax.Array
    col: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.row, self.col), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, row, col = children
        return cls(data=data, row=row, col=col, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseBoxCSR:
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices, indptr = children
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseBoxBCOO:
    data: jax.Array
    indices: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices = children
        return cls(data=data, indices=indices, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseMatvecPlan:
    storage: str
    payload: object
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.payload,), {"storage": self.storage, "rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload,) = children
        return cls(
            storage=aux_data["storage"],
            payload=payload,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseQRFactor:
    reflectors: jax.Array
    taus: jax.Array
    r_factor: SparseCSR
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.reflectors, self.taus, self.r_factor), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        reflectors, taus, r_factor = children
        return cls(
            reflectors=reflectors,
            taus=taus,
            r_factor=r_factor,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseLUSolvePlan:
    p: SparseCOO | SparseCSR | SparseBCOO
    l: SparseCOO | SparseCSR | SparseBCOO
    u: SparseCOO | SparseCSR | SparseBCOO
    rows: int
    algebra: str

    def tree_flatten(self):
        return (self.p, self.l, self.u), {"rows": self.rows, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        p, l, u = children
        return cls(p=p, l=l, u=u, rows=aux_data["rows"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseCholeskySolvePlan:
    factor: SparseCOO | SparseCSR | SparseBCOO
    rows: int
    algebra: str
    structure: str

    def tree_flatten(self):
        return (self.factor,), {"rows": self.rows, "algebra": self.algebra, "structure": self.structure}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (factor,) = children
        return cls(factor=factor, rows=aux_data["rows"], algebra=aux_data["algebra"], structure=aux_data["structure"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockSparseCOO:
    data: jax.Array
    row: jax.Array
    col: jax.Array
    block_rows: int
    block_cols: int
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.row, self.col), {
            "block_rows": self.block_rows,
            "block_cols": self.block_cols,
            "rows": self.rows,
            "cols": self.cols,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, row, col = children
        return cls(
            data=data,
            row=row,
            col=col,
            block_rows=aux_data["block_rows"],
            block_cols=aux_data["block_cols"],
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockSparseCSR:
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    block_rows: int
    block_cols: int
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr), {
            "block_rows": self.block_rows,
            "block_cols": self.block_cols,
            "rows": self.rows,
            "cols": self.cols,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices, indptr = children
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            block_rows=aux_data["block_rows"],
            block_cols=aux_data["block_cols"],
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockSparseMatvecPlan:
    storage: str
    payload: object
    block_rows: int
    block_cols: int
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.payload,), {
            "storage": self.storage,
            "block_rows": self.block_rows,
            "block_cols": self.block_cols,
            "rows": self.rows,
            "cols": self.cols,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload,) = children
        return cls(
            storage=aux_data["storage"],
            payload=payload,
            block_rows=aux_data["block_rows"],
            block_cols=aux_data["block_cols"],
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VariableBlockSparseCOO:
    data: jax.Array
    row: jax.Array
    col: jax.Array
    row_block_sizes: jax.Array
    col_block_sizes: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.row, self.col, self.row_block_sizes, self.col_block_sizes), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, row, col, row_block_sizes, col_block_sizes = children
        return cls(
            data=data,
            row=row,
            col=col,
            row_block_sizes=row_block_sizes,
            col_block_sizes=col_block_sizes,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VariableBlockSparseCSR:
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    row_block_sizes: jax.Array
    col_block_sizes: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr, self.row_block_sizes, self.col_block_sizes), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, indices, indptr, row_block_sizes, col_block_sizes = children
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            row_block_sizes=row_block_sizes,
            col_block_sizes=col_block_sizes,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VariableBlockSparseMatvecPlan:
    storage: str
    payload: object
    row_block_sizes: jax.Array
    col_block_sizes: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.payload, self.row_block_sizes, self.col_block_sizes), {
            "storage": self.storage,
            "rows": self.rows,
            "cols": self.cols,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        payload, row_block_sizes, col_block_sizes = children
        return cls(
            storage=aux_data["storage"],
            payload=payload,
            row_block_sizes=row_block_sizes,
            col_block_sizes=col_block_sizes,
            rows=aux_data["rows"],
            cols=aux_data["cols"],
            algebra=aux_data["algebra"],
        )


def as_sparse_coo(x: SparseCOO, *, algebra: str, label: str) -> SparseCOO:
    checks.check(isinstance(x, SparseCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.row).ndim, 1, f"{label}.row_ndim")
    checks.check_equal(jnp.asarray(x.col).ndim, 1, f"{label}.col_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 1, f"{label}.data_ndim")
    checks.check_equal(x.row.shape[0], x.col.shape[0], f"{label}.nnz_idx")
    checks.check_equal(x.row.shape[0], x.data.shape[0], f"{label}.nnz_data")
    return x


def as_sparse_csr(x: SparseCSR, *, algebra: str, label: str) -> SparseCSR:
    checks.check(isinstance(x, SparseCSR), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.indices).ndim, 1, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.indptr).ndim, 1, f"{label}.indptr_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 1, f"{label}.data_ndim")
    checks.check_equal(x.indices.shape[0], x.data.shape[0], f"{label}.nnz")
    checks.check_equal(x.indptr.shape[0], x.rows + 1, f"{label}.indptr_rows")
    return x


def as_sparse_bcoo(x: SparseBCOO, *, algebra: str, label: str) -> SparseBCOO:
    checks.check(isinstance(x, SparseBCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.indices).ndim, 2, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 1, f"{label}.data_ndim")
    checks.check_equal(x.indices.shape[0], x.data.shape[0], f"{label}.nnz")
    checks.check_equal(x.indices.shape[1], 2, f"{label}.index_width")
    return x


def as_sparse_interval_coo(x: SparseIntervalCOO, *, algebra: str, label: str) -> SparseIntervalCOO:
    checks.check(isinstance(x, SparseIntervalCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.row).ndim, 1, f"{label}.row_ndim")
    checks.check_equal(jnp.asarray(x.col).ndim, 1, f"{label}.col_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 2, f"{label}.data_ndim")
    checks.check_equal(x.data.shape[-1], 2, f"{label}.data_tail")
    checks.check_equal(x.row.shape[0], x.col.shape[0], f"{label}.nnz_idx")
    checks.check_equal(x.row.shape[0], x.data.shape[0], f"{label}.nnz_data")
    return x


def as_sparse_interval_csr(x: SparseIntervalCSR, *, algebra: str, label: str) -> SparseIntervalCSR:
    checks.check(isinstance(x, SparseIntervalCSR), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.indices).ndim, 1, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.indptr).ndim, 1, f"{label}.indptr_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 2, f"{label}.data_ndim")
    checks.check_equal(x.data.shape[-1], 2, f"{label}.data_tail")
    checks.check_equal(x.indices.shape[0], x.data.shape[0], f"{label}.nnz")
    checks.check_equal(x.indptr.shape[0], x.rows + 1, f"{label}.indptr_rows")
    return x


def as_sparse_interval_bcoo(x: SparseIntervalBCOO, *, algebra: str, label: str) -> SparseIntervalBCOO:
    checks.check(isinstance(x, SparseIntervalBCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.indices).ndim, 2, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 2, f"{label}.data_ndim")
    checks.check_equal(x.data.shape[-1], 2, f"{label}.data_tail")
    checks.check_equal(x.indices.shape[0], x.data.shape[0], f"{label}.nnz")
    checks.check_equal(x.indices.shape[1], 2, f"{label}.index_width")
    return x


def as_sparse_box_coo(x: SparseBoxCOO, *, algebra: str, label: str) -> SparseBoxCOO:
    checks.check(isinstance(x, SparseBoxCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.row).ndim, 1, f"{label}.row_ndim")
    checks.check_equal(jnp.asarray(x.col).ndim, 1, f"{label}.col_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 2, f"{label}.data_ndim")
    checks.check_equal(x.data.shape[-1], 4, f"{label}.data_tail")
    checks.check_equal(x.row.shape[0], x.col.shape[0], f"{label}.nnz_idx")
    checks.check_equal(x.row.shape[0], x.data.shape[0], f"{label}.nnz_data")
    return x


def as_sparse_box_csr(x: SparseBoxCSR, *, algebra: str, label: str) -> SparseBoxCSR:
    checks.check(isinstance(x, SparseBoxCSR), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.indices).ndim, 1, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.indptr).ndim, 1, f"{label}.indptr_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 2, f"{label}.data_ndim")
    checks.check_equal(x.data.shape[-1], 4, f"{label}.data_tail")
    checks.check_equal(x.indices.shape[0], x.data.shape[0], f"{label}.nnz")
    checks.check_equal(x.indptr.shape[0], x.rows + 1, f"{label}.indptr_rows")
    return x


def as_sparse_box_bcoo(x: SparseBoxBCOO, *, algebra: str, label: str) -> SparseBoxBCOO:
    checks.check(isinstance(x, SparseBoxBCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.indices).ndim, 2, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.data).ndim, 2, f"{label}.data_ndim")
    checks.check_equal(x.data.shape[-1], 4, f"{label}.data_tail")
    checks.check_equal(x.indices.shape[0], x.data.shape[0], f"{label}.nnz")
    checks.check_equal(x.indices.shape[1], 2, f"{label}.index_width")
    return x


def csr_row_ids(indptr: jax.Array, *, rows: int, nnz: int) -> jax.Array:
    counts = indptr[1:] - indptr[:-1]
    return jnp.repeat(jnp.arange(rows, dtype=indptr.dtype), counts, total_repeat_length=nnz)


def coo_to_bcoo(x: SparseCOO) -> SparseBCOO:
    x = as_sparse_coo(x, algebra=x.algebra, label="sparse_common.coo_to_bcoo")
    return SparseBCOO(
        data=x.data,
        indices=jnp.stack([x.row, x.col], axis=-1),
        rows=x.rows,
        cols=x.cols,
        algebra=x.algebra,
    )


def bcoo_to_sparse_bcoo(x: SparseBCOO, *, algebra: str) -> SparseBCOO:
    x = as_sparse_bcoo(x, algebra=x.algebra, label="sparse_common.bcoo_to_sparse_bcoo")
    return SparseBCOO(data=x.data, indices=x.indices, rows=x.rows, cols=x.cols, algebra=algebra)


def dense_to_sparse_bcoo(a: jax.Array, *, algebra: str, tol: float = 0.0) -> SparseBCOO:
    arr = jnp.asarray(a)
    mask = jnp.abs(arr) > jnp.asarray(tol, dtype=arr.dtype)
    row, col = jnp.nonzero(mask, size=int(mask.size), fill_value=-1)
    valid = row >= 0
    safe_row = jnp.where(valid, row, 0)
    safe_col = jnp.where(valid, col, 0)
    data = jnp.where(valid, arr[safe_row, safe_col], jnp.zeros_like(safe_row, dtype=arr.dtype))
    indices = jnp.stack([safe_row, safe_col], axis=-1)
    return SparseBCOO(
        data=data,
        indices=jnp.asarray(indices, dtype=jnp.int32),
        rows=int(arr.shape[0]),
        cols=int(arr.shape[1]),
        algebra=algebra,
    )


def scipy_csr_to_sparse_bcoo(csr, *, algebra: str, dtype=None) -> SparseBCOO:
    coo = csr.tocoo()
    target_dtype = dtype if dtype is not None else coo.data.dtype
    data = jnp.asarray(coo.data, dtype=target_dtype)
    indices = jnp.stack(
        [
            jnp.asarray(coo.row, dtype=jnp.int32),
            jnp.asarray(coo.col, dtype=jnp.int32),
        ],
        axis=-1,
    )
    return SparseBCOO(
        data=data,
        indices=indices,
        rows=int(coo.shape[0]),
        cols=int(coo.shape[1]),
        algebra=algebra,
    )


def sparse_bcoo_to_dense(x: SparseBCOO, *, algebra: str, label: str) -> jax.Array:
    x = as_sparse_bcoo(x, algebra=algebra, label=label)
    out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
    return out.at[(x.indices[:, 0], x.indices[:, 1])].add(x.data)


def sparse_bcoo_matvec(x: SparseBCOO, v: jax.Array, *, algebra: str, label: str) -> jax.Array:
    x = as_sparse_bcoo(x, algebra=algebra, label=label)
    vv = jnp.asarray(v, dtype=x.data.dtype)
    checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
    contrib = x.data * vv[x.indices[:, 1]]
    return jax.ops.segment_sum(contrib, x.indices[:, 0], num_segments=x.rows)


def sparse_bcoo_matmul_dense_rhs(x: SparseBCOO, b: jax.Array, *, algebra: str, label: str) -> jax.Array:
    x = as_sparse_bcoo(x, algebra=algebra, label=label)
    bb = jnp.asarray(b, dtype=x.data.dtype)
    checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
    contrib = x.data[:, None] * bb[x.indices[:, 1], :]
    return jax.ops.segment_sum(contrib, x.indices[:, 0], num_segments=x.rows)


def sparse_bcoo_add(x: SparseBCOO, y: SparseBCOO, *, algebra: str, label: str) -> SparseBCOO:
    x = as_sparse_bcoo(x, algebra=algebra, label=f"{label}.x")
    y = as_sparse_bcoo(y, algebra=algebra, label=f"{label}.y")
    checks.check_equal(x.rows, y.rows, f"{label}.rows")
    checks.check_equal(x.cols, y.cols, f"{label}.cols")
    data = jnp.concatenate([x.data, y.data], axis=0)
    indices = jnp.concatenate([x.indices, y.indices], axis=0)
    return SparseBCOO(data=data, indices=indices, rows=x.rows, cols=x.cols, algebra=algebra)


def sparse_bcoo_matmul_sparse(x: SparseBCOO, y: SparseBCOO, *, algebra: str, label: str) -> SparseBCOO:
    x = as_sparse_bcoo(x, algebra=algebra, label=f"{label}.x")
    y = as_sparse_bcoo(y, algebra=algebra, label=f"{label}.y")
    checks.check_equal(x.cols, y.rows, f"{label}.inner")
    product = sparse_bcoo_to_dense(x, algebra=algebra, label=f"{label}.x_dense") @ sparse_bcoo_to_dense(
        y,
        algebra=algebra,
        label=f"{label}.y_dense",
    )
    return dense_to_sparse_bcoo(product, algebra=algebra)


def sparse_matvec_plan_from_sparse(
    x: SparseCOO | SparseCSR | SparseBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseMatvecPlan:
    if isinstance(x, SparseCOO):
        x = as_sparse_coo(x, algebra=algebra, label=label)
        payload = (x.data, x.row, x.col)
        storage = "coo"
    elif isinstance(x, SparseCSR):
        x = as_sparse_csr(x, algebra=algebra, label=label)
        payload = (x.data, x.indices, csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0]))
        storage = "csr"
    else:
        x = as_sparse_bcoo(x, algebra=algebra, label=label)
        payload = x
        storage = "bcoo"
    return SparseMatvecPlan(storage=storage, payload=payload, rows=x.rows, cols=x.cols, algebra=algebra)


def sparse_matvec_plan_apply(plan: SparseMatvecPlan, v: jax.Array, *, algebra: str, label: str) -> jax.Array:
    plan = as_sparse_matvec_plan(plan, algebra=algebra, label=label)
    vv = jnp.asarray(v)
    checks.check_equal(vv.ndim, 1, f"{label}.vector_ndim")
    checks.check_equal(plan.cols, vv.shape[0], f"{label}.inner")
    if plan.storage == "coo":
        data, row, col = plan.payload
        return jax.ops.segment_sum(data * vv[col], row, num_segments=plan.rows)
    if plan.storage == "csr":
        data, indices, row_ids = plan.payload
        return jax.ops.segment_sum(data * vv[indices], row_ids, num_segments=plan.rows)
    return sparse_bcoo_matvec(plan.payload, vv, algebra=algebra, label=f"{label}.bcoo")


def sparse_interval_matvec_plan_from_sparse(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseMatvecPlan:
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        payload = (x.data, x.row, x.col)
        storage = "coo"
    elif isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        payload = (x.data, x.indices, csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0]))
        storage = "csr"
    else:
        x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
        payload = x
        storage = "bcoo"
    return SparseMatvecPlan(storage=storage, payload=payload, rows=x.rows, cols=x.cols, algebra=algebra)


def sparse_interval_matvec_plan_apply(plan: SparseMatvecPlan, v: jax.Array, *, algebra: str, label: str) -> jax.Array:
    plan = as_sparse_matvec_plan(plan, algebra=algebra, label=label)
    vv = mat_common.as_interval_vector(v, f"{label}.v")
    checks.check_equal(plan.cols, vv.shape[0], f"{label}.inner")
    if plan.storage == "coo":
        data, row, col = plan.payload
        return _segment_interval_sum(di.fast_mul(data, vv[col]), row, num_segments=plan.rows)
    if plan.storage == "csr":
        data, indices, row_ids = plan.payload
        return _segment_interval_sum(di.fast_mul(data, vv[indices]), row_ids, num_segments=plan.rows)
    return sparse_interval_matvec(plan.payload, vv, algebra=algebra, label=f"{label}.bcoo")


def sparse_box_matvec_plan_from_sparse(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseMatvecPlan:
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        payload = (x.data, x.row, x.col)
        storage = "coo"
    elif isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        payload = (x.data, x.indices, csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0]))
        storage = "csr"
    else:
        x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
        payload = x
        storage = "bcoo"
    return SparseMatvecPlan(storage=storage, payload=payload, rows=x.rows, cols=x.cols, algebra=algebra)


def sparse_box_matvec_plan_apply(plan: SparseMatvecPlan, v: jax.Array, *, algebra: str, label: str) -> jax.Array:
    plan = as_sparse_matvec_plan(plan, algebra=algebra, label=label)
    vv = mat_common.as_box_vector(v, f"{label}.v")
    checks.check_equal(plan.cols, vv.shape[0], f"{label}.inner")
    if plan.storage == "coo":
        data, row, col = plan.payload
        return _segment_box_sum(acb_core.acb_mul(data, vv[col]), row, num_segments=plan.rows)
    if plan.storage == "csr":
        data, indices, row_ids = plan.payload
        return _segment_box_sum(acb_core.acb_mul(data, vv[indices]), row_ids, num_segments=plan.rows)
    return sparse_box_matvec(plan.payload, vv, algebra=algebra, label=f"{label}.bcoo")


def as_sparse_matvec_plan(x: SparseMatvecPlan, *, algebra: str, label: str) -> SparseMatvecPlan:
    checks.check(isinstance(x, SparseMatvecPlan), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check(x.storage in ("coo", "csr", "bcoo"), f"{label}.storage")
    return x


def as_sparse_lu_solve_plan(x: SparseLUSolvePlan | tuple, *, algebra: str, label: str) -> SparseLUSolvePlan:
    if isinstance(x, SparseLUSolvePlan):
        checks.check(x.algebra == algebra, f"{label}.algebra")
        return x
    checks.check(isinstance(x, tuple) and len(x) == 3, f"{label}.tuple")
    p, l, u = x
    rows_p, cols_p = sparse_shape(p, algebra=algebra, label=f"{label}.p")
    rows_l, cols_l = sparse_shape(l, algebra=algebra, label=f"{label}.l")
    rows_u, cols_u = sparse_shape(u, algebra=algebra, label=f"{label}.u")
    checks.check_equal(rows_p, cols_p, f"{label}.p_square")
    checks.check_equal(rows_l, cols_l, f"{label}.l_square")
    checks.check_equal(rows_u, cols_u, f"{label}.u_square")
    checks.check_equal(rows_p, rows_l, f"{label}.rows_l")
    checks.check_equal(rows_p, rows_u, f"{label}.rows_u")
    return SparseLUSolvePlan(p=p, l=l, u=u, rows=int(rows_p), algebra=algebra)


def sparse_lu_solve_plan_from_factors(
    p: SparseCOO | SparseCSR | SparseBCOO,
    l: SparseCOO | SparseCSR | SparseBCOO,
    u: SparseCOO | SparseCSR | SparseBCOO,
    *,
    algebra: str,
) -> SparseLUSolvePlan:
    return as_sparse_lu_solve_plan((p, l, u), algebra=algebra, label="sparse_common.sparse_lu_solve_plan_from_factors")


def as_sparse_cholesky_solve_plan(
    x: SparseCholeskySolvePlan | SparseCOO | SparseCSR | SparseBCOO,
    *,
    algebra: str,
    structure: str,
    label: str,
) -> SparseCholeskySolvePlan:
    if isinstance(x, SparseCholeskySolvePlan):
        checks.check(x.algebra == algebra, f"{label}.algebra")
        checks.check(x.structure == structure, f"{label}.structure")
        return x
    rows, cols = sparse_shape(x, algebra=algebra, label=label)
    checks.check_equal(rows, cols, f"{label}.square")
    return SparseCholeskySolvePlan(factor=x, rows=int(rows), algebra=algebra, structure=structure)


def sparse_cholesky_solve_plan_from_factor(
    factor: SparseCOO | SparseCSR | SparseBCOO,
    *,
    algebra: str,
    structure: str,
) -> SparseCholeskySolvePlan:
    return as_sparse_cholesky_solve_plan(
        factor,
        algebra=algebra,
        structure=structure,
        label="sparse_common.sparse_cholesky_solve_plan_from_factor",
    )


def as_sparse_qr_factor(x: SparseQRFactor, *, algebra: str, label: str) -> SparseQRFactor:
    checks.check(isinstance(x, SparseQRFactor), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.reflectors).ndim, 2, f"{label}.reflectors_ndim")
    checks.check_equal(jnp.asarray(x.taus).ndim, 1, f"{label}.taus_ndim")
    checks.check_equal(x.reflectors.shape[0], x.rows, f"{label}.reflector_rows")
    checks.check_equal(x.reflectors.shape[1], x.taus.shape[0], f"{label}.reflector_cols")
    return x


def sparse_shape(
    x: SparseCOO | SparseCSR | SparseBCOO | SparseMatvecPlan | SparseLUSolvePlan | SparseCholeskySolvePlan,
    *,
    algebra: str,
    label: str,
) -> tuple[int, int]:
    if isinstance(x, SparseMatvecPlan):
        plan = as_sparse_matvec_plan(x, algebra=algebra, label=label)
        return plan.rows, plan.cols
    if isinstance(x, SparseLUSolvePlan):
        plan = as_sparse_lu_solve_plan(x, algebra=algebra, label=label)
        return plan.rows, plan.rows
    if isinstance(x, SparseCholeskySolvePlan):
        plan = as_sparse_cholesky_solve_plan(x, algebra=algebra, structure=x.structure, label=label)
        return plan.rows, plan.rows
    if isinstance(x, SparseCOO):
        x = as_sparse_coo(x, algebra=algebra, label=label)
        return x.rows, x.cols
    if isinstance(x, SparseCSR):
        x = as_sparse_csr(x, algebra=algebra, label=label)
        return x.rows, x.cols
    x = as_sparse_bcoo(x, algebra=algebra, label=label)
    return x.rows, x.cols


def sparse_det_from_lu(
    x,
    *,
    lu_fn,
    diag_fn,
    to_dense_fn,
):
    p, _, u = lu_fn(x)
    return jnp.linalg.det(to_dense_fn(p)) * jnp.prod(diag_fn(u))


def sparse_inv_via_solve(
    x,
    *,
    algebra: str,
    label: str,
    dtype,
    solve_fn,
):
    rows, _ = sparse_shape(x, algebra=algebra, label=label)
    eye = jnp.eye(rows, dtype=dtype)
    return jax.vmap(lambda col: solve_fn(x, col), in_axes=1, out_axes=1)(eye)


def sparse_square_via_matmul(x, *, matmul_sparse_fn):
    return matmul_sparse_fn(x, x)


def sparse_to_dense(x, *, algebra: str, label: str) -> jax.Array:
    if isinstance(x, SparseCOO):
        x = as_sparse_coo(x, algebra=algebra, label=label)
        out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
        return out.at[(x.row, x.col)].add(x.data)
    if isinstance(x, SparseCSR):
        x = as_sparse_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        out = jnp.zeros((x.rows, x.cols), dtype=x.data.dtype)
        return out.at[(row_ids, x.indices)].add(x.data)
    return sparse_bcoo_to_dense(x, algebra=algebra, label=label)


def sparse_real_to_interval_matrix(x, *, algebra: str, label: str) -> jax.Array:
    return mat_common.interval_from_point(sparse_to_dense(x, algebra=algebra, label=label))


def sparse_complex_to_box_matrix(x, *, algebra: str, label: str) -> jax.Array:
    return mat_common.box_from_point(sparse_to_dense(x, algebra=algebra, label=label))


def sparse_real_to_interval_sparse(
    x: SparseCOO | SparseCSR | SparseBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO:
    if isinstance(x, SparseCOO):
        x = as_sparse_coo(x, algebra=algebra, label=label)
        return SparseIntervalCOO(
            data=mat_common.interval_from_point(x.data),
            row=x.row,
            col=x.col,
            rows=x.rows,
            cols=x.cols,
            algebra=algebra,
        )
    if isinstance(x, SparseCSR):
        x = as_sparse_csr(x, algebra=algebra, label=label)
        return SparseIntervalCSR(
            data=mat_common.interval_from_point(x.data),
            indices=x.indices,
            indptr=x.indptr,
            rows=x.rows,
            cols=x.cols,
            algebra=algebra,
        )
    x = as_sparse_bcoo(x, algebra=algebra, label=label)
    return SparseIntervalBCOO(
        data=mat_common.interval_from_point(x.data),
        indices=x.indices,
        rows=x.rows,
        cols=x.cols,
        algebra=algebra,
    )


def sparse_complex_to_box_sparse(
    x: SparseCOO | SparseCSR | SparseBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO:
    if isinstance(x, SparseCOO):
        x = as_sparse_coo(x, algebra=algebra, label=label)
        return SparseBoxCOO(
            data=mat_common.box_from_point(x.data),
            row=x.row,
            col=x.col,
            rows=x.rows,
            cols=x.cols,
            algebra=algebra,
        )
    if isinstance(x, SparseCSR):
        x = as_sparse_csr(x, algebra=algebra, label=label)
        return SparseBoxCSR(
            data=mat_common.box_from_point(x.data),
            indices=x.indices,
            indptr=x.indptr,
            rows=x.rows,
            cols=x.cols,
            algebra=algebra,
        )
    x = as_sparse_bcoo(x, algebra=algebra, label=label)
    return SparseBoxBCOO(
        data=mat_common.box_from_point(x.data),
        indices=x.indices,
        rows=x.rows,
        cols=x.cols,
        algebra=algebra,
    )


def sparse_interval_to_dense(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    zero = mat_common.interval_from_point(jnp.zeros((x.rows, x.cols), dtype=jnp.asarray(x.data).dtype))
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        return zero.at[(x.row, x.col)].add(x.data)
    if isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        return zero.at[(row_ids, x.indices)].add(x.data)
    x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
    return zero.at[(x.indices[:, 0], x.indices[:, 1])].add(x.data)


def sparse_box_to_dense(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    zero = mat_common.box_from_point(jnp.zeros((x.rows, x.cols), dtype=jnp.complex128))
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        return zero.at[(x.row, x.col)].add(x.data)
    if isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        return zero.at[(row_ids, x.indices)].add(x.data)
    x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
    return zero.at[(x.indices[:, 0], x.indices[:, 1])].add(x.data)


def _segment_interval_sum(values: jax.Array, segment_ids: jax.Array, *, num_segments: int) -> jax.Array:
    lo = jax.ops.segment_sum(values[..., 0], segment_ids, num_segments=num_segments)
    hi = jax.ops.segment_sum(values[..., 1], segment_ids, num_segments=num_segments)
    return di.interval(di._below(lo), di._above(hi))


def _segment_box_sum(values: jax.Array, segment_ids: jax.Array, *, num_segments: int) -> jax.Array:
    re = _segment_interval_sum(values[..., 0:2], segment_ids, num_segments=num_segments)
    im = _segment_interval_sum(values[..., 2:4], segment_ids, num_segments=num_segments)
    return acb_core.acb_box(re, im)


def sparse_interval_trace(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    zero = mat_common.interval_from_point(jnp.asarray(0.0, dtype=jnp.float64))
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        mask = x.row == x.col
        return mat_common.interval_sum(jnp.where(mask[:, None], x.data, zero[None, :]), axis=0)
    if isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        mask = row_ids == x.indices
        return mat_common.interval_sum(jnp.where(mask[:, None], x.data, zero[None, :]), axis=0)
    x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
    mask = x.indices[:, 0] == x.indices[:, 1]
    return mat_common.interval_sum(jnp.where(mask[:, None], x.data, zero[None, :]), axis=0)


def sparse_box_trace(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    zero = mat_common.box_from_point(jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        mask = x.row == x.col
        return mat_common.box_sum(jnp.where(mask[:, None], x.data, zero[None, :]), axis=0)
    if isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        mask = row_ids == x.indices
        return mat_common.box_sum(jnp.where(mask[:, None], x.data, zero[None, :]), axis=0)
    x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
    mask = x.indices[:, 0] == x.indices[:, 1]
    return mat_common.box_sum(jnp.where(mask[:, None], x.data, zero[None, :]), axis=0)


def sparse_interval_norm_1(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(arb_core.arb_abs(x.data), x.col, num_segments=x.cols)
    elif isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(arb_core.arb_abs(x.data), x.indices, num_segments=x.cols)
    else:
        x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(arb_core.arb_abs(x.data), x.indices[:, 1], num_segments=x.cols)
    return mat_common.interval_from_point(jnp.max(di.midpoint(sums), axis=-1))


def sparse_interval_norm_inf(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(arb_core.arb_abs(x.data), x.row, num_segments=x.rows)
    elif isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        sums = _segment_interval_sum(arb_core.arb_abs(x.data), row_ids, num_segments=x.rows)
    else:
        x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(arb_core.arb_abs(x.data), x.indices[:, 0], num_segments=x.rows)
    return mat_common.interval_from_point(jnp.max(di.midpoint(sums), axis=-1))


def sparse_box_norm_1(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(acb_core.acb_abs(x.data), x.col, num_segments=x.cols)
    elif isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(acb_core.acb_abs(x.data), x.indices, num_segments=x.cols)
    else:
        x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(acb_core.acb_abs(x.data), x.indices[:, 1], num_segments=x.cols)
    return mat_common.box_from_point(jnp.max(di.midpoint(sums), axis=-1))


def sparse_box_norm_inf(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(acb_core.acb_abs(x.data), x.row, num_segments=x.rows)
    elif isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        sums = _segment_interval_sum(acb_core.acb_abs(x.data), row_ids, num_segments=x.rows)
    else:
        x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
        sums = _segment_interval_sum(acb_core.acb_abs(x.data), x.indices[:, 0], num_segments=x.rows)
    return mat_common.box_from_point(jnp.max(di.midpoint(sums), axis=-1))


def sparse_interval_matvec(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    v: jax.Array,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    vv = mat_common.as_interval_vector(v, f"{label}.v")
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
        contrib = di.fast_mul(x.data, vv[x.col])
        return _segment_interval_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = di.fast_mul(x.data, vv[x.indices])
        return _segment_interval_sum(contrib, row_ids, num_segments=x.rows)
    x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
    checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
    contrib = di.fast_mul(x.data, vv[x.indices[:, 1]])
    return _segment_interval_sum(contrib, x.indices[:, 0], num_segments=x.rows)


def sparse_box_matvec(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    v: jax.Array,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    vv = mat_common.as_box_vector(v, f"{label}.v")
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
        contrib = acb_core.acb_mul(x.data, vv[x.col])
        return _segment_box_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = acb_core.acb_mul(x.data, vv[x.indices])
        return _segment_box_sum(contrib, row_ids, num_segments=x.rows)
    x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
    checks.check_equal(x.cols, vv.shape[0], f"{label}.inner")
    contrib = acb_core.acb_mul(x.data, vv[x.indices[:, 1]])
    return _segment_box_sum(contrib, x.indices[:, 0], num_segments=x.rows)


def sparse_interval_matmul_dense_rhs(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    b: jax.Array,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    bb = mat_common.as_interval_rhs(b, f"{label}.b")
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
        contrib = di.fast_mul(x.data[:, None, :], bb[x.col, :, :])
        return _segment_interval_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = di.fast_mul(x.data[:, None, :], bb[x.indices, :, :])
        return _segment_interval_sum(contrib, row_ids, num_segments=x.rows)
    x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
    checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
    contrib = di.fast_mul(x.data[:, None, :], bb[x.indices[:, 1], :, :])
    return _segment_interval_sum(contrib, x.indices[:, 0], num_segments=x.rows)


def sparse_box_matmul_dense_rhs(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    b: jax.Array,
    *,
    algebra: str,
    label: str,
) -> jax.Array:
    bb = mat_common.as_box_rhs(b, f"{label}.b")
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
        contrib = acb_core.acb_mul(x.data[:, None, :], bb[x.col, :, :])
        return _segment_box_sum(contrib, x.row, num_segments=x.rows)
    if isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        contrib = acb_core.acb_mul(x.data[:, None, :], bb[x.indices, :, :])
        return _segment_box_sum(contrib, row_ids, num_segments=x.rows)
    x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
    checks.check_equal(x.cols, bb.shape[0], f"{label}.inner")
    contrib = acb_core.acb_mul(x.data[:, None, :], bb[x.indices[:, 1], :, :])
    return _segment_box_sum(contrib, x.indices[:, 0], num_segments=x.rows)


def sparse_interval_transpose(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseIntervalBCOO:
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label=label)
        return SparseIntervalBCOO(
            data=x.data,
            indices=jnp.stack([x.col, x.row], axis=-1),
            rows=x.cols,
            cols=x.rows,
            algebra=algebra,
        )
    if isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        return SparseIntervalBCOO(
            data=x.data,
            indices=jnp.stack([x.indices, row_ids], axis=-1),
            rows=x.cols,
            cols=x.rows,
            algebra=algebra,
        )
    x = as_sparse_interval_bcoo(x, algebra=algebra, label=label)
    return SparseIntervalBCOO(data=x.data, indices=x.indices[:, ::-1], rows=x.cols, cols=x.rows, algebra=algebra)


def sparse_box_transpose(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseBoxBCOO:
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label=label)
        return SparseBoxBCOO(
            data=x.data,
            indices=jnp.stack([x.col, x.row], axis=-1),
            rows=x.cols,
            cols=x.rows,
            algebra=algebra,
        )
    if isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label=label)
        row_ids = csr_row_ids(x.indptr, rows=x.rows, nnz=x.data.shape[0])
        return SparseBoxBCOO(
            data=x.data,
            indices=jnp.stack([x.indices, row_ids], axis=-1),
            rows=x.cols,
            cols=x.rows,
            algebra=algebra,
        )
    x = as_sparse_box_bcoo(x, algebra=algebra, label=label)
    return SparseBoxBCOO(data=x.data, indices=x.indices[:, ::-1], rows=x.cols, cols=x.rows, algebra=algebra)


def sparse_box_conjugate_transpose(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseBoxBCOO:
    tx = sparse_box_transpose(x, algebra=algebra, label=label)
    return SparseBoxBCOO(data=acb_core.acb_conj(tx.data), indices=tx.indices, rows=tx.rows, cols=tx.cols, algebra=algebra)


def _as_sparse_interval_bcoo(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseIntervalBCOO:
    if isinstance(x, SparseIntervalBCOO):
        return as_sparse_interval_bcoo(x, algebra=algebra, label=label)
    return sparse_interval_transpose(sparse_interval_transpose(x, algebra=algebra, label=label), algebra=algebra, label=f"{label}.bcoo")


def _as_sparse_box_bcoo(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseBoxBCOO:
    if isinstance(x, SparseBoxBCOO):
        return as_sparse_box_bcoo(x, algebra=algebra, label=label)
    return sparse_box_transpose(sparse_box_transpose(x, algebra=algebra, label=label), algebra=algebra, label=f"{label}.bcoo")


def sparse_interval_add(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    y: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseIntervalBCOO:
    xb = _as_sparse_interval_bcoo(x, algebra=algebra, label=f"{label}.x")
    yb = _as_sparse_interval_bcoo(y, algebra=algebra, label=f"{label}.y")
    checks.check_equal(xb.rows, yb.rows, f"{label}.rows")
    checks.check_equal(xb.cols, yb.cols, f"{label}.cols")
    return SparseIntervalBCOO(
        data=jnp.concatenate([xb.data, yb.data], axis=0),
        indices=jnp.concatenate([xb.indices, yb.indices], axis=0),
        rows=xb.rows,
        cols=xb.cols,
        algebra=algebra,
    )


def sparse_box_add(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    y: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    *,
    algebra: str,
    label: str,
) -> SparseBoxBCOO:
    xb = _as_sparse_box_bcoo(x, algebra=algebra, label=f"{label}.x")
    yb = _as_sparse_box_bcoo(y, algebra=algebra, label=f"{label}.y")
    checks.check_equal(xb.rows, yb.rows, f"{label}.rows")
    checks.check_equal(xb.cols, yb.cols, f"{label}.cols")
    return SparseBoxBCOO(
        data=jnp.concatenate([xb.data, yb.data], axis=0),
        indices=jnp.concatenate([xb.indices, yb.indices], axis=0),
        rows=xb.rows,
        cols=xb.cols,
        algebra=algebra,
    )


def sparse_interval_scale(
    x: SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO,
    alpha: jax.Array,
    *,
    algebra: str,
) -> SparseIntervalCOO | SparseIntervalCSR | SparseIntervalBCOO:
    alpha_iv = mat_common.interval_from_point(jnp.asarray(alpha, dtype=jnp.float64))
    if isinstance(x, SparseIntervalCOO):
        x = as_sparse_interval_coo(x, algebra=algebra, label="sparse_common.sparse_interval_scale")
        return SparseIntervalCOO(data=di.fast_mul(x.data, alpha_iv), row=x.row, col=x.col, rows=x.rows, cols=x.cols, algebra=algebra)
    if isinstance(x, SparseIntervalCSR):
        x = as_sparse_interval_csr(x, algebra=algebra, label="sparse_common.sparse_interval_scale")
        return SparseIntervalCSR(data=di.fast_mul(x.data, alpha_iv), indices=x.indices, indptr=x.indptr, rows=x.rows, cols=x.cols, algebra=algebra)
    x = as_sparse_interval_bcoo(x, algebra=algebra, label="sparse_common.sparse_interval_scale")
    return SparseIntervalBCOO(data=di.fast_mul(x.data, alpha_iv), indices=x.indices, rows=x.rows, cols=x.cols, algebra=algebra)


def sparse_box_scale(
    x: SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO,
    alpha: jax.Array,
    *,
    algebra: str,
) -> SparseBoxCOO | SparseBoxCSR | SparseBoxBCOO:
    alpha_box = mat_common.box_from_point(jnp.asarray(alpha, dtype=jnp.complex128))
    if isinstance(x, SparseBoxCOO):
        x = as_sparse_box_coo(x, algebra=algebra, label="sparse_common.sparse_box_scale")
        return SparseBoxCOO(data=acb_core.acb_mul(x.data, alpha_box), row=x.row, col=x.col, rows=x.rows, cols=x.cols, algebra=algebra)
    if isinstance(x, SparseBoxCSR):
        x = as_sparse_box_csr(x, algebra=algebra, label="sparse_common.sparse_box_scale")
        return SparseBoxCSR(data=acb_core.acb_mul(x.data, alpha_box), indices=x.indices, indptr=x.indptr, rows=x.rows, cols=x.cols, algebra=algebra)
    x = as_sparse_box_bcoo(x, algebra=algebra, label="sparse_common.sparse_box_scale")
    return SparseBoxBCOO(data=acb_core.acb_mul(x.data, alpha_box), indices=x.indices, rows=x.rows, cols=x.cols, algebra=algebra)


def pad_batch_repeat_last(args: tuple, *, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)


def pad_validated_batch(x: jax.Array, *, pad_to: int, validate, label: str) -> jax.Array:
    (padded,), _ = pad_batch_repeat_last((validate(x, label),), pad_to=pad_to)
    return padded


def vmapped_batch_fixed(x: jax.Array, *, validate, label: str, apply):
    x = validate(x, label)
    return jax.vmap(apply)(x)


def vmapped_batch_padded(x: jax.Array, *, pad_to: int, validate, label: str, apply):
    padded = pad_validated_batch(x, pad_to=pad_to, validate=validate, label=label)
    return jax.vmap(apply)(padded)


def as_block_sparse_coo(x: BlockSparseCOO, *, algebra: str, label: str) -> BlockSparseCOO:
    checks.check(isinstance(x, BlockSparseCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.data).ndim, 3, f"{label}.data_ndim")
    checks.check_equal(jnp.asarray(x.row).ndim, 1, f"{label}.row_ndim")
    checks.check_equal(jnp.asarray(x.col).ndim, 1, f"{label}.col_ndim")
    checks.check_equal(x.data.shape[0], x.row.shape[0], f"{label}.nnzb_row")
    checks.check_equal(x.data.shape[0], x.col.shape[0], f"{label}.nnzb_col")
    checks.check_equal(x.data.shape[1], x.block_rows, f"{label}.block_rows")
    checks.check_equal(x.data.shape[2], x.block_cols, f"{label}.block_cols")
    return x


def as_block_sparse_csr(x: BlockSparseCSR, *, algebra: str, label: str) -> BlockSparseCSR:
    checks.check(isinstance(x, BlockSparseCSR), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.data).ndim, 3, f"{label}.data_ndim")
    checks.check_equal(jnp.asarray(x.indices).ndim, 1, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.indptr).ndim, 1, f"{label}.indptr_ndim")
    checks.check_equal(x.data.shape[0], x.indices.shape[0], f"{label}.nnzb")
    checks.check_equal(x.indptr.shape[0] - 1, x.rows // x.block_rows, f"{label}.block_row_ptrs")
    checks.check_equal(x.data.shape[1], x.block_rows, f"{label}.block_rows")
    checks.check_equal(x.data.shape[2], x.block_cols, f"{label}.block_cols")
    return x


def as_block_sparse_matvec_plan(x: BlockSparseMatvecPlan, *, algebra: str, label: str) -> BlockSparseMatvecPlan:
    checks.check(isinstance(x, BlockSparseMatvecPlan), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check(x.storage == "bcsr", f"{label}.storage")
    return x


def as_variable_block_sparse_coo(x: VariableBlockSparseCOO, *, algebra: str, label: str) -> VariableBlockSparseCOO:
    checks.check(isinstance(x, VariableBlockSparseCOO), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.data).ndim, 3, f"{label}.data_ndim")
    checks.check_equal(jnp.asarray(x.row).ndim, 1, f"{label}.row_ndim")
    checks.check_equal(jnp.asarray(x.col).ndim, 1, f"{label}.col_ndim")
    checks.check_equal(jnp.asarray(x.row_block_sizes).ndim, 1, f"{label}.row_block_sizes_ndim")
    checks.check_equal(jnp.asarray(x.col_block_sizes).ndim, 1, f"{label}.col_block_sizes_ndim")
    checks.check_equal(x.data.shape[0], x.row.shape[0], f"{label}.nnzb_row")
    checks.check_equal(x.data.shape[0], x.col.shape[0], f"{label}.nnzb_col")
    return x


def as_variable_block_sparse_csr(x: VariableBlockSparseCSR, *, algebra: str, label: str) -> VariableBlockSparseCSR:
    checks.check(isinstance(x, VariableBlockSparseCSR), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.data).ndim, 3, f"{label}.data_ndim")
    checks.check_equal(jnp.asarray(x.indices).ndim, 1, f"{label}.indices_ndim")
    checks.check_equal(jnp.asarray(x.indptr).ndim, 1, f"{label}.indptr_ndim")
    checks.check_equal(jnp.asarray(x.row_block_sizes).ndim, 1, f"{label}.row_block_sizes_ndim")
    checks.check_equal(jnp.asarray(x.col_block_sizes).ndim, 1, f"{label}.col_block_sizes_ndim")
    checks.check_equal(x.data.shape[0], x.indices.shape[0], f"{label}.nnzb")
    checks.check_equal(x.indptr.shape[0], x.row_block_sizes.shape[0] + 1, f"{label}.indptr_row_blocks")
    return x


def as_variable_block_sparse_matvec_plan(x: VariableBlockSparseMatvecPlan, *, algebra: str, label: str) -> VariableBlockSparseMatvecPlan:
    checks.check(isinstance(x, VariableBlockSparseMatvecPlan), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check(x.storage == "vcsr", f"{label}.storage")
    checks.check_equal(jnp.asarray(x.row_block_sizes).ndim, 1, f"{label}.row_block_sizes_ndim")
    checks.check_equal(jnp.asarray(x.col_block_sizes).ndim, 1, f"{label}.col_block_sizes_ndim")
    return x


__all__ = [
    "SparseCOO",
    "SparseCSR",
    "SparseBCOO",
    "SparseIntervalCOO",
    "SparseIntervalCSR",
    "SparseIntervalBCOO",
    "SparseBoxCOO",
    "SparseBoxCSR",
    "SparseBoxBCOO",
    "SparseMatvecPlan",
    "SparseLUSolvePlan",
    "SparseCholeskySolvePlan",
    "SparseQRFactor",
    "BlockSparseCOO",
    "BlockSparseCSR",
    "BlockSparseMatvecPlan",
    "VariableBlockSparseCOO",
    "VariableBlockSparseCSR",
    "VariableBlockSparseMatvecPlan",
    "as_sparse_coo",
    "as_sparse_csr",
    "as_sparse_bcoo",
    "as_sparse_interval_coo",
    "as_sparse_interval_csr",
    "as_sparse_interval_bcoo",
    "as_sparse_box_coo",
    "as_sparse_box_csr",
    "as_sparse_box_bcoo",
    "as_sparse_matvec_plan",
    "as_sparse_lu_solve_plan",
    "sparse_lu_solve_plan_from_factors",
    "as_sparse_cholesky_solve_plan",
    "sparse_cholesky_solve_plan_from_factor",
    "as_sparse_qr_factor",
    "sparse_shape",
    "sparse_det_from_lu",
    "sparse_inv_via_solve",
    "sparse_square_via_matmul",
    "sparse_to_dense",
    "sparse_real_to_interval_matrix",
    "sparse_complex_to_box_matrix",
    "sparse_real_to_interval_sparse",
    "sparse_complex_to_box_sparse",
    "sparse_interval_to_dense",
    "sparse_box_to_dense",
    "sparse_interval_trace",
    "sparse_box_trace",
    "sparse_interval_norm_1",
    "sparse_interval_norm_inf",
    "sparse_box_norm_1",
    "sparse_box_norm_inf",
    "sparse_interval_matvec",
    "sparse_box_matvec",
    "sparse_interval_matmul_dense_rhs",
    "sparse_box_matmul_dense_rhs",
    "sparse_interval_transpose",
    "sparse_box_transpose",
    "sparse_box_conjugate_transpose",
    "sparse_interval_add",
    "sparse_box_add",
    "sparse_interval_scale",
    "sparse_box_scale",
    "pad_batch_repeat_last",
    "as_block_sparse_coo",
    "as_block_sparse_csr",
    "as_block_sparse_matvec_plan",
    "as_variable_block_sparse_coo",
    "as_variable_block_sparse_csr",
    "as_variable_block_sparse_matvec_plan",
    "csr_row_ids",
    "coo_to_bcoo",
    "bcoo_to_sparse_bcoo",
    "dense_to_sparse_bcoo",
    "scipy_csr_to_sparse_bcoo",
    "sparse_bcoo_to_dense",
    "sparse_bcoo_matvec",
    "sparse_bcoo_matmul_dense_rhs",
    "sparse_bcoo_add",
    "sparse_bcoo_matmul_sparse",
    "sparse_matvec_plan_from_sparse",
    "sparse_matvec_plan_apply",
    "sparse_interval_matvec_plan_from_sparse",
    "sparse_interval_matvec_plan_apply",
    "sparse_box_matvec_plan_from_sparse",
    "sparse_box_matvec_plan_apply",
]
