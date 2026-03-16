from __future__ import annotations

from dataclasses import dataclass

import jax
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

from . import checks

jax.config.update("jax_enable_x64", True)


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


def csr_row_ids(indptr: jax.Array, *, rows: int, nnz: int) -> jax.Array:
    counts = indptr[1:] - indptr[:-1]
    return jnp.repeat(jnp.arange(rows, dtype=indptr.dtype), counts, total_repeat_length=nnz)


def coo_to_bcoo(x: SparseCOO) -> jsparse.BCOO:
    return jsparse.BCOO((x.data, jnp.stack([x.row, x.col], axis=-1)), shape=(x.rows, x.cols))


def bcoo_to_sparse_bcoo(x: jsparse.BCOO, *, algebra: str) -> SparseBCOO:
    return SparseBCOO(data=x.data, indices=x.indices, rows=int(x.shape[0]), cols=int(x.shape[1]), algebra=algebra)


def as_sparse_matvec_plan(x: SparseMatvecPlan, *, algebra: str, label: str) -> SparseMatvecPlan:
    checks.check(isinstance(x, SparseMatvecPlan), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check(x.storage in ("coo", "csr", "bcoo"), f"{label}.storage")
    return x


def as_sparse_qr_factor(x: SparseQRFactor, *, algebra: str, label: str) -> SparseQRFactor:
    checks.check(isinstance(x, SparseQRFactor), f"{label}.type")
    checks.check(x.algebra == algebra, f"{label}.algebra")
    checks.check_equal(jnp.asarray(x.reflectors).ndim, 2, f"{label}.reflectors_ndim")
    checks.check_equal(jnp.asarray(x.taus).ndim, 1, f"{label}.taus_ndim")
    checks.check_equal(x.reflectors.shape[0], x.rows, f"{label}.reflector_rows")
    checks.check_equal(x.reflectors.shape[1], x.taus.shape[0], f"{label}.reflector_cols")
    return x


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
    "SparseMatvecPlan",
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
    "as_sparse_matvec_plan",
    "as_sparse_qr_factor",
    "as_block_sparse_coo",
    "as_block_sparse_csr",
    "as_block_sparse_matvec_plan",
    "as_variable_block_sparse_coo",
    "as_variable_block_sparse_csr",
    "as_variable_block_sparse_matvec_plan",
    "csr_row_ids",
    "coo_to_bcoo",
    "bcoo_to_sparse_bcoo",
]
