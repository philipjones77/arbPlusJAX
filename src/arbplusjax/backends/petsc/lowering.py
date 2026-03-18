from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy import sparse

from ... import acb_core
from ... import double_interval as di
from ... import mat_common
from ... import sparse_common
from .runtime import get_petsc_module


def _get_native_vec_array(native_vec) -> np.ndarray:
    if hasattr(native_vec, "getArray"):
        try:
            return np.asarray(native_vec.getArray(readonly=True))
        except TypeError:
            return np.asarray(native_vec.getArray())
    if hasattr(native_vec, "array"):
        return np.asarray(native_vec.array)
    raise TypeError("Unsupported native PETSc vector interface")


def _set_native_vec_array(native_vec, values) -> None:
    array = np.asarray(values)
    if hasattr(native_vec, "setArray"):
        native_vec.setArray(array)
        return
    if hasattr(native_vec, "array"):
        native_vec.array = array
        return
    if hasattr(native_vec, "getArray"):
        target = native_vec.getArray()
        target[...] = array
        return
    raise TypeError("Unsupported native PETSc vector interface")


class _CallablePythonMatContext:
    def __init__(self, matvec: Callable[[jnp.ndarray], jnp.ndarray]) -> None:
        self._matvec = matvec

    def mult(self, mat, x, y) -> None:
        del mat
        result = self._matvec(jnp.asarray(_get_native_vec_array(x)))
        _set_native_vec_array(y, np.asarray(result))


def to_petsc_vec(values, *, petsc=None):
    module = get_petsc_module() if petsc is None else petsc
    values = getattr(values, "native", values)
    if hasattr(values, "getArray") and hasattr(values, "getSize"):
        return values
    vector = jnp.asarray(values)
    if vector.ndim != 1:
        raise ValueError("PETSc lowering expects a one-dimensional point vector")
    return module.Vec().createWithArray(np.asarray(vector))


def to_petsc_mat(operator, *, shape: tuple[int, int] | None = None, petsc=None):
    module = get_petsc_module() if petsc is None else petsc
    operator = getattr(operator, "native", operator)
    if hasattr(operator, "mult") and hasattr(operator, "getSize"):
        return operator
    if sparse.issparse(operator):
        return _sparse_matrix_to_petsc(operator.tocsr(), petsc=module)
    if isinstance(operator, mat_common.DenseMatvecPlan):
        return _dense_matrix_to_petsc(operator.matrix, petsc=module)
    if isinstance(operator, sparse_common.SparseMatvecPlan):
        return to_petsc_mat(operator.payload, petsc=module)
    if isinstance(operator, (sparse_common.SparseCOO, sparse_common.SparseCSR, sparse_common.SparseBCOO)):
        return _arb_sparse_to_petsc(operator, petsc=module)
    if isinstance(
        operator,
        (
            sparse_common.BlockSparseCOO,
            sparse_common.BlockSparseCSR,
            sparse_common.BlockSparseMatvecPlan,
            sparse_common.VariableBlockSparseCOO,
            sparse_common.VariableBlockSparseCSR,
            sparse_common.VariableBlockSparseMatvecPlan,
        ),
    ):
        return _matrix_free_to_petsc(
            shape=(int(operator.rows), int(operator.cols)),
            matvec=_structured_matvec(operator),
            petsc=module,
        )
    if callable(operator) or hasattr(operator, "matvec"):
        matvec = operator if callable(operator) else operator.matvec
        resolved_shape = shape or getattr(operator, "shape", None)
        if resolved_shape is None:
            raise ValueError("Matrix-free PETSc lowering requires an explicit shape")
        return _matrix_free_to_petsc(
            shape=(int(resolved_shape[0]), int(resolved_shape[1])),
            matvec=matvec,
            petsc=module,
        )
    return _dense_matrix_to_petsc(operator, petsc=module)


def _dense_matrix_to_petsc(matrix, *, petsc):
    dense = _as_point_dense_matrix(matrix)
    return petsc.Mat().createDense(size=dense.shape, array=np.asarray(dense))


def _as_point_dense_matrix(matrix) -> jnp.ndarray:
    array = jnp.asarray(matrix)
    if array.ndim == 2:
        return array
    if array.ndim >= 3 and array.shape[-1] == 2 and array.shape[-2] == array.shape[-3]:
        return di.midpoint(array)
    if array.ndim >= 3 and array.shape[-1] == 4 and array.shape[-2] == array.shape[-3]:
        return acb_core.acb_midpoint(array)
    raise ValueError("PETSc dense lowering expects a 2D point matrix or a square interval/box matrix")


def _sparse_matrix_to_petsc(matrix, *, petsc):
    csr = matrix.tocsr()
    return petsc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr, csr.indices, csr.data),
    )


def _arb_sparse_to_petsc(operator, *, petsc):
    if isinstance(operator, sparse_common.SparseCSR):
        return petsc.Mat().createAIJ(
            size=(operator.rows, operator.cols),
            csr=(np.asarray(operator.indptr), np.asarray(operator.indices), np.asarray(operator.data)),
        )
    if isinstance(operator, sparse_common.SparseCOO):
        csr = sparse.csr_matrix(
            (np.asarray(operator.data), (np.asarray(operator.row), np.asarray(operator.col))),
            shape=(operator.rows, operator.cols),
        )
        return _sparse_matrix_to_petsc(csr, petsc=petsc)
    indices = np.asarray(operator.indices)
    csr = sparse.csr_matrix(
        (np.asarray(operator.data), (indices[:, 0], indices[:, 1])),
        shape=(operator.rows, operator.cols),
    )
    return _sparse_matrix_to_petsc(csr, petsc=petsc)


def _matrix_free_to_petsc(*, shape: tuple[int, int], matvec: Callable[[jnp.ndarray], jnp.ndarray], petsc):
    context = _CallablePythonMatContext(matvec)
    native_matrix = petsc.Mat()
    try:
        created = native_matrix.createPython(shape, context=context)
    except TypeError:
        created = native_matrix.createPython(shape, context)
    matrix = created if created is not None else native_matrix
    if hasattr(matrix, "setPythonContext"):
        matrix.setPythonContext(context)
    try:
        setattr(matrix, "_arbplusjax_python_context", context)
    except Exception:
        pass
    if hasattr(matrix, "setUp"):
        matrix.setUp()
    return matrix


def _structured_matvec(operator):
    if isinstance(operator, (sparse_common.BlockSparseCOO, sparse_common.BlockSparseCSR)):
        if jnp.iscomplexobj(jnp.asarray(operator.data)):
            from ... import scb_block_mat

            return lambda vector: scb_block_mat.scb_block_mat_matvec(operator, jnp.asarray(vector))
        from ... import srb_block_mat

        return lambda vector: srb_block_mat.srb_block_mat_matvec(operator, jnp.asarray(vector))
    if isinstance(operator, sparse_common.BlockSparseMatvecPlan):
        if _payload_is_complex(operator.payload):
            from ... import scb_block_mat

            return lambda vector: scb_block_mat.scb_block_mat_matvec_cached_apply(operator, jnp.asarray(vector))
        from ... import srb_block_mat

        return lambda vector: srb_block_mat.srb_block_mat_matvec_cached_apply(operator, jnp.asarray(vector))
    if isinstance(operator, (sparse_common.VariableBlockSparseCOO, sparse_common.VariableBlockSparseCSR)):
        if jnp.iscomplexobj(jnp.asarray(operator.data)):
            from ... import scb_vblock_mat

            return lambda vector: scb_vblock_mat.scb_vblock_mat_matvec(operator, jnp.asarray(vector))
        from ... import srb_vblock_mat

        return lambda vector: srb_vblock_mat.srb_vblock_mat_matvec(operator, jnp.asarray(vector))
    if _payload_is_complex(operator.payload):
        from ... import scb_vblock_mat

        return lambda vector: scb_vblock_mat.scb_vblock_mat_matvec_cached_apply(operator, jnp.asarray(vector))
    from ... import srb_vblock_mat

    return lambda vector: srb_vblock_mat.srb_vblock_mat_matvec_cached_apply(operator, jnp.asarray(vector))


def _payload_is_complex(payload: Any) -> bool:
    if hasattr(payload, "data"):
        return bool(jnp.iscomplexobj(jnp.asarray(payload.data)))
    return False
