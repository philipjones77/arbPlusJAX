from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from arbplusjax import sparse_common
from arbplusjax.backends.petsc import lowering as petsc_lowering
from arbplusjax.backends.petsc.native import (
    create_dmplex_from_cell_list,
    create_petsc_object,
    unwrap_petsc_object,
)
from arbplusjax.backends.petsc.solve import LinearSolveConfig, solve_linear_system
from arbplusjax.backends.slepc.eigs import EigensolveConfig, solve_eigenproblem
from arbplusjax.backends.slepc.native import create_slepc_object, unwrap_slepc_object
from arbplusjax.backends.slepc import eigs as slepc_eigs


def _materialize_fake_mat(matrix) -> np.ndarray:
    if matrix._dense is not None:
        return np.asarray(matrix._dense)
    basis = np.eye(matrix._shape[1], dtype=np.float64)
    return np.stack([matrix._apply(basis[:, column]) for column in range(basis.shape[1])], axis=1)


class _FakeVec:
    def __init__(self) -> None:
        self._array = None

    def createWithArray(self, array):
        self._array = np.asarray(array)
        return self

    def createSeq(self, size: int):
        self._array = np.zeros((int(size),), dtype=np.float64)
        return self

    def getArray(self, readonly: bool = False):
        del readonly
        return self._array

    def setArray(self, array) -> None:
        self._array = np.asarray(array)

    def getSize(self) -> int:
        return int(self._array.shape[0])


class _FakeMat:
    def __init__(self) -> None:
        self._dense = None
        self._shape = None
        self._python_context = None

    def createDense(self, size=None, array=None):
        if array is not None:
            self._dense = np.asarray(array)
        else:
            self._dense = np.zeros(tuple(int(axis) for axis in size), dtype=np.float64)
        self._shape = tuple(int(axis) for axis in self._dense.shape)
        return self

    def createAIJ(self, size, csr):
        indptr, indices, data = csr
        from scipy import sparse

        csr_matrix = sparse.csr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=size)
        self._dense = csr_matrix.toarray()
        self._shape = tuple(int(axis) for axis in size)
        return self

    def createPython(self, size, context=None):
        self._shape = tuple(int(axis) for axis in size)
        self._python_context = context
        return self

    def setPythonContext(self, context) -> None:
        self._python_context = context

    def setUp(self) -> None:
        return None

    def getSize(self):
        return self._shape

    def createVecRight(self):
        return _FakeVec().createSeq(self._shape[1])

    def mult(self, vector, result=None):
        applied = self._apply(np.asarray(vector.getArray()))
        target = _FakeVec().createWithArray(applied) if result is None else result
        if result is not None:
            result.setArray(applied)
        return target

    def _apply(self, values):
        if self._dense is not None:
            return np.asarray(self._dense @ values)
        target = _FakeVec().createSeq(self._shape[0])
        self._python_context.mult(self, _FakeVec().createWithArray(values), target)
        return np.asarray(target.getArray())


class _FakePC:
    def __init__(self) -> None:
        self.type = None

    def setType(self, pc_type) -> None:
        self.type = pc_type


class _FakeCreatable:
    def __init__(self) -> None:
        self.created = False
        self.type = None

    def create(self, *args, **kwargs):
        self.created = True
        self.create_args = args
        self.create_kwargs = kwargs
        return self

    def setType(self, object_type) -> None:
        self.type = object_type


class _FakeKSP:
    def __init__(self) -> None:
        self._operator = None
        self._preconditioner = None
        self._pc = _FakePC()
        self.type = None
        self.tolerances = {}

    def create(self):
        return self

    def setType(self, ksp_type) -> None:
        self.type = ksp_type

    def setOperators(self, operator, preconditioner=None) -> None:
        self._operator = operator
        self._preconditioner = preconditioner or operator

    def getPC(self):
        return self._pc

    def setTolerances(self, **kwargs) -> None:
        self.tolerances = kwargs

    def solve(self, rhs, solution) -> None:
        matrix = _materialize_fake_mat(self._operator)
        rhs_array = np.asarray(rhs.getArray())
        solution.setArray(np.linalg.solve(matrix, rhs_array))


class _FakeST:
    def __init__(self) -> None:
        self.type = None
        self.shift = None
        self.ksp = None

    def create(self):
        return self

    def setType(self, st_type) -> None:
        self.type = st_type

    def setShift(self, shift: float) -> None:
        self.shift = shift

    def setKSP(self, ksp) -> None:
        self.ksp = ksp


class _FakeEPS:
    def __init__(self) -> None:
        self._operator = None
        self._mass = None
        self._nev = 1
        self._which = "SMALLEST_MAGNITUDE"
        self._st = None
        self._eigenvalues = np.asarray([])
        self._eigenvectors = np.zeros((0, 0))

    def create(self):
        return self

    def setOperators(self, operator, mass=None) -> None:
        self._operator = operator
        self._mass = mass

    def getOperators(self):
        return self._operator, self._mass

    def setProblemType(self, problem_type) -> None:
        self.problem_type = problem_type

    def setType(self, eps_type) -> None:
        self.eps_type = eps_type

    def setDimensions(self, nev: int) -> None:
        self._nev = int(nev)

    def setWhichEigenpairs(self, which) -> None:
        self._which = which

    def setTarget(self, target) -> None:
        self.target = target

    def setST(self, st) -> None:
        self._st = st

    def solve(self) -> None:
        matrix = _materialize_fake_mat(self._operator)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if self._which == "LARGEST_MAGNITUDE":
            order = np.argsort(-np.abs(eigenvalues))
        elif self._which == "LARGEST_REAL":
            order = np.argsort(-np.real(eigenvalues))
        elif self._which == "SMALLEST_REAL":
            order = np.argsort(np.real(eigenvalues))
        else:
            order = np.argsort(np.abs(eigenvalues))
        limited = order[: self._nev]
        self._eigenvalues = np.asarray(eigenvalues[limited])
        self._eigenvectors = np.asarray(eigenvectors[:, limited])

    def getConverged(self) -> int:
        return int(self._eigenvalues.shape[0])

    def getEigenpair(self, index: int, real_vector, imag_vector=None):
        real_vector.setArray(self._eigenvectors[:, int(index)])
        if imag_vector is not None:
            imag_vector.setArray(np.zeros_like(self._eigenvectors[:, int(index)]))
        return self._eigenvalues[int(index)]


class _FakeSys:
    @staticmethod
    def getVersion():
        return (3, 0, 0)


class _FakePETScModule:
    Vec = _FakeVec
    Mat = _FakeMat
    KSP = _FakeKSP
    PC = _FakeCreatable
    SNES = _FakeCreatable
    TS = _FakeCreatable
    DM = _FakeCreatable
    IS = _FakeCreatable
    AO = _FakeCreatable
    Sys = _FakeSys


class _FakeDMPlex(_FakeCreatable):
    def createFromCellList(self, dim, cells, coordinates, *args, **kwargs):
        self.dim = int(dim)
        self.cells = np.asarray(cells)
        self.coordinates = np.asarray(coordinates)
        self.cell_args = args
        self.cell_kwargs = kwargs
        return self


_FakePETScModule.DMPlex = _FakeDMPlex


class _FakeSLEPcModule:
    EPS = _FakeEPS
    ST = _FakeST
    PEP = _FakeCreatable
    NEP = _FakeCreatable
    SVD = _FakeCreatable
    MFN = _FakeCreatable
    BV = _FakeCreatable
    RG = _FakeCreatable
    FN = _FakeCreatable
    DS = _FakeCreatable
    Sys = _FakeSys


def test_to_petsc_mat_lowers_dense_sparse_and_block_sparse_without_new_matrix_api(monkeypatch) -> None:
    monkeypatch.setattr(petsc_lowering, "get_petsc_module", lambda: _FakePETScModule)

    dense = petsc_lowering.to_petsc_mat(jnp.array([[3.0, -1.0], [-1.0, 2.0]]))
    sparse_bcoo = sparse_common.dense_to_sparse_bcoo(jnp.array([[3.0, 0.0], [0.0, 4.0]]), algebra="srb")
    sparse_native = petsc_lowering.to_petsc_mat(sparse_bcoo)

    from arbplusjax import srb_block_mat

    block = srb_block_mat.srb_block_mat_from_dense_csr(
        jnp.array([[2.0, 1.0], [0.0, 3.0]]),
        block_shape=(1, 1),
    )
    block_native = petsc_lowering.to_petsc_mat(block)

    assert np.allclose(_materialize_fake_mat(dense), np.array([[3.0, -1.0], [-1.0, 2.0]]))
    assert np.allclose(_materialize_fake_mat(sparse_native), np.array([[3.0, 0.0], [0.0, 4.0]]))
    assert np.allclose(_materialize_fake_mat(block_native), np.array([[2.0, 1.0], [0.0, 3.0]]))


def test_solve_linear_system_uses_native_petsc_ksp(monkeypatch) -> None:
    from arbplusjax.backends.petsc import solve as petsc_solve

    monkeypatch.setattr(petsc_lowering, "get_petsc_module", lambda: _FakePETScModule)
    monkeypatch.setattr(petsc_solve, "get_petsc_module", lambda: _FakePETScModule)

    matrix = sparse_common.dense_to_sparse_bcoo(
        jnp.array([[4.0, 1.0], [1.0, 3.0]]),
        algebra="srb",
    )
    solution = solve_linear_system(
        matrix,
        jnp.array([1.0, 2.0]),
        config=LinearSolveConfig(ksp_type="cg", pc_type="jacobi", rtol=1e-10, max_it=20),
    )

    assert jnp.allclose(solution, jnp.array([0.09090909, 0.63636364]), atol=1e-6)


def test_solve_eigenproblem_uses_native_slepc_backend(monkeypatch) -> None:
    from arbplusjax.backends.slepc import runtime as slepc_runtime

    monkeypatch.setattr(petsc_lowering, "get_petsc_module", lambda: _FakePETScModule)
    monkeypatch.setattr(slepc_eigs, "get_petsc_module", lambda: _FakePETScModule)
    monkeypatch.setattr(slepc_eigs, "get_slepc_module", lambda: _FakeSLEPcModule)
    monkeypatch.setattr(slepc_runtime, "get_slepc_module", lambda: _FakeSLEPcModule)

    result = solve_eigenproblem(
        jnp.diag(jnp.array([4.0, 1.0, 2.0])),
        config=EigensolveConfig(nev=2, which="SMALLEST_MAGNITUDE", st_type="SINVERT", shift=0.1),
    )

    assert result.converged == 2
    assert jnp.allclose(result.eigenvalues, jnp.array([1.0, 2.0]), atol=1e-6)
    assert result.eigenvectors.shape == (3, 2)


def test_petsc_native_wrapper_exposes_generic_object_creation_and_dmplex(monkeypatch) -> None:
    from arbplusjax.backends.petsc import native as petsc_native

    monkeypatch.setattr(petsc_native, "get_petsc_module", lambda: _FakePETScModule)

    snes = create_petsc_object("SNES")
    snes.setType("newtonls")
    native_snes = unwrap_petsc_object(snes)
    plex = create_dmplex_from_cell_list(
        jnp.array([[0, 1, 2]], dtype=jnp.int32),
        jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        interpolate=True,
    )
    native_plex = unwrap_petsc_object(plex)

    assert native_snes.created is True
    assert native_snes.type == "newtonls"
    assert native_plex.dim == 2
    assert native_plex.cells.shape == (1, 3)
    assert native_plex.coordinates.shape == (3, 2)
    assert native_plex.cell_kwargs["interpolate"] is True


def test_slepc_native_wrapper_exposes_generic_object_creation(monkeypatch) -> None:
    from arbplusjax.backends.slepc import native as slepc_native

    monkeypatch.setattr(slepc_native, "get_slepc_module", lambda: _FakeSLEPcModule)

    pep = create_slepc_object("PEP")
    pep.setType("toar")
    native_pep = unwrap_slepc_object(pep)

    assert native_pep.created is True
    assert native_pep.type == "toar"
