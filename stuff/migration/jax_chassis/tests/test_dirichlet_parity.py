import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import dirichlet

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _default_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None
    return _find_lib(
        build_dir,
        ["dirichlet_ref.dll", "libdirichlet_ref.dll", "libdirichlet_ref.so", "libdirichlet_ref.dylib"],
    )


def _load_lib():
    lib_env = os.getenv("DIRICHLET_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.dirichlet_zeta_ref.argtypes = [DI, ctypes.c_int]
    lib.dirichlet_zeta_ref.restype = DI
    lib.dirichlet_eta_ref.argtypes = [DI, ctypes.c_int]
    lib.dirichlet_eta_ref.restype = DI
    lib.dirichlet_zeta_batch_ref.argtypes = [
        ctypes.POINTER(DI),
        ctypes.POINTER(DI),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.dirichlet_eta_batch_ref.argtypes = [
        ctypes.POINTER(DI),
        ctypes.POINTER(DI),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    return lib


def _random_intervals(rng: np.random.Generator, n: int) -> np.ndarray:
    mid = rng.uniform(1.2, 4.0, size=n)
    half = rng.uniform(0.0, 0.2, size=n)
    lo = mid - half
    hi = mid + half
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def test_dirichlet_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2024)
    count = 2000
    n_terms = 32

    s = _random_intervals(rng, count)
    s_struct = (DI * count)(*([DI(float(a), float(b)) for a, b in s]))

    zeta_out = (DI * count)()
    eta_out = (DI * count)()

    lib.dirichlet_zeta_batch_ref(s_struct, zeta_out, count, n_terms)
    lib.dirichlet_eta_batch_ref(s_struct, eta_out, count, n_terms)

    zeta_c = np.array([[zeta_out[i].a, zeta_out[i].b] for i in range(count)], dtype=np.float64)
    eta_c = np.array([[eta_out[i].a, eta_out[i].b] for i in range(count)], dtype=np.float64)

    zeta_j = np.asarray(dirichlet.dirichlet_zeta_batch_jit(jnp.asarray(s), n_terms=n_terms))
    eta_j = np.asarray(dirichlet.dirichlet_eta_batch_jit(jnp.asarray(s), n_terms=n_terms))

    np.testing.assert_allclose(zeta_c, zeta_j, rtol=1e-12, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(eta_c, eta_j, rtol=1e-12, atol=0.0, equal_nan=True)
    assert np.all(zeta_j[:, 0] <= zeta_j[:, 1])
    assert np.all(eta_j[:, 0] <= eta_j[:, 1])
