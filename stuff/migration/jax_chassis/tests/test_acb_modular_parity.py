import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import acb_modular

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


class ACB(ctypes.Structure):
    _fields_ = [("real", DI), ("imag", DI)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _default_libs() -> tuple[Path | None, Path | None]:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    lib = _find_lib(build_dir, ["acb_modular_ref.dll", "libacb_modular_ref.dll", "libacb_modular_ref.so", "libacb_modular_ref.dylib"])
    return di, lib


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    lib_env = os.getenv("ACB_MODULAR_REF_LIB")
    d_di, d_lib = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.acb_modular_j_ref
    fn.argtypes = [ACB]
    fn.restype = ACB
    return lib


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def _random_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    re = _random_intervals(rng, n, 0.0, 0.5)
    im = _random_intervals(rng, n, 0.5, 1.2)
    return np.concatenate([re, im], axis=-1)


def _call_unary(lib, x: np.ndarray) -> np.ndarray:
    fn = lib.acb_modular_j_ref
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(ACB(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3]))))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def test_acb_modular_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2161)
    tau = _random_boxes(rng, 3000)

    c_j = _call_unary(lib, tau)
    j_j = np.asarray(acb_modular.acb_modular_j_batch_jit(jnp.asarray(tau)))
    np.testing.assert_allclose(c_j, j_j, rtol=5e-12, atol=2e-12, equal_nan=True)
