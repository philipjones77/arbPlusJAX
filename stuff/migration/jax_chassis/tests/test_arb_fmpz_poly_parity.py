import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import arb_fmpz_poly

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


def _default_libs() -> tuple[Path | None, Path | None]:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    lib = _find_lib(build_dir, ["arb_fmpz_poly_ref.dll", "libarb_fmpz_poly_ref.dll", "libarb_fmpz_poly_ref.so", "libarb_fmpz_poly_ref.dylib"])
    return di, lib


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    lib_env = os.getenv("ARB_FMPZ_POLY_REF_LIB")
    d_di, d_lib = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.arb_fmpz_poly_eval_cubic_ref
    fn.argtypes = [ctypes.POINTER(DI), DI]
    fn.restype = DI
    return lib


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def test_arb_fmpz_poly_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2201)
    coeffs = _random_intervals(rng, 4 * 2000, -0.5, 0.5).reshape(2000, 4, 2)
    x = _random_intervals(rng, 2000, -0.3, 0.3)

    out_c = np.empty((2000, 2), dtype=np.float64)
    fn = lib.arb_fmpz_poly_eval_cubic_ref
    for i in range(2000):
        buf = (DI * 4)()
        for k in range(4):
            buf[k].a = float(coeffs[i, k, 0])
            buf[k].b = float(coeffs[i, k, 1])
        r = fn(buf, DI(float(x[i, 0]), float(x[i, 1])))
        out_c[i, 0] = r.a
        out_c[i, 1] = r.b

    out_j = np.asarray(arb_fmpz_poly.arb_fmpz_poly_eval_cubic_batch_jit(jnp.asarray(coeffs), jnp.asarray(x)))
    np.testing.assert_allclose(out_c, out_j, rtol=5e-12, atol=2e-12, equal_nan=True)
