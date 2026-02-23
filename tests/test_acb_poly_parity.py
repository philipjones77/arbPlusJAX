import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import acb_poly

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


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
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    lib = _find_lib(build_dir, ["acb_poly_ref.dll", "libacb_poly_ref.dll", "libacb_poly_ref.so", "libacb_poly_ref.dylib"])
    return di, lib


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    lib_env = os.getenv("ACB_POLY_REF_LIB")
    d_di, d_lib = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.acb_poly_eval_cubic_ref
    fn.argtypes = [ctypes.POINTER(ACB), ACB]
    fn.restype = ACB
    return lib


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def _random_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    re = _random_intervals(rng, n, -0.5, 0.5)
    im = _random_intervals(rng, n, -0.3, 0.3)
    return np.concatenate([re, im], axis=-1)


def test_acb_poly_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2171)
    coeffs = _random_boxes(rng, 4 * 2000).reshape(2000, 4, 4)
    z = _random_boxes(rng, 2000)

    out_c = np.empty((2000, 4), dtype=np.float64)
    fn = lib.acb_poly_eval_cubic_ref
    for i in range(2000):
        buf = (ACB * 4)()
        for k in range(4):
            buf[k].real.a = float(coeffs[i, k, 0])
            buf[k].real.b = float(coeffs[i, k, 1])
            buf[k].imag.a = float(coeffs[i, k, 2])
            buf[k].imag.b = float(coeffs[i, k, 3])
        r = fn(buf, ACB(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3]))))
        out_c[i, 0] = r.real.a
        out_c[i, 1] = r.real.b
        out_c[i, 2] = r.imag.a
        out_c[i, 3] = r.imag.b

    out_j = np.asarray(acb_poly.acb_poly_eval_cubic_batch_jit(jnp.asarray(coeffs), jnp.asarray(z)))
    np.testing.assert_allclose(out_c, out_j, rtol=5e-12, atol=2e-12, equal_nan=True)
