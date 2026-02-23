import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import arb_mat

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


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
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    lib = _find_lib(build_dir, ["arb_mat_ref.dll", "libarb_mat_ref.dll", "libarb_mat_ref.so", "libarb_mat_ref.dylib"])
    return di, lib


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    lib_env = os.getenv("ARB_MAT_REF_LIB")
    d_di, d_lib = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("arb_mat_2x2_det_ref", "arb_mat_2x2_trace_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.POINTER(DI)]
        fn.restype = DI
    return lib


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def _random_mats(rng: np.random.Generator, n: int) -> np.ndarray:
    mats = _random_intervals(rng, n * 4, -0.5, 0.5).reshape(n, 2, 2, 2)
    return mats


def _call_unary(lib, fn_name: str, mats: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty((mats.shape[0], 2), dtype=np.float64)
    for i in range(mats.shape[0]):
        buf = (DI * 4)()
        flat = mats[i].reshape(4, 2)
        for k in range(4):
            buf[k].a = float(flat[k, 0])
            buf[k].b = float(flat[k, 1])
        r = fn(buf)
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def test_arb_mat_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2221)
    mats = _random_mats(rng, 2000)

    c_det = _call_unary(lib, "arb_mat_2x2_det_ref", mats)
    j_det = np.asarray(arb_mat.arb_mat_2x2_det_batch_jit(jnp.asarray(mats)))
    np.testing.assert_allclose(c_det, j_det, rtol=5e-12, atol=2e-12, equal_nan=True)

    c_tr = _call_unary(lib, "arb_mat_2x2_trace_ref", mats)
    j_tr = np.asarray(arb_mat.arb_mat_2x2_trace_batch_jit(jnp.asarray(mats)))
    np.testing.assert_allclose(c_tr, j_tr, rtol=5e-12, atol=2e-12, equal_nan=True)
