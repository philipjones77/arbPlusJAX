import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import acb_elliptic

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
    lib = _find_lib(build_dir, ["acb_elliptic_ref.dll", "libacb_elliptic_ref.dll", "libacb_elliptic_ref.so", "libacb_elliptic_ref.dylib"])
    return di, lib


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    lib_env = os.getenv("ACB_ELLIPTIC_REF_LIB")
    d_di, d_lib = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("acb_elliptic_k_ref", "acb_elliptic_e_ref"):
        fn = getattr(lib, fn_name)
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
    re = _random_intervals(rng, n, 0.0, 0.9)
    im = _random_intervals(rng, n, -0.2, 0.2)
    return np.concatenate([re, im], axis=-1)


def _call_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(ACB(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3]))))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def test_acb_elliptic_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2141)
    m = _random_boxes(rng, 3000)

    c_k = _call_unary(lib, "acb_elliptic_k_ref", m)
    j_k = np.asarray(acb_elliptic.acb_elliptic_k_batch_jit(jnp.asarray(m)))
    np.testing.assert_allclose(c_k, j_k, rtol=5e-12, atol=2e-12, equal_nan=True)

    c_e = _call_unary(lib, "acb_elliptic_e_ref", m)
    j_e = np.asarray(acb_elliptic.acb_elliptic_e_batch_jit(jnp.asarray(m)))
    np.testing.assert_allclose(c_e, j_e, rtol=5e-12, atol=2e-12, equal_nan=True)
