import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import arb_calc

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
    lib = _find_lib(build_dir, ["arb_calc_ref.dll", "libarb_calc_ref.dll", "libarb_calc_ref.so", "libarb_calc_ref.dylib"])
    return di, lib


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    lib_env = os.getenv("ARB_CALC_REF_LIB")
    d_di, d_lib = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.arb_calc_integrate_line_ref
    fn.argtypes = [DI, DI, ctypes.c_int, ctypes.c_int]
    fn.restype = DI
    return lib


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def _call_unary(lib, a: np.ndarray, b: np.ndarray, integrand_id: int, n_steps: int) -> np.ndarray:
    fn = lib.arb_calc_integrate_line_ref
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(b[i, 0]), float(b[i, 1])), integrand_id, n_steps)
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def test_arb_calc_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2191)
    a = _random_intervals(rng, 3000, -0.5, 0.5)
    b = _random_intervals(rng, 3000, 0.2, 1.0)
    n_steps = 48

    c_exp = _call_unary(lib, a, b, 0, n_steps)
    j_exp = np.asarray(arb_calc.arb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="exp", n_steps=n_steps))
    np.testing.assert_allclose(c_exp, j_exp, rtol=5e-12, atol=2e-12, equal_nan=True)

    c_sin = _call_unary(lib, a, b, 1, n_steps)
    j_sin = np.asarray(arb_calc.arb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="sin", n_steps=n_steps))
    np.testing.assert_allclose(c_sin, j_sin, rtol=5e-12, atol=2e-12, equal_nan=True)

    c_cos = _call_unary(lib, a, b, 2, n_steps)
    j_cos = np.asarray(arb_calc.arb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="cos", n_steps=n_steps))
    np.testing.assert_allclose(c_cos, j_cos, rtol=5e-12, atol=2e-12, equal_nan=True)
