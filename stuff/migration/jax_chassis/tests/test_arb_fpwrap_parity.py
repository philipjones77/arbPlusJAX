import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import arb_fpwrap

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class CDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _default_lib() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["arb_fpwrap_ref.dll", "libarb_fpwrap_ref.dll", "libarb_fpwrap_ref.so", "libarb_fpwrap_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("ARB_FPWRAP_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.arb_fpwrap_double_exp_ref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    lib.arb_fpwrap_double_exp_ref.restype = ctypes.c_int
    lib.arb_fpwrap_double_log_ref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    lib.arb_fpwrap_double_log_ref.restype = ctypes.c_int
    lib.arb_fpwrap_cdouble_exp_ref.argtypes = [ctypes.POINTER(CDouble), CDouble]
    lib.arb_fpwrap_cdouble_exp_ref.restype = ctypes.c_int
    lib.arb_fpwrap_cdouble_log_ref.argtypes = [ctypes.POINTER(CDouble), CDouble]
    lib.arb_fpwrap_cdouble_log_ref.restype = ctypes.c_int
    return lib


def test_arb_fpwrap_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2211)
    x = rng.uniform(0.1, 2.0, size=2000)
    z = rng.normal(size=2000) + 1j * rng.normal(size=2000)

    out_exp = np.empty_like(x)
    out_log = np.empty_like(x)
    for i in range(2000):
        val = ctypes.c_double()
        lib.arb_fpwrap_double_exp_ref(ctypes.byref(val), float(x[i]))
        out_exp[i] = val.value
        val2 = ctypes.c_double()
        lib.arb_fpwrap_double_log_ref(ctypes.byref(val2), float(x[i]))
        out_log[i] = val2.value

    j_exp = np.asarray(arb_fpwrap.arb_fpwrap_double_exp_jit(jnp.asarray(x)))
    j_log = np.asarray(arb_fpwrap.arb_fpwrap_double_log_jit(jnp.asarray(x)))
    np.testing.assert_allclose(out_exp, j_exp, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out_log, j_log, rtol=0.0, atol=0.0)

    out_c_exp = np.empty_like(z)
    out_c_log = np.empty_like(z)
    for i in range(2000):
        c = CDouble(float(z[i].real), float(z[i].imag))
        r = CDouble()
        lib.arb_fpwrap_cdouble_exp_ref(ctypes.byref(r), c)
        out_c_exp[i] = r.real + 1j * r.imag
        r2 = CDouble()
        lib.arb_fpwrap_cdouble_log_ref(ctypes.byref(r2), c)
        out_c_log[i] = r2.real + 1j * r2.imag

    j_c_exp = np.asarray(arb_fpwrap.arb_fpwrap_cdouble_exp_jit(jnp.asarray(z)))
    j_c_log = np.asarray(arb_fpwrap.arb_fpwrap_cdouble_log_jit(jnp.asarray(z)))
    np.testing.assert_allclose(out_c_exp, j_c_exp, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out_c_log, j_c_log, rtol=0.0, atol=0.0)
