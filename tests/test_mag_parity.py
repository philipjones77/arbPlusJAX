import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import mag

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _default_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["mag_ref.dll", "libmag_ref.dll", "libmag_ref.so", "libmag_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("MAG_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.mag_add_ref.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.mag_add_ref.restype = ctypes.c_double
    lib.mag_mul_ref.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.mag_mul_ref.restype = ctypes.c_double
    return lib


def test_mag_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2103)
    count = 12000
    a = rng.normal(size=count).astype(np.float64)
    b = rng.normal(size=count).astype(np.float64)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.mag_add_ref
    mul_fn = lib.mag_mul_ref
    for i in range(count):
        out_add[i] = add_fn(float(a[i]), float(b[i]))
        out_mul[i] = mul_fn(float(a[i]), float(b[i]))

    j_add = np.asarray(mag.mag_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(mag.mag_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    np.testing.assert_allclose(out_add, j_add, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out_mul, j_mul, rtol=0.0, atol=0.0)
