import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import arf

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class ARF(ctypes.Structure):
    _fields_ = [("v", ctypes.c_double)]


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
    return _find_lib(build_dir, ["arf_ref.dll", "libarf_ref.dll", "libarf_ref.so", "libarf_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("ARF_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("arf_add_ref", "arf_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ARF, ARF]
        fn.restype = ARF
    return lib


def test_arf_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2241)
    a = rng.normal(size=3000)
    b = rng.normal(size=3000)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.arf_add_ref
    mul_fn = lib.arf_mul_ref
    for i in range(3000):
        r = add_fn(ARF(float(a[i])), ARF(float(b[i])))
        out_add[i] = r.v
        r = mul_fn(ARF(float(a[i])), ARF(float(b[i])))
        out_mul[i] = r.v

    j_add = np.asarray(arf.arf_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(arf.arf_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    np.testing.assert_allclose(out_add, j_add, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out_mul, j_mul, rtol=0.0, atol=0.0)
