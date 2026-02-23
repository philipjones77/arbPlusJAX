import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import fmpr

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class FMPR(ctypes.Structure):
    _fields_ = [("v", ctypes.c_double)]


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
    return _find_lib(build_dir, ["fmpr_ref.dll", "libfmpr_ref.dll", "libfmpr_ref.so", "libfmpr_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("FMPR_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("fmpr_add_ref", "fmpr_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [FMPR, FMPR]
        fn.restype = FMPR
    return lib


def test_fmpr_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2026)
    count = 20000
    a = rng.normal(size=count).astype(np.float64)
    b = rng.normal(size=count).astype(np.float64)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.fmpr_add_ref
    mul_fn = lib.fmpr_mul_ref
    for i in range(count):
        out_add[i] = add_fn(FMPR(float(a[i])), FMPR(float(b[i]))).v
        out_mul[i] = mul_fn(FMPR(float(a[i])), FMPR(float(b[i]))).v

    j_add = np.asarray(fmpr.fmpr_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(fmpr.fmpr_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    np.testing.assert_allclose(out_add, j_add, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(out_mul, j_mul, rtol=1e-12, atol=0.0)
