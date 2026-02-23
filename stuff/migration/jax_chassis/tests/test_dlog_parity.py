import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import dlog

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


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
    return _find_lib(build_dir, ["dlog_ref.dll", "libdlog_ref.dll", "libdlog_ref.so", "libdlog_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("DLOG_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.dlog_log1p_ref.argtypes = [ctypes.c_double]
    lib.dlog_log1p_ref.restype = ctypes.c_double
    lib.dlog_log1p_batch_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    return lib


def test_dlog_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2025)
    count = 8000
    x = rng.uniform(-0.9, 5.0, size=count).astype(np.float64)

    out_c = np.empty_like(x)
    lib.dlog_log1p_batch_ref(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        count,
    )

    out_j = np.asarray(dlog.dlog_log1p_batch_jit(jnp.asarray(x)))
    np.testing.assert_allclose(out_c, out_j, rtol=1e-12, atol=0.0, equal_nan=True)
