import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import bool_mat

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
    return _find_lib(
        build_dir,
        ["bool_mat_ref.dll", "libbool_mat_ref.dll", "libbool_mat_ref.so", "libbool_mat_ref.dylib"],
    )


def _load_lib():
    lib_env = os.getenv("BOOL_MAT_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.bool_mat_2x2_det_ref.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.bool_mat_2x2_det_ref.restype = ctypes.c_uint8
    lib.bool_mat_2x2_trace_ref.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.bool_mat_2x2_trace_ref.restype = ctypes.c_uint8
    lib.bool_mat_2x2_det_batch_ref.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.bool_mat_2x2_trace_batch_ref.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    return lib


def test_bool_mat_parity():
    lib = _load_lib()
    rng = np.random.default_rng(123)
    count = 4096
    mats = rng.integers(0, 2, size=(count, 4), dtype=np.uint8)

    flat = np.ascontiguousarray(mats.reshape(-1))
    det_out = np.empty(count, dtype=np.uint8)
    tr_out = np.empty(count, dtype=np.uint8)

    lib.bool_mat_2x2_det_batch_ref(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        det_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        count,
    )
    lib.bool_mat_2x2_trace_batch_ref(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        tr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        count,
    )

    j_det = np.asarray(bool_mat.bool_mat_2x2_det_batch_jit(jnp.asarray(mats)))
    j_tr = np.asarray(bool_mat.bool_mat_2x2_trace_batch_jit(jnp.asarray(mats)))

    np.testing.assert_array_equal(det_out, j_det)
    np.testing.assert_array_equal(tr_out, j_tr)
