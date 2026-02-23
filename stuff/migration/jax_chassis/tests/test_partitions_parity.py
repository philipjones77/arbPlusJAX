import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import partitions

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
        [
            "partitions_ref.dll",
            "libpartitions_ref.dll",
            "libpartitions_ref.so",
            "libpartitions_ref.dylib",
        ],
    )


def _load_lib():
    lib_env = os.getenv("PARTITIONS_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.partitions_p_ref.argtypes = [ctypes.c_int]
    lib.partitions_p_ref.restype = ctypes.c_uint64
    return lib


def test_partitions_parity():
    lib = _load_lib()
    n = np.arange(0, 15, dtype=np.int64)
    out_c = np.array([lib.partitions_p_ref(int(k)) for k in n], dtype=np.uint64)
    out_j = np.asarray(partitions.partitions_p_batch_jit(jnp.asarray(n)))
    np.testing.assert_array_equal(out_c, out_j)
