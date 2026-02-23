import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import fmpzi

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class FMPZI(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_int64), ("hi", ctypes.c_int64)]


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
    return _find_lib(build_dir, ["fmpzi_ref.dll", "libfmpzi_ref.dll", "libfmpzi_ref.so", "libfmpzi_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("FMPZI_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    lib = ctypes.CDLL(str(lib_path))
    lib.fmpzi_add_ref.argtypes = [FMPZI, FMPZI]
    lib.fmpzi_add_ref.restype = FMPZI
    lib.fmpzi_sub_ref.argtypes = [FMPZI, FMPZI]
    lib.fmpzi_sub_ref.restype = FMPZI
    return lib


def test_fmpzi_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2102)
    count = 6000
    lo = rng.integers(-50, 50, size=count, dtype=np.int64)
    hi = lo + rng.integers(0, 50, size=count, dtype=np.int64)
    lo2 = rng.integers(-50, 50, size=count, dtype=np.int64)
    hi2 = lo2 + rng.integers(0, 50, size=count, dtype=np.int64)

    a = np.stack([lo, hi], axis=-1)
    b = np.stack([lo2, hi2], axis=-1)

    out_add = np.empty_like(a)
    out_sub = np.empty_like(a)
    add_fn = lib.fmpzi_add_ref
    sub_fn = lib.fmpzi_sub_ref
    for i in range(count):
        r_add = add_fn(FMPZI(int(a[i, 0]), int(a[i, 1])), FMPZI(int(b[i, 0]), int(b[i, 1])))
        r_sub = sub_fn(FMPZI(int(a[i, 0]), int(a[i, 1])), FMPZI(int(b[i, 0]), int(b[i, 1])))
        out_add[i, 0] = r_add.lo
        out_add[i, 1] = r_add.hi
        out_sub[i, 0] = r_sub.lo
        out_sub[i, 1] = r_sub.hi

    j_add = np.asarray(fmpzi.fmpzi_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_sub = np.asarray(fmpzi.fmpzi_sub_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    np.testing.assert_array_equal(out_add, j_add)
    np.testing.assert_array_equal(out_sub, j_sub)
