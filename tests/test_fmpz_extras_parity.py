import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import fmpz_extras

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class FMPZ(ctypes.Structure):
    _fields_ = [("v", ctypes.c_int64)]


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
    return _find_lib(
        build_dir,
        [
            "fmpz_extras_ref.dll",
            "libfmpz_extras_ref.dll",
            "libfmpz_extras_ref.so",
            "libfmpz_extras_ref.dylib",
        ],
    )


def _load_lib():
    lib_env = os.getenv("FMPZ_EXTRAS_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("fmpz_extras_add_ref", "fmpz_extras_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [FMPZ, FMPZ]
        fn.restype = FMPZ
    return lib


def test_fmpz_extras_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2101)
    count = 8000
    a = rng.integers(-1000, 1000, size=count, dtype=np.int64)
    b = rng.integers(-1000, 1000, size=count, dtype=np.int64)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.fmpz_extras_add_ref
    mul_fn = lib.fmpz_extras_mul_ref
    for i in range(count):
        out_add[i] = add_fn(FMPZ(int(a[i])), FMPZ(int(b[i]))).v
        out_mul[i] = mul_fn(FMPZ(int(a[i])), FMPZ(int(b[i]))).v

    j_add = np.asarray(fmpz_extras.fmpz_extras_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(fmpz_extras.fmpz_extras_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    np.testing.assert_array_equal(out_add, j_add)
    np.testing.assert_array_equal(out_mul, j_mul)
