import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import bernoulli

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


def _default_lib() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["bernoulli_ref.dll", "libbernoulli_ref.dll", "libbernoulli_ref.so", "libbernoulli_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("BERNOULLI_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bernoulli_number_ref
    fn.argtypes = [ctypes.c_int]
    fn.restype = ctypes.c_double
    return lib


def test_bernoulli_parity():
    lib = _load_lib()
    n = np.array([0, 1, 2, 4, 6], dtype=np.int64)
    out_c = np.array([lib.bernoulli_number_ref(int(k)) for k in n], dtype=np.float64)
    out_j = np.asarray(bernoulli.bernoulli_number_batch_jit(jnp.asarray(n)))
    np.testing.assert_allclose(out_c, out_j, rtol=0.0, atol=0.0)
