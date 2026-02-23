import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import acf

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class ACF(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


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
    return _find_lib(build_dir, ["acf_ref.dll", "libacf_ref.dll", "libacf_ref.so", "libacf_ref.dylib"])


def _load_lib():
    lib_env = os.getenv("ACF_REF_LIB")
    lib_path = Path(lib_env) if lib_env else _default_lib()
    if lib_path is None or not lib_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("acf_add_ref", "acf_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ACF, ACF]
        fn.restype = ACF
    return lib


def test_acf_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2181)
    a = rng.normal(size=3000) + 1j * rng.normal(size=3000)
    b = rng.normal(size=3000) + 1j * rng.normal(size=3000)

    out_add = np.empty(3000, dtype=np.complex128)
    out_mul = np.empty(3000, dtype=np.complex128)
    add_fn = lib.acf_add_ref
    mul_fn = lib.acf_mul_ref
    for i in range(3000):
        r = add_fn(ACF(float(a[i].real), float(a[i].imag)), ACF(float(b[i].real), float(b[i].imag)))
        out_add[i] = r.re + 1j * r.im
        r = mul_fn(ACF(float(a[i].real), float(a[i].imag)), ACF(float(b[i].real), float(b[i].imag)))
        out_mul[i] = r.re + 1j * r.im

    j_add = np.asarray(acf.acf_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(acf.acf_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    np.testing.assert_allclose(out_add, j_add, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(out_mul, j_mul, rtol=1e-14, atol=1e-14)
