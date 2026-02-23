import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import acb_core

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


class ACB(ctypes.Structure):
    _fields_ = [("real", DI), ("imag", DI)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _default_libs() -> tuple[Path | None, Path | None]:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    core = _find_lib(build_dir, ["acb_core_ref.dll", "libacb_core_ref.dll", "libacb_core_ref.so", "libacb_core_ref.dylib"])
    return di, core


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    core_env = os.getenv("ACB_CORE_REF_LIB")
    d_di, d_core = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    core_path = Path(core_env) if core_env else d_core
    if di_path is None or core_path is None or not di_path.exists() or not core_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(core_path))
    for fn_name in (
        "acb_exp_ref",
        "acb_log_ref",
        "acb_sqrt_ref",
        "acb_sin_ref",
        "acb_cos_ref",
        "acb_tan_ref",
        "acb_sinh_ref",
        "acb_cosh_ref",
        "acb_tanh_ref",
    ):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ACB]
        fn.restype = ACB
    return lib


def _random_intervals(rng: np.random.Generator, n: int, scale: float = 6.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def _random_boxes(rng: np.random.Generator, n: int, scale: float = 6.0) -> np.ndarray:
    re = _random_intervals(rng, n, scale)
    im = _random_intervals(rng, n, scale)
    return np.concatenate([re, im], axis=-1)


def _call_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(ACB(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3]))))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def test_acb_core_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2111)
    x = _random_boxes(rng, 4000, 5.0)

    c_exp = _call_unary(lib, "acb_exp_ref", x)
    c_log = _call_unary(lib, "acb_log_ref", x)
    c_sqrt = _call_unary(lib, "acb_sqrt_ref", x)
    c_sin = _call_unary(lib, "acb_sin_ref", x)
    c_cos = _call_unary(lib, "acb_cos_ref", x)
    c_tan = _call_unary(lib, "acb_tan_ref", x)
    c_sinh = _call_unary(lib, "acb_sinh_ref", x)
    c_cosh = _call_unary(lib, "acb_cosh_ref", x)
    c_tanh = _call_unary(lib, "acb_tanh_ref", x)

    j_exp = np.asarray(acb_core.acb_exp_batch_jit(jnp.asarray(x)))
    j_log = np.asarray(acb_core.acb_log_batch_jit(jnp.asarray(x)))
    j_sqrt = np.asarray(acb_core.acb_sqrt_batch_jit(jnp.asarray(x)))
    j_sin = np.asarray(acb_core.acb_sin_batch_jit(jnp.asarray(x)))
    j_cos = np.asarray(acb_core.acb_cos_batch_jit(jnp.asarray(x)))
    j_tan = np.asarray(acb_core.acb_tan_batch_jit(jnp.asarray(x)))
    j_sinh = np.asarray(acb_core.acb_sinh_batch_jit(jnp.asarray(x)))
    j_cosh = np.asarray(acb_core.acb_cosh_batch_jit(jnp.asarray(x)))
    j_tanh = np.asarray(acb_core.acb_tanh_batch_jit(jnp.asarray(x)))

    np.testing.assert_allclose(c_exp, j_exp, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(c_log, j_log, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(c_sqrt, j_sqrt, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(c_sin, j_sin, rtol=3e-12, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_cos, j_cos, rtol=3e-12, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_tan, j_tan, rtol=5e-12, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_sinh, j_sinh, rtol=3e-12, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_cosh, j_cosh, rtol=3e-12, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_tanh, j_tanh, rtol=3e-12, atol=2e-14, equal_nan=True)
