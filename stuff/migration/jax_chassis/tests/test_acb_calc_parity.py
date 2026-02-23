import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import acb_calc

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


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
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    calc = _find_lib(build_dir, ["acb_calc_ref.dll", "libacb_calc_ref.dll", "libacb_calc_ref.so", "libacb_calc_ref.dylib"])
    return di, calc


def _load_lib():
    di_env = os.getenv("DI_REF_LIB")
    calc_env = os.getenv("ACB_CALC_REF_LIB")
    d_di, d_calc = _default_libs()
    di_path = Path(di_env) if di_env else d_di
    calc_path = Path(calc_env) if calc_env else d_calc
    if di_path is None or calc_path is None or not di_path.exists() or not calc_path.exists():
        pytest.skip("C reference libraries not found. Build migration/c_chassis first.")

    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(calc_path))
    fn = lib.acb_calc_integrate_line_ref
    fn.argtypes = [ACB, ACB, ctypes.c_int, ctypes.c_int]
    fn.restype = ACB
    return lib


def _random_intervals(rng: np.random.Generator, n: int, scale: float = 2.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def _random_boxes(rng: np.random.Generator, n: int, scale: float = 2.0) -> np.ndarray:
    re = _random_intervals(rng, n, scale)
    im = _random_intervals(rng, n, scale)
    return np.concatenate([re, im], axis=-1)


def _call_unary(lib, x: np.ndarray, integrand_id: int, n_steps: int) -> np.ndarray:
    fn = lib.acb_calc_integrate_line_ref
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(
            ACB(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3]))),
            ACB(DI(float(x[i, 4]), float(x[i, 5])), DI(float(x[i, 6]), float(x[i, 7]))),
            integrand_id,
            n_steps,
        )
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def test_acb_calc_parity():
    lib = _load_lib()
    rng = np.random.default_rng(2121)
    a = _random_boxes(rng, 3000, 1.2)
    b = _random_boxes(rng, 3000, 1.2)
    n_steps = 48

    c_exp = _call_unary(lib, np.concatenate([a, b], axis=-1), 0, n_steps)
    j_exp = np.asarray(acb_calc.acb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="exp", n_steps=n_steps))
    np.testing.assert_allclose(c_exp, j_exp, rtol=5e-12, atol=2e-12, equal_nan=True)

    c_sin = _call_unary(lib, np.concatenate([a, b], axis=-1), 1, n_steps)
    j_sin = np.asarray(acb_calc.acb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="sin", n_steps=n_steps))
    np.testing.assert_allclose(c_sin, j_sin, rtol=5e-12, atol=2e-12, equal_nan=True)

    c_cos = _call_unary(lib, np.concatenate([a, b], axis=-1), 2, n_steps)
    j_cos = np.asarray(acb_calc.acb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="cos", n_steps=n_steps))
    np.testing.assert_allclose(c_cos, j_cos, rtol=5e-12, atol=2e-12, equal_nan=True)
