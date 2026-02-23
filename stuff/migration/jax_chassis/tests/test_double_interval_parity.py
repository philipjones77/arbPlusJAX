import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from arbjax import double_interval as di

pytestmark = pytest.mark.parity
if os.getenv("ARBJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def _default_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None

    candidates = [
        "double_interval_ref.dll",
        "libdouble_interval_ref.dll",
        "libdouble_interval_ref.so",
        "libdouble_interval_ref.dylib",
    ]
    for name in candidates:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _load_lib():
    explicit = os.getenv("DI_REF_LIB")
    path = Path(explicit) if explicit else _default_lib_path()
    if path is None or not path.exists():
        pytest.skip("C reference library not found. Build migration/c_chassis first.")

    lib = ctypes.CDLL(str(path))
    lib.di_fast_add.argtypes = [DI, DI]
    lib.di_fast_add.restype = DI
    lib.di_fast_mul.argtypes = [DI, DI]
    lib.di_fast_mul.restype = DI
    lib.di_fast_div.argtypes = [DI, DI]
    lib.di_fast_div.restype = DI
    lib.di_fast_sqr.argtypes = [DI]
    lib.di_fast_sqr.restype = DI
    return lib


def _random_intervals(rng: np.random.Generator, n: int, scale: float = 1e6) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def _random_nonzero_denominators(rng: np.random.Generator, n: int) -> np.ndarray:
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
    lo_mag = rng.uniform(1e-8, 1e4, size=n)
    hi_mag = lo_mag + rng.uniform(0.0, 1e4, size=n)
    lo = np.where(sign > 0.0, lo_mag, -hi_mag)
    hi = np.where(sign > 0.0, hi_mag, -lo_mag)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def _call_binary(lib, fn_name: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(xs)
    for i in range(xs.shape[0]):
        r = fn(DI(xs[i, 0], xs[i, 1]), DI(ys[i, 0], ys[i, 1]))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _call_unary(lib, fn_name: str, xs: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(xs)
    for i in range(xs.shape[0]):
        r = fn(DI(xs[i, 0], xs[i, 1]))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _assert_close(c_out: np.ndarray, j_out: np.ndarray):
    np.testing.assert_allclose(c_out, j_out, rtol=2e-15, atol=0.0, equal_nan=True)
    assert np.all(j_out[:, 0] <= j_out[:, 1])


def test_parity_add_mul_div_sqr():
    lib = _load_lib()
    rng = np.random.default_rng(123)
    n = 4000

    x = _random_intervals(rng, n)
    y = _random_intervals(rng, n)
    d = _random_nonzero_denominators(rng, n)

    xj = jnp.asarray(x)
    yj = jnp.asarray(y)
    dj = jnp.asarray(d)

    add_c = _call_binary(lib, "di_fast_add", x, y)
    mul_c = _call_binary(lib, "di_fast_mul", x, y)
    div_c = _call_binary(lib, "di_fast_div", x, d)
    sqr_c = _call_unary(lib, "di_fast_sqr", x)

    add_j = np.asarray(di.batch_fast_add(xj, yj))
    mul_j = np.asarray(di.batch_fast_mul(xj, yj))
    div_j = np.asarray(di.batch_fast_div(xj, dj))
    sqr_j = np.asarray(di.batch_fast_sqr(xj))

    _assert_close(add_c, add_j)
    _assert_close(mul_c, mul_j)
    _assert_close(div_c, div_j)
    _assert_close(sqr_c, sqr_j)
