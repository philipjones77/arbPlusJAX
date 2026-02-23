from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from arbplusjax import double_interval as di


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def default_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None

    for name in ("double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"):
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def load_lib(path: Path):
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


def random_intervals(rng: np.random.Generator, n: int, scale: float = 1e6) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_nonzero_denominators(rng: np.random.Generator, n: int) -> np.ndarray:
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
    lo_mag = rng.uniform(1e-8, 1e4, size=n)
    hi_mag = lo_mag + rng.uniform(0.0, 1e4, size=n)
    lo = np.where(sign > 0.0, lo_mag, -hi_mag)
    hi = np.where(sign > 0.0, hi_mag, -lo_mag)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def call_binary(lib, fn_name: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(xs)
    for i in range(xs.shape[0]):
        r = fn(DI(xs[i, 0], xs[i, 1]), DI(ys[i, 0], ys[i, 1]))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_unary(lib, fn_name: str, xs: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(xs)
    for i in range(xs.shape[0]):
        r = fn(DI(xs[i, 0], xs[i, 1]))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def report(op_name: str, c_out: np.ndarray, j_out: np.ndarray) -> bool:
    ok = np.allclose(c_out, j_out, rtol=2e-15, atol=0.0, equal_nan=True) and np.all(j_out[:, 0] <= j_out[:, 1])
    diff = np.max(np.abs(c_out - j_out))
    print(f"{op_name:8s} | ok={ok} | max_abs_diff={diff:.3e}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare double_interval C and JAX kernels.")
    parser.add_argument("--lib", type=str, default=os.getenv("DI_REF_LIB", ""), help="Path to C reference shared library")
    parser.add_argument("--samples", type=int, default=20000, help="Number of random samples")
    args = parser.parse_args()

    lib_path = Path(args.lib) if args.lib else default_lib_path()
    if lib_path is None or not lib_path.exists():
        print("C reference library not found. Build C reference libraries in the Arb workspace first.")
        return 1

    lib = load_lib(lib_path)
    rng = np.random.default_rng(12345)
    n = args.samples

    x = random_intervals(rng, n)
    y = random_intervals(rng, n)
    d = random_nonzero_denominators(rng, n)

    xj = jnp.asarray(x)
    yj = jnp.asarray(y)
    dj = jnp.asarray(d)

    add_ok = report("fast_add", call_binary(lib, "di_fast_add", x, y), np.asarray(di.batch_fast_add(xj, yj)))
    mul_ok = report("fast_mul", call_binary(lib, "di_fast_mul", x, y), np.asarray(di.batch_fast_mul(xj, yj)))
    div_ok = report("fast_div", call_binary(lib, "di_fast_div", x, d), np.asarray(di.batch_fast_div(xj, dj)))
    sqr_ok = report("fast_sqr", call_unary(lib, "di_fast_sqr", x), np.asarray(di.batch_fast_sqr(xj)))

    all_ok = add_ok and mul_ok and div_ok and sqr_ok
    print(f"\nresult: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
