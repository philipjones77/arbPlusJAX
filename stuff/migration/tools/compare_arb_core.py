from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from arbjax import arb_core


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def default_paths() -> tuple[Path | None, Path | None]:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    core = _find_lib(build_dir, ["arb_core_ref.dll", "libarb_core_ref.dll", "libarb_core_ref.so", "libarb_core_ref.dylib"])
    return di, core


def load_lib(di_path: Path, core_path: Path):
    ctypes.CDLL(str(di_path))
    core = ctypes.CDLL(str(core_path))
    for fn_name in (
        "arb_exp_ref",
        "arb_log_ref",
        "arb_sqrt_ref",
        "arb_sin_ref",
        "arb_cos_ref",
        "arb_tan_ref",
        "arb_sinh_ref",
        "arb_cosh_ref",
        "arb_tanh_ref",
    ):
        fn = getattr(core, fn_name)
        fn.argtypes = [DI]
        fn.restype = DI
    return core


def random_intervals(rng: np.random.Generator, n: int, scale: float = 8.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_positive_intervals(rng: np.random.Generator, n: int, lo: float = 0.01, hi: float = 8.0) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def call_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def report(name: str, c_out: np.ndarray, j_out: np.ndarray, rtol: float, atol: float = 0.0) -> bool:
    ok = np.allclose(c_out, j_out, rtol=rtol, atol=atol, equal_nan=True)
    with np.errstate(invalid="ignore"):
        diff = np.abs(c_out - j_out)
    finite = np.isfinite(diff)
    max_diff = float(np.max(diff[finite])) if np.any(finite) else 0.0
    print(f"{name:16s} | ok={ok} | max_abs_diff={max_diff:.3e}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare arb core C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=12000)
    args = parser.parse_args()

    di_env = os.getenv("DI_REF_LIB", "")
    core_env = os.getenv("ARB_CORE_REF_LIB", "")
    d_di, d_core = default_paths()
    di_path = Path(di_env) if di_env else d_di
    core_path = Path(core_env) if core_env else d_core
    if di_path is None or core_path is None or not di_path.exists() or not core_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(di_path, core_path)
    rng = np.random.default_rng(2101)
    x = random_intervals(rng, args.samples, 8.0)
    xp = random_positive_intervals(rng, args.samples, 0.01, 8.0)

    ok = True
    ok &= report("arb_exp", call_unary(lib, "arb_exp_ref", x), np.asarray(arb_core.arb_exp_batch_jit(jnp.asarray(x))), rtol=3e-13, atol=2e-14)
    ok &= report("arb_log", call_unary(lib, "arb_log_ref", xp), np.asarray(arb_core.arb_log_batch_jit(jnp.asarray(xp))), rtol=3e-13, atol=2e-14)
    ok &= report("arb_sqrt", call_unary(lib, "arb_sqrt_ref", xp), np.asarray(arb_core.arb_sqrt_batch_jit(jnp.asarray(xp))), rtol=3e-13, atol=2e-14)
    ok &= report("arb_sin", call_unary(lib, "arb_sin_ref", x), np.asarray(arb_core.arb_sin_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("arb_cos", call_unary(lib, "arb_cos_ref", x), np.asarray(arb_core.arb_cos_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("arb_tan", call_unary(lib, "arb_tan_ref", x), np.asarray(arb_core.arb_tan_batch_jit(jnp.asarray(x))), rtol=5e-12, atol=2e-14)
    ok &= report("arb_sinh", call_unary(lib, "arb_sinh_ref", x), np.asarray(arb_core.arb_sinh_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("arb_cosh", call_unary(lib, "arb_cosh_ref", x), np.asarray(arb_core.arb_cosh_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("arb_tanh", call_unary(lib, "arb_tanh_ref", x), np.asarray(arb_core.arb_tanh_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)

    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
