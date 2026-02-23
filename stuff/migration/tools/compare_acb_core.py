from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from arbjax import acb_core


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


def default_paths() -> tuple[Path | None, Path | None]:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    core = _find_lib(build_dir, ["acb_core_ref.dll", "libacb_core_ref.dll", "libacb_core_ref.so", "libacb_core_ref.dylib"])
    return di, core


def load_lib(di_path: Path, core_path: Path):
    ctypes.CDLL(str(di_path))
    core = ctypes.CDLL(str(core_path))
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
        fn = getattr(core, fn_name)
        fn.argtypes = [ACB]
        fn.restype = ACB
    return core


def random_intervals(rng: np.random.Generator, n: int, scale: float = 6.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_boxes(rng: np.random.Generator, n: int, scale: float = 6.0) -> np.ndarray:
    re = random_intervals(rng, n, scale)
    im = random_intervals(rng, n, scale)
    return np.concatenate([re, im], axis=-1)


def call_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(ACB(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3]))))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
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
    parser = argparse.ArgumentParser(description="Compare acb core C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=12000)
    args = parser.parse_args()

    di_env = os.getenv("DI_REF_LIB", "")
    core_env = os.getenv("ACB_CORE_REF_LIB", "")
    d_di, d_core = default_paths()
    di_path = Path(di_env) if di_env else d_di
    core_path = Path(core_env) if core_env else d_core
    if di_path is None or core_path is None or not di_path.exists() or not core_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(di_path, core_path)
    rng = np.random.default_rng(2112)
    x = random_boxes(rng, args.samples, 6.0)

    ok = True
    ok &= report("acb_exp", call_unary(lib, "acb_exp_ref", x), np.asarray(acb_core.acb_exp_batch_jit(jnp.asarray(x))), rtol=3e-13, atol=2e-14)
    ok &= report("acb_log", call_unary(lib, "acb_log_ref", x), np.asarray(acb_core.acb_log_batch_jit(jnp.asarray(x))), rtol=3e-13, atol=2e-14)
    ok &= report("acb_sqrt", call_unary(lib, "acb_sqrt_ref", x), np.asarray(acb_core.acb_sqrt_batch_jit(jnp.asarray(x))), rtol=3e-13, atol=2e-14)
    ok &= report("acb_sin", call_unary(lib, "acb_sin_ref", x), np.asarray(acb_core.acb_sin_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("acb_cos", call_unary(lib, "acb_cos_ref", x), np.asarray(acb_core.acb_cos_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("acb_tan", call_unary(lib, "acb_tan_ref", x), np.asarray(acb_core.acb_tan_batch_jit(jnp.asarray(x))), rtol=5e-12, atol=2e-14)
    ok &= report("acb_sinh", call_unary(lib, "acb_sinh_ref", x), np.asarray(acb_core.acb_sinh_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("acb_cosh", call_unary(lib, "acb_cosh_ref", x), np.asarray(acb_core.acb_cosh_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)
    ok &= report("acb_tanh", call_unary(lib, "acb_tanh_ref", x), np.asarray(acb_core.acb_tanh_batch_jit(jnp.asarray(x))), rtol=3e-12, atol=2e-14)

    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
