from __future__ import annotations

import argparse
import ctypes
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import jax.numpy as jnp
import numpy as np

from arbplusjax import acb_calc


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
    calc = _find_lib(build_dir, ["acb_calc_ref.dll", "libacb_calc_ref.dll", "libacb_calc_ref.so", "libacb_calc_ref.dylib"])
    return di, calc


def load_lib(di_path: Path, calc_path: Path):
    ctypes.CDLL(str(di_path))
    calc = ctypes.CDLL(str(calc_path))
    fn = calc.acb_calc_integrate_line_ref
    fn.argtypes = [ACB, ACB, ctypes.c_int, ctypes.c_int]
    fn.restype = ACB
    return calc


def random_intervals(rng: np.random.Generator, n: int, scale: float = 2.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_boxes(rng: np.random.Generator, n: int, scale: float = 2.0) -> np.ndarray:
    re = random_intervals(rng, n, scale)
    im = random_intervals(rng, n, scale)
    return np.concatenate([re, im], axis=-1)


def call_unary(lib, a: np.ndarray, b: np.ndarray, integrand_id: int, n_steps: int) -> np.ndarray:
    fn = lib.acb_calc_integrate_line_ref
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        r = fn(
            ACB(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3]))),
            ACB(DI(float(b[i, 0]), float(b[i, 1])), DI(float(b[i, 2]), float(b[i, 3]))),
            integrand_id,
            n_steps,
        )
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


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        )
    except Exception:
        return "unknown"


def _log_run(tool: str, command: str, notes: str = "") -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "migration" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "runs.csv"
    if not log_path.exists():
        log_path.write_text("run_id,timestamp_utc,tool,command,commit,platform,notes\n", encoding="ascii")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"{tool}-{timestamp}"
    commit = _git_commit(repo_root)
    plat = platform.platform()
    line = f"{run_id},{timestamp},{tool},\"{command}\",{commit},\"{plat}\",\"{notes}\"\n"
    with log_path.open("a", encoding="ascii") as f:
        f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare acb_calc C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()

    di_env = os.getenv("DI_REF_LIB", "")
    calc_env = os.getenv("ACB_CALC_REF_LIB", "")
    d_di, d_calc = default_paths()
    di_path = Path(di_env) if di_env else d_di
    calc_path = Path(calc_env) if calc_env else d_calc
    if di_path is None or calc_path is None or not di_path.exists() or not calc_path.exists():
        print("Reference libraries not found. Build C reference libraries in the Arb workspace first.")
        return 1

    lib = load_lib(di_path, calc_path)
    rng = np.random.default_rng(2122)
    a = random_boxes(rng, args.samples, 1.5)
    b = random_boxes(rng, args.samples, 1.5)
    n_steps = args.steps

    ok = True
    ok &= report(
        "acb_calc_exp",
        call_unary(lib, a, b, 0, n_steps),
        np.asarray(acb_calc.acb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="exp", n_steps=n_steps)),
        rtol=5e-12,
        atol=2e-12,
    )
    ok &= report(
        "acb_calc_sin",
        call_unary(lib, a, b, 1, n_steps),
        np.asarray(acb_calc.acb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="sin", n_steps=n_steps)),
        rtol=5e-12,
        atol=2e-12,
    )
    ok &= report(
        "acb_calc_cos",
        call_unary(lib, a, b, 2, n_steps),
        np.asarray(acb_calc.acb_calc_integrate_line_batch_jit(jnp.asarray(a), jnp.asarray(b), integrand="cos", n_steps=n_steps)),
        rtol=5e-12,
        atol=2e-12,
    )

    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_acb_calc", f"compare_acb_calc.py --samples {args.samples} --steps {args.steps}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
