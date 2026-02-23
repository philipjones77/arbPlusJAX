from __future__ import annotations

import argparse
import ctypes
import gc
import os
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
import psutil

from arbjax import hypgeom


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


class ACBBox(ctypes.Structure):
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
    hyp = _find_lib(build_dir, ["hypgeom_ref.dll", "libhypgeom_ref.dll", "libhypgeom_ref.so", "libhypgeom_ref.dylib"])
    return di, hyp


def load_hypgeom_lib(di_path: Path, hyp_path: Path):
    ctypes.CDLL(str(di_path))
    hyp = ctypes.CDLL(str(hyp_path))
    hyp.arb_hypgeom_rising_ui_ref.argtypes = [DI, ctypes.c_ulonglong]
    hyp.arb_hypgeom_rising_ui_ref.restype = DI
    hyp.acb_hypgeom_rising_ui_ref.argtypes = [ACBBox, ctypes.c_ulonglong]
    hyp.acb_hypgeom_rising_ui_ref.restype = ACBBox
    hyp.arb_hypgeom_lgamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_lgamma_ref.restype = DI
    hyp.acb_hypgeom_lgamma_ref.argtypes = [ACBBox]
    hyp.acb_hypgeom_lgamma_ref.restype = ACBBox
    hyp.arb_hypgeom_gamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_gamma_ref.restype = DI
    hyp.acb_hypgeom_gamma_ref.argtypes = [ACBBox]
    hyp.acb_hypgeom_gamma_ref.restype = ACBBox
    hyp.arb_hypgeom_rgamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_rgamma_ref.restype = DI
    hyp.acb_hypgeom_rgamma_ref.argtypes = [ACBBox]
    hyp.acb_hypgeom_rgamma_ref.restype = ACBBox
    hyp.arb_hypgeom_erf_ref.argtypes = [DI]
    hyp.arb_hypgeom_erf_ref.restype = DI
    hyp.acb_hypgeom_erf_ref.argtypes = [ACBBox]
    hyp.acb_hypgeom_erf_ref.restype = ACBBox
    hyp.arb_hypgeom_erfc_ref.argtypes = [DI]
    hyp.arb_hypgeom_erfc_ref.restype = DI
    hyp.acb_hypgeom_erfc_ref.argtypes = [ACBBox]
    hyp.acb_hypgeom_erfc_ref.restype = ACBBox
    hyp.arb_hypgeom_erfi_ref.argtypes = [DI]
    hyp.arb_hypgeom_erfi_ref.restype = DI
    hyp.acb_hypgeom_erfi_ref.argtypes = [ACBBox]
    hyp.acb_hypgeom_erfi_ref.restype = ACBBox
    hyp.arb_hypgeom_1f1_ref.argtypes = [DI, DI, DI]
    hyp.arb_hypgeom_1f1_ref.restype = DI
    hyp.acb_hypgeom_1f1_ref.argtypes = [ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_1f1_ref.restype = ACBBox
    hyp.arb_hypgeom_2f1_ref.argtypes = [DI, DI, DI, DI]
    hyp.arb_hypgeom_2f1_ref.restype = DI
    hyp.acb_hypgeom_2f1_ref.argtypes = [ACBBox, ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_2f1_ref.restype = ACBBox
    return hyp


def random_intervals(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_positive_intervals(rng: np.random.Generator, n: int, lo: float = 0.05, hi: float = 10.0) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def random_acb_boxes(rng: np.random.Generator, n: int, scale: float = 3.0) -> np.ndarray:
    re = random_intervals(rng, n, scale=scale)
    im = random_intervals(rng, n, scale=scale)
    return np.concatenate([re, im], axis=-1)


def random_acb_boxes_away_from_poles(rng: np.random.Generator, n: int) -> np.ndarray:
    re = random_intervals(rng, n, scale=4.0)
    im_lo = rng.uniform(0.15, 3.0, size=n)
    im_hi = im_lo + rng.uniform(0.0, 0.5, size=n)
    im = np.stack([im_lo, im_hi], axis=-1)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def random_hyp_params_real(rng: np.random.Generator, n: int):
    a = random_positive_intervals(rng, n, lo=0.5, hi=2.0)
    b = random_positive_intervals(rng, n, lo=1.5, hi=3.5)
    c = random_positive_intervals(rng, n, lo=1.8, hi=4.0)
    z = random_intervals(rng, n, scale=0.5)
    return a, b, c, z


def random_hyp_params_complex(rng: np.random.Generator, n: int):
    a = random_acb_boxes(rng, n, scale=1.0)
    b = random_acb_boxes(rng, n, scale=1.0)
    c = random_acb_boxes(rng, n, scale=1.0)
    b[:, 0:2] += 2.2
    c[:, 0:2] += 2.5
    z = random_acb_boxes(rng, n, scale=0.5)
    return a, b, c, z


def call_real_scalar(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_scalar(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xb = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
        r = fn(xb)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_rising(lib, x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = lib.arb_hypgeom_rising_ui_ref(DI(float(x[i, 0]), float(x[i, 1])), n)
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_rising(lib, x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xb = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
        r = lib.acb_hypgeom_rising_ui_ref(xb, n)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_ternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(b[i, 0]), float(b[i, 1])), DI(float(z[i, 0]), float(z[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_ternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        aa = ACBBox(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3])))
        bb = ACBBox(DI(float(b[i, 0]), float(b[i, 1])), DI(float(b[i, 2]), float(b[i, 3])))
        zz = ACBBox(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3])))
        r = fn(aa, bb, zz)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_quaternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        r = fn(
            DI(float(a[i, 0]), float(a[i, 1])),
            DI(float(b[i, 0]), float(b[i, 1])),
            DI(float(c[i, 0]), float(c[i, 1])),
            DI(float(z[i, 0]), float(z[i, 1])),
        )
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_quaternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        aa = ACBBox(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3])))
        bb = ACBBox(DI(float(b[i, 0]), float(b[i, 1])), DI(float(b[i, 2]), float(b[i, 3])))
        cc = ACBBox(DI(float(c[i, 0]), float(c[i, 1])), DI(float(c[i, 2]), float(c[i, 3])))
        zz = ACBBox(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3])))
        r = fn(aa, bb, cc, zz)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def _rss_bytes(proc: psutil.Process) -> int:
    gc.collect()
    return proc.memory_info().rss


def _time_c(fn, repeats: int = 3):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), out


def _time_jax(fn, repeats: int = 5):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), out


def _run_case(name: str, c_fn, j_fn, rtol: float, atol: float, proc: psutil.Process):
    rss0 = _rss_bytes(proc)
    t0 = time.perf_counter()
    j_first = j_fn()
    j_first.block_until_ready()
    t1 = time.perf_counter()
    rss1 = _rss_bytes(proc)
    j_compile_ms = (t1 - t0) * 1000.0
    j_compile_rss_mb = (rss1 - rss0) / (1024.0 * 1024.0)

    rss2 = _rss_bytes(proc)
    j_exec_s, j_out = _time_jax(j_fn)
    rss3 = _rss_bytes(proc)
    j_exec_ms = j_exec_s * 1000.0
    j_exec_rss_mb = (rss3 - rss2) / (1024.0 * 1024.0)
    j_np = np.asarray(j_out)

    rss4 = _rss_bytes(proc)
    c_exec_s, c_out = _time_c(c_fn)
    rss5 = _rss_bytes(proc)
    c_exec_ms = c_exec_s * 1000.0
    c_exec_rss_mb = (rss5 - rss4) / (1024.0 * 1024.0)

    with np.errstate(invalid="ignore"):
        diff = np.abs(c_out - j_np)
    finite = np.isfinite(diff)
    max_diff = float(np.max(diff[finite])) if np.any(finite) else 0.0
    ok = np.allclose(c_out, j_np, rtol=rtol, atol=atol, equal_nan=True)
    speedup = (c_exec_ms / j_exec_ms) if j_exec_ms > 0.0 else np.inf
    return (
        name,
        ok,
        max_diff,
        c_exec_ms,
        j_compile_ms,
        j_exec_ms,
        speedup,
        c_exec_rss_mb,
        j_compile_rss_mb,
        j_exec_rss_mb,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark hypgeom C element-wise reference vs JAX batched kernels.")
    parser.add_argument("--samples-real", type=int, default=3000)
    parser.add_argument("--samples-complex", type=int, default=1200)
    args = parser.parse_args()

    di_env = os.getenv("DI_REF_LIB", "")
    hyp_env = os.getenv("HYPGEOM_REF_LIB", "")
    d_di, d_hyp = default_paths()
    di_path = Path(di_env) if di_env else d_di
    hyp_path = Path(hyp_env) if hyp_env else d_hyp
    if di_path is None or hyp_path is None or not di_path.exists() or not hyp_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_hypgeom_lib(di_path, hyp_path)
    proc = psutil.Process()
    rng = np.random.default_rng(2026)

    xr = random_intervals(rng, args.samples_real, 10.0)
    xc = random_acb_boxes(rng, args.samples_complex, 3.0)
    xr_lg = random_positive_intervals(rng, args.samples_real)
    xc_lg = random_acb_boxes_away_from_poles(rng, args.samples_complex)
    xr_e = random_intervals(rng, args.samples_real, 1.5)
    xc_e = random_acb_boxes(rng, args.samples_complex, 1.5)
    a_r, b_r, c_r, z_r = random_hyp_params_real(rng, args.samples_real)
    a_c, b_c, c_c, z_c = random_hyp_params_complex(rng, args.samples_complex)

    xr_j = jnp.asarray(xr)
    xc_j = jnp.asarray(xc)
    xr_lg_j = jnp.asarray(xr_lg)
    xc_lg_j = jnp.asarray(xc_lg)
    xr_e_j = jnp.asarray(xr_e)
    xc_e_j = jnp.asarray(xc_e)
    a_r_j = jnp.asarray(a_r)
    b_r_j = jnp.asarray(b_r)
    c_r_j = jnp.asarray(c_r)
    z_r_j = jnp.asarray(z_r)
    a_c_j = jnp.asarray(a_c)
    b_c_j = jnp.asarray(b_c)
    c_c_j = jnp.asarray(c_c)
    z_c_j = jnp.asarray(z_c)

    rows = []
    rows.append(_run_case(
        "arb_rising_ui(n=15)",
        lambda: call_real_rising(lib, xr, 15),
        lambda: hypgeom.arb_hypgeom_rising_ui_batch_jit(xr_j, n=15),
        rtol=2e-13,
        atol=0.0,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_rising_ui(n=15)",
        lambda: call_complex_rising(lib, xc, 15),
        lambda: hypgeom.acb_hypgeom_rising_ui_batch_jit(xc_j, n=15),
        rtol=3e-13,
        atol=0.0,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_lgamma",
        lambda: call_real_scalar(lib, "arb_hypgeom_lgamma_ref", xr_lg),
        lambda: hypgeom.arb_hypgeom_lgamma_batch_jit(xr_lg_j),
        rtol=2e-12,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_lgamma",
        lambda: call_complex_scalar(lib, "acb_hypgeom_lgamma_ref", xc_lg),
        lambda: hypgeom.acb_hypgeom_lgamma_batch_jit(xc_lg_j),
        rtol=1e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_gamma",
        lambda: call_real_scalar(lib, "arb_hypgeom_gamma_ref", xr_lg),
        lambda: hypgeom.arb_hypgeom_gamma_batch_jit(xr_lg_j),
        rtol=4e-12,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_gamma",
        lambda: call_complex_scalar(lib, "acb_hypgeom_gamma_ref", xc_lg),
        lambda: hypgeom.acb_hypgeom_gamma_batch_jit(xc_lg_j),
        rtol=2e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_rgamma",
        lambda: call_real_scalar(lib, "arb_hypgeom_rgamma_ref", xr_lg),
        lambda: hypgeom.arb_hypgeom_rgamma_batch_jit(xr_lg_j),
        rtol=4e-12,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_rgamma",
        lambda: call_complex_scalar(lib, "acb_hypgeom_rgamma_ref", xc_lg),
        lambda: hypgeom.acb_hypgeom_rgamma_batch_jit(xc_lg_j),
        rtol=2e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_erf",
        lambda: call_real_scalar(lib, "arb_hypgeom_erf_ref", xr_e),
        lambda: hypgeom.arb_hypgeom_erf_batch_jit(xr_e_j),
        rtol=2e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_erf",
        lambda: call_complex_scalar(lib, "acb_hypgeom_erf_ref", xc_e),
        lambda: hypgeom.acb_hypgeom_erf_batch_jit(xc_e_j),
        rtol=3e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_erfc",
        lambda: call_real_scalar(lib, "arb_hypgeom_erfc_ref", xr_e),
        lambda: hypgeom.arb_hypgeom_erfc_batch_jit(xr_e_j),
        rtol=2e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_erfc",
        lambda: call_complex_scalar(lib, "acb_hypgeom_erfc_ref", xc_e),
        lambda: hypgeom.acb_hypgeom_erfc_batch_jit(xc_e_j),
        rtol=3e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_erfi",
        lambda: call_real_scalar(lib, "arb_hypgeom_erfi_ref", xr_e),
        lambda: hypgeom.arb_hypgeom_erfi_batch_jit(xr_e_j),
        rtol=2e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_erfi",
        lambda: call_complex_scalar(lib, "acb_hypgeom_erfi_ref", xc_e),
        lambda: hypgeom.acb_hypgeom_erfi_batch_jit(xc_e_j),
        rtol=3e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_1f1",
        lambda: call_real_ternary(lib, "arb_hypgeom_1f1_ref", a_r, b_r, z_r),
        lambda: hypgeom.arb_hypgeom_1f1_batch_jit(a_r_j, b_r_j, z_r_j),
        rtol=2e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_1f1",
        lambda: call_complex_ternary(lib, "acb_hypgeom_1f1_ref", a_c, b_c, z_c),
        lambda: hypgeom.acb_hypgeom_1f1_batch_jit(a_c_j, b_c_j, z_c_j),
        rtol=4e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "arb_2f1",
        lambda: call_real_quaternary(lib, "arb_hypgeom_2f1_ref", a_r, b_r, c_r, z_r),
        lambda: hypgeom.arb_hypgeom_2f1_batch_jit(a_r_j, b_r_j, c_r_j, z_r_j),
        rtol=3e-11,
        atol=2e-14,
        proc=proc,
    ))
    rows.append(_run_case(
        "acb_2f1",
        lambda: call_complex_quaternary(lib, "acb_hypgeom_2f1_ref", a_c, b_c, c_c, z_c),
        lambda: hypgeom.acb_hypgeom_2f1_batch_jit(a_c_j, b_c_j, c_c_j, z_c_j),
        rtol=5e-11,
        atol=2e-14,
        proc=proc,
    ))

    print(
        "kernel               | ok | max_abs_diff | c_ms | jax_compile_ms | jax_exec_ms | speedup(c/jax) | c_rss_mb | jax_compile_rss_mb | jax_exec_rss_mb"
    )
    print("-" * 150)
    all_ok = True
    for row in rows:
        name, ok, max_diff, c_ms, j_compile_ms, j_exec_ms, speedup, c_rss_mb, j_compile_rss_mb, j_exec_rss_mb = row
        all_ok &= ok
        print(
            f"{name:20s} | {str(ok):2s} | {max_diff:11.3e} | {c_ms:7.2f} | {j_compile_ms:14.2f} | {j_exec_ms:11.2f} | {speedup:13.2f} |"
            f" {c_rss_mb:8.2f} | {j_compile_rss_mb:18.2f} | {j_exec_rss_mb:14.2f}"
        )

    print(f"\nresult: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
