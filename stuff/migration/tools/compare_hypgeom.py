from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

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
    hyp.arb_hypgeom_0f1_ref.argtypes = [DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_0f1_ref.restype = DI
    hyp.acb_hypgeom_0f1_ref.argtypes = [ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_0f1_ref.restype = ACBBox
    hyp.arb_hypgeom_m_ref.argtypes = [DI, DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_m_ref.restype = DI
    hyp.acb_hypgeom_m_ref.argtypes = [ACBBox, ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_m_ref.restype = ACBBox
    hyp.arb_hypgeom_1f1_ref.argtypes = [DI, DI, DI]
    hyp.arb_hypgeom_1f1_ref.restype = DI
    hyp.arb_hypgeom_1f1_integration_ref.argtypes = [DI, DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_1f1_integration_ref.restype = DI
    hyp.acb_hypgeom_1f1_ref.argtypes = [ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_1f1_ref.restype = ACBBox
    hyp.acb_hypgeom_1f1_integration_ref.argtypes = [ACBBox, ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_1f1_integration_ref.restype = ACBBox
    hyp.arb_hypgeom_1f1_full_ref.argtypes = [DI, DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_1f1_full_ref.restype = DI
    hyp.acb_hypgeom_1f1_full_ref.argtypes = [ACBBox, ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_1f1_full_ref.restype = ACBBox
    hyp.arb_hypgeom_2f1_ref.argtypes = [DI, DI, DI, DI]
    hyp.arb_hypgeom_2f1_ref.restype = DI
    hyp.arb_hypgeom_2f1_integration_ref.argtypes = [DI, DI, DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_2f1_integration_ref.restype = DI
    hyp.acb_hypgeom_2f1_ref.argtypes = [ACBBox, ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_2f1_ref.restype = ACBBox
    hyp.acb_hypgeom_2f1_integration_ref.argtypes = [ACBBox, ACBBox, ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_2f1_integration_ref.restype = ACBBox
    hyp.arb_hypgeom_2f1_full_ref.argtypes = [DI, DI, DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_2f1_full_ref.restype = DI
    hyp.acb_hypgeom_2f1_full_ref.argtypes = [ACBBox, ACBBox, ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_2f1_full_ref.restype = ACBBox
    hyp.arb_hypgeom_bessel_j_ref.argtypes = [DI, DI]
    hyp.arb_hypgeom_bessel_j_ref.restype = DI
    hyp.arb_hypgeom_bessel_y_ref.argtypes = [DI, DI]
    hyp.arb_hypgeom_bessel_y_ref.restype = DI
    hyp.arb_hypgeom_bessel_i_ref.argtypes = [DI, DI]
    hyp.arb_hypgeom_bessel_i_ref.restype = DI
    hyp.arb_hypgeom_bessel_k_ref.argtypes = [DI, DI]
    hyp.arb_hypgeom_bessel_k_ref.restype = DI
    hyp.arb_hypgeom_u_ref.argtypes = [DI, DI, DI]
    hyp.arb_hypgeom_u_ref.restype = DI
    hyp.arb_hypgeom_u_integration_ref.argtypes = [DI, DI, DI]
    hyp.arb_hypgeom_u_integration_ref.restype = DI
    hyp.arb_hypgeom_erfinv_ref.argtypes = [DI]
    hyp.arb_hypgeom_erfinv_ref.restype = DI
    hyp.arb_hypgeom_erfcinv_ref.argtypes = [DI]
    hyp.arb_hypgeom_erfcinv_ref.restype = DI
    hyp.acb_hypgeom_bessel_j_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_j_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_y_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_y_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_i_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_i_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_k_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_k_ref.restype = ACBBox
    hyp.acb_hypgeom_u_ref.argtypes = [ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_u_ref.restype = ACBBox
    hyp.acb_hypgeom_u_integration_ref.argtypes = [ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_u_integration_ref.restype = ACBBox
    if hasattr(hyp, "hypgeom_ref_set_bessel_real_mode"):
        hyp.hypgeom_ref_set_bessel_real_mode.argtypes = [ctypes.c_int]
        hyp.hypgeom_ref_set_bessel_real_mode.restype = None
    return hyp


def random_intervals(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_acb_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    re = random_intervals(rng, n, scale=3.0)
    im = random_intervals(rng, n, scale=3.0)
    return np.concatenate([re, im], axis=-1)


def random_positive_intervals(rng: np.random.Generator, n: int, lo: float = 0.05, hi: float = 10.0) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def random_acb_boxes_away_from_poles(rng: np.random.Generator, n: int) -> np.ndarray:
    re = random_intervals(rng, n, scale=4.0)
    im_lo = rng.uniform(0.15, 3.0, size=n)
    im_hi = im_lo + rng.uniform(0.0, 0.5, size=n)
    im = np.stack([im_lo, im_hi], axis=-1)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def random_small_intervals(rng: np.random.Generator, n: int, scale: float = 1.5) -> np.ndarray:
    return random_intervals(rng, n, scale=scale)


def random_small_acb_boxes(rng: np.random.Generator, n: int, scale: float = 1.5) -> np.ndarray:
    re = random_intervals(rng, n, scale=scale)
    im = random_intervals(rng, n, scale=scale)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def random_hyp_params_real(rng: np.random.Generator, n: int):
    a = random_positive_intervals(rng, n)
    b = random_positive_intervals(rng, n)
    c = random_positive_intervals(rng, n)
    z = random_small_intervals(rng, n, scale=0.8)
    return a, b, c, z


def random_hyp_params_complex(rng: np.random.Generator, n: int):
    a = random_small_acb_boxes(rng, n, scale=1.0)
    b = random_small_acb_boxes(rng, n, scale=1.0)
    c = random_small_acb_boxes(rng, n, scale=1.0)
    b[:, 0:2] += 2.2
    c[:, 0:2] += 2.5
    z = random_small_acb_boxes(rng, n, scale=0.5)
    return a, b, c, z


def random_bessel_params_real(rng: np.random.Generator, n: int):
    nu = random_positive_intervals(rng, n, lo=0.2, hi=2.5)
    z = random_small_intervals(rng, n, scale=2.0)
    return nu, z


def random_bessel_params_complex(rng: np.random.Generator, n: int):
    nu = random_small_acb_boxes(rng, n, scale=1.0)
    z = random_small_acb_boxes(rng, n, scale=2.0)
    return nu, z


def call_real(lib, x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = lib.arb_hypgeom_rising_ui_ref(DI(float(x[i, 0]), float(x[i, 1])), n)
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex(lib, x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xb = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
        r = lib.acb_hypgeom_rising_ui_ref(xb, n)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_lgamma(lib, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = lib.arb_hypgeom_lgamma_ref(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_lgamma(lib, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xb = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
        r = lib.acb_hypgeom_lgamma_ref(xb)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
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


def call_real_binary(lib, fn_name: str, a: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        if flag is None:
            r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(z[i, 0]), float(z[i, 1])))
        else:
            r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(z[i, 0]), float(z[i, 1])), int(flag))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_binary(lib, fn_name: str, a: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        aa = ACBBox(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3])))
        zz = ACBBox(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3])))
        if flag is None:
            r = fn(aa, zz)
        else:
            r = fn(aa, zz, int(flag))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_ternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        if flag is None:
            r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(b[i, 0]), float(b[i, 1])), DI(float(z[i, 0]), float(z[i, 1])))
        else:
            r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(b[i, 0]), float(b[i, 1])), DI(float(z[i, 0]), float(z[i, 1])), int(flag))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_ternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        aa = ACBBox(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3])))
        bb = ACBBox(DI(float(b[i, 0]), float(b[i, 1])), DI(float(b[i, 2]), float(b[i, 3])))
        zz = ACBBox(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3])))
        if flag is None:
            r = fn(aa, bb, zz)
        else:
            r = fn(aa, bb, zz, int(flag))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_quaternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        if flag is None:
            r = fn(
                DI(float(a[i, 0]), float(a[i, 1])),
                DI(float(b[i, 0]), float(b[i, 1])),
                DI(float(c[i, 0]), float(c[i, 1])),
                DI(float(z[i, 0]), float(z[i, 1])),
            )
        else:
            r = fn(
                DI(float(a[i, 0]), float(a[i, 1])),
                DI(float(b[i, 0]), float(b[i, 1])),
                DI(float(c[i, 0]), float(c[i, 1])),
                DI(float(z[i, 0]), float(z[i, 1])),
                int(flag),
            )
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_quaternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        aa = ACBBox(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3])))
        bb = ACBBox(DI(float(b[i, 0]), float(b[i, 1])), DI(float(b[i, 2]), float(b[i, 3])))
        cc = ACBBox(DI(float(c[i, 0]), float(c[i, 1])), DI(float(c[i, 2]), float(c[i, 3])))
        zz = ACBBox(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3])))
        if flag is None:
            r = fn(aa, bb, cc, zz)
        else:
            r = fn(aa, bb, cc, zz, int(flag))
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def call_real_binary(lib, fn_name: str, a: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        if flag is None:
            r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(z[i, 0]), float(z[i, 1])))
        else:
            r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(z[i, 0]), float(z[i, 1])), int(flag))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def call_complex_binary(lib, fn_name: str, a: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        aa = ACBBox(DI(float(a[i, 0]), float(a[i, 1])), DI(float(a[i, 2]), float(a[i, 3])))
        zz = ACBBox(DI(float(z[i, 0]), float(z[i, 1])), DI(float(z[i, 2]), float(z[i, 3])))
        if flag is None:
            r = fn(aa, zz)
        else:
            r = fn(aa, zz, int(flag))
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
    print(f"{name:22s} | ok={ok} | max_abs_diff={max_diff:.3e}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare hypgeom rising_ui C and JAX kernels.")
    parser.add_argument("--samples-real", type=int, default=10000)
    parser.add_argument("--samples-complex", type=int, default=5000)
    parser.add_argument("--bessel-real-mode", choices=("sample", "midpoint"), default="sample")
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
    if hasattr(lib, "hypgeom_ref_set_bessel_real_mode"):
        lib.hypgeom_ref_set_bessel_real_mode(0 if args.bessel_real_mode == "midpoint" else 1)
    rng = np.random.default_rng(2026)
    xr = random_intervals(rng, args.samples_real, 10.0)
    xc = random_acb_boxes(rng, args.samples_complex)

    ok = True
    for n in (0, 1, 2, 5, 9, 15):
        c_real = call_real(lib, xr, n)
        j_real = np.asarray(hypgeom.arb_hypgeom_rising_ui_batch_jit(jnp.asarray(xr), n=n))
        c_cplx = call_complex(lib, xc, n)
        j_cplx = np.asarray(hypgeom.acb_hypgeom_rising_ui_batch_jit(jnp.asarray(xc), n=n))

        ok &= report(f"arb_rising_ui n={n}", c_real, j_real, rtol=2e-13)
        ok &= report(f"acb_rising_ui n={n}", c_cplx, j_cplx, rtol=3e-13)

    xr_lg = random_positive_intervals(rng, args.samples_real)
    xc_lg = random_acb_boxes_away_from_poles(rng, args.samples_complex)
    c_real_lg = call_real_lgamma(lib, xr_lg)
    j_real_lg = np.asarray(hypgeom.arb_hypgeom_lgamma_batch_jit(jnp.asarray(xr_lg)))
    c_cplx_lg = call_complex_lgamma(lib, xc_lg)
    j_cplx_lg = np.asarray(hypgeom.acb_hypgeom_lgamma_batch_jit(jnp.asarray(xc_lg)))

    ok &= report("arb_lgamma", c_real_lg, j_real_lg, rtol=2e-12, atol=2e-14)
    ok &= report("acb_lgamma", c_cplx_lg, j_cplx_lg, rtol=1e-11, atol=2e-14)

    c_real_g = call_real_unary(lib, "arb_hypgeom_gamma_ref", xr_lg)
    j_real_g = np.asarray(hypgeom.arb_hypgeom_gamma_batch_jit(jnp.asarray(xr_lg)))
    c_cplx_g = call_complex_unary(lib, "acb_hypgeom_gamma_ref", xc_lg)
    j_cplx_g = np.asarray(hypgeom.acb_hypgeom_gamma_batch_jit(jnp.asarray(xc_lg)))
    ok &= report("arb_gamma", c_real_g, j_real_g, rtol=4e-12, atol=2e-14)
    ok &= report("acb_gamma", c_cplx_g, j_cplx_g, rtol=2e-11, atol=2e-14)

    c_real_rg = call_real_unary(lib, "arb_hypgeom_rgamma_ref", xr_lg)
    j_real_rg = np.asarray(hypgeom.arb_hypgeom_rgamma_batch_jit(jnp.asarray(xr_lg)))
    c_cplx_rg = call_complex_unary(lib, "acb_hypgeom_rgamma_ref", xc_lg)
    j_cplx_rg = np.asarray(hypgeom.acb_hypgeom_rgamma_batch_jit(jnp.asarray(xc_lg)))
    ok &= report("arb_rgamma", c_real_rg, j_real_rg, rtol=4e-12, atol=2e-14)
    ok &= report("acb_rgamma", c_cplx_rg, j_cplx_rg, rtol=2e-11, atol=2e-14)

    xr_e = random_small_intervals(rng, args.samples_real, 1.5)
    xc_e = random_small_acb_boxes(rng, args.samples_complex, 1.5)

    c_erf_r = call_real_unary(lib, "arb_hypgeom_erf_ref", xr_e)
    c_erf_c = call_complex_unary(lib, "acb_hypgeom_erf_ref", xc_e)
    j_erf_r = np.asarray(hypgeom.arb_hypgeom_erf_batch_jit(jnp.asarray(xr_e)))
    j_erf_c = np.asarray(hypgeom.acb_hypgeom_erf_batch_jit(jnp.asarray(xc_e)))
    ok &= report("arb_erf", c_erf_r, j_erf_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_erf", c_erf_c, j_erf_c, rtol=3e-11, atol=2e-14)

    c_erfc_r = call_real_unary(lib, "arb_hypgeom_erfc_ref", xr_e)
    c_erfc_c = call_complex_unary(lib, "acb_hypgeom_erfc_ref", xc_e)
    j_erfc_r = np.asarray(hypgeom.arb_hypgeom_erfc_batch_jit(jnp.asarray(xr_e)))
    j_erfc_c = np.asarray(hypgeom.acb_hypgeom_erfc_batch_jit(jnp.asarray(xc_e)))
    ok &= report("arb_erfc", c_erfc_r, j_erfc_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_erfc", c_erfc_c, j_erfc_c, rtol=3e-11, atol=2e-14)

    c_erfi_r = call_real_unary(lib, "arb_hypgeom_erfi_ref", xr_e)
    c_erfi_c = call_complex_unary(lib, "acb_hypgeom_erfi_ref", xc_e)
    j_erfi_r = np.asarray(hypgeom.arb_hypgeom_erfi_batch_jit(jnp.asarray(xr_e)))
    j_erfi_c = np.asarray(hypgeom.acb_hypgeom_erfi_batch_jit(jnp.asarray(xc_e)))
    ok &= report("arb_erfi", c_erfi_r, j_erfi_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_erfi", c_erfi_c, j_erfi_c, rtol=3e-11, atol=2e-14)

    xr_inv = random_intervals(rng, args.samples_real, 0.9)
    xr_erfc = random_positive_intervals(rng, args.samples_real, lo=0.1, hi=1.9)
    c_erfinv_r = call_real_unary(lib, "arb_hypgeom_erfinv_ref", xr_inv)
    c_erfcinv_r = call_real_unary(lib, "arb_hypgeom_erfcinv_ref", xr_erfc)
    j_erfinv_r = np.asarray(hypgeom.arb_hypgeom_erfinv_batch_jit(jnp.asarray(xr_inv)))
    j_erfcinv_r = np.asarray(hypgeom.arb_hypgeom_erfcinv_batch_jit(jnp.asarray(xr_erfc)))
    ok &= report("arb_erfinv", c_erfinv_r, j_erfinv_r, rtol=5e-10, atol=2e-14)
    ok &= report("arb_erfcinv", c_erfcinv_r, j_erfcinv_r, rtol=5e-10, atol=2e-14)

    a_r, b_r, c_r, z_r = random_hyp_params_real(rng, args.samples_real)
    a_c, b_c, c_c, z_c = random_hyp_params_complex(rng, args.samples_complex)

    c_0f1_r = call_real_binary(lib, "arb_hypgeom_0f1_ref", b_r, z_r, flag=1)
    c_0f1_c = call_complex_binary(lib, "acb_hypgeom_0f1_ref", b_c, z_c, flag=1)
    j_0f1_r = np.asarray(hypgeom.arb_hypgeom_0f1_batch_jit(jnp.asarray(b_r), jnp.asarray(z_r), regularized=True))
    j_0f1_c = np.asarray(hypgeom.acb_hypgeom_0f1_batch_jit(jnp.asarray(b_c), jnp.asarray(z_c), regularized=True))
    ok &= report("arb_0f1_reg", c_0f1_r, j_0f1_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_0f1_reg", c_0f1_c, j_0f1_c, rtol=4e-11, atol=2e-14)

    c_m_r = call_real_ternary(lib, "arb_hypgeom_m_ref", a_r, b_r, z_r, flag=1)
    c_m_c = call_complex_ternary(lib, "acb_hypgeom_m_ref", a_c, b_c, z_c, flag=1)
    j_m_r = np.asarray(hypgeom.arb_hypgeom_m_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(z_r), regularized=True))
    j_m_c = np.asarray(hypgeom.acb_hypgeom_m_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(z_c), regularized=True))
    ok &= report("arb_m_reg", c_m_r, j_m_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_m_reg", c_m_c, j_m_c, rtol=4e-11, atol=2e-14)

    c_1f1_r = call_real_ternary(lib, "arb_hypgeom_1f1_ref", a_r, b_r, z_r)
    c_1f1_c = call_complex_ternary(lib, "acb_hypgeom_1f1_ref", a_c, b_c, z_c)
    j_1f1_r = np.asarray(hypgeom.arb_hypgeom_1f1_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(z_r)))
    j_1f1_c = np.asarray(hypgeom.acb_hypgeom_1f1_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(z_c)))
    ok &= report("arb_1f1", c_1f1_r, j_1f1_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_1f1", c_1f1_c, j_1f1_c, rtol=4e-11, atol=2e-14)

    c_1f1r_r = call_real_ternary(lib, "arb_hypgeom_1f1_full_ref", a_r, b_r, z_r, flag=1)
    c_1f1r_c = call_complex_ternary(lib, "acb_hypgeom_1f1_full_ref", a_c, b_c, z_c, flag=1)
    j_1f1r_r = np.asarray(hypgeom.arb_hypgeom_1f1_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(z_r), regularized=True))
    j_1f1r_c = np.asarray(hypgeom.acb_hypgeom_1f1_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(z_c), regularized=True))
    ok &= report("arb_1f1_reg", c_1f1r_r, j_1f1r_r, rtol=2e-11, atol=2e-14)
    ok &= report("acb_1f1_reg", c_1f1r_c, j_1f1r_c, rtol=4e-11, atol=2e-14)

    c_2f1_r = call_real_quaternary(lib, "arb_hypgeom_2f1_ref", a_r, b_r, c_r, z_r)
    c_2f1_c = call_complex_quaternary(lib, "acb_hypgeom_2f1_ref", a_c, b_c, c_c, z_c)
    j_2f1_r = np.asarray(hypgeom.arb_hypgeom_2f1_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(c_r), jnp.asarray(z_r)))
    j_2f1_c = np.asarray(hypgeom.acb_hypgeom_2f1_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(c_c), jnp.asarray(z_c)))
    ok &= report("arb_2f1", c_2f1_r, j_2f1_r, rtol=3e-11, atol=2e-14)
    ok &= report("acb_2f1", c_2f1_c, j_2f1_c, rtol=5e-11, atol=2e-14)

    c_2f1r_r = call_real_quaternary(lib, "arb_hypgeom_2f1_full_ref", a_r, b_r, c_r, z_r, flag=1)
    c_2f1r_c = call_complex_quaternary(lib, "acb_hypgeom_2f1_full_ref", a_c, b_c, c_c, z_c, flag=1)
    j_2f1r_r = np.asarray(hypgeom.arb_hypgeom_2f1_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(c_r), jnp.asarray(z_r), regularized=True))
    j_2f1r_c = np.asarray(hypgeom.acb_hypgeom_2f1_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(c_c), jnp.asarray(z_c), regularized=True))
    ok &= report("arb_2f1_reg", c_2f1r_r, j_2f1r_r, rtol=3e-11, atol=2e-14)
    ok &= report("acb_2f1_reg", c_2f1r_c, j_2f1r_c, rtol=5e-11, atol=2e-14)

    c_1f1i_r = call_real_ternary(lib, "arb_hypgeom_1f1_integration_ref", a_r, b_r, z_r, flag=1)
    c_1f1i_c = call_complex_ternary(lib, "acb_hypgeom_1f1_integration_ref", a_c, b_c, z_c, flag=1)
    j_1f1i_r = np.asarray(
        hypgeom.arb_hypgeom_1f1_integration_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(z_r), regularized=True)
    )
    j_1f1i_c = np.asarray(
        hypgeom.acb_hypgeom_1f1_integration_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(z_c), regularized=True)
    )
    ok &= report("arb_1f1_int", c_1f1i_r, j_1f1i_r, rtol=3e-11, atol=2e-14)
    ok &= report("acb_1f1_int", c_1f1i_c, j_1f1i_c, rtol=5e-11, atol=2e-14)

    c_2f1i_r = call_real_quaternary(lib, "arb_hypgeom_2f1_integration_ref", a_r, b_r, c_r, z_r, flag=1)
    c_2f1i_c = call_complex_quaternary(lib, "acb_hypgeom_2f1_integration_ref", a_c, b_c, c_c, z_c, flag=1)
    j_2f1i_r = np.asarray(
        hypgeom.arb_hypgeom_2f1_integration_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(c_r), jnp.asarray(z_r), regularized=True)
    )
    j_2f1i_c = np.asarray(
        hypgeom.acb_hypgeom_2f1_integration_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(c_c), jnp.asarray(z_c), regularized=True)
    )
    ok &= report("arb_2f1_int", c_2f1i_r, j_2f1i_r, rtol=3e-11, atol=2e-14)
    ok &= report("acb_2f1_int", c_2f1i_c, j_2f1i_c, rtol=5e-11, atol=2e-14)

    c_u_r = call_real_ternary(lib, "arb_hypgeom_u_ref", a_r, b_r, z_r)
    c_u_c = call_complex_ternary(lib, "acb_hypgeom_u_ref", a_c, b_c, z_c)
    j_u_r = np.asarray(hypgeom.arb_hypgeom_u_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(z_r)))
    j_u_c = np.asarray(hypgeom.acb_hypgeom_u_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(z_c)))
    ok &= report("arb_u", c_u_r, j_u_r, rtol=5e-10, atol=2e-14)
    ok &= report("acb_u", c_u_c, j_u_c, rtol=5e-10, atol=2e-14)

    c_ui_r = call_real_ternary(lib, "arb_hypgeom_u_integration_ref", a_r, b_r, z_r)
    c_ui_c = call_complex_ternary(lib, "acb_hypgeom_u_integration_ref", a_c, b_c, z_c)
    j_ui_r = np.asarray(hypgeom.arb_hypgeom_u_integration_batch_jit(jnp.asarray(a_r), jnp.asarray(b_r), jnp.asarray(z_r)))
    j_ui_c = np.asarray(hypgeom.acb_hypgeom_u_integration_batch_jit(jnp.asarray(a_c), jnp.asarray(b_c), jnp.asarray(z_c)))
    ok &= report("arb_u_int", c_ui_r, j_ui_r, rtol=5e-10, atol=2e-14)
    ok &= report("acb_u_int", c_ui_c, j_ui_c, rtol=5e-10, atol=2e-14)

    nu_r, z_r = random_bessel_params_real(rng, args.samples_real)
    nu_c, z_c = random_bessel_params_complex(rng, args.samples_complex)

    c_bj_r = call_real_binary(lib, "arb_hypgeom_bessel_j_ref", nu_r, z_r)
    c_by_r = call_real_binary(lib, "arb_hypgeom_bessel_y_ref", nu_r, z_r)
    c_bi_r = call_real_binary(lib, "arb_hypgeom_bessel_i_ref", nu_r, z_r)
    c_bk_r = call_real_binary(lib, "arb_hypgeom_bessel_k_ref", nu_r, z_r)
    j_bj_r = np.asarray(
        hypgeom.arb_hypgeom_bessel_j_batch_jit(jnp.asarray(nu_r), jnp.asarray(z_r), mode=args.bessel_real_mode)
    )
    j_by_r = np.asarray(
        hypgeom.arb_hypgeom_bessel_y_batch_jit(jnp.asarray(nu_r), jnp.asarray(z_r), mode=args.bessel_real_mode)
    )
    j_bi_r = np.asarray(
        hypgeom.arb_hypgeom_bessel_i_batch_jit(jnp.asarray(nu_r), jnp.asarray(z_r), mode=args.bessel_real_mode)
    )
    j_bk_r = np.asarray(
        hypgeom.arb_hypgeom_bessel_k_batch_jit(jnp.asarray(nu_r), jnp.asarray(z_r), mode=args.bessel_real_mode)
    )
    ok &= report("arb_bessel_j", c_bj_r, j_bj_r, rtol=2e-11, atol=2e-14)
    ok &= report("arb_bessel_y", c_by_r, j_by_r, rtol=2e-11, atol=2e-14)
    ok &= report("arb_bessel_i", c_bi_r, j_bi_r, rtol=2e-11, atol=2e-14)
    ok &= report("arb_bessel_k", c_bk_r, j_bk_r, rtol=2e-11, atol=2e-14)

    c_bj_c = call_complex_binary(lib, "acb_hypgeom_bessel_j_ref", nu_c, z_c)
    c_by_c = call_complex_binary(lib, "acb_hypgeom_bessel_y_ref", nu_c, z_c)
    c_bi_c = call_complex_binary(lib, "acb_hypgeom_bessel_i_ref", nu_c, z_c)
    c_bk_c = call_complex_binary(lib, "acb_hypgeom_bessel_k_ref", nu_c, z_c)
    j_bj_c = np.asarray(hypgeom.acb_hypgeom_bessel_j_batch_jit(jnp.asarray(nu_c), jnp.asarray(z_c)))
    j_by_c = np.asarray(hypgeom.acb_hypgeom_bessel_y_batch_jit(jnp.asarray(nu_c), jnp.asarray(z_c)))
    j_bi_c = np.asarray(hypgeom.acb_hypgeom_bessel_i_batch_jit(jnp.asarray(nu_c), jnp.asarray(z_c)))
    j_bk_c = np.asarray(hypgeom.acb_hypgeom_bessel_k_batch_jit(jnp.asarray(nu_c), jnp.asarray(z_c)))
    ok &= report("acb_bessel_j", c_bj_c, j_bj_c, rtol=4e-11, atol=2e-14)
    ok &= report("acb_bessel_y", c_by_c, j_by_c, rtol=4e-11, atol=2e-14)
    ok &= report("acb_bessel_i", c_bi_c, j_bi_c, rtol=4e-11, atol=2e-14)
    ok &= report("acb_bessel_k", c_bk_c, j_bk_c, rtol=4e-11, atol=2e-14)

    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
