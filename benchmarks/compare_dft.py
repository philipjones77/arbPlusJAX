from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from arbplusjax import dft


class CPLX(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


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


def default_dft_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["dft_ref.dll", "libdft_ref.dll", "libdft_ref.so", "libdft_ref.dylib"])


def load_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    cp = ctypes.POINTER(CPLX)
    ap = ctypes.POINTER(ACBBox)
    sp = ctypes.POINTER(ctypes.c_size_t)
    lib.cplx_dft_naive_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_dft_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_dft_rad2_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_dft_prod_ref.argtypes = [cp, cp, sp, ctypes.c_size_t]
    lib.cplx_convol_circular_naive_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_dft_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_rad2_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.acb_dft_naive_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_dft_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_dft_rad2_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_dft_prod_ref.argtypes = [ap, ap, sp, ctypes.c_size_t]
    lib.acb_convol_circular_naive_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_dft_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_rad2_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    return lib


def rand_complex(rng: np.random.Generator, n: int) -> np.ndarray:
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)


def to_c_array(x: np.ndarray):
    n = x.shape[0]
    arr_t = CPLX * n
    arr = arr_t()
    for i in range(n):
        arr[i] = CPLX(float(np.real(x[i])), float(np.imag(x[i])))
    return arr


def from_c_array(arr, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = arr[i].re + 1j * arr[i].im
    return out


def call_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xin = to_c_array(x)
    arr_t = CPLX * n
    out = arr_t()
    getattr(lib, fn_name)(xin, out, n)
    return from_c_array(out, n)


def call_prod(lib, x: np.ndarray, cyc: list[int]) -> np.ndarray:
    n = x.shape[0]
    xin = to_c_array(x)
    arr_t = CPLX * n
    out = arr_t()
    cyc_t = ctypes.c_size_t * len(cyc)
    ccyc = cyc_t(*cyc)
    lib.cplx_dft_prod_ref(xin, out, ccyc, len(cyc))
    return from_c_array(out, n)


def call_conv(lib, fn_name: str, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    ff = to_c_array(f)
    gg = to_c_array(g)
    arr_t = CPLX * n
    out = arr_t()
    getattr(lib, fn_name)(ff, gg, out, n)
    return from_c_array(out, n)


def to_boxes(x: np.ndarray) -> np.ndarray:
    re = np.real(x)
    im = np.imag(x)
    return np.stack([re, re, im, im], axis=-1).astype(np.float64)


def to_acb_array(x: np.ndarray):
    n = x.shape[0]
    arr_t = ACBBox * n
    arr = arr_t()
    for i in range(n):
        arr[i] = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
    return arr


def from_acb_array(arr, n: int) -> np.ndarray:
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        out[i, 0] = arr[i].real.a
        out[i, 1] = arr[i].real.b
        out[i, 2] = arr[i].imag.a
        out[i, 3] = arr[i].imag.b
    return out


def call_acb_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xin = to_acb_array(x)
    arr_t = ACBBox * n
    out = arr_t()
    getattr(lib, fn_name)(xin, out, n)
    return from_acb_array(out, n)


def call_acb_prod(lib, x: np.ndarray, cyc: list[int]) -> np.ndarray:
    n = x.shape[0]
    xin = to_acb_array(x)
    arr_t = ACBBox * n
    out = arr_t()
    cyc_t = ctypes.c_size_t * len(cyc)
    ccyc = cyc_t(*cyc)
    lib.acb_dft_prod_ref(xin, out, ccyc, len(cyc))
    return from_acb_array(out, n)


def call_acb_conv(lib, fn_name: str, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    ff = to_acb_array(f)
    gg = to_acb_array(g)
    arr_t = ACBBox * n
    out = arr_t()
    getattr(lib, fn_name)(ff, gg, out, n)
    return from_acb_array(out, n)


def report(name: str, c_out: np.ndarray, j_out: np.ndarray, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    ok = np.allclose(c_out, j_out, rtol=rtol, atol=atol)
    diff = np.abs(c_out - j_out)
    max_diff = float(np.max(diff)) if diff.size else 0.0
    print(f"{name:24s} | ok={ok} | max_abs_diff={max_diff:.3e}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DFT C and JAX kernels.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--n-prod", type=int, default=12)
    args = parser.parse_args()

    dft_env = os.getenv("DFT_REF_LIB", "")
    path = Path(dft_env) if dft_env else default_dft_path()
    if path is None or not path.exists():
        print("DFT reference library not found. Build C reference libraries in the Arb workspace first.")
        return 1

    lib = load_lib(path)
    rng = np.random.default_rng(314159)
    x = rand_complex(rng, args.n)
    xprod = rand_complex(rng, args.n_prod)
    f = rand_complex(rng, args.n)
    g = rand_complex(rng, args.n)
    xb = to_boxes(x)
    xprodb = to_boxes(xprod)
    fb = to_boxes(f)
    gb = to_boxes(g)

    ok = True
    ok &= report("dft_naive", call_unary(lib, "cplx_dft_naive_ref", xprod), np.asarray(dft.dft_naive_jit(jnp.asarray(xprod), inverse=False)))
    ok &= report("dft_main", call_unary(lib, "cplx_dft_ref", x), np.asarray(dft.dft_jit(jnp.asarray(x))))
    ok &= report("dft_rad2", call_unary(lib, "cplx_dft_rad2_ref", x), np.asarray(dft.dft_rad2_jit(jnp.asarray(x))))
    ok &= report("dft_prod", call_prod(lib, xprod, [3, 4]), np.asarray(dft.dft_prod_jit(jnp.asarray(xprod), cyc=(3, 4))))
    ok &= report("conv_naive", call_conv(lib, "cplx_convol_circular_naive_ref", f, g), np.asarray(dft.convol_circular_naive_jit(jnp.asarray(f), jnp.asarray(g))))
    ok &= report("conv_dft", call_conv(lib, "cplx_convol_circular_dft_ref", f, g), np.asarray(dft.convol_circular_dft_jit(jnp.asarray(f), jnp.asarray(g))))
    ok &= report("conv_rad2", call_conv(lib, "cplx_convol_circular_rad2_ref", f, g), np.asarray(dft.convol_circular_rad2_jit(jnp.asarray(f), jnp.asarray(g))))
    ok &= report("conv_main", call_conv(lib, "cplx_convol_circular_ref", f, g), np.asarray(dft.convol_circular_jit(jnp.asarray(f), jnp.asarray(g))))
    ok &= report("acb_dft_naive", call_acb_unary(lib, "acb_dft_naive_ref", xprodb), np.asarray(dft.acb_dft_naive_jit(jnp.asarray(xprodb), inverse=False)))
    ok &= report("acb_dft_main", call_acb_unary(lib, "acb_dft_ref", xb), np.asarray(dft.acb_dft_jit(jnp.asarray(xb))))
    ok &= report("acb_dft_rad2", call_acb_unary(lib, "acb_dft_rad2_ref", xb), np.asarray(dft.acb_dft_rad2_jit(jnp.asarray(xb))))
    ok &= report("acb_dft_prod", call_acb_prod(lib, xprodb, [3, 4]), np.asarray(dft.acb_dft_prod_jit(jnp.asarray(xprodb), cyc=(3, 4))))
    ok &= report("acb_conv_naive", call_acb_conv(lib, "acb_convol_circular_naive_ref", fb, gb), np.asarray(dft.acb_convol_circular_naive_jit(jnp.asarray(fb), jnp.asarray(gb))))
    ok &= report("acb_conv_dft", call_acb_conv(lib, "acb_convol_circular_dft_ref", fb, gb), np.asarray(dft.acb_convol_circular_dft_jit(jnp.asarray(fb), jnp.asarray(gb))))
    ok &= report("acb_conv_rad2", call_acb_conv(lib, "acb_convol_circular_rad2_ref", fb, gb), np.asarray(dft.acb_convol_circular_rad2_jit(jnp.asarray(fb), jnp.asarray(gb))))
    ok &= report("acb_conv_main", call_acb_conv(lib, "acb_convol_circular_ref", fb, gb), np.asarray(dft.acb_convol_circular_jit(jnp.asarray(fb), jnp.asarray(gb))))

    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
