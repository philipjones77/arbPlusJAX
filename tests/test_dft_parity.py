import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import dft

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


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


def _default_dft_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["dft_ref.dll", "libdft_ref.dll", "libdft_ref.so", "libdft_ref.dylib"])


def _load_lib():
    dft_env = os.getenv("DFT_REF_LIB")
    path = Path(dft_env) if dft_env else _default_dft_lib_path()
    if path is None or not path.exists():
        pytest.skip("DFT reference library not found. Build C reference libraries in the Arb workspace first.")

    lib = ctypes.CDLL(str(path))
    cp = ctypes.POINTER(CPLX)
    ap = ctypes.POINTER(ACBBox)
    sp = ctypes.POINTER(ctypes.c_size_t)

    lib.cplx_dft_naive_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_dft_naive_ref.restype = None
    lib.cplx_idft_naive_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_idft_naive_ref.restype = None
    lib.cplx_dft_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_dft_ref.restype = None
    lib.cplx_idft_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_idft_ref.restype = None
    lib.cplx_dft_rad2_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_dft_rad2_ref.restype = None
    lib.cplx_idft_rad2_ref.argtypes = [cp, cp, ctypes.c_size_t]
    lib.cplx_idft_rad2_ref.restype = None
    lib.cplx_dft_prod_ref.argtypes = [cp, cp, sp, ctypes.c_size_t]
    lib.cplx_dft_prod_ref.restype = None
    lib.cplx_convol_circular_naive_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_naive_ref.restype = None
    lib.cplx_convol_circular_dft_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_dft_ref.restype = None
    lib.cplx_convol_circular_rad2_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_rad2_ref.restype = None
    lib.cplx_convol_circular_ref.argtypes = [cp, cp, cp, ctypes.c_size_t]
    lib.cplx_convol_circular_ref.restype = None

    lib.acb_dft_naive_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_dft_naive_ref.restype = None
    lib.acb_idft_naive_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_idft_naive_ref.restype = None
    lib.acb_dft_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_dft_ref.restype = None
    lib.acb_idft_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_idft_ref.restype = None
    lib.acb_dft_rad2_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_dft_rad2_ref.restype = None
    lib.acb_idft_rad2_ref.argtypes = [ap, ap, ctypes.c_size_t]
    lib.acb_idft_rad2_ref.restype = None
    lib.acb_dft_prod_ref.argtypes = [ap, ap, sp, ctypes.c_size_t]
    lib.acb_dft_prod_ref.restype = None
    lib.acb_convol_circular_naive_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_naive_ref.restype = None
    lib.acb_convol_circular_dft_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_dft_ref.restype = None
    lib.acb_convol_circular_rad2_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_rad2_ref.restype = None
    lib.acb_convol_circular_ref.argtypes = [ap, ap, ap, ctypes.c_size_t]
    lib.acb_convol_circular_ref.restype = None
    return lib


def _rand_complex(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)


def _to_c_array(x: np.ndarray):
    n = x.shape[0]
    arr_t = CPLX * n
    arr = arr_t()
    for i in range(n):
        arr[i] = CPLX(float(np.real(x[i])), float(np.imag(x[i])))
    return arr


def _from_c_array(arr, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = arr[i].re + 1j * arr[i].im
    return out


def _call_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xin = _to_c_array(x)
    arr_t = CPLX * n
    out = arr_t()
    getattr(lib, fn_name)(xin, out, n)
    return _from_c_array(out, n)


def _call_prod(lib, x: np.ndarray, cyc: list[int]) -> np.ndarray:
    n = x.shape[0]
    xin = _to_c_array(x)
    arr_t = CPLX * n
    out = arr_t()
    cyc_t = ctypes.c_size_t * len(cyc)
    ccyc = cyc_t(*cyc)
    lib.cplx_dft_prod_ref(xin, out, ccyc, len(cyc))
    return _from_c_array(out, n)


def _call_conv(lib, fn_name: str, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    ff = _to_c_array(f)
    gg = _to_c_array(g)
    arr_t = CPLX * n
    out = arr_t()
    getattr(lib, fn_name)(ff, gg, out, n)
    return _from_c_array(out, n)


def _to_boxes(x: np.ndarray) -> np.ndarray:
    re = np.real(x)
    im = np.imag(x)
    return np.stack([re, re, im, im], axis=-1).astype(np.float64)


def _to_acb_array(x: np.ndarray):
    n = x.shape[0]
    arr_t = ACBBox * n
    arr = arr_t()
    for i in range(n):
        arr[i] = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
    return arr


def _from_acb_array(arr, n: int) -> np.ndarray:
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        out[i, 0] = arr[i].real.a
        out[i, 1] = arr[i].real.b
        out[i, 2] = arr[i].imag.a
        out[i, 3] = arr[i].imag.b
    return out


def _call_acb_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xin = _to_acb_array(x)
    arr_t = ACBBox * n
    out = arr_t()
    getattr(lib, fn_name)(xin, out, n)
    return _from_acb_array(out, n)


def _call_acb_prod(lib, x: np.ndarray, cyc: list[int]) -> np.ndarray:
    n = x.shape[0]
    xin = _to_acb_array(x)
    arr_t = ACBBox * n
    out = arr_t()
    cyc_t = ctypes.c_size_t * len(cyc)
    ccyc = cyc_t(*cyc)
    lib.acb_dft_prod_ref(xin, out, ccyc, len(cyc))
    return _from_acb_array(out, n)


def _call_acb_conv(lib, fn_name: str, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    ff = _to_acb_array(f)
    gg = _to_acb_array(g)
    arr_t = ACBBox * n
    out = arr_t()
    getattr(lib, fn_name)(ff, gg, out, n)
    return _from_acb_array(out, n)


def test_dft_main_and_fft_parity():
    lib = _load_lib()
    x8 = _rand_complex(8, seed=41)
    x12 = _rand_complex(12, seed=42)

    c_main8 = _call_unary(lib, "cplx_dft_ref", x8)
    c_main12 = _call_unary(lib, "cplx_dft_ref", x12)
    c_rad2 = _call_unary(lib, "cplx_dft_rad2_ref", x8)
    c_naive = _call_unary(lib, "cplx_dft_naive_ref", x12)

    j_main8 = np.asarray(dft.dft_jit(jnp.asarray(x8)))
    j_main12 = np.asarray(dft.dft_jit(jnp.asarray(x12)))
    j_rad2 = np.asarray(dft.dft_rad2_jit(jnp.asarray(x8)))
    j_naive = np.asarray(dft.dft_naive_jit(jnp.asarray(x12), inverse=False))

    np.testing.assert_allclose(c_main8, j_main8, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_main12, j_main12, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_rad2, j_rad2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_naive, j_naive, rtol=1e-12, atol=1e-12)


def test_dft_product_and_convolution_parity():
    lib = _load_lib()
    x = _rand_complex(12, seed=51)
    f = _rand_complex(8, seed=52)
    g = _rand_complex(8, seed=53)

    c_prod = _call_prod(lib, x, [3, 4])
    j_prod = np.asarray(dft.dft_prod_jit(jnp.asarray(x), cyc=(3, 4)))
    np.testing.assert_allclose(c_prod, j_prod, rtol=1e-12, atol=1e-12)

    c_cn = _call_conv(lib, "cplx_convol_circular_naive_ref", f, g)
    c_cd = _call_conv(lib, "cplx_convol_circular_dft_ref", f, g)
    c_cr = _call_conv(lib, "cplx_convol_circular_rad2_ref", f, g)
    c_c = _call_conv(lib, "cplx_convol_circular_ref", f, g)

    j_cn = np.asarray(dft.convol_circular_naive_jit(jnp.asarray(f), jnp.asarray(g)))
    j_cd = np.asarray(dft.convol_circular_dft_jit(jnp.asarray(f), jnp.asarray(g)))
    j_cr = np.asarray(dft.convol_circular_rad2_jit(jnp.asarray(f), jnp.asarray(g)))
    j_c = np.asarray(dft.convol_circular_jit(jnp.asarray(f), jnp.asarray(g)))

    np.testing.assert_allclose(c_cn, j_cn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_cd, j_cd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_cr, j_cr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_c, j_c, rtol=1e-12, atol=1e-12)


def test_acb_dft_and_convolution_parity():
    lib = _load_lib()
    x8 = _to_boxes(_rand_complex(8, seed=61))
    x12 = _to_boxes(_rand_complex(12, seed=62))
    f = _to_boxes(_rand_complex(8, seed=63))
    g = _to_boxes(_rand_complex(8, seed=64))

    c_main8 = _call_acb_unary(lib, "acb_dft_ref", x8)
    c_main12 = _call_acb_unary(lib, "acb_dft_ref", x12)
    c_rad2 = _call_acb_unary(lib, "acb_dft_rad2_ref", x8)
    c_naive = _call_acb_unary(lib, "acb_dft_naive_ref", x12)
    c_prod = _call_acb_prod(lib, x12, [3, 4])

    j_main8 = np.asarray(dft.acb_dft_jit(jnp.asarray(x8)))
    j_main12 = np.asarray(dft.acb_dft_jit(jnp.asarray(x12)))
    j_rad2 = np.asarray(dft.acb_dft_rad2_jit(jnp.asarray(x8)))
    j_naive = np.asarray(dft.acb_dft_naive_jit(jnp.asarray(x12), inverse=False))
    j_prod = np.asarray(dft.acb_dft_prod_jit(jnp.asarray(x12), cyc=(3, 4)))

    np.testing.assert_allclose(c_main8, j_main8, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_main12, j_main12, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_rad2, j_rad2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_naive, j_naive, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_prod, j_prod, rtol=1e-12, atol=1e-12)

    c_cn = _call_acb_conv(lib, "acb_convol_circular_naive_ref", f, g)
    c_cd = _call_acb_conv(lib, "acb_convol_circular_dft_ref", f, g)
    c_cr = _call_acb_conv(lib, "acb_convol_circular_rad2_ref", f, g)
    c_c = _call_acb_conv(lib, "acb_convol_circular_ref", f, g)

    j_cn = np.asarray(dft.acb_convol_circular_naive_jit(jnp.asarray(f), jnp.asarray(g)))
    j_cd = np.asarray(dft.acb_convol_circular_dft_jit(jnp.asarray(f), jnp.asarray(g)))
    j_cr = np.asarray(dft.acb_convol_circular_rad2_jit(jnp.asarray(f), jnp.asarray(g)))
    j_c = np.asarray(dft.acb_convol_circular_jit(jnp.asarray(f), jnp.asarray(g)))

    np.testing.assert_allclose(c_cn, j_cn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_cd, j_cd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_cr, j_cr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_c, j_c, rtol=1e-12, atol=1e-12)
