import ctypes
import os
from pathlib import Path

from tests._arb_c_chassis import get_c_ref_build_dir

import jax.numpy as jnp
import numpy as np
import pytest

from arbplusjax import hypgeom

from tests._test_checks import _check
pytestmark = pytest.mark.parity
if os.getenv("ARBPLUSJAX_RUN_PARITY", "0") != "1":
    pytest.skip("Parity tests disabled. Set ARBPLUSJAX_RUN_PARITY=1 to enable.", allow_module_level=True)


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


def _default_hypgeom_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None

    return _find_lib(build_dir, ["hypgeom_ref.dll", "libhypgeom_ref.dll", "libhypgeom_ref.so", "libhypgeom_ref.dylib"])


def _default_double_interval_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = get_c_ref_build_dir()
    if not build_dir.exists():
        return None

    return _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])


def _load_libs():
    di_path = Path(os.getenv("DI_REF_LIB", "")) if os.getenv("DI_REF_LIB") else _default_double_interval_lib_path()
    hyp_path = Path(os.getenv("HYPGEOM_REF_LIB", "")) if os.getenv("HYPGEOM_REF_LIB") else _default_hypgeom_lib_path()

    if di_path is None or not di_path.exists() or hyp_path is None or not hyp_path.exists():
        pytest.skip("C reference libraries not found. Build C reference libraries in the Arb workspace first.")

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
    hyp.acb_hypgeom_1f1_ref.argtypes = [ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_1f1_ref.restype = ACBBox
    hyp.arb_hypgeom_1f1_full_ref.argtypes = [DI, DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_1f1_full_ref.restype = DI
    hyp.acb_hypgeom_1f1_full_ref.argtypes = [ACBBox, ACBBox, ACBBox, ctypes.c_int]
    hyp.acb_hypgeom_1f1_full_ref.restype = ACBBox
    hyp.arb_hypgeom_2f1_ref.argtypes = [DI, DI, DI, DI]
    hyp.arb_hypgeom_2f1_ref.restype = DI
    hyp.acb_hypgeom_2f1_ref.argtypes = [ACBBox, ACBBox, ACBBox, ACBBox]
    hyp.acb_hypgeom_2f1_ref.restype = ACBBox
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
    hyp.arb_hypgeom_bessel_i_scaled_ref.argtypes = [DI, DI]
    hyp.arb_hypgeom_bessel_i_scaled_ref.restype = DI
    hyp.arb_hypgeom_bessel_k_scaled_ref.argtypes = [DI, DI]
    hyp.arb_hypgeom_bessel_k_scaled_ref.restype = DI
    hyp.arb_hypgeom_bessel_i_integration_ref.argtypes = [DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_bessel_i_integration_ref.restype = DI
    hyp.arb_hypgeom_bessel_k_integration_ref.argtypes = [DI, DI, ctypes.c_int]
    hyp.arb_hypgeom_bessel_k_integration_ref.restype = DI
    hyp.acb_hypgeom_bessel_j_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_j_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_y_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_y_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_i_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_i_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_k_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_k_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_i_scaled_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_i_scaled_ref.restype = ACBBox
    hyp.acb_hypgeom_bessel_k_scaled_ref.argtypes = [ACBBox, ACBBox]
    hyp.acb_hypgeom_bessel_k_scaled_ref.restype = ACBBox

    return hyp


def _random_intervals(rng: np.random.Generator, n: int, scale: float = 10.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=n)
    b = rng.uniform(-scale, scale, size=n)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=-1).astype(np.float64)


def _random_acb_boxes(rng: np.random.Generator, n: int, scale: float = 3.0) -> np.ndarray:
    re = _random_intervals(rng, n, scale=scale)
    im = _random_intervals(rng, n, scale=scale)
    return np.concatenate([re, im], axis=-1)


def _random_positive_intervals(rng: np.random.Generator, n: int, lo: float = 0.05, hi: float = 10.0) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def _random_acb_boxes_away_from_poles(rng: np.random.Generator, n: int) -> np.ndarray:
    re = _random_intervals(rng, n, scale=4.0)
    im_lo = rng.uniform(0.15, 3.0, size=n)
    im_hi = im_lo + rng.uniform(0.0, 0.5, size=n)
    im = np.stack([im_lo, im_hi], axis=-1)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def _random_small_intervals(rng: np.random.Generator, n: int, scale: float = 1.5) -> np.ndarray:
    return _random_intervals(rng, n, scale=scale)


def _random_small_acb_boxes(rng: np.random.Generator, n: int, scale: float = 1.5) -> np.ndarray:
    re = _random_intervals(rng, n, scale=scale)
    im = _random_intervals(rng, n, scale=scale)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def _random_hyp_params_real(rng: np.random.Generator, n: int):
    a = _random_positive_intervals(rng, n, lo=0.5, hi=2.0)
    b = _random_positive_intervals(rng, n, lo=1.5, hi=3.5)
    c = _random_positive_intervals(rng, n, lo=1.8, hi=4.0)
    z = _random_small_intervals(rng, n, scale=0.5)
    return a, b, c, z


def _random_hyp_params_complex(rng: np.random.Generator, n: int):
    a = _random_small_acb_boxes(rng, n, scale=1.0)
    b = _random_small_acb_boxes(rng, n, scale=1.0)
    c = _random_small_acb_boxes(rng, n, scale=1.0)
    # Shift real parts of denominator params away from zero/poles
    b[:, 0:2] += 2.2
    c[:, 0:2] += 2.5
    z = _random_small_acb_boxes(rng, n, scale=0.5)
    return a, b, c, z


def _random_bessel_params_real(rng: np.random.Generator, n: int):
    nu = _random_positive_intervals(rng, n, lo=0.2, hi=2.5)
    z = _random_small_intervals(rng, n, scale=2.0)
    return nu, z


def _random_bessel_params_complex(rng: np.random.Generator, n: int):
    nu = _random_small_acb_boxes(rng, n, scale=1.0)
    z = _random_small_acb_boxes(rng, n, scale=2.0)
    return nu, z


def _call_arb_scalar(lib, x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = lib.arb_hypgeom_rising_ui_ref(DI(float(x[i, 0]), float(x[i, 1])), n)
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _call_acb_scalar(lib, x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xb = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
        r = lib.acb_hypgeom_rising_ui_ref(xb, n)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def _call_arb_lgamma(lib, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = lib.arb_hypgeom_lgamma_ref(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _call_acb_lgamma(lib, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xb = ACBBox(DI(float(x[i, 0]), float(x[i, 1])), DI(float(x[i, 2]), float(x[i, 3])))
        r = lib.acb_hypgeom_lgamma_ref(xb)
        out[i, 0] = r.real.a
        out[i, 1] = r.real.b
        out[i, 2] = r.imag.a
        out[i, 3] = r.imag.b
    return out


def _call_arb_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _call_acb_unary(lib, fn_name: str, x: np.ndarray) -> np.ndarray:
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


def _call_arb_binary(lib, fn_name: str, a: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
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


def _call_acb_binary(lib, fn_name: str, a: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
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


def _call_arb_ternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
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


def _call_acb_ternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
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


def _call_arb_quaternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
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


def _call_acb_quaternary(lib, fn_name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray, flag: int | None = None) -> np.ndarray:
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


def test_rising_ui_real_and_complex_parity():
    lib = _load_libs()
    rng = np.random.default_rng(1234)
    xr = _random_intervals(rng, 1200)
    xc = _random_acb_boxes(rng, 800)

    for n in (0, 1, 2, 5, 9, 15):
        c_real = _call_arb_scalar(lib, xr, n)
        j_real = np.asarray(hypgeom.arb_hypgeom_rising_ui_batch_jit(jnp.asarray(xr), n=n))

        c_cplx = _call_acb_scalar(lib, xc, n)
        j_cplx = np.asarray(hypgeom.acb_hypgeom_rising_ui_batch_jit(jnp.asarray(xc), n=n))

        np.testing.assert_allclose(c_real, j_real, rtol=2e-13, atol=0.0, equal_nan=True)
        np.testing.assert_allclose(c_cplx, j_cplx, rtol=3e-13, atol=0.0, equal_nan=True)

        _check(np.all(j_real[:, 0] <= j_real[:, 1]))
        _check(np.all(j_cplx[:, 0] <= j_cplx[:, 1]))
        _check(np.all(j_cplx[:, 2] <= j_cplx[:, 3]))


def test_lgamma_real_and_complex_parity():
    lib = _load_libs()
    rng = np.random.default_rng(2027)
    xr = _random_positive_intervals(rng, 1400)
    xc = _random_acb_boxes_away_from_poles(rng, 1000)

    c_real = _call_arb_lgamma(lib, xr)
    j_real = np.asarray(hypgeom.arb_hypgeom_lgamma_batch_jit(jnp.asarray(xr)))
    c_cplx = _call_acb_lgamma(lib, xc)
    j_cplx = np.asarray(hypgeom.acb_hypgeom_lgamma_batch_jit(jnp.asarray(xc)))

    np.testing.assert_allclose(c_real, j_real, rtol=2e-12, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(c_cplx, j_cplx, rtol=1e-11, atol=0.0, equal_nan=True)

    _check(np.all(j_real[:, 0] <= j_real[:, 1]))
    _check(np.all(j_cplx[:, 0] <= j_cplx[:, 1]))
    _check(np.all(j_cplx[:, 2] <= j_cplx[:, 3]))


def test_gamma_rgamma_real_and_complex_parity():
    lib = _load_libs()
    rng = np.random.default_rng(2041)
    xr = _random_positive_intervals(rng, 1200)
    xc = _random_acb_boxes_away_from_poles(rng, 900)

    c_rg = _call_arb_unary(lib, "arb_hypgeom_gamma_ref", xr)
    j_rg = np.asarray(hypgeom.arb_hypgeom_gamma_batch_jit(jnp.asarray(xr)))
    c_cg = _call_acb_unary(lib, "acb_hypgeom_gamma_ref", xc)
    j_cg = np.asarray(hypgeom.acb_hypgeom_gamma_batch_jit(jnp.asarray(xc)))

    c_rr = _call_arb_unary(lib, "arb_hypgeom_rgamma_ref", xr)
    j_rr = np.asarray(hypgeom.arb_hypgeom_rgamma_batch_jit(jnp.asarray(xr)))
    c_cr = _call_acb_unary(lib, "acb_hypgeom_rgamma_ref", xc)
    j_cr = np.asarray(hypgeom.acb_hypgeom_rgamma_batch_jit(jnp.asarray(xc)))

    np.testing.assert_allclose(c_rg, j_rg, rtol=4e-12, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(c_cg, j_cg, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_rr, j_rr, rtol=4e-12, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(c_cr, j_cr, rtol=2e-11, atol=2e-14, equal_nan=True)

    _check(np.all(j_rg[:, 0] <= j_rg[:, 1]))
    _check(np.all(j_cg[:, 0] <= j_cg[:, 1]))
    _check(np.all(j_cg[:, 2] <= j_cg[:, 3]))
    _check(np.all(j_rr[:, 0] <= j_rr[:, 1]))
    _check(np.all(j_cr[:, 0] <= j_cr[:, 1]))
    _check(np.all(j_cr[:, 2] <= j_cr[:, 3]))


def test_erf_erfc_erfi_real_and_complex_parity():
    lib = _load_libs()
    rng = np.random.default_rng(2051)
    xr = _random_small_intervals(rng, 1400, scale=1.5)
    xc = _random_small_acb_boxes(rng, 1000, scale=1.5)

    c_erf_r = _call_arb_unary(lib, "arb_hypgeom_erf_ref", xr)
    c_erf_c = _call_acb_unary(lib, "acb_hypgeom_erf_ref", xc)
    j_erf_r = np.asarray(hypgeom.arb_hypgeom_erf_batch_jit(jnp.asarray(xr)))
    j_erf_c = np.asarray(hypgeom.acb_hypgeom_erf_batch_jit(jnp.asarray(xc)))

    c_erfc_r = _call_arb_unary(lib, "arb_hypgeom_erfc_ref", xr)
    c_erfc_c = _call_acb_unary(lib, "acb_hypgeom_erfc_ref", xc)
    j_erfc_r = np.asarray(hypgeom.arb_hypgeom_erfc_batch_jit(jnp.asarray(xr)))
    j_erfc_c = np.asarray(hypgeom.acb_hypgeom_erfc_batch_jit(jnp.asarray(xc)))

    c_erfi_r = _call_arb_unary(lib, "arb_hypgeom_erfi_ref", xr)
    c_erfi_c = _call_acb_unary(lib, "acb_hypgeom_erfi_ref", xc)
    j_erfi_r = np.asarray(hypgeom.arb_hypgeom_erfi_batch_jit(jnp.asarray(xr)))
    j_erfi_c = np.asarray(hypgeom.acb_hypgeom_erfi_batch_jit(jnp.asarray(xc)))

    np.testing.assert_allclose(c_erf_r, j_erf_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_erf_c, j_erf_c, rtol=3e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_erfc_r, j_erfc_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_erfc_c, j_erfc_c, rtol=3e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_erfi_r, j_erfi_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_erfi_c, j_erfi_c, rtol=3e-11, atol=2e-14, equal_nan=True)


def test_0f1_m_and_regularized_parity():
    lib = _load_libs()
    rng = np.random.default_rng(2059)
    ar, br, cr, zr = _random_hyp_params_real(rng, 900)
    ac, bc, cc, zc = _random_hyp_params_complex(rng, 650)

    c_0f1_r = _call_arb_binary(lib, "arb_hypgeom_0f1_ref", br, zr, flag=1)
    c_0f1_c = _call_acb_binary(lib, "acb_hypgeom_0f1_ref", bc, zc, flag=1)
    j_0f1_r = np.asarray(hypgeom.arb_hypgeom_0f1_batch_jit(jnp.asarray(br), jnp.asarray(zr), regularized=True))
    j_0f1_c = np.asarray(hypgeom.acb_hypgeom_0f1_batch_jit(jnp.asarray(bc), jnp.asarray(zc), regularized=True))

    c_m_r = _call_arb_ternary(lib, "arb_hypgeom_m_ref", ar, br, zr, flag=1)
    c_m_c = _call_acb_ternary(lib, "acb_hypgeom_m_ref", ac, bc, zc, flag=1)
    j_m_r = np.asarray(hypgeom.arb_hypgeom_m_batch_jit(jnp.asarray(ar), jnp.asarray(br), jnp.asarray(zr), regularized=True))
    j_m_c = np.asarray(hypgeom.acb_hypgeom_m_batch_jit(jnp.asarray(ac), jnp.asarray(bc), jnp.asarray(zc), regularized=True))

    c_1f1_r = _call_arb_ternary(lib, "arb_hypgeom_1f1_full_ref", ar, br, zr, flag=1)
    c_1f1_c = _call_acb_ternary(lib, "acb_hypgeom_1f1_full_ref", ac, bc, zc, flag=1)
    j_1f1_r = np.asarray(hypgeom.arb_hypgeom_1f1_batch_jit(jnp.asarray(ar), jnp.asarray(br), jnp.asarray(zr), regularized=True))
    j_1f1_c = np.asarray(hypgeom.acb_hypgeom_1f1_batch_jit(jnp.asarray(ac), jnp.asarray(bc), jnp.asarray(zc), regularized=True))

    c_2f1_r = _call_arb_quaternary(lib, "arb_hypgeom_2f1_full_ref", ar, br, cr, zr, flag=1)
    c_2f1_c = _call_acb_quaternary(lib, "acb_hypgeom_2f1_full_ref", ac, bc, cc, zc, flag=1)
    j_2f1_r = np.asarray(
        hypgeom.arb_hypgeom_2f1_batch_jit(jnp.asarray(ar), jnp.asarray(br), jnp.asarray(cr), jnp.asarray(zr), regularized=True)
    )
    j_2f1_c = np.asarray(
        hypgeom.acb_hypgeom_2f1_batch_jit(jnp.asarray(ac), jnp.asarray(bc), jnp.asarray(cc), jnp.asarray(zc), regularized=True)
    )

    np.testing.assert_allclose(c_0f1_r, j_0f1_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_0f1_c, j_0f1_c, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_m_r, j_m_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_m_c, j_m_c, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_1f1_r, j_1f1_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_1f1_c, j_1f1_c, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_2f1_r, j_2f1_r, rtol=3e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_2f1_c, j_2f1_c, rtol=5e-11, atol=2e-14, equal_nan=True)


def test_1f1_2f1_real_and_complex_parity():
    lib = _load_libs()
    rng = np.random.default_rng(2061)
    ar, br, cr, zr = _random_hyp_params_real(rng, 1000)
    ac, bc, cc, zc = _random_hyp_params_complex(rng, 700)

    c_1f1_r = _call_arb_ternary(lib, "arb_hypgeom_1f1_ref", ar, br, zr)
    j_1f1_r = np.asarray(hypgeom.arb_hypgeom_1f1_batch_jit(jnp.asarray(ar), jnp.asarray(br), jnp.asarray(zr)))
    c_1f1_c = _call_acb_ternary(lib, "acb_hypgeom_1f1_ref", ac, bc, zc)
    j_1f1_c = np.asarray(hypgeom.acb_hypgeom_1f1_batch_jit(jnp.asarray(ac), jnp.asarray(bc), jnp.asarray(zc)))

    c_2f1_r = _call_arb_quaternary(lib, "arb_hypgeom_2f1_ref", ar, br, cr, zr)
    j_2f1_r = np.asarray(hypgeom.arb_hypgeom_2f1_batch_jit(jnp.asarray(ar), jnp.asarray(br), jnp.asarray(cr), jnp.asarray(zr)))
    c_2f1_c = _call_acb_quaternary(lib, "acb_hypgeom_2f1_ref", ac, bc, cc, zc)
    j_2f1_c = np.asarray(hypgeom.acb_hypgeom_2f1_batch_jit(jnp.asarray(ac), jnp.asarray(bc), jnp.asarray(cc), jnp.asarray(zc)))

    np.testing.assert_allclose(c_1f1_r, j_1f1_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_1f1_c, j_1f1_c, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_2f1_r, j_2f1_r, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_2f1_c, j_2f1_c, rtol=4e-11, atol=2e-14, equal_nan=True)


def test_bessel_real_and_complex_parity():
    lib = _load_libs()
    rng = np.random.default_rng(2077)
    nr, zr = _random_bessel_params_real(rng, 900)
    nc, zc = _random_bessel_params_complex(rng, 650)

    c_jr = _call_arb_binary(lib, "arb_hypgeom_bessel_j_ref", nr, zr)
    c_yr = _call_arb_binary(lib, "arb_hypgeom_bessel_y_ref", nr, zr)
    c_ir = _call_arb_binary(lib, "arb_hypgeom_bessel_i_ref", nr, zr)
    c_kr = _call_arb_binary(lib, "arb_hypgeom_bessel_k_ref", nr, zr)
    c_is = _call_arb_binary(lib, "arb_hypgeom_bessel_i_scaled_ref", nr, zr)
    c_ks = _call_arb_binary(lib, "arb_hypgeom_bessel_k_scaled_ref", nr, zr)
    c_ii = _call_arb_binary(lib, "arb_hypgeom_bessel_i_integration_ref", nr, zr, flag=1)
    c_ki = _call_arb_binary(lib, "arb_hypgeom_bessel_k_integration_ref", nr, zr, flag=1)

    j_jr = np.asarray(hypgeom.arb_hypgeom_bessel_j_batch_jit(jnp.asarray(nr), jnp.asarray(zr)))
    j_yr = np.asarray(hypgeom.arb_hypgeom_bessel_y_batch_jit(jnp.asarray(nr), jnp.asarray(zr)))
    j_ir = np.asarray(hypgeom.arb_hypgeom_bessel_i_batch_jit(jnp.asarray(nr), jnp.asarray(zr)))
    j_kr = np.asarray(hypgeom.arb_hypgeom_bessel_k_batch_jit(jnp.asarray(nr), jnp.asarray(zr)))
    j_is = np.asarray(hypgeom.arb_hypgeom_bessel_i_scaled_batch_jit(jnp.asarray(nr), jnp.asarray(zr)))
    j_ks = np.asarray(hypgeom.arb_hypgeom_bessel_k_scaled_batch_jit(jnp.asarray(nr), jnp.asarray(zr)))
    j_ii = np.asarray(hypgeom.arb_hypgeom_bessel_i_integration_batch_jit(jnp.asarray(nr), jnp.asarray(zr), scaled=True))
    j_ki = np.asarray(hypgeom.arb_hypgeom_bessel_k_integration_batch_jit(jnp.asarray(nr), jnp.asarray(zr), scaled=True))

    np.testing.assert_allclose(c_jr, j_jr, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_yr, j_yr, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_ir, j_ir, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_kr, j_kr, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_is, j_is, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_ks, j_ks, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_ii, j_ii, rtol=2e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_ki, j_ki, rtol=2e-11, atol=2e-14, equal_nan=True)

    c_jc = _call_acb_binary(lib, "acb_hypgeom_bessel_j_ref", nc, zc)
    c_yc = _call_acb_binary(lib, "acb_hypgeom_bessel_y_ref", nc, zc)
    c_ic = _call_acb_binary(lib, "acb_hypgeom_bessel_i_ref", nc, zc)
    c_kc = _call_acb_binary(lib, "acb_hypgeom_bessel_k_ref", nc, zc)
    c_isc = _call_acb_binary(lib, "acb_hypgeom_bessel_i_scaled_ref", nc, zc)
    c_ksc = _call_acb_binary(lib, "acb_hypgeom_bessel_k_scaled_ref", nc, zc)

    j_jc = np.asarray(hypgeom.acb_hypgeom_bessel_j_batch_jit(jnp.asarray(nc), jnp.asarray(zc)))
    j_yc = np.asarray(hypgeom.acb_hypgeom_bessel_y_batch_jit(jnp.asarray(nc), jnp.asarray(zc)))
    j_ic = np.asarray(hypgeom.acb_hypgeom_bessel_i_batch_jit(jnp.asarray(nc), jnp.asarray(zc)))
    j_kc = np.asarray(hypgeom.acb_hypgeom_bessel_k_batch_jit(jnp.asarray(nc), jnp.asarray(zc)))
    j_isc = np.asarray(hypgeom.acb_hypgeom_bessel_i_scaled_batch_jit(jnp.asarray(nc), jnp.asarray(zc)))
    j_ksc = np.asarray(hypgeom.acb_hypgeom_bessel_k_scaled_batch_jit(jnp.asarray(nc), jnp.asarray(zc)))

    np.testing.assert_allclose(c_jc, j_jc, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_yc, j_yc, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_ic, j_ic, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_kc, j_kc, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_isc, j_isc, rtol=4e-11, atol=2e-14, equal_nan=True)
    np.testing.assert_allclose(c_ksc, j_ksc, rtol=4e-11, atol=2e-14, equal_nan=True)
