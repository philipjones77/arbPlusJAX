from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import mpmath as mp

from arbplusjax import hypgeom, acb_core
from arbplusjax import double_interval as di


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


def load_c_libs(arb_repo: Path):
    build_dir = arb_repo / "migration" / "c_chassis" / "build"
    di_path = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    hyp_path = _find_lib(build_dir, ["hypgeom_ref.dll", "libhypgeom_ref.dll", "libhypgeom_ref.so", "libhypgeom_ref.dylib"])
    if not di_path or not hyp_path:
        raise RuntimeError("C reference libraries not found in Arb workspace build.")
    ctypes.CDLL(str(di_path))
    hyp = ctypes.CDLL(str(hyp_path))
    hyp.arb_hypgeom_lgamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_lgamma_ref.restype = DI
    hyp.acb_hypgeom_lgamma_ref.argtypes = [ACB]
    hyp.acb_hypgeom_lgamma_ref.restype = ACB
    return hyp


def _interval(lo, hi):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _mid_real(x):
    return 0.5 * (x[0] + x[1])


def _mid_box(box):
    re = 0.5 * (box[0] + box[1])
    im = 0.5 * (box[2] + box[3])
    return re + 1j * im


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arb-repo", type=str, required=True)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range-lo", type=float, default=0.1)
    parser.add_argument("--range-hi", type=float, default=8.0)
    parser.add_argument("--imag-range", type=float, default=6.0)
    parser.add_argument("--rad", type=float, default=0.05)
    parser.add_argument("--mp-dps", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mids = rng.uniform(args.range_lo, args.range_hi, size=args.iters)
    xs = np.stack([mids - args.rad, mids + args.rad], axis=-1)

    re_mid = rng.uniform(args.range_lo, args.range_hi, size=args.iters)
    im_mid = rng.uniform(-args.imag_range, args.imag_range, size=args.iters)
    re_box = np.stack([re_mid - args.rad, re_mid + args.rad], axis=-1)
    im_box = np.stack([im_mid - args.rad, im_mid + args.rad], axis=-1)

    hyp = load_c_libs(Path(args.arb_repo))

    # C reference (real)
    c_real = np.zeros((args.iters, 2), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(args.iters):
        lo, hi = xs[i]
        v = hyp.arb_hypgeom_lgamma_ref(DI(lo, hi))
        c_real[i] = [v.a, v.b]
    t_c_real = time.perf_counter() - t0
    c_real_mid = np.mean(c_real, axis=1)

    # C reference (complex)
    c_cplx = np.zeros((args.iters, 4), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(args.iters):
        rlo, rhi = re_box[i]
        ilo, ihi = im_box[i]
        v = hyp.acb_hypgeom_lgamma_ref(ACB(DI(rlo, rhi), DI(ilo, ihi)))
        c_cplx[i] = [v.real.a, v.real.b, v.imag.a, v.imag.b]
    t_c_cplx = time.perf_counter() - t0
    c_cplx_mid = _mid_box(c_cplx.T).T if c_cplx.ndim == 2 else _mid_box(c_cplx)

    # mpmath
    mp.mp.dps = args.mp_dps
    t0 = time.perf_counter()
    mp_real = np.asarray([float(mp.loggamma(mp.mpf(m))) for m in mids], dtype=np.float64)
    t_mp_real = time.perf_counter() - t0
    t0 = time.perf_counter()
    mp_cplx = np.asarray([complex(mp.loggamma(mp.mpf(re_mid[i]) + 1j * mp.mpf(im_mid[i]))) for i in range(args.iters)], dtype=np.complex128)
    t_mp_cplx = time.perf_counter() - t0

    # JAX (real)
    x_jax = jnp.asarray(xs, dtype=jnp.float64)

    @jax.jit
    def jax_real_basic(x):
        return jax.vmap(lambda t: _mid_real(hypgeom.arb_hypgeom_lgamma(t)))(x)

    @jax.jit
    def jax_real_point(x):
        mids = 0.5 * (x[:, 0] + x[:, 1])
        return jax.vmap(lambda t: jax.lax.lgamma(t))(mids)

    # warmup
    jax_real_basic(x_jax).block_until_ready()
    jax_real_point(x_jax).block_until_ready()

    t0 = time.perf_counter()
    jax_real_basic_vals = np.array(jax_real_basic(x_jax).block_until_ready())
    t_jax_real_basic = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax_real_point_vals = np.array(jax_real_point(x_jax).block_until_ready())
    t_jax_real_point = time.perf_counter() - t0

    # JAX (complex)
    re_jax = jnp.asarray(re_box, dtype=jnp.float64)
    im_jax = jnp.asarray(im_box, dtype=jnp.float64)
    z_box = acb_core.acb_box(re_jax, im_jax)

    @jax.jit
    def jax_cplx_basic(z):
        return jax.vmap(lambda t: acb_core.acb_midpoint(hypgeom.acb_hypgeom_lgamma(t)))(z)

    @jax.jit
    def jax_cplx_point(z):
        mids = acb_core.acb_midpoint(z)
        return jax.vmap(lambda t: hypgeom._complex_loggamma(t))(mids)

    jax_cplx_basic(z_box).block_until_ready()
    jax_cplx_point(z_box).block_until_ready()

    t0 = time.perf_counter()
    jax_cplx_basic_vals = np.array(jax_cplx_basic(z_box).block_until_ready())
    t_jax_cplx_basic = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax_cplx_point_vals = np.array(jax_cplx_point(z_box).block_until_ready())
    t_jax_cplx_point = time.perf_counter() - t0

    def err_stats(vals, ref):
        err = np.abs(vals - ref)
        err = err[np.isfinite(err)]
        if err.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(err)), float(np.max(err))

    def contain_real(vals, ref_boxes):
        ok = (ref_boxes[:, 0] <= vals) & (vals <= ref_boxes[:, 1])
        ok = ok & np.isfinite(vals)
        return float(np.mean(ok))

    def contain_cplx(vals, ref_boxes):
        re = vals.real
        im = vals.imag
        ok_re = (ref_boxes[:, 0] <= re) & (re <= ref_boxes[:, 1])
        ok_im = (ref_boxes[:, 2] <= im) & (im <= ref_boxes[:, 3])
        ok = ok_re & ok_im & np.isfinite(re) & np.isfinite(im)
        return float(np.mean(ok))

    # public-JAX real reference (point)
    jax_native_real = jax.jit(jax.vmap(lambda t: lax.lgamma(t)))
    jax_native_cplx = None
    jax_native_real(jnp.asarray(mids, dtype=jnp.float64)).block_until_ready()
    # public JAX lgamma remains real-only here
    t0 = time.perf_counter()
    jax_native_real_vals = np.array(jax_native_real(jnp.asarray(mids, dtype=jnp.float64)).block_until_ready())
    t_jax_native_real = time.perf_counter() - t0
    jax_native_cplx_vals = None
    t_jax_native_cplx = None

    print(f"loggamma compare vs C midpoint (iters={args.iters}):")
    print("real:")
    for name, vals, t in [
        ("C", c_real_mid, t_c_real),
        ("mpmath", mp_real, t_mp_real),
        ("jax_basic", jax_real_basic_vals, t_jax_real_basic),
        ("jax_point", jax_real_point_vals, t_jax_real_point),
        ("jax_native", jax_native_real_vals, t_jax_native_real),
    ]:
        if name == "C":
            print(f"{name:10s} time={t:.3f}s (truth)")
            continue
        mean_err, max_err = err_stats(vals, c_real_mid)
        contain = contain_real(vals, c_real)
        print(f"{name:10s} time={t:.3f}s mean_err={mean_err:.3e} max_err={max_err:.3e} contain={contain:.3f}")

    print("complex:")
    for name, vals, t in [
        ("C", c_cplx_mid, t_c_cplx),
        ("mpmath", mp_cplx, t_mp_cplx),
        ("jax_basic", jax_cplx_basic_vals, t_jax_cplx_basic),
        ("jax_point", jax_cplx_point_vals, t_jax_cplx_point),
        ("jax_native", jax_native_cplx_vals, t_jax_native_cplx),
    ]:
        if name == "C":
            print(f"{name:10s} time={t:.3f}s (truth)")
            continue
        if vals is None:
            print(f"{name:10s} unavailable (complex not supported)")
            continue
        mean_err, max_err = err_stats(vals, c_cplx_mid)
        contain = contain_cplx(vals, c_cplx)
        print(f"{name:10s} time={t:.3f}s mean_err={mean_err:.3e} max_err={max_err:.3e} contain={contain:.3f}")

    # stress test near branch cuts (real negative, complex near negative real axis)
    neg = rng.uniform(-8.0, -0.1, size=args.iters)
    neg_boxes = np.stack([neg - args.rad, neg + args.rad], axis=-1)
    re_cut = rng.uniform(-8.0, -0.1, size=args.iters)
    im_cut = rng.uniform(-0.05, 0.05, size=args.iters)
    re_cut_box = np.stack([re_cut - args.rad, re_cut + args.rad], axis=-1)
    im_cut_box = np.stack([im_cut - args.rad, im_cut + args.rad], axis=-1)

    # C ref near branch cuts
    c_real_cut = np.zeros((args.iters, 2), dtype=np.float64)
    for i in range(args.iters):
        lo, hi = neg_boxes[i]
        v = hyp.arb_hypgeom_lgamma_ref(DI(lo, hi))
        c_real_cut[i] = [v.a, v.b]
    c_real_cut_mid = np.mean(c_real_cut, axis=1)

    c_cplx_cut = np.zeros((args.iters, 4), dtype=np.float64)
    for i in range(args.iters):
        rlo, rhi = re_cut_box[i]
        ilo, ihi = im_cut_box[i]
        v = hyp.acb_hypgeom_lgamma_ref(ACB(DI(rlo, rhi), DI(ilo, ihi)))
        c_cplx_cut[i] = [v.real.a, v.real.b, v.imag.a, v.imag.b]
    c_cplx_cut_mid = _mid_box(c_cplx_cut.T).T if c_cplx_cut.ndim == 2 else _mid_box(c_cplx_cut)

    # JAX/misc near branch cut
    neg_jax = jnp.asarray(neg_boxes, dtype=jnp.float64)
    cut_box = acb_core.acb_box(jnp.asarray(re_cut_box, dtype=jnp.float64), jnp.asarray(im_cut_box, dtype=jnp.float64))
    jax_real_cut = np.array(jax_real_basic(neg_jax).block_until_ready())
    jax_point_cut = np.array(jax_real_point(neg_jax).block_until_ready())
    jax_native_real_cut = np.array(jax_native_real(jnp.asarray(neg, dtype=jnp.float64)).block_until_ready())
    mp_real_cut = []
    for v in neg:
        val = mp.loggamma(mp.mpf(v))
        if hasattr(val, "imag") and abs(val.imag) > 0:
            mp_real_cut.append(np.nan)
        else:
            mp_real_cut.append(float(val))
    mp_real_cut = np.asarray(mp_real_cut, dtype=np.float64)

    jax_cplx_cut = np.array(jax_cplx_basic(cut_box).block_until_ready())
    jax_cplx_point_cut = np.array(jax_cplx_point(cut_box).block_until_ready())
    jax_native_cplx_cut = None
    mp_cplx_cut = np.asarray([complex(mp.loggamma(mp.mpf(re_cut[i]) + 1j * mp.mpf(im_cut[i]))) for i in range(args.iters)], dtype=np.complex128)

    print("real (negative axis stress):")
    for name, vals in [
        ("mpmath", mp_real_cut),
        ("jax_basic", jax_real_cut),
        ("jax_point", jax_point_cut),
        ("jax_native", jax_native_real_cut),
    ]:
        mean_err, max_err = err_stats(vals, c_real_cut_mid)
        contain = contain_real(vals, c_real_cut)
        print(f"{name:10s} mean_err={mean_err:.3e} max_err={max_err:.3e} contain={contain:.3f}")

    print("complex (near negative real axis):")
    for name, vals in [
        ("mpmath", mp_cplx_cut),
        ("jax_basic", jax_cplx_cut),
        ("jax_point", jax_cplx_point_cut),
        ("jax_native", jax_native_cplx_cut),
    ]:
        if vals is None:
            print(f"{name:10s} unavailable (complex not supported)")
            continue
        mean_err, max_err = err_stats(vals, c_cplx_cut_mid)
        contain = contain_cplx(vals, c_cplx_cut)
        print(f"{name:10s} mean_err={mean_err:.3e} max_err={max_err:.3e} contain={contain:.3f}")


if __name__ == "__main__":
    main()
