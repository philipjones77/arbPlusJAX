from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mpmath as mp

from arbplusjax import hypgeom, arb_core, ball_wrappers, baseline_wrappers
from arbplusjax import double_interval as di


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


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
    hyp.arb_hypgeom_gamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_gamma_ref.restype = DI
    return hyp


def _interval(lo, hi):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _mid(x):
    return 0.5 * (x[0] + x[1])


def _timeit(fn, iters: int) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arb-repo", type=str, required=True)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range-lo", type=float, default=0.1)
    parser.add_argument("--range-hi", type=float, default=3.0)
    parser.add_argument("--rad", type=float, default=0.01)
    parser.add_argument("--mp-dps", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mids = rng.uniform(args.range_lo, args.range_hi, size=args.iters)
    xs = np.stack([mids - args.rad, mids + args.rad], axis=-1)

    hyp = load_c_libs(Path(args.arb_repo))

    # C baseline (truth)
    c_boxes = np.zeros((args.iters, 2), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(args.iters):
        lo, hi = xs[i]
        v = hyp.arb_hypgeom_gamma_ref(DI(lo, hi))
        c_boxes[i] = [v.a, v.b]
    t_c = time.perf_counter() - t0
    c_mid = np.mean(c_boxes, axis=1)

    # mpmath
    mp.mp.dps = args.mp_dps
    mp_vals = np.zeros(args.iters, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(args.iters):
        m = mp.mpf(mids[i])
        mp_vals[i] = float(mp.gamma(m))
    t_mp = time.perf_counter() - t0

    # JAX vectorized (interval midpoint)
    x_jax = jnp.asarray(xs, dtype=jnp.float64)

    def vec_mid(fn):
        def f(x):
            return _mid(fn(x))
        return jax.jit(jax.vmap(f))

    base_fn = vec_mid(hypgeom.arb_hypgeom_gamma)
    rig_fn = vec_mid(ball_wrappers.arb_ball_gamma)
    adapt_fn = vec_mid(ball_wrappers.arb_ball_gamma_adaptive)
    mp_mode_fn = vec_mid(lambda x: baseline_wrappers.arb_gamma_mp(x, mode="baseline", dps=50))

    # warmup
    base_fn(x_jax).block_until_ready()
    rig_fn(x_jax).block_until_ready()
    adapt_fn(x_jax).block_until_ready()
    mp_mode_fn(x_jax).block_until_ready()

    t0 = time.perf_counter()
    base_vals = np.array(base_fn(x_jax).block_until_ready())
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    rig_vals = np.array(rig_fn(x_jax).block_until_ready())
    t_rig = time.perf_counter() - t0

    t0 = time.perf_counter()
    adapt_vals = np.array(adapt_fn(x_jax).block_until_ready())
    t_adapt = time.perf_counter() - t0

    t0 = time.perf_counter()
    mp_mode_vals = np.array(mp_mode_fn(x_jax).block_until_ready())
    t_mp_mode = time.perf_counter() - t0

    # jax.special (point, vectorized)
    js_fn = jax.jit(jax.vmap(lambda t: jax.scipy.special.gamma(t)))
    js_fn(jnp.asarray(mids, dtype=jnp.float64)).block_until_ready()
    t0 = time.perf_counter()
    js_vals = np.array(js_fn(jnp.asarray(mids, dtype=jnp.float64)).block_until_ready())
    t_js = time.perf_counter() - t0

    def err_stats(vals):
        err = np.abs(vals - c_mid)
        return float(np.mean(err)), float(np.max(err))

    def containment_fraction(vals):
        return float(np.mean([(c_boxes[i, 0] <= vals[i] <= c_boxes[i, 1]) for i in range(args.iters)]))

    print("Gamma compare vs C midpoint (iters={}, vectorized JAX):".format(args.iters))
    for name, vals, t in [
        ("C", c_mid, t_c),
        ("mpmath", mp_vals, t_mp),
        ("jax_baseline", base_vals, t_base),
        ("jax_rigorous", rig_vals, t_rig),
        ("jax_adaptive", adapt_vals, t_adapt),
        ("jax_mp_mode", mp_mode_vals, t_mp_mode),
        ("jax.special", js_vals, t_js),
    ]:
        if name == "C":
            print(f"{name:12s} time={t:.3f}s (truth)")
            continue
        mean_err, max_err = err_stats(vals)
        contain = containment_fraction(vals)
        print(f"{name:12s} time={t:.3f}s mean_err={mean_err:.3e} max_err={max_err:.3e} contain={contain:.3f}")


if __name__ == "__main__":
    main()
