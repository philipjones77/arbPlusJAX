from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import arb_core, acb_core, hypgeom, ball_wrappers
from arbplusjax import double_interval as di


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


def default_paths(arb_repo: Path) -> tuple[Path | None, Path | None, Path | None]:
    build_dir = arb_repo / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None, None
    di_path = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    arb_core_path = _find_lib(build_dir, ["arb_core_ref.dll", "libarb_core_ref.dll", "libarb_core_ref.so", "libarb_core_ref.dylib"])
    hyp_path = _find_lib(build_dir, ["hypgeom_ref.dll", "libhypgeom_ref.dll", "libhypgeom_ref.so", "libhypgeom_ref.dylib"])
    return di_path, arb_core_path, hyp_path


def load_c_libs(di_path: Path, arb_core_path: Path, hyp_path: Path):
    ctypes.CDLL(str(di_path))
    core = ctypes.CDLL(str(arb_core_path))
    hyp = ctypes.CDLL(str(hyp_path))

    core.arb_exp_ref.argtypes = [DI]
    core.arb_exp_ref.restype = DI
    core.arb_log_ref.argtypes = [DI]
    core.arb_log_ref.restype = DI
    core.arb_sin_ref.argtypes = [DI]
    core.arb_sin_ref.restype = DI

    hyp.arb_hypgeom_gamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_gamma_ref.restype = DI

    return core, hyp


def _interval(lo, hi):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _contains(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (a[0] <= b[0]) & (a[1] >= b[1])


def _width(x: jnp.ndarray) -> float:
    return float(x[1] - x[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arb-repo", type=str, default=os.getenv("ARB_REPO", ""))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.samples

    # ranges
    exp_range = (-2.0, 2.0)
    sin_range = (-3.0, 3.0)
    log_range = (0.1, 3.0)
    gamma_range = (0.1, 3.0)
    rad = 0.01

    def sample_interval(r):
        lo, hi = r
        mid = rng.uniform(lo, hi, size=n)
        return np.stack([mid - rad, mid + rad], axis=-1)

    exp_x = sample_interval(exp_range)
    sin_x = sample_interval(sin_range)
    log_x = sample_interval(log_range)
    gam_x = sample_interval(gamma_range)

    # JAX baseline
    def jax_eval(fn, xs):
        return np.array([fn(_interval(a, b)) for a, b in xs])

    base_exp = jax_eval(arb_core.arb_exp, exp_x)
    base_log = jax_eval(arb_core.arb_log, log_x)
    base_sin = jax_eval(arb_core.arb_sin, sin_x)
    base_gamma = jax_eval(hypgeom.arb_hypgeom_gamma, gam_x)

    rig_exp = jax_eval(ball_wrappers.arb_ball_exp, exp_x)
    rig_log = jax_eval(ball_wrappers.arb_ball_log, log_x)
    rig_sin = jax_eval(ball_wrappers.arb_ball_sin, sin_x)
    rig_gamma = jax_eval(ball_wrappers.arb_ball_gamma, gam_x)

    ad_exp = jax_eval(ball_wrappers.arb_ball_exp_adaptive, exp_x)
    ad_log = jax_eval(ball_wrappers.arb_ball_log_adaptive, log_x)
    ad_sin = jax_eval(ball_wrappers.arb_ball_sin_adaptive, sin_x)
    ad_gamma = jax_eval(ball_wrappers.arb_ball_gamma_adaptive, gam_x)

    print("Real interval widths (avg):")
    for name, base, rig, ad in [
        ("exp", base_exp, rig_exp, ad_exp),
        ("log", base_log, rig_log, ad_log),
        ("sin", base_sin, rig_sin, ad_sin),
        ("gamma", base_gamma, rig_gamma, ad_gamma),
    ]:
        b = np.mean([_width(x) for x in base])
        r = np.mean([_width(x) for x in rig])
        a = np.mean([_width(x) for x in ad])
        print(f"{name:6s} base={b:.3e} rig={r:.3e} adapt={a:.3e}")

    if args.arb_repo:
        di_path, arb_core_path, hyp_path = default_paths(Path(args.arb_repo))
        if di_path and arb_core_path and hyp_path:
            core, hyp = load_c_libs(di_path, arb_core_path, hyp_path)

            def c_eval(fn, xs):
                out = []
                for a, b in xs:
                    v = fn(DI(a, b))
                    out.append(np.array([v.a, v.b], dtype=np.float64))
                return np.array(out)

            c_exp = c_eval(core.arb_exp_ref, exp_x)
            c_log = c_eval(core.arb_log_ref, log_x)
            c_sin = c_eval(core.arb_sin_ref, sin_x)
            c_gamma = c_eval(hyp.arb_hypgeom_gamma_ref, gam_x)

            print("\nContainment vs C (fraction):")
            for name, base, rig, ad, cref in [
                ("exp", base_exp, rig_exp, ad_exp, c_exp),
                ("log", base_log, rig_log, ad_log, c_log),
                ("sin", base_sin, rig_sin, ad_sin, c_sin),
                ("gamma", base_gamma, rig_gamma, ad_gamma, c_gamma),
            ]:
                base_c = np.mean([_contains(b, c) for b, c in zip(base, cref)])
                rig_c = np.mean([_contains(b, c) for b, c in zip(rig, cref)])
                ad_c = np.mean([_contains(b, c) for b, c in zip(ad, cref)])
                print(f"{name:6s} base={base_c:.3f} rig={rig_c:.3f} adapt={ad_c:.3f}")
        else:
            print("C reference libraries not found under arb repo. Skipping C compare.")
    else:
        print("No arb repo path set. Skipping C compare.")

    # Complex comparison (no C ref by default)
    cmid_re = rng.uniform(-2.0, 2.0, size=n)
    cmid_im = rng.uniform(-2.0, 2.0, size=n)
    cboxes = np.stack(
        [cmid_re - rad, cmid_re + rad, cmid_im - rad, cmid_im + rad],
        axis=-1,
    )

    def jax_eval_box(fn, xs):
        return np.array([fn(jnp.array(x, dtype=jnp.float64)) for x in xs])

    base_c_exp = jax_eval_box(acb_core.acb_exp, cboxes)
    rig_c_exp = jax_eval_box(ball_wrappers.acb_ball_exp, cboxes)
    ad_c_exp = jax_eval_box(ball_wrappers.acb_ball_exp_adaptive, cboxes)

    def box_width(x):
        return float((x[1] - x[0]) + (x[3] - x[2]))

    print("\nComplex box widths (avg, exp):")
    print(
        f"exp base={np.mean([box_width(x) for x in base_c_exp]):.3e} "
        f"rig={np.mean([box_width(x) for x in rig_c_exp]):.3e} "
        f"adapt={np.mean([box_width(x) for x in ad_c_exp]):.3e}"
    )


if __name__ == "__main__":
    main()
