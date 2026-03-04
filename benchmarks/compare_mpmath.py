from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import mpmath as mp

from arbplusjax import arb_core, acb_core, hypgeom, ball_wrappers
from arbplusjax import barnesg
from arbplusjax import double_interval as di


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def default_paths(arb_repo: Path) -> tuple[Path | None, Path | None]:
    build_dir = arb_repo / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None
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
    core.arb_cos_ref.argtypes = [DI]
    core.arb_cos_ref.restype = DI
    core.arb_tan_ref.argtypes = [DI]
    core.arb_tan_ref.restype = DI
    core.arb_sinh_ref.argtypes = [DI]
    core.arb_sinh_ref.restype = DI
    core.arb_cosh_ref.argtypes = [DI]
    core.arb_cosh_ref.restype = DI
    core.arb_tanh_ref.argtypes = [DI]
    core.arb_tanh_ref.restype = DI
    core.arb_sqrt_ref.argtypes = [DI]
    core.arb_sqrt_ref.restype = DI

    hyp.arb_hypgeom_gamma_ref.argtypes = [DI]
    hyp.arb_hypgeom_gamma_ref.restype = DI
    hyp.arb_hypgeom_erf_ref.argtypes = [DI]
    hyp.arb_hypgeom_erf_ref.restype = DI
    hyp.arb_hypgeom_erfc_ref.argtypes = [DI]
    hyp.arb_hypgeom_erfc_ref.restype = DI
    hyp.arb_hypgeom_erfi_ref.argtypes = [DI]
    hyp.arb_hypgeom_erfi_ref.restype = DI

    return core, hyp


def _interval(lo, hi):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _contains(a: np.ndarray, b: np.ndarray) -> bool:
    return (a[0] <= b[0]) and (a[1] >= b[1])


def _point_in_interval(x: float, b: np.ndarray) -> bool:
    return (b[0] <= x) and (x <= b[1])


def _width(x: np.ndarray) -> float:
    return float(x[1] - x[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arb-repo", type=str, default=os.getenv("ARB_REPO", ""))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dps", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.samples
    mp.mp.dps = args.dps

    ranges = {
        "exp": (-2.0, 2.0),
        "log": (0.1, 3.0),
        "sqrt": (0.01, 4.0),
        "sin": (-3.0, 3.0),
        "cos": (-3.0, 3.0),
        "tan": (-1.0, 1.0),
        "sinh": (-2.0, 2.0),
        "cosh": (-2.0, 2.0),
        "tanh": (-2.0, 2.0),
        "gamma": (0.1, 3.0),
        "erf": (-2.0, 2.0),
        "erfc": (-2.0, 2.0),
        "erfi": (-2.0, 2.0),
        "barnesg": (0.5, 3.5),
    }
    rad = 0.01

    def sample_interval(r):
        lo, hi = r
        mid = rng.uniform(lo, hi, size=n)
        return np.stack([mid - rad, mid + rad], axis=-1)

    xs = {k: sample_interval(v) for k, v in ranges.items()}

    def jax_eval(fn, xs):
        return np.array([fn(_interval(a, b)) for a, b in xs])

    base = {
        "exp": jax_eval(arb_core.arb_exp, xs["exp"]),
        "log": jax_eval(arb_core.arb_log, xs["log"]),
        "sqrt": jax_eval(arb_core.arb_sqrt, xs["sqrt"]),
        "sin": jax_eval(arb_core.arb_sin, xs["sin"]),
        "cos": jax_eval(arb_core.arb_cos, xs["cos"]),
        "tan": jax_eval(arb_core.arb_tan, xs["tan"]),
        "sinh": jax_eval(arb_core.arb_sinh, xs["sinh"]),
        "cosh": jax_eval(arb_core.arb_cosh, xs["cosh"]),
        "tanh": jax_eval(arb_core.arb_tanh, xs["tanh"]),
        "gamma": jax_eval(hypgeom.arb_hypgeom_gamma, xs["gamma"]),
        "erf": jax_eval(hypgeom.arb_hypgeom_erf, xs["erf"]),
        "erfc": jax_eval(hypgeom.arb_hypgeom_erfc, xs["erfc"]),
        "erfi": jax_eval(hypgeom.arb_hypgeom_erfi, xs["erfi"]),
        "barnesg": jax_eval(hypgeom.arb_hypgeom_barnesg, xs["barnesg"]),
    }

    rig = {
        "exp": jax_eval(ball_wrappers.arb_ball_exp, xs["exp"]),
        "log": jax_eval(ball_wrappers.arb_ball_log, xs["log"]),
        "sin": jax_eval(ball_wrappers.arb_ball_sin, xs["sin"]),
        "gamma": jax_eval(ball_wrappers.arb_ball_gamma, xs["gamma"]),
        "barnesg": jax_eval(ball_wrappers.arb_ball_barnesg, xs["barnesg"]),
    }
    adapt = {
        "exp": jax_eval(ball_wrappers.arb_ball_exp_adaptive, xs["exp"]),
        "log": jax_eval(ball_wrappers.arb_ball_log_adaptive, xs["log"]),
        "sin": jax_eval(ball_wrappers.arb_ball_sin_adaptive, xs["sin"]),
        "gamma": jax_eval(ball_wrappers.arb_ball_gamma_adaptive, xs["gamma"]),
        "barnesg": jax_eval(ball_wrappers.arb_ball_barnesg_adaptive, xs["barnesg"]),
    }

    def mp_eval(fn, xs):
        out = []
        for a, b in xs:
            m = mp.mpf(0.5 * (a + b))
            v = fn(m)
            out.append(np.array([float(v), float(v)], dtype=np.float64))
        return np.array(out)

    mp_map = {
        "exp": mp.exp,
        "log": mp.log,
        "sqrt": mp.sqrt,
        "sin": mp.sin,
        "cos": mp.cos,
        "tan": mp.tan,
        "sinh": mp.sinh,
        "cosh": mp.cosh,
        "tanh": mp.tanh,
        "gamma": mp.gamma,
        "erf": mp.erf,
        "erfc": mp.erfc,
        "erfi": mp.erfi,
        "barnesg": mp.barnesg,
    }
    mp_vals = {k: mp_eval(fn, xs[k]) for k, fn in mp_map.items()}

    print("mpmath midpoint widths (avg):")
    for name in mp_vals:
        w = np.mean([_width(v) for v in mp_vals[name]])
        print(f"{name:6s} width={w:.3e}")

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

            c = {
                "exp": c_eval(core.arb_exp_ref, xs["exp"]),
                "log": c_eval(core.arb_log_ref, xs["log"]),
                "sqrt": c_eval(core.arb_sqrt_ref, xs["sqrt"]),
                "sin": c_eval(core.arb_sin_ref, xs["sin"]),
                "cos": c_eval(core.arb_cos_ref, xs["cos"]),
                "tan": c_eval(core.arb_tan_ref, xs["tan"]),
                "sinh": c_eval(core.arb_sinh_ref, xs["sinh"]),
                "cosh": c_eval(core.arb_cosh_ref, xs["cosh"]),
                "tanh": c_eval(core.arb_tanh_ref, xs["tanh"]),
                "gamma": c_eval(hyp.arb_hypgeom_gamma_ref, xs["gamma"]),
                "erf": c_eval(hyp.arb_hypgeom_erf_ref, xs["erf"]),
                "erfc": c_eval(hyp.arb_hypgeom_erfc_ref, xs["erfc"]),
                "erfi": c_eval(hyp.arb_hypgeom_erfi_ref, xs["erfi"]),
            }

            print("\nContainment vs C (fraction):")
            print("\nContainment vs C (fraction):")
            for name in c:
                base_c = np.mean([_contains(b, c0) for b, c0 in zip(base[name], c[name])])
                mp_c = np.mean([_point_in_interval(float(m[0]), c0) for m, c0 in zip(mp_vals[name], c[name])])
                if name in rig:
                    rig_c = np.mean([_contains(b, c0) for b, c0 in zip(rig[name], c[name])])
                    ad_c = np.mean([_contains(b, c0) for b, c0 in zip(adapt[name], c[name])])
                    print(f"{name:6s} base={base_c:.3f} rig={rig_c:.3f} adapt={ad_c:.3f} mpmath_point={mp_c:.3f}")
                else:
                    print(f"{name:6s} base={base_c:.3f} mpmath_point={mp_c:.3f}")
        else:
            print("C reference libraries not found under arb repo. Skipping C compare.")
    else:
        print("No arb repo path set. Skipping C compare.")

    # Complex midpoint error vs JAX midpoint
    cmid_re = rng.uniform(-2.0, 2.0, size=n)
    cmid_im = rng.uniform(-2.0, 2.0, size=n)
    cboxes = np.stack([cmid_re - rad, cmid_re + rad, cmid_im - rad, cmid_im + rad], axis=-1)

    def jax_eval_box(fn, xs):
        return np.array([fn(jnp.array(x, dtype=jnp.float64)) for x in xs])

    base_c_exp = jax_eval_box(acb_core.acb_exp, cboxes)
    base_c_log = jax_eval_box(acb_core.acb_log, cboxes)
    base_c_sin = jax_eval_box(acb_core.acb_sin, cboxes)

    def mp_eval_c(fn, xs):
        out = []
        for x in xs:
            m = mp.mpf(0.5 * (x[0] + x[1])) + 1j * mp.mpf(0.5 * (x[2] + x[3]))
            v = fn(m)
            out.append(complex(v))
        return np.array(out, dtype=np.complex128)

    mp_c_exp = mp_eval_c(mp.exp, cboxes)
    mp_c_log = mp_eval_c(mp.log, cboxes)
    mp_c_sin = mp_eval_c(mp.sin, cboxes)

    def midpoint_from_box(box):
        return 0.5 * (box[0] + box[1]) + 1j * 0.5 * (box[2] + box[3])

    mid_exp = np.array([midpoint_from_box(b) for b in base_c_exp])
    mid_log = np.array([midpoint_from_box(b) for b in base_c_log])
    mid_sin = np.array([midpoint_from_box(b) for b in base_c_sin])

    print("\nComplex midpoint error vs JAX midpoint (avg |delta|):")
    for name, jv, mv in [
        ("exp", mid_exp, mp_c_exp),
        ("log", mid_log, mp_c_log),
        ("sin", mid_sin, mp_c_sin),
    ]:
        err = np.mean(np.abs(jv - mv))
        print(f"{name:6s} err={err:.3e}")


if __name__ == "__main__":
    main()
