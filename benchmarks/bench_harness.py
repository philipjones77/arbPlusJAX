from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import platform

from tools.runtime_manifest import collect_runtime_manifest
from tools.runtime_manifest import write_runtime_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _maybe_add_repo(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if p.exists():
        sys.path.insert(0, str(p))


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def _auto_detect_c_ref_dir(repo_root: Path) -> Path | None:
    env = os.getenv("ARB_C_REF_DIR", "")
    if env:
        p = Path(env)
        if p.exists():
            return p

    candidates = [
        repo_root / "stuff" / "migration" / "c_chassis" / "build",
    ]

    parent = repo_root.parent
    for root in (parent / "flint", parent / "arb"):
        candidates.extend(
            [
                root / "build",
                root / "build" / "Release",
                root / "build" / "Debug",
                root / "out" / "build",
                root / "out" / "Release",
                root / "out" / "Debug",
            ]
        )

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_c_libs(c_ref_dir: Path) -> dict[str, ctypes.CDLL]:
    libs: dict[str, ctypes.CDLL] = {}
    if not c_ref_dir.exists():
        return libs

    sysname = platform.system().lower()
    if sysname.startswith("win"):
        exts = [".dll"]
    elif sysname == "darwin":
        exts = [".dylib", ".so"]
    else:
        exts = [".so", ".dylib"]

    def _names(base: str) -> list[str]:
        names: list[str] = []
        if base.startswith("lib"):
            roots = [base]
        else:
            roots = [base, f"lib{base}"]
        for root in roots:
            for ext in exts:
                names.append(f"{root}{ext}")
        # allow cross-OS builds to be referenced explicitly
        names.extend(
            [
                f"{base}.dll",
                f"lib{base}.dll",
                f"lib{base}.so",
                f"lib{base}.dylib",
            ]
        )
        return list(dict.fromkeys(names))

    di_lib = _find_lib(c_ref_dir, [
        *(_names("double_interval_ref")),
    ])
    if di_lib is not None:
        ctypes.CDLL(str(di_lib))

    core_lib = _find_lib(c_ref_dir, [
        *(_names("arb_core_ref")),
    ])
    hyp_lib = _find_lib(c_ref_dir, [
        *(_names("hypgeom_ref")),
    ])

    if core_lib is not None:
        core = ctypes.CDLL(str(core_lib))
        for fn_name in (
            "arb_exp_ref",
            "arb_log_ref",
            "arb_sqrt_ref",
            "arb_sin_ref",
            "arb_cos_ref",
            "arb_tan_ref",
            "arb_sinh_ref",
            "arb_cosh_ref",
            "arb_tanh_ref",
        ):
            fn = getattr(core, fn_name, None)
            if fn is not None:
                fn.argtypes = [DI]
                fn.restype = DI
        libs["arb_core_ref"] = core

    if hyp_lib is not None:
        hyp = ctypes.CDLL(str(hyp_lib))
        for fn_name in (
            "arb_hypgeom_gamma_ref",
            "arb_hypgeom_erf_ref",
            "arb_hypgeom_erfc_ref",
            "arb_hypgeom_erfi_ref",
        ):
            fn = getattr(hyp, fn_name, None)
            if fn is not None:
                fn.argtypes = [DI]
                fn.restype = DI
        for fn_name in (
            "arb_hypgeom_bessel_j_ref",
            "arb_hypgeom_bessel_y_ref",
            "arb_hypgeom_bessel_i_ref",
            "arb_hypgeom_bessel_k_ref",
        ):
            fn = getattr(hyp, fn_name, None)
            if fn is not None:
                fn.argtypes = [DI, DI]
                fn.restype = DI
        libs["hypgeom_ref"] = hyp

    return libs


def _call_c_unary(lib: ctypes.CDLL, fn_name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        r = fn(DI(float(x[i, 0]), float(x[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _call_c_bivariate(lib: ctypes.CDLL, fn_name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        r = fn(DI(float(a[i, 0]), float(a[i, 1])), DI(float(b[i, 0]), float(b[i, 1])))
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


def _interval_mid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x[:, 0] + x[:, 1])


def _interval_width(x: np.ndarray) -> np.ndarray:
    return x[:, 1] - x[:, 0]


def _contains(intervals: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (intervals[:, 0] <= points) & (points <= intervals[:, 1])


def _contains_interval(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return (outer[:, 0] <= inner[:, 0]) & (outer[:, 1] >= inner[:, 1])


def _pad_first_dim(x: np.ndarray, target: int) -> np.ndarray:
    n = x.shape[0]
    if target <= n:
        return x
    pad = np.repeat(x[-1:, ...], target - n, axis=0)
    return np.concatenate([x, pad], axis=0)


def _trim_first_dim(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n]


def _stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def _write_hist(path: Path, data: np.ndarray, bins: int = 25) -> None:
    if data.size == 0:
        return
    counts, edges = np.histogram(data, bins=bins)
    lines = ["bin_left,bin_right,count"]
    for i in range(len(counts)):
        lines.append(f"{edges[i]:.6g},{edges[i+1]:.6g},{int(counts[i])}")
    path.write_text("\n".join(lines))


def _resolve_scipy_fn(name: str):
    import scipy
    import scipy.special
    if hasattr(scipy.special, name):
        return getattr(scipy.special, name)
    if hasattr(np, name):
        return getattr(np, name)
    return None


def _resolve_jax_fn(name: str):
    import jax.numpy as jnp
    import jax.scipy.special as jsp
    if hasattr(jsp, name):
        return getattr(jsp, name)
    if hasattr(jnp, name):
        return getattr(jnp, name)
    return None


def _load_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as metadata
        for pkg in ("jax", "jaxlib", "numpy", "scipy", "mpmath"):
            try:
                versions[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                pass
    except Exception:
        pass
    return versions


def _find_wolframscript_windows(wolfram_dir: str | None) -> Path | None:
    if not wolfram_dir:
        return None
    cand = Path(wolfram_dir) / "wolframscript.exe"
    return cand if cand.exists() else None


def _find_wolframscript_linux(wolfram_dir: str | None) -> Path | None:
    if not wolfram_dir:
        return None
    cand = Path(wolfram_dir) / "wolframscript"
    return cand if cand.exists() else None


def _wolfram_local_eval(wolfram: Path, fn_name: str, xs: np.ndarray, log_path: Path | None) -> np.ndarray | None:
    import tempfile

    payload = {"fn": fn_name, "x": xs.tolist()}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as tmp:
            json.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        path_str = str(tmp_path).replace("\\", "\\\\")
        wl_code = f"""
data = Import["{path_str}", "RawJSON"];
fn = data["fn"];
xs = data["x"];
f = Which[
  fn == "exp", Exp,
  fn == "log", Log,
  fn == "sqrt", Sqrt,
  fn == "sin", Sin,
  fn == "cos", Cos,
  fn == "tan", Tan,
  fn == "sinh", Sinh,
  fn == "cosh", Cosh,
  fn == "tanh", Tanh,
  fn == "gamma", Gamma,
  fn == "erf", Erf,
  fn == "erfc", Erfc,
  True, $Failed
];
If[f === $Failed, Print["__FAIL__"], Print[ExportString[f /@ xs, "JSON"]]];
"""
    except Exception as exc:
        if log_path is not None:
            log_path.write_text(f"local_payload_error: {exc}\n")
        return None
    try:
        out = subprocess.check_output([str(wolfram), "-code", wl_code], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        if log_path is not None:
            log_path.write_text(f"local_error: {exc}\n")
        return None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    out = out.strip()
    if not out or "__FAIL__" in out:
        if log_path is not None:
            log_path.write_text(f"local_empty_or_fail: {out}\n")
        return None
    try:
        snippet = out
        if "[" in out and "]" in out:
            start = out.find("[")
            end = out.rfind("]")
            if end > start:
                snippet = out[start : end + 1]
        return np.asarray(json.loads(snippet), dtype=np.float64)
    except Exception:
        if log_path is not None:
            log_path.write_text(f"local_parse_error: {out}\n")
        return None


def _wolfram_cloud_eval(url: str, fn_name: str, xs: np.ndarray, api_key: str | None, log_path: Path | None) -> np.ndarray | None:
    import urllib.parse
    import urllib.request

    results = []
    for x in xs:
        params = {"fn": fn_name, "x": float(x)}
        if api_key:
            params["_key"] = api_key
        query = urllib.parse.urlencode(params)
        req_url = f"{url}?{query}"
        req = urllib.request.Request(req_url)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = resp.read().decode("utf-8")
            if "<!DOCTYPE" in payload or "<html" in payload.lower():
                if log_path is not None:
                    log_path.write_text(f"cloud_html_response: {payload[:500]}\n")
                return None
            try:
                val = json.loads(payload)
            except Exception:
                val = payload
            try:
                results.append(float(val))
            except Exception:
                if log_path is not None:
                    log_path.write_text(f"cloud_parse_error: {payload}\n")
                return None
        except Exception:
            if log_path is not None:
                log_path.write_text(f"cloud_request_error: {req_url}\n")
            return None
    return np.asarray(results, dtype=np.float64)


def _eval_jax_interval(name: str, *intervals: np.ndarray, mode: str, prec_bits: int):
    import jax.numpy as jnp
    from arbplusjax import api

    arrs = tuple(jnp.asarray(x) for x in intervals)
    return api.eval_interval(name, *arrs, mode=mode, prec_bits=prec_bits, dps=None)


def _eval_jax_interval_batch(name: str, *intervals: np.ndarray, mode: str, prec_bits: int):
    import jax.numpy as jnp
    from arbplusjax import api
    from arbplusjax import hypgeom

    arrs = tuple(jnp.asarray(x) for x in intervals)
    bessel_fixed_real = {
        "besselj": hypgeom.arb_hypgeom_bessel_j_batch_fixed_prec,
        "bessely": hypgeom.arb_hypgeom_bessel_y_batch_fixed_prec,
        "besseli": hypgeom.arb_hypgeom_bessel_i_batch_fixed_prec,
        "besselk": hypgeom.arb_hypgeom_bessel_k_batch_fixed_prec,
    }
    fn = bessel_fixed_real.get(name)
    if fn is not None and len(arrs) == 2 and mode == "basic":
        return fn(arrs[0], arrs[1], prec_bits=prec_bits, mode="sample")
    return api.eval_interval_batch(name, *arrs, mode=mode, prec_bits=prec_bits, dps=None)


def _eval_jax_point_batch(name: str, *xs: np.ndarray):
    import jax.numpy as jnp
    from arbplusjax import api

    arrs = tuple(jnp.asarray(x) for x in xs)
    return api.eval_point_batch(name, *arrs)


def _split_command(cmd: str) -> list[str]:
    if not cmd.strip():
        return []
    return shlex.split(cmd, posix=(platform.system().lower() != "windows"))


def _boost_eval(
    boost_cmd: str,
    fn_name: str,
    x: np.ndarray,
    nu: np.ndarray | None = None,
    z: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    argv = _split_command(boost_cmd)
    if not argv:
        return None, "boost_command_empty"
    payload: dict[str, Any] = {"function": fn_name, "x": x.tolist()}
    if nu is not None and z is not None:
        payload["nu"] = nu.tolist()
        payload["z"] = z.tolist()
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        return None, f"boost_launch_error: {exc}"
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        return None, f"boost_nonzero_exit({proc.returncode}): {stderr[:240]}"
    stdout = proc.stdout.strip()
    if not stdout:
        return None, "boost_empty_output"
    try:
        vals = json.loads(stdout)
        arr = np.asarray(vals, dtype=np.float64)
    except Exception as exc:
        return None, f"boost_parse_error: {exc}"
    if arr.ndim != 1:
        return None, "boost_output_must_be_1d_array"
    if arr.shape[0] != x.shape[0]:
        return None, f"boost_length_mismatch: got {arr.shape[0]} expected {x.shape[0]}"
    return arr, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend benchmark harness.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-samples", type=str, default="")
    parser.add_argument("--sweep-seeds", type=str, default="")
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--prec-bits", type=int, default=200)
    parser.add_argument("--c-ref-dir", type=str, default=os.getenv("ARB_C_REF_DIR", ""))
    parser.add_argument("--scipy-repo", type=str, default=os.getenv("SCIPY_REPO", ""))
    parser.add_argument("--jax-repo", type=str, default=os.getenv("JAX_REPO", ""))
    parser.add_argument("--mpmath-repo", type=str, default=os.getenv("MPMATH_REPO", ""))
    parser.add_argument(
        "--boost-ref-cmd",
        type=str,
        default=os.getenv("BOOST_REF_CMD", ""),
        help="Optional command for Boost baseline. Reads JSON from stdin and prints JSON array to stdout.",
    )
    parser.add_argument("--wolfram-cloud-url", type=str, default=os.getenv("WOLFRAM_CLOUD_URL", ""))
    parser.add_argument("--wolfram-windows-dir", type=str, default=os.getenv("WOLFRAM_WINDOWS_DIR", ""))
    parser.add_argument("--wolfram-linux-dir", type=str, default=os.getenv("WOLFRAM_LINUX_DIR", ""))
    parser.add_argument("--functions", type=str, default="")
    parser.add_argument("--jax-batch", action="store_true", help="JIT a single batched JAX call per function.")
    parser.add_argument("--jax-point-batch", action="store_true", help="JIT a batched point-evaluation JAX kernel per function.")
    parser.add_argument("--jax-warmup", action="store_true", help="Warm up JAX kernels before timing (excludes compile cost).")
    parser.add_argument(
        "--jax-dtype",
        type=str,
        choices=("float64", "float32"),
        default="float64",
        help="JAX input dtype for JAX backends.",
    )
    parser.add_argument(
        "--jax-fixed-batch-size",
        type=int,
        default=0,
        help="If > 0, pad JAX inputs to a fixed leading dimension to reduce recompiles across sweep sample sizes.",
    )
    parser.add_argument("--outdir", type=str, default="")
    args = parser.parse_args()

    _maybe_add_repo(args.scipy_repo)
    _maybe_add_repo(args.jax_repo)
    _maybe_add_repo(args.mpmath_repo)

    from benchmarks.bench_registry import FUNCTIONS, by_name

    fn_filter = None
    if args.functions:
        fn_filter = {name.strip() for name in args.functions.split(",") if name.strip()}

    specs = FUNCTIONS
    if fn_filter:
        lookup = by_name()
        specs = [lookup[name] for name in fn_filter if name in lookup]

    if not specs:
        print("No functions selected.")
        return 1

    sweep_samples = [int(args.samples)]
    if args.sweep_samples:
        sweep_samples = [int(x) for x in args.sweep_samples.split(",") if x.strip()]

    sweep_seeds = [int(args.seed)]
    if args.sweep_seeds:
        sweep_seeds = [int(x) for x in args.sweep_seeds.split(",") if x.strip()]

    base_outdir = Path(args.outdir) if args.outdir else None
    if base_outdir is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base_outdir = Path("results") / "benchmarks" / f"run_{stamp}"
    base_outdir.mkdir(parents=True, exist_ok=True)
    manifest = collect_runtime_manifest(REPO_ROOT, jax_mode=os.getenv("JAX_PLATFORMS", "auto"), python_path=sys.executable)
    write_runtime_manifest(base_outdir, manifest)

    if args.c_ref_dir:
        c_ref_dir = Path(args.c_ref_dir)
    else:
        c_ref_dir = _auto_detect_c_ref_dir(REPO_ROOT)
    c_libs = _load_c_libs(c_ref_dir) if c_ref_dir else {}

    import jax
    import jax.numpy as jnp
    jax_np_dtype = np.float32 if args.jax_dtype == "float32" else np.float64

    try:
        import mpmath as mp
    except Exception:
        mp = None

    wolfram_cloud_url = args.wolfram_cloud_url.strip()
    wolfram_cloud_key = os.getenv("WOLFRAM_CLOUD_API_KEY", "")
    wolfram_win = _find_wolframscript_windows(args.wolfram_windows_dir)
    wolfram_linux = _find_wolframscript_linux(args.wolfram_linux_dir)
    wolfram_local = wolfram_win or wolfram_linux

    sweep_runs: list[dict[str, Any]] = []

    for samples in sweep_samples:
        for seed in sweep_seeds:
            rng = np.random.default_rng(seed)
            run_outdir = base_outdir / f"samples_{samples}_seed_{seed}"
            run_outdir.mkdir(parents=True, exist_ok=True)

            results: list[dict[str, Any]] = []

            for spec in specs:
                is_bivariate = spec.name in {"besselj", "bessely", "besseli", "besselk", "CubesselK"}
                lo, hi = spec.range
                x = rng.uniform(lo, hi, size=samples).astype(np.float64)
                if spec.category == "real_positive":
                    x = np.clip(x, min(lo, hi), max(lo, hi))
                radius = float(spec.radius)
                intervals = np.stack([x - radius, x + radius], axis=-1)
                if is_bivariate:
                    nu = rng.uniform(0.1, 5.0, size=samples).astype(np.float64)
                    nu_intervals = np.stack([nu - radius, nu + radius], axis=-1)
                    # keep z positive to avoid complex-valued outputs for non-integer nu
                    z = np.abs(x)
                    z = np.clip(z, 1e-6, None)
                    z_intervals = np.stack([z - radius, z + radius], axis=-1)

                # Use a stable leading shape for JAX runs so compiled kernels can be reused across sweep sizes.
                jax_n = max(samples, args.jax_fixed_batch_size) if args.jax_fixed_batch_size > 0 else samples
                x_jax = _pad_first_dim(x.astype(jax_np_dtype, copy=False), jax_n)
                intervals_jax = _pad_first_dim(intervals.astype(jax_np_dtype, copy=False), jax_n)
                if is_bivariate:
                    nu_jax = _pad_first_dim(nu.astype(jax_np_dtype, copy=False), jax_n)
                    z_jax = _pad_first_dim(z.astype(jax_np_dtype, copy=False), jax_n)
                    nu_intervals_jax = _pad_first_dim(nu_intervals.astype(jax_np_dtype, copy=False), jax_n)
                    z_intervals_jax = _pad_first_dim(z_intervals.astype(jax_np_dtype, copy=False), jax_n)

                c_out = None
                if spec.c_lib and spec.c_fn and spec.c_lib in c_libs:
                    t0 = time.perf_counter()
                    if is_bivariate:
                        c_out = _call_c_bivariate(c_libs[spec.c_lib], spec.c_fn, nu_intervals, z_intervals)
                    else:
                        c_out = _call_c_unary(c_libs[spec.c_lib], spec.c_fn, intervals)
                    t1 = time.perf_counter()
                    results.append({
                        "function": spec.name,
                        "backend": "c_arb",
                        "samples": samples,
                        "mean_abs_err": "",
                        "max_abs_err": "",
                        "containment_rate": "",
                        "mean_width": float(np.mean(_interval_width(c_out))),
                        "time_ms": (t1 - t0) * 1e3,
                        "notes": "",
                    })

                jax_interval_out: dict[str, np.ndarray] = {}
                for mode, label in (("basic", "jax_basic"), ("adaptive", "jax_adaptive"), ("rigorous", "jax_rigorous")):
                    fn_name = getattr(spec, f"jax_{mode}")
                    if not fn_name:
                        continue
                    if args.jax_batch and args.jax_warmup:
                        if is_bivariate:
                            _ = _eval_jax_interval_batch(
                                spec.name, nu_intervals_jax, z_intervals_jax, mode=mode, prec_bits=args.prec_bits
                            )
                        else:
                            _ = _eval_jax_interval_batch(spec.name, intervals_jax, mode=mode, prec_bits=args.prec_bits)
                        jax.block_until_ready(_)
                    t0 = time.perf_counter()
                    if args.jax_batch:
                        if is_bivariate:
                            out = np.asarray(
                                _eval_jax_interval_batch(
                                    spec.name, nu_intervals_jax, z_intervals_jax, mode=mode, prec_bits=args.prec_bits
                                )
                            )
                        else:
                            out = np.asarray(_eval_jax_interval_batch(spec.name, intervals_jax, mode=mode, prec_bits=args.prec_bits))
                    else:
                        if is_bivariate:
                            out = np.asarray(
                                _eval_jax_interval(spec.name, nu_intervals_jax, z_intervals_jax, mode=mode, prec_bits=args.prec_bits)
                            )
                        else:
                            out = np.asarray(_eval_jax_interval(spec.name, intervals_jax, mode=mode, prec_bits=args.prec_bits))
                    t1 = time.perf_counter()
                    out = _trim_first_dim(out, samples)
                    jax_interval_out[label] = out
                    mean_width = float(np.mean(_interval_width(out)))
                    mean_abs_err = ""
                    max_abs_err = ""
                    containment_rate = ""
                    if c_out is not None:
                        c_mid = _interval_mid(c_out)
                        j_mid = _interval_mid(out)
                        err = np.abs(j_mid - c_mid)
                        mean_abs_err = float(np.mean(err))
                        max_abs_err = float(np.max(err))
                        containment_rate = float(np.mean(_contains_interval(c_out, out)))
                    results.append({
                        "function": spec.name,
                        "backend": label,
                        "samples": samples,
                        "mean_abs_err": mean_abs_err,
                        "max_abs_err": max_abs_err,
                        "containment_rate": containment_rate,
                        "mean_width": mean_width,
                        "time_ms": (t1 - t0) * 1e3,
                        "notes": "",
                    })

                if spec.scipy:
                    fn = _resolve_scipy_fn(spec.scipy)
                    if fn is not None:
                        t0 = time.perf_counter()
                        y = fn(nu, z) if is_bivariate else fn(x)
                        t1 = time.perf_counter()
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "scipy",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": "",
                        })

                if spec.jax_scipy:
                    fn = _resolve_jax_fn(spec.jax_scipy)
                    if fn is not None:
                        t0 = time.perf_counter()
                        if is_bivariate:
                            y = np.asarray(fn(jnp.asarray(nu_jax), jnp.asarray(z_jax)))
                        else:
                            y = np.asarray(fn(jnp.asarray(x_jax)))
                        t1 = time.perf_counter()
                        y = _trim_first_dim(y, samples)
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "jax_scipy",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": "",
                        })

                    if args.jax_point_batch and spec.jax_point:
                        try:
                            if args.jax_warmup:
                                if is_bivariate:
                                    _ = _eval_jax_point_batch(spec.name, nu_jax, z_jax)
                                else:
                                    _ = _eval_jax_point_batch(spec.name, x_jax)
                                jax.block_until_ready(_)
                            t0 = time.perf_counter()
                            if is_bivariate:
                                y = np.asarray(_eval_jax_point_batch(spec.name, nu_jax, z_jax))
                            else:
                                y = np.asarray(_eval_jax_point_batch(spec.name, x_jax))
                            t1 = time.perf_counter()
                            y = _trim_first_dim(y, samples)
                        except Exception as exc:
                            results.append({
                                "function": spec.name,
                                "backend": "jax_point",
                                "samples": samples,
                                "mean_abs_err": "",
                                "max_abs_err": "",
                                "containment_rate": "",
                                "mean_width": "",
                                "time_ms": "",
                                "notes": f"jax_point_unavailable: {exc}",
                            })
                            y = None
                            t1 = t0 = None
                        if y is None:
                            continue
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "jax_point",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": "",
                        })

                if spec.mpmath and mp is not None:
                    mp.mp.dps = args.dps
                    fn = getattr(mp, spec.mpmath, None)
                    if fn is not None:
                        t0 = time.perf_counter()
                        if is_bivariate:
                            y = np.asarray([float(fn(nu[i], z[i])) for i in range(samples)], dtype=np.float64)
                        else:
                            y = np.asarray([float(fn(val)) for val in x], dtype=np.float64)
                        t1 = time.perf_counter()
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "mpmath",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": "",
                        })

                if args.boost_ref_cmd:
                    t0 = time.perf_counter()
                    if is_bivariate:
                        y, boost_note = _boost_eval(args.boost_ref_cmd, spec.name, x, nu=nu, z=z)
                    else:
                        y, boost_note = _boost_eval(args.boost_ref_cmd, spec.name, x)
                    t1 = time.perf_counter()
                    if y is not None:
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "boost",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": "",
                        })
                    else:
                        results.append({
                            "function": spec.name,
                            "backend": "boost",
                            "samples": samples,
                            "mean_abs_err": "",
                            "max_abs_err": "",
                            "containment_rate": "",
                            "mean_width": "",
                            "time_ms": "",
                            "notes": boost_note,
                        })

                if wolfram_local is not None:
                    t0 = time.perf_counter()
                    log_path = run_outdir / f"mathematica_local_{spec.name}.log"
                    y = _wolfram_local_eval(wolfram_local, spec.name, x, log_path)
                    t1 = time.perf_counter()
                    if y is not None:
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "mathematica_local",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": str(wolfram_local),
                        })

                if wolfram_cloud_url:
                    t0 = time.perf_counter()
                    log_path = run_outdir / f"mathematica_cloud_{spec.name}.log"
                    y = _wolfram_cloud_eval(wolfram_cloud_url, spec.name, x, wolfram_cloud_key or None, log_path)
                    t1 = time.perf_counter()
                    if y is not None:
                        mean_abs_err = ""
                        max_abs_err = ""
                        containment_rate = ""
                        if c_out is not None:
                            c_mid = _interval_mid(c_out)
                            err = np.abs(y - c_mid)
                            mean_abs_err = float(np.mean(err))
                            max_abs_err = float(np.max(err))
                            containment_rate = float(np.mean(_contains(c_out, y)))
                        results.append({
                            "function": spec.name,
                            "backend": "mathematica_cloud",
                            "samples": samples,
                            "mean_abs_err": mean_abs_err,
                            "max_abs_err": max_abs_err,
                            "containment_rate": containment_rate,
                            "mean_width": "",
                            "time_ms": (t1 - t0) * 1e3,
                            "notes": "wolfram_cloud",
                        })

                if c_out is not None:
                    c_mid = _interval_mid(c_out)
                    for mode, label in (("basic", "jax_basic"), ("adaptive", "jax_adaptive"), ("rigorous", "jax_rigorous")):
                        fn_name = getattr(spec, f"jax_{mode}")
                        if not fn_name:
                            continue
                        out = jax_interval_out.get(label)
                        if out is None:
                            continue
                        j_mid = _interval_mid(out)
                        err = np.abs(j_mid - c_mid)
                        stats = _stats(err)
                        detail = {
                            "function": spec.name,
                            "backend": label,
                            "samples": samples,
                            "seed": seed,
                            "error_stats": stats,
                            "containment_rate": float(np.mean(_contains_interval(c_out, out))),
                            "mean_width": float(np.mean(_interval_width(out))),
                        }
                        detail_path = run_outdir / f"{spec.name}_{label}_detail.json"
                        detail_path.write_text(json.dumps(detail, indent=2))
                        _write_hist(run_outdir / f"{spec.name}_{label}_err_hist.csv", err)

            meta = {
                "samples": samples,
                "seed": seed,
                "dps": args.dps,
                "prec_bits": args.prec_bits,
                "jax_dtype": args.jax_dtype,
                "jax_fixed_batch_size": args.jax_fixed_batch_size,
                "c_ref_dir": str(c_ref_dir) if c_ref_dir else "",
                "boost_ref_cmd": args.boost_ref_cmd,
                "versions": _load_versions(),
            }

            (run_outdir / "summary.json").write_text(json.dumps({"meta": meta, "rows": results}, indent=2))

            if results:
                headers = list(results[0].keys())
                lines = [",".join(headers)]
                for row in results:
                    vals = []
                    for h in headers:
                        v = row[h]
                        if isinstance(v, float):
                            vals.append(f"{v:.6g}")
                        else:
                            vals.append(str(v))
                    lines.append(",".join(vals))
                (run_outdir / "summary.csv").write_text("\n".join(lines))

            (run_outdir / "README.md").write_text(
                "# Benchmark Run\n\n"
                "This folder contains benchmark outputs produced by benchmarks/bench_harness.py.\n"
            )

            sweep_runs.append({"samples": samples, "seed": seed, "path": str(run_outdir)})

    (base_outdir / "sweep_index.json").write_text(json.dumps(sweep_runs, indent=2))
    print(f"Wrote results to {base_outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
