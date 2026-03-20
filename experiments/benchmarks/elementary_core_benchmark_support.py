from __future__ import annotations

from functools import lru_cache
import json
import platform
import math
import os
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import sys

import jax
import jax.numpy as jnp
import numpy as np
try:
    import pandas as pd
except Exception:  # pragma: no cover - optional notebook dependency
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from tools.reference_backends import apply_reference_env
from tools.reference_backends import boost_root
from tools.reference_backends import reference_prefix
from tools.reference_backends import flint_root
from tools.reference_backends import wolfram_linux_dir

apply_reference_env(REPO_ROOT)

from arbplusjax import arb_core
from arbplusjax import elementary
from arbplusjax import jax_diagnostics
from arbplusjax import point_wrappers
from benchmarks import bench_harness

jax.config.update("jax_enable_x64", True)

OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "elementary_core"


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    detail: str


@dataclass(frozen=True)
class CoreCase:
    name: str
    arb_name: str
    domain: tuple[float, float]
    scipy_expr: str | None = None
    mathematica_expr: str | None = None
    c_ref_name: str | None = None


@dataclass(frozen=True)
class ElementaryCase:
    name: str
    kind: str
    domain_a: tuple[float, float]
    domain_b: tuple[float, float] | None = None
    length: int = 16
    scipy_expr: str | None = None


CORE_CASES: tuple[CoreCase, ...] = (
    CoreCase("exp", "arb_exp_batch_jit", (-20.0, 20.0), "exp", "Exp[#]&", "arb_exp_ref"),
    CoreCase("log", "arb_log_batch_jit", (1e-6, 20.0), "log", "Log[#]&", "arb_log_ref"),
    CoreCase("sqrt", "arb_sqrt_batch_jit", (0.0, 100.0), "sqrt", "Sqrt[#]&", "arb_sqrt_ref"),
    CoreCase("sin", "arb_sin_batch_jit", (-20.0, 20.0), "sin", "Sin[#]&", "arb_sin_ref"),
    CoreCase("cos", "arb_cos_batch_jit", (-20.0, 20.0), "cos", "Cos[#]&", "arb_cos_ref"),
    CoreCase("tan", "arb_tan_batch_jit", (-1.2, 1.2), "tan", "Tan[#]&", "arb_tan_ref"),
    CoreCase("sinh", "arb_sinh_batch_jit", (-10.0, 10.0), "sinh", "Sinh[#]&", "arb_sinh_ref"),
    CoreCase("cosh", "arb_cosh_batch_jit", (-10.0, 10.0), "cosh", "Cosh[#]&", "arb_cosh_ref"),
    CoreCase("tanh", "arb_tanh_batch_jit", (-10.0, 10.0), "tanh", "Tanh[#]&", "arb_tanh_ref"),
    CoreCase("log1p", "arb_log1p_batch_jit", (-0.99, 20.0), "log1p", "Log[1 + #]&", "arb_log1p_ref"),
    CoreCase("expm1", "arb_expm1_batch_jit", (-20.0, 20.0), "expm1", "(Exp[#] - 1)&", "arb_expm1_ref"),
    CoreCase("sin_pi", "arb_sin_pi_batch_jit", (-4.0, 4.0), None, "Sin[Pi #]&", None),
    CoreCase("cos_pi", "arb_cos_pi_batch_jit", (-4.0, 4.0), None, "Cos[Pi #]&", None),
    CoreCase("tan_pi", "arb_tan_pi_batch_jit", (-0.45, 0.45), None, "Tan[Pi #]&", None),
    CoreCase("sinc", "arb_sinc_batch_jit", (-20.0, 20.0), None, "Sinc[#/Pi]&", None),
    CoreCase("sinc_pi", "arb_sinc_pi_batch_jit", (-20.0, 20.0), None, "Sinc[#]&", None),
)


ELEMENTARY_CASES: tuple[ElementaryCase, ...] = (
    ElementaryCase("logaddexp", "binary_real", (-20.0, 20.0), (-20.0, 20.0), scipy_expr="logaddexp"),
    ElementaryCase("logsubexp", "binary_real", (-20.0, 20.0), (-40.0, 0.0), scipy_expr=None),
    ElementaryCase("logsumexp", "vector_real", (-20.0, 20.0), length=32, scipy_expr="logsumexp"),
    ElementaryCase("log1mexp", "unary_real", (-20.0, -1e-6), scipy_expr=None),
    ElementaryCase("logexpm1", "unary_real", (1e-6, 20.0), scipy_expr=None),
    ElementaryCase("log_abs", "unary_complex", (-4.0, 4.0), scipy_expr=None),
    ElementaryCase("log_pow_abs", "binary_real", (1e-6, 20.0), (-3.0, 3.0), scipy_expr=None),
    ElementaryCase("x_pow_a", "binary_real", (1e-6, 20.0), (-3.0, 3.0), scipy_expr=None),
    ElementaryCase("cis", "unary_real", (-20.0, 20.0), scipy_expr=None),
    ElementaryCase("sinc", "unary_real", (-20.0, 20.0), scipy_expr=None),
    ElementaryCase("sinc_pi", "unary_real", (-20.0, 20.0), scipy_expr=None),
    ElementaryCase("sin_pi", "unary_real", (-4.0, 4.0), scipy_expr=None),
    ElementaryCase("cos_pi", "unary_real", (-4.0, 4.0), scipy_expr=None),
    ElementaryCase("tan_pi", "unary_real", (-0.45, 0.45), scipy_expr=None),
    ElementaryCase("exp_pi_i", "unary_real", (-4.0, 4.0), scipy_expr=None),
    ElementaryCase("log_sin_pi", "unary_real", (0.05, 0.95), scipy_expr=None),
    ElementaryCase("clog", "unary_complex", (-4.0, 4.0), scipy_expr=None),
    ElementaryCase("cpow", "binary_complex", (-3.0, 3.0), (-2.0, 2.0), scipy_expr=None),
    ElementaryCase("z_to_minus_s", "binary_complex", (-3.0, 3.0), (-2.0, 2.0), scipy_expr=None),
)


def _intervalize(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    return np.stack([xx, xx], axis=-1)


def _midpoint(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x[..., 0] + x[..., 1])


def _stats(values: np.ndarray) -> dict[str, float]:
    vv = np.asarray(values, dtype=np.float64).reshape(-1)
    if vv.size == 0:
        return {"max_abs": 0.0, "median_abs": 0.0, "max_rel": 0.0}
    return {
        "max_abs": float(np.max(np.abs(vv))),
        "median_abs": float(np.median(np.abs(vv))),
        "max_rel": float(np.max(np.abs(vv))),
    }


def _error_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    aa = np.asarray(actual)
    rr = np.asarray(reference)
    abs_err = np.abs(aa - rr)
    denom = np.maximum(np.abs(rr), 1e-30)
    rel_err = abs_err / denom
    out = _stats(abs_err)
    out["max_rel"] = float(np.max(rel_err)) if rel_err.size else 0.0
    return out


def _format_math_number(x: complex) -> str:
    if isinstance(x, complex) or np.iscomplexobj(x):
        z = complex(x)
        return f"({z.real:.17g}+{z.imag:.17g} I)"
    return f"{float(x):.17g}"


def detect_backends() -> list[BackendStatus]:
    statuses: list[BackendStatus] = []
    flint = flint_root()
    statuses.append(
        BackendStatus(
            "flint_install",
            flint is not None,
            str(flint) if flint is not None else "FLINT source install not found",
        )
    )
    statuses.append(BackendStatus("arb_flint_c_ref", _load_c_refs() is not None, "ctypes C reference loader"))
    boost_ok, boost_detail = _boost_adapter_status()
    statuses.append(BackendStatus("boost", boost_ok, boost_detail))
    statuses.append(BackendStatus("mpmath", _module_available("mpmath"), "python module"))
    statuses.append(BackendStatus("scipy", _module_available("scipy"), "python module"))
    statuses.append(BackendStatus("mathematica", _wolfram_available(), _wolfram_detail()))
    return statuses


def _make_table(rows: list[dict[str, Any]]):
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def _table_rows(table: Any) -> list[dict[str, Any]]:
    if pd is not None and hasattr(table, "to_dict"):
        return table.to_dict(orient="records")
    return list(table)


def render_table(table: Any, *, max_rows: int = 20) -> Any:
    if pd is not None:
        return table
    rows = _table_rows(table)[:max_rows]
    if not rows:
        return ""
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    rule = "-|-".join("-" * widths[c] for c in cols)
    body = [" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([header, rule, *body])


def backend_status_frame():
    return _make_table([{"backend": s.name, "available": s.available, "detail": s.detail} for s in detect_backends()])


def backend_resolution_frame():
    rows = [
        {"item": "repo_root", "value": str(REPO_ROOT)},
        {"item": "output_dir", "value": str(OUTPUT_DIR)},
        {"item": "ref_prefix", "value": str(reference_prefix())},
        {"item": "flint_root", "value": str(flint_root()) if flint_root() is not None else ""},
        {"item": "boost_root", "value": str(boost_root()) if boost_root() is not None else ""},
        {"item": "boost_ref_cmd", "value": os.getenv("BOOST_REF_CMD", "")},
        {"item": "wolfram_linux_dir", "value": str(wolfram_linux_dir()) if wolfram_linux_dir() is not None else ""},
        {"item": "arb_c_ref_dir", "value": str(bench_harness._auto_detect_c_ref_dir(REPO_ROOT) or "")},
    ]
    return _make_table(rows)


def runtime_version_frame():
    rows: list[dict[str, Any]] = [
        {"component": "python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
        {"component": "machine", "version": platform.machine()},
    ]
    try:
        import importlib.metadata as metadata

        for pkg in ("jax", "jaxlib", "numpy", "scipy", "mpmath", "pandas"):
            try:
                version = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                version = ""
            rows.append({"component": pkg, "version": version})
    except Exception:
        pass

    try:
        rows.append({"component": "jax_default_backend", "version": jax.default_backend()})
        rows.append({"component": "jax_devices", "version": ", ".join(d.platform for d in jax.devices())})
    except Exception:
        pass

    if _wolfram_available():
        rows.append({"component": "mathematica", "version": _wolfram_detail()})

    return _make_table(rows)


@lru_cache(maxsize=None)
def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _wolfram_available() -> bool:
    exe = shutil.which("wolframscript")
    if exe is None:
        return False
    try:
        proc = subprocess.run([exe, "-code", "$Version"], capture_output=True, text=True, timeout=15)
    except Exception:
        return False
    return proc.returncode == 0


@lru_cache(maxsize=1)
def _wolfram_detail() -> str:
    exe = shutil.which("wolframscript")
    if exe is None:
        return "wolframscript not found"
    try:
        proc = subprocess.run([exe, "-code", "$Version"], capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return f"wolframscript error: {type(exc).__name__}"
    if proc.returncode == 0:
        return proc.stdout.strip() or "activated"
    msg = (proc.stdout + proc.stderr).strip().splitlines()
    return msg[0] if msg else "wolframscript unavailable"


def _load_c_refs() -> dict[str, Any] | None:
    build_dir = bench_harness._auto_detect_c_ref_dir(REPO_ROOT)
    if build_dir is None:
        return None
    libs = bench_harness._load_c_libs(build_dir)
    return libs or None


@lru_cache(maxsize=1)
def _boost_adapter_status() -> tuple[bool, str]:
    cmd = os.getenv("BOOST_REF_CMD", "").strip()
    if not cmd:
        return False, "BOOST_REF_CMD not configured"
    y, note = bench_harness._boost_eval(cmd, "exp", np.asarray([0.0, 1.0], dtype=np.float64))
    if y is None:
        return False, note or "boost adapter unavailable"
    return True, cmd


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_real(domain: tuple[float, float], size: int, seed: int) -> np.ndarray:
    lo, hi = domain
    return _rng(seed).uniform(lo, hi, size=size).astype(np.float64)


def _sample_complex(domain: tuple[float, float], size: int, seed: int) -> np.ndarray:
    rr = _sample_real(domain, size, seed)
    ii = _sample_real(domain, size, seed + 17)
    return rr + 1j * ii


def _block(x: Any) -> Any:
    if isinstance(x, tuple):
        for item in x:
            jax.block_until_ready(item)
        return x
    return jax.block_until_ready(x)


def _profile_python_call(fn: Callable[[Any], Any], arg: Any, repeats: int) -> tuple[Any, dict[str, float]]:
    rss0 = jax_diagnostics.peak_rss_mb()
    t0 = time.perf_counter()
    out = fn(arg)
    t1 = time.perf_counter()
    rss1 = jax_diagnostics.peak_rss_mb()
    times: list[float] = []
    for _ in range(repeats):
        s0 = time.perf_counter()
        fn(arg)
        times.append((time.perf_counter() - s0) * 1e3)
    return out, {
        "compile_ms": 0.0,
        "steady_ms_median": float(np.median(times)) if times else (t1 - t0) * 1e3,
        "steady_ms_p95": float(np.percentile(times, 95)) if times else (t1 - t0) * 1e3,
        "recompile_new_shape_ms": 0.0,
        "peak_rss_delta_mb": max(rss1 - rss0, 0.0),
    }


def _run_wolfram_real(expr_template: str, xs: np.ndarray) -> np.ndarray:
    exe = shutil.which("wolframscript")
    if exe is None:
        raise RuntimeError("wolframscript not found")
    x_list = ",".join(_format_math_number(x) for x in xs.tolist())
    code = f'ExportString[N[Map[{expr_template}, {{{x_list}}}], 50], "JSON"]'
    proc = subprocess.run([exe, "-code", code], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout + proc.stderr).strip())
    return np.asarray(json.loads(proc.stdout), dtype=np.float64)


def _core_reference_values(case: CoreCase, xs: np.ndarray, accuracy_sample: int) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    sample = xs[:accuracy_sample]
    if case.scipy_expr is not None:
        import scipy.special

        if hasattr(np, case.scipy_expr):
            refs["scipy"] = np.asarray(getattr(np, case.scipy_expr)(sample))
        elif hasattr(scipy.special, case.scipy_expr):
            refs["scipy"] = np.asarray(getattr(scipy.special, case.scipy_expr)(sample))
    if case.name in {"sin_pi", "cos_pi", "tan_pi", "sinc", "sinc_pi"}:
        pi_sample = np.pi * sample
        if case.name == "sin_pi":
            refs["scipy"] = np.sin(pi_sample)
        elif case.name == "cos_pi":
            refs["scipy"] = np.cos(pi_sample)
        elif case.name == "tan_pi":
            refs["scipy"] = np.tan(pi_sample)
        elif case.name == "sinc":
            refs["scipy"] = np.sinc(sample / np.pi)
        elif case.name == "sinc_pi":
            refs["scipy"] = np.sinc(sample)
    if _module_available("mpmath"):
        import mpmath as mp

        mp.mp.dps = 80
        mapping = {
            "exp": mp.exp,
            "log": mp.log,
            "sqrt": mp.sqrt,
            "sin": mp.sin,
            "cos": mp.cos,
            "tan": mp.tan,
            "sinh": mp.sinh,
            "cosh": mp.cosh,
            "tanh": mp.tanh,
            "log1p": mp.log1p,
            "expm1": mp.expm1,
            "sin_pi": lambda x: mp.sin(mp.pi * x),
            "cos_pi": lambda x: mp.cos(mp.pi * x),
            "tan_pi": lambda x: mp.tan(mp.pi * x),
            "sinc": lambda x: mp.sin(x) / x if x != 0 else mp.mpf("1"),
            "sinc_pi": lambda x: mp.sin(mp.pi * x) / (mp.pi * x) if x != 0 else mp.mpf("1"),
        }
        refs["mpmath"] = np.asarray([float(mapping[case.name](float(x))) for x in sample], dtype=np.float64)
    if case.mathematica_expr is not None and _wolfram_available():
        try:
            refs["mathematica"] = _run_wolfram_real(case.mathematica_expr, sample)
        except Exception:
            pass
    return refs


def run_core_function_sweep(
    *,
    sample_size: int = 4096,
    repeats: int = 8,
    accuracy_sample: int = 128,
    diagnostics: jax_diagnostics.JaxDiagnosticsConfig | None = None,
) -> Any:
    c_refs = _load_c_refs() or {}
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(CORE_CASES):
        xs = _sample_real(case.domain, sample_size, seed=idx)
        x_iv = jnp.asarray(_intervalize(xs))
        alt_iv = x_iv[: max(8, sample_size // 2)]
        fn = getattr(arb_core, case.arb_name)
        out, timing = jax_diagnostics.profile_jitted_function(
            fn,
            x_iv,
            alt_iv,
            repeats=repeats,
            name=f"core_{case.name}",
            config=diagnostics,
        )
        out_np = np.asarray(out)
        midpoint = _midpoint(out_np)
        widths = out_np[:, 1] - out_np[:, 0]
        refs = _core_reference_values(case, xs, accuracy_sample)
        row: dict[str, Any] = {
            "suite": "core",
            "function": case.name,
            "sample_size": sample_size,
            "compile_ms": timing["compile_ms"],
            "steady_ms_median": timing["steady_ms_median"],
            "steady_ms_p95": timing["steady_ms_p95"],
            "recompile_new_shape_ms": timing["recompile_new_shape_ms"],
            "peak_rss_delta_mb": timing["peak_rss_delta_mb"],
            "input_mb": float(x_iv.size * x_iv.dtype.itemsize) / (1024.0 * 1024.0),
            "output_mb": float(out_np.size * out_np.dtype.itemsize) / (1024.0 * 1024.0),
            "mean_width": float(np.mean(widths)),
            "max_width": float(np.max(widths)),
            "available_refs": ",".join(sorted(refs)),
        }
        sample_mid = midpoint[:accuracy_sample]
        for ref_name, ref_values in refs.items():
            err = _error_stats(sample_mid, ref_values)
            row[f"{ref_name}_max_abs"] = err["max_abs"]
            row[f"{ref_name}_median_abs"] = err["median_abs"]
            row[f"{ref_name}_max_rel"] = err["max_rel"]
        if case.c_ref_name is not None and "arb_core_ref" in c_refs:
            try:
                c_out = bench_harness._call_c_unary(c_refs["arb_core_ref"], case.c_ref_name, np.asarray(x_iv))
                contained = (out_np[:accuracy_sample, 0] <= c_out[:accuracy_sample, 0]) & (
                    out_np[:accuracy_sample, 1] >= c_out[:accuracy_sample, 1]
                )
                row["arb_flint_interval_containment_rate"] = float(np.mean(contained))
                row["arb_flint_width_ratio_mean"] = float(
                    np.mean((out_np[:accuracy_sample, 1] - out_np[:accuracy_sample, 0]) / np.maximum(c_out[:accuracy_sample, 1] - c_out[:accuracy_sample, 0], 1e-30))
                )
            except Exception:
                row["arb_flint_interval_containment_rate"] = np.nan
                row["arb_flint_width_ratio_mean"] = np.nan
        else:
            row["arb_flint_interval_containment_rate"] = np.nan
            row["arb_flint_width_ratio_mean"] = np.nan
        rows.append(row)
    return _make_table(rows)


def _elementary_inputs(case: ElementaryCase, sample_size: int, seed: int) -> tuple[Any, Any]:
    if case.kind == "unary_real":
        x = _sample_real(case.domain_a, sample_size, seed)
        return x, x[: max(8, sample_size // 2)]
    if case.kind == "binary_real":
        a = _sample_real(case.domain_a, sample_size, seed)
        b = _sample_real(case.domain_b or case.domain_a, sample_size, seed + 1)
        if case.name == "logsubexp":
            b = np.minimum(a - 1e-6, b)
        if case.name == "log_pow_abs":
            a, b = b, a
        return (a, b), (a[: max(8, sample_size // 2)], b[: max(8, sample_size // 2)])
    if case.kind == "vector_real":
        v = _sample_real(case.domain_a, sample_size * case.length, seed).reshape(sample_size, case.length)
        return v, v[: max(8, sample_size // 2)]
    if case.kind == "unary_complex":
        z = _sample_complex(case.domain_a, sample_size, seed)
        return z, z[: max(8, sample_size // 2)]
    if case.kind == "binary_complex":
        z = _sample_complex(case.domain_a, sample_size, seed)
        a = _sample_complex(case.domain_b or case.domain_a, sample_size, seed + 1)
        return (z, a), (z[: max(8, sample_size // 2)], a[: max(8, sample_size // 2)])
    raise ValueError(f"unsupported case kind: {case.kind}")


def _elementary_callable(name: str) -> Callable[..., Any]:
    return getattr(elementary, name)


def _jit_wrap(case: ElementaryCase) -> Callable[[Any], Any]:
    fn = _elementary_callable(case.name)
    if case.kind == "unary_real":
        return jax.jit(lambda x: fn(jnp.asarray(x, dtype=jnp.float64)))
    if case.kind == "binary_real":
        return jax.jit(lambda ab: fn(jnp.asarray(ab[0], dtype=jnp.float64), jnp.asarray(ab[1], dtype=jnp.float64)))
    if case.kind == "vector_real":
        return jax.jit(lambda x: fn(jnp.asarray(x, dtype=jnp.float64), axis=-1))
    if case.kind == "unary_complex":
        return jax.jit(lambda z: fn(jnp.asarray(z, dtype=jnp.complex128)))
    if case.kind == "binary_complex":
        return jax.jit(lambda za: fn(jnp.asarray(za[0], dtype=jnp.complex128), jnp.asarray(za[1], dtype=jnp.complex128)))
    raise ValueError(f"unsupported case kind: {case.kind}")


def _elementary_reference(case: ElementaryCase, sample: Any) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    if case.kind == "unary_real":
        x = np.asarray(sample, dtype=np.float64)
        if case.name == "log1mexp":
            refs["numpy"] = np.where(x < -np.log(2.0), np.log1p(-np.exp(x)), np.log(-np.expm1(x)))
        elif case.name == "logexpm1":
            refs["numpy"] = np.where(x > np.log(2.0), x + np.log1p(-np.exp(-x)), np.log(np.expm1(x)))
        elif case.name == "sinc":
            refs["numpy"] = np.sinc(x / np.pi)
        elif case.name == "sinc_pi":
            refs["numpy"] = np.sinc(x)
        elif case.name == "sin_pi":
            refs["numpy"] = np.sin(np.pi * x)
        elif case.name == "cos_pi":
            refs["numpy"] = np.cos(np.pi * x)
        elif case.name == "tan_pi":
            refs["numpy"] = np.tan(np.pi * x)
        elif case.name == "exp_pi_i":
            refs["numpy"] = np.exp(1j * np.pi * x)
        elif case.name == "log_sin_pi":
            refs["numpy"] = np.log(np.sin(np.pi * x))
        elif case.scipy_expr is not None:
            import scipy.special

            if hasattr(np, case.scipy_expr):
                refs["numpy"] = getattr(np, case.scipy_expr)(x)
            elif hasattr(scipy.special, case.scipy_expr):
                refs["scipy"] = getattr(scipy.special, case.scipy_expr)(x)
    elif case.kind == "binary_real":
        a, b = sample
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if case.name == "logaddexp":
            refs["numpy"] = np.logaddexp(a, b)
        elif case.name == "logsubexp":
            refs["numpy"] = a + np.log1p(-np.exp(b - a))
        elif case.name == "log_pow_abs":
            refs["numpy"] = a * np.log(np.abs(b))
        elif case.name == "x_pow_a":
            refs["numpy"] = np.exp(b * np.log(a))
    elif case.kind == "vector_real":
        import scipy.special

        v = np.asarray(sample, dtype=np.float64)
        refs["scipy"] = scipy.special.logsumexp(v, axis=-1)
    elif case.kind == "unary_complex":
        z = np.asarray(sample, dtype=np.complex128)
        if case.name == "log_abs":
            refs["numpy"] = np.log(np.abs(z))
        elif case.name == "cis":
            refs["numpy"] = np.cos(z.real) + 1j * np.sin(z.real)
        elif case.name == "clog":
            refs["numpy"] = np.log(np.abs(z)) + 1j * np.angle(z)
    elif case.kind == "binary_complex":
        z, a = sample
        z = np.asarray(z, dtype=np.complex128)
        a = np.asarray(a, dtype=np.complex128)
        if case.name == "cpow":
            refs["numpy"] = np.exp(a * np.log(np.abs(z) + 0j) + 1j * a * np.angle(z))
        elif case.name == "z_to_minus_s":
            refs["numpy"] = np.exp(-a * (np.log(np.abs(z) + 0j) + 1j * np.angle(z)))
    if _module_available("mpmath"):
        import mpmath as mp

        mp.mp.dps = 80
        if case.kind == "unary_real":
            x = np.asarray(sample, dtype=np.float64)
            mapping = {
                "log1mexp": lambda t: mp.log(1 - mp.e**t),
                "logexpm1": lambda t: mp.log(mp.e**t - 1),
                "sinc": lambda t: mp.sin(t) / t if t != 0 else mp.mpf("1"),
                "sinc_pi": lambda t: mp.sin(mp.pi * t) / (mp.pi * t) if t != 0 else mp.mpf("1"),
                "sin_pi": lambda t: mp.sin(mp.pi * t),
                "cos_pi": lambda t: mp.cos(mp.pi * t),
                "tan_pi": lambda t: mp.tan(mp.pi * t),
                "exp_pi_i": lambda t: mp.e ** (1j * mp.pi * t),
                "log_sin_pi": lambda t: mp.log(mp.sin(mp.pi * t)),
            }
            if case.name in mapping:
                refs["mpmath"] = np.asarray([
                    complex(mapping[case.name](float(t) if not isinstance(t, complex) else t))
                    for t in x
                ], dtype=np.complex128 if case.name in {"exp_pi_i"} else np.float64)
    return refs


def run_elementary_function_sweep(
    *,
    sample_size: int = 2048,
    repeats: int = 8,
    accuracy_sample: int = 128,
    diagnostics: jax_diagnostics.JaxDiagnosticsConfig | None = None,
) -> Any:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(ELEMENTARY_CASES):
        arg, alt_arg = _elementary_inputs(case, sample_size, seed=100 + idx)
        jitted = _jit_wrap(case)
        out, timing = jax_diagnostics.profile_jitted_function(
            jitted,
            arg,
            alt_arg,
            repeats=repeats,
            name=f"elementary_{case.name}",
            config=diagnostics,
        )
        out_np = np.asarray(out)
        sample = out_np[:accuracy_sample]
        arg_sample = (
            arg[:accuracy_sample]
            if isinstance(arg, np.ndarray)
            else tuple(np.asarray(part)[:accuracy_sample] for part in arg)
        )
        refs = _elementary_reference(case, arg_sample)
        row: dict[str, Any] = {
            "suite": "elementary",
            "function": case.name,
            "kind": case.kind,
            "sample_size": sample_size,
            "compile_ms": timing["compile_ms"],
            "steady_ms_median": timing["steady_ms_median"],
            "steady_ms_p95": timing["steady_ms_p95"],
            "recompile_new_shape_ms": timing["recompile_new_shape_ms"],
            "peak_rss_delta_mb": timing["peak_rss_delta_mb"],
            "available_refs": ",".join(sorted(refs)),
        }
        for ref_name, ref_values in refs.items():
            err = _error_stats(sample, ref_values)
            row[f"{ref_name}_max_abs"] = err["max_abs"]
            row[f"{ref_name}_median_abs"] = err["median_abs"]
            row[f"{ref_name}_max_rel"] = err["max_rel"]
        rows.append(row)
    return _make_table(rows)


def run_core_mode_sweep(
    *,
    sample_size: int = 2048,
    repeats: int = 8,
    diagnostics: jax_diagnostics.JaxDiagnosticsConfig | None = None,
) -> Any:
    rows: list[dict[str, Any]] = []
    mode_fns = {
        "point": {
            "exp": point_wrappers.arb_exp_point,
            "log": point_wrappers.arb_log_point,
            "sqrt": point_wrappers.arb_sqrt_point,
            "sin": point_wrappers.arb_sin_point,
            "cos": point_wrappers.arb_cos_point,
            "tan": point_wrappers.arb_tan_point,
            "sinh": point_wrappers.arb_sinh_point,
            "cosh": point_wrappers.arb_cosh_point,
            "tanh": point_wrappers.arb_tanh_point,
            "log1p": point_wrappers.arb_log1p_point,
            "expm1": point_wrappers.arb_expm1_point,
            "sin_pi": point_wrappers.arb_sin_pi_point,
            "cos_pi": point_wrappers.arb_cos_pi_point,
            "tan_pi": point_wrappers.arb_tan_pi_point,
            "sinc": point_wrappers.arb_sinc_point,
            "sinc_pi": point_wrappers.arb_sinc_pi_point,
        },
        "basic": {
            "exp": arb_core.arb_exp_batch_jit,
            "log": arb_core.arb_log_batch_jit,
            "sqrt": arb_core.arb_sqrt_batch_jit,
            "sin": arb_core.arb_sin_batch_jit,
            "cos": arb_core.arb_cos_batch_jit,
            "tan": arb_core.arb_tan_batch_jit,
            "sinh": arb_core.arb_sinh_batch_jit,
            "cosh": arb_core.arb_cosh_batch_jit,
            "tanh": arb_core.arb_tanh_batch_jit,
            "log1p": arb_core.arb_log1p_batch_jit,
            "expm1": arb_core.arb_expm1_batch_jit,
            "sin_pi": arb_core.arb_sin_pi_batch_jit,
            "cos_pi": arb_core.arb_cos_pi_batch_jit,
            "tan_pi": arb_core.arb_tan_pi_batch_jit,
            "sinc": arb_core.arb_sinc_batch_jit,
            "sinc_pi": arb_core.arb_sinc_pi_batch_jit,
        },
        "adaptive": {
            "exp": lambda x: arb_core.arb_exp_batch_prec(x, prec_bits=106),
            "log": lambda x: arb_core.arb_log_batch_prec(x, prec_bits=106),
            "sqrt": lambda x: arb_core.arb_sqrt_batch_prec(x, prec_bits=106),
            "sin": lambda x: arb_core.arb_sin_batch_prec(x, prec_bits=106),
            "cos": lambda x: arb_core.arb_cos_batch_prec(x, prec_bits=106),
            "tan": lambda x: arb_core.arb_tan_batch_prec(x, prec_bits=106),
            "sinh": lambda x: arb_core.arb_sinh_batch_prec(x, prec_bits=106),
            "cosh": lambda x: arb_core.arb_cosh_batch_prec(x, prec_bits=106),
            "tanh": lambda x: arb_core.arb_tanh_batch_prec(x, prec_bits=106),
            "log1p": lambda x: arb_core.arb_log1p_batch_prec(x, prec_bits=106),
            "expm1": lambda x: arb_core.arb_expm1_batch_prec(x, prec_bits=106),
            "sin_pi": lambda x: arb_core.arb_sin_pi_batch_prec(x, prec_bits=106),
            "cos_pi": lambda x: arb_core.arb_cos_pi_batch_prec(x, prec_bits=106),
            "tan_pi": lambda x: arb_core.arb_tan_pi_batch_prec(x, prec_bits=106),
            "sinc": lambda x: arb_core.arb_sinc_batch_prec(x, prec_bits=106),
            "sinc_pi": lambda x: arb_core.arb_sinc_pi_batch_prec(x, prec_bits=106),
        },
        "rigorous": {
            "exp": lambda x: arb_core.arb_exp_batch_prec(x, prec_bits=212),
            "log": lambda x: arb_core.arb_log_batch_prec(x, prec_bits=212),
            "sqrt": lambda x: arb_core.arb_sqrt_batch_prec(x, prec_bits=212),
            "sin": lambda x: arb_core.arb_sin_batch_prec(x, prec_bits=212),
            "cos": lambda x: arb_core.arb_cos_batch_prec(x, prec_bits=212),
            "tan": lambda x: arb_core.arb_tan_batch_prec(x, prec_bits=212),
            "sinh": lambda x: arb_core.arb_sinh_batch_prec(x, prec_bits=212),
            "cosh": lambda x: arb_core.arb_cosh_batch_prec(x, prec_bits=212),
            "tanh": lambda x: arb_core.arb_tanh_batch_prec(x, prec_bits=212),
            "log1p": lambda x: arb_core.arb_log1p_batch_prec(x, prec_bits=212),
            "expm1": lambda x: arb_core.arb_expm1_batch_prec(x, prec_bits=212),
            "sin_pi": lambda x: arb_core.arb_sin_pi_batch_prec(x, prec_bits=212),
            "cos_pi": lambda x: arb_core.arb_cos_pi_batch_prec(x, prec_bits=212),
            "tan_pi": lambda x: arb_core.arb_tan_pi_batch_prec(x, prec_bits=212),
            "sinc": lambda x: arb_core.arb_sinc_batch_prec(x, prec_bits=212),
            "sinc_pi": lambda x: arb_core.arb_sinc_pi_batch_prec(x, prec_bits=212),
        },
    }
    for idx, case in enumerate(CORE_CASES):
        xs = _sample_real(case.domain, sample_size, seed=200 + idx)
        x_point = jnp.asarray(xs)
        x_iv = jnp.asarray(_intervalize(xs))
        for mode, mapping in mode_fns.items():
            fn = jax.jit(mapping[case.name]) if mode != "basic" else mapping[case.name]
            arg = x_point if mode == "point" else x_iv
            alt_arg = arg[: max(8, sample_size // 2)]
            out, timing = jax_diagnostics.profile_jitted_function(
                fn,
                arg,
                alt_arg,
                repeats=repeats,
                name=f"mode_{mode}_{case.name}",
                config=diagnostics,
            )
            arr = np.asarray(out)
            width_mean = 0.0 if mode == "point" else float(np.mean(arr[:, 1] - arr[:, 0]))
            rows.append(
                {
                    "function": case.name,
                    "mode": mode,
                    "sample_size": sample_size,
                    "compile_ms": timing["compile_ms"],
                    "steady_ms_median": timing["steady_ms_median"],
                    "recompile_new_shape_ms": timing["recompile_new_shape_ms"],
                    "peak_rss_delta_mb": timing["peak_rss_delta_mb"],
                    "mean_width": width_mean,
                }
            )
    return _make_table(rows)


def run_core_ad_sweep(
    *,
    sample_size: int = 256,
    repeats: int = 6,
    fd_eps: float = 1e-6,
    diagnostics: jax_diagnostics.JaxDiagnosticsConfig | None = None,
) -> Any:
    rows: list[dict[str, Any]] = []
    point_map = {
        "exp": point_wrappers.arb_exp_point,
        "log": point_wrappers.arb_log_point,
        "sqrt": point_wrappers.arb_sqrt_point,
        "sin": point_wrappers.arb_sin_point,
        "cos": point_wrappers.arb_cos_point,
        "tan": point_wrappers.arb_tan_point,
        "sinh": point_wrappers.arb_sinh_point,
        "cosh": point_wrappers.arb_cosh_point,
        "tanh": point_wrappers.arb_tanh_point,
        "log1p": point_wrappers.arb_log1p_point,
        "expm1": point_wrappers.arb_expm1_point,
        "sin_pi": point_wrappers.arb_sin_pi_point,
        "cos_pi": point_wrappers.arb_cos_pi_point,
        "tan_pi": point_wrappers.arb_tan_pi_point,
        "sinc": point_wrappers.arb_sinc_point,
        "sinc_pi": point_wrappers.arb_sinc_pi_point,
    }
    basic_map = {
        "exp": arb_core.arb_exp_batch_jit,
        "log": arb_core.arb_log_batch_jit,
        "sqrt": arb_core.arb_sqrt_batch_jit,
        "sin": arb_core.arb_sin_batch_jit,
        "cos": arb_core.arb_cos_batch_jit,
        "tan": arb_core.arb_tan_batch_jit,
        "sinh": arb_core.arb_sinh_batch_jit,
        "cosh": arb_core.arb_cosh_batch_jit,
        "tanh": arb_core.arb_tanh_batch_jit,
        "log1p": arb_core.arb_log1p_batch_jit,
        "expm1": arb_core.arb_expm1_batch_jit,
        "sin_pi": arb_core.arb_sin_pi_batch_jit,
        "cos_pi": arb_core.arb_cos_pi_batch_jit,
        "tan_pi": arb_core.arb_tan_pi_batch_jit,
        "sinc": arb_core.arb_sinc_batch_jit,
        "sinc_pi": arb_core.arb_sinc_pi_batch_jit,
    }
    adaptive_map = {name: jax.jit(lambda x, _fn=fn: _fn(x, prec_bits=106)) for name, fn in {
        "exp": arb_core.arb_exp_batch_prec,
        "log": arb_core.arb_log_batch_prec,
        "sqrt": arb_core.arb_sqrt_batch_prec,
        "sin": arb_core.arb_sin_batch_prec,
        "cos": arb_core.arb_cos_batch_prec,
        "tan": arb_core.arb_tan_batch_prec,
        "sinh": arb_core.arb_sinh_batch_prec,
        "cosh": arb_core.arb_cosh_batch_prec,
        "tanh": arb_core.arb_tanh_batch_prec,
        "log1p": arb_core.arb_log1p_batch_prec,
        "expm1": arb_core.arb_expm1_batch_prec,
        "sin_pi": arb_core.arb_sin_pi_batch_prec,
        "cos_pi": arb_core.arb_cos_pi_batch_prec,
        "tan_pi": arb_core.arb_tan_pi_batch_prec,
        "sinc": arb_core.arb_sinc_batch_prec,
        "sinc_pi": arb_core.arb_sinc_pi_batch_prec,
    }.items()}
    rigorous_map = {name: jax.jit(lambda x, _fn=fn: _fn(x, prec_bits=212)) for name, fn in {
        "exp": arb_core.arb_exp_batch_prec,
        "log": arb_core.arb_log_batch_prec,
        "sqrt": arb_core.arb_sqrt_batch_prec,
        "sin": arb_core.arb_sin_batch_prec,
        "cos": arb_core.arb_cos_batch_prec,
        "tan": arb_core.arb_tan_batch_prec,
        "sinh": arb_core.arb_sinh_batch_prec,
        "cosh": arb_core.arb_cosh_batch_prec,
        "tanh": arb_core.arb_tanh_batch_prec,
        "log1p": arb_core.arb_log1p_batch_prec,
        "expm1": arb_core.arb_expm1_batch_prec,
        "sin_pi": arb_core.arb_sin_pi_batch_prec,
        "cos_pi": arb_core.arb_cos_pi_batch_prec,
        "tan_pi": arb_core.arb_tan_pi_batch_prec,
        "sinc": arb_core.arb_sinc_batch_prec,
        "sinc_pi": arb_core.arb_sinc_pi_batch_prec,
    }.items()}
    mode_maps = {
        "point": point_map,
        "basic": basic_map,
        "adaptive": adaptive_map,
        "rigorous": rigorous_map,
    }

    for idx, case in enumerate(CORE_CASES):
        x = jnp.asarray(_sample_real(case.domain, sample_size, seed=500 + idx))
        direction = jnp.asarray(_sample_real((-1.0, 1.0), sample_size, seed=900 + idx))
        direction = direction / jnp.linalg.norm(direction)
        alt_x = x[: max(8, sample_size // 2)]
        for mode, mapping in mode_maps.items():
            base_fn = mapping[case.name]
            if mode == "point":
                def loss(v, fn=base_fn):
                    return jnp.sum(fn(v))
            else:
                def loss(v, fn=base_fn):
                    out = fn(jnp.stack([v, v], axis=-1))
                    return jnp.sum(0.5 * (out[..., 0] + out[..., 1]))
            grad_fn = jax.jit(jax.grad(loss))
            grad_out, timing = jax_diagnostics.profile_jitted_function(
                grad_fn,
                x,
                alt_x,
                repeats=repeats,
                name=f"ad_{mode}_{case.name}",
                config=diagnostics,
            )
            fd_forward = float(loss(x + fd_eps * direction))
            fd_backward = float(loss(x - fd_eps * direction))
            fd_dir = (fd_forward - fd_backward) / (2.0 * fd_eps)
            ad_dir = float(jnp.vdot(grad_out, direction).real)
            rows.append(
                {
                    "function": case.name,
                    "mode": mode,
                    "sample_size": sample_size,
                    "grad_compile_ms": timing["compile_ms"],
                    "grad_steady_ms_median": timing["steady_ms_median"],
                    "grad_recompile_new_shape_ms": timing["recompile_new_shape_ms"],
                    "peak_rss_delta_mb": timing["peak_rss_delta_mb"],
                    "directional_fd_abs_err": abs(ad_dir - fd_dir),
                    "directional_fd_rel_err": abs(ad_dir - fd_dir) / max(abs(fd_dir), 1e-12),
                }
            )
    return _make_table(rows)


def summarize_results(df: Any, sort_by: str = "steady_ms_median") -> Any:
    cols = [c for c in [
        "suite",
        "function",
        "kind",
        "sample_size",
        "compile_ms",
        "steady_ms_median",
        "recompile_new_shape_ms",
        "peak_rss_delta_mb",
        "mean_width",
        "max_width",
        "scipy_max_rel",
        "numpy_max_rel",
        "mpmath_max_rel",
        "mathematica_max_rel",
        "arb_flint_interval_containment_rate",
        "available_refs",
    ] if (pd is not None and c in df.columns) or (pd is None and any(c in r for r in _table_rows(df)))]
    if pd is not None:
        return df[cols].sort_values(sort_by, kind="stable").reset_index(drop=True)
    rows = _table_rows(df)
    rows = [{c: row.get(c, "") for c in cols} for row in rows]
    rows.sort(key=lambda row: row.get(sort_by, 0.0))
    return rows


def write_result_tables(name: str, df: Any) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{name}.csv"
    json_path = OUTPUT_DIR / f"{name}.json"
    rows = _table_rows(df)
    if pd is not None:
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
    else:
        import csv

        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(json.dumps(rows, indent=2))
    return {"csv": csv_path, "json": json_path}


__all__ = [
    "CORE_CASES",
    "ELEMENTARY_CASES",
    "OUTPUT_DIR",
    "backend_status_frame",
    "detect_backends",
    "render_table",
    "run_core_function_sweep",
    "run_core_ad_sweep",
    "run_core_mode_sweep",
    "run_elementary_function_sweep",
    "summarize_results",
    "write_result_tables",
]
