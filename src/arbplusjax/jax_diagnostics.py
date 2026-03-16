from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import jax


@dataclass(frozen=True)
class JaxDiagnosticsConfig:
    enabled: bool = False
    capture_jaxpr: bool = False
    capture_hlo: bool = False
    trace_execution: bool = False
    trace_dir: str | None = None


def config_from_env(prefix: str = "ARBPLUSJAX_JAX_DIAGNOSTICS_") -> JaxDiagnosticsConfig:
    enabled = os.getenv(f"{prefix}ENABLED", "").lower() in {"1", "true", "yes", "on"}
    capture_jaxpr = os.getenv(f"{prefix}JAXPR", "").lower() in {"1", "true", "yes", "on"}
    capture_hlo = os.getenv(f"{prefix}HLO", "").lower() in {"1", "true", "yes", "on"}
    trace_execution = os.getenv(f"{prefix}TRACE", "").lower() in {"1", "true", "yes", "on"}
    trace_dir = os.getenv(f"{prefix}TRACE_DIR", "") or None
    return JaxDiagnosticsConfig(
        enabled=enabled,
        capture_jaxpr=capture_jaxpr,
        capture_hlo=capture_hlo,
        trace_execution=trace_execution,
        trace_dir=trace_dir,
    )


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _block(x: Any) -> Any:
    if isinstance(x, tuple):
        for item in x:
            jax.block_until_ready(item)
        return x
    return jax.block_until_ready(x)


@contextmanager
def maybe_trace(name: str, config: JaxDiagnosticsConfig | None = None):
    cfg = config or config_from_env()
    if not (cfg.enabled and cfg.trace_execution):
        yield None
        return
    trace_root = Path(cfg.trace_dir) if cfg.trace_dir else Path(tempfile.gettempdir()) / "arbplusjax_jax_traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_path = trace_root / name
    with jax.profiler.trace(str(trace_path), create_perfetto_link=False):
        yield trace_path


def collect_compilation_artifacts(
    fn: Callable[..., Any],
    *args: Any,
    name: str = "jax_fn",
    config: JaxDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    cfg = config or config_from_env()
    out: dict[str, Any] = {"name": name, "enabled": cfg.enabled}
    if not cfg.enabled:
        return out
    if cfg.capture_jaxpr:
        try:
            out["jaxpr"] = str(jax.make_jaxpr(fn)(*args))
        except Exception as exc:
            out["jaxpr_error"] = f"{type(exc).__name__}: {exc}"
    if cfg.capture_hlo:
        try:
            lowered = jax.jit(fn).lower(*args)
            out["hlo"] = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        except Exception as exc:
            out["hlo_error"] = f"{type(exc).__name__}: {exc}"
    return out


def profile_jitted_function(
    fn: Callable[[Any], Any],
    arg: Any,
    alt_arg: Any,
    *,
    repeats: int = 8,
    name: str = "jax_fn",
    config: JaxDiagnosticsConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    cfg = config or config_from_env()
    diagnostics: dict[str, Any] = {
        "name": name,
        "diagnostics_enabled": cfg.enabled,
        "compile_ms": 0.0,
        "steady_ms_median": 0.0,
        "steady_ms_p95": 0.0,
        "recompile_new_shape_ms": 0.0,
        "peak_rss_delta_mb": 0.0,
    }
    rss0 = peak_rss_mb()
    with maybe_trace(f"{name}_compile", cfg):
        t0 = time.perf_counter()
        out = _block(fn(arg))
        t1 = time.perf_counter()
    rss1 = peak_rss_mb()
    diagnostics["compile_ms"] = (t1 - t0) * 1e3
    diagnostics["peak_rss_delta_mb"] = max(rss1 - rss0, 0.0)
    times: list[float] = []
    with maybe_trace(f"{name}_steady", cfg):
        for _ in range(repeats):
            s0 = time.perf_counter()
            _block(fn(arg))
            times.append((time.perf_counter() - s0) * 1e3)
    diagnostics["steady_ms_median"] = float(sorted(times)[len(times) // 2]) if times else 0.0
    if times:
        times_sorted = sorted(times)
        diagnostics["steady_ms_p95"] = float(times_sorted[min(len(times_sorted) - 1, max(0, math.ceil(0.95 * len(times_sorted)) - 1))])
    with maybe_trace(f"{name}_recompile", cfg):
        r0 = time.perf_counter()
        _block(fn(alt_arg))
        r1 = time.perf_counter()
    diagnostics["recompile_new_shape_ms"] = (r1 - r0) * 1e3
    diagnostics.update(collect_compilation_artifacts(fn, arg, name=name, config=cfg))
    return out, diagnostics


__all__ = [
    "JaxDiagnosticsConfig",
    "collect_compilation_artifacts",
    "config_from_env",
    "maybe_trace",
    "peak_rss_mb",
    "profile_jitted_function",
]
