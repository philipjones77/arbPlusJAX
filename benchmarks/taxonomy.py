from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BenchmarkScriptMetadata:
    script_name: str
    intent: str
    category: str
    default_device: str
    notes: str = ""


def _category_for_script(script_name: str) -> str:
    if "core_scalar" in script_name:
        return "scalar"
    if "matrix_service_api" in script_name:
        return "matrix"
    if "special_function_service_api" in script_name:
        return "special"
    if "matrix_backend" in script_name:
        return "backend_matrix"
    if "nufft_backends" in script_name:
        return "backend_transform"
    if "matrix_free" in script_name:
        return "matrix_free"
    if "dense_matrix_surface" in script_name:
        return "matrix_dense"
    if "sparse_matrix_surface" in script_name:
        return "matrix_sparse"
    if "block_sparse_matrix_surface" in script_name:
        return "matrix_block_sparse"
    if "vblock_sparse_matrix_surface" in script_name:
        return "matrix_vblock_sparse"
    if "matrix_stack" in script_name or "matrix_suite" in script_name:
        return "matrix"
    if "fft" in script_name or "nufft" in script_name:
        return "transform"
    if "calc" in script_name:
        return "integration"
    if "api_surface" in script_name:
        return "api"
    if "_mat" in script_name:
        return "matrix"
    if any(token in script_name for token in ("gamma", "hypgeom", "dirichlet", "elliptic", "modular", "barnes")):
        return "special"
    if any(token in script_name for token in ("bernoulli", "partitions", "dlog")):
        return "combinatorics"
    return "scalar"


def _intent_for_script(script_name: str) -> str:
    if script_name.startswith("compare_"):
        return "compare"
    if "diagnostics" in script_name:
        return "compile"
    if "krylov" in script_name:
        return "ad"
    if "backend_candidates" in script_name or "backends" in script_name:
        return "compare"
    if "compare" in script_name:
        return "accuracy"
    return "perf"


def _default_device_for_script(script_name: str, *, category: str) -> str:
    if "nufft_backends" in script_name or "fft_nufft" in script_name:
        return "gpu_optional"
    if category.startswith("backend_"):
        return "cpu"
    if "matrix_free" in script_name or "matrix_stack" in script_name:
        return "cpu"
    return "cpu"


def discover_benchmark_scripts() -> list[str]:
    names: list[str] = []
    for path in sorted(BENCHMARK_ROOT.glob("*.py")):
        if path.name in {"__init__.py", "_source_tree_bootstrap.py", "sitecustomize.py", "taxonomy.py"}:
            continue
        if path.name.startswith("benchmark_") or path.name.startswith("compare_") or path.name == "bench_harness.py":
            names.append(path.name)
    return names


def build_benchmark_taxonomy() -> dict[str, BenchmarkScriptMetadata]:
    taxonomy: dict[str, BenchmarkScriptMetadata] = {}
    for script_name in discover_benchmark_scripts():
        category = _category_for_script(script_name)
        intent = _intent_for_script(script_name)
        taxonomy[script_name] = BenchmarkScriptMetadata(
            script_name=script_name,
            intent=intent,
            category=category,
            default_device=_default_device_for_script(script_name, category=category),
        )
    return taxonomy


BENCHMARK_TAXONOMY = build_benchmark_taxonomy()

OFFICIAL_BENCHMARKS: dict[str, str] = {
    "api_speed": "benchmark_api_surface.py",
    "core_accuracy": "bench_harness.py",
    "matrix_speed": "benchmark_matrix_suite.py",
    "matrix_compile": "benchmark_matrix_stack_diagnostics.py",
    "matrix_ad": "benchmark_matrix_free_krylov.py",
    "matrix_backend_compare": "benchmark_matrix_backend_candidates.py",
    "transform_speed": "benchmark_fft_nufft.py",
    "transform_backend_compare": "benchmark_nufft_backends.py",
    "transform_gpu": "benchmark_fft_nufft.py",
}


def official_roles_for_script(script_name: str) -> tuple[str, ...]:
    return tuple(role for role, target in OFFICIAL_BENCHMARKS.items() if target == script_name)


def marker_names_for_script(script_name: str) -> tuple[str, ...]:
    meta = BENCHMARK_TAXONOMY[script_name]
    markers = ["benchmark", f"benchmark_{meta.intent}", f"benchmark_{meta.category}"]
    if official_roles_for_script(script_name):
        markers.append("benchmark_official")
    if meta.default_device == "cpu":
        markers.append("benchmark_cpu")
    elif meta.default_device == "gpu_optional":
        markers.append("benchmark_gpu")
    return tuple(markers)


def smoke_script_names() -> tuple[str, ...]:
    return (
        "bench_harness.py",
        "benchmark_api_surface.py",
        "benchmark_core_scalar_service_api.py",
        "benchmark_special_function_service_api.py",
        "benchmark_matrix_service_api.py",
        "benchmark_matrix_backend_candidates.py",
        "benchmark_nufft_backends.py",
    )
