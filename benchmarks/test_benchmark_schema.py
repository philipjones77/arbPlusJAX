from __future__ import annotations

import json
from pathlib import Path

from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report


def test_benchmark_record_serializes_core_fields() -> None:
    record = BenchmarkRecord(
        benchmark_name="benchmark_matrix_stack_diagnostics.py",
        concern="matrix_compile",
        category="matrix",
        implementation="jrb_mat",
        operation="logdet",
        device="cpu",
        dtype="float64",
        cold_time_s=1.2,
        warm_time_s=0.01,
        recompile_time_s=0.8,
        python_overhead_s=0.001,
        memory_bytes=1024,
        accuracy_abs=1e-9,
        accuracy_rel=1e-8,
        residual=1e-10,
        ad_forward_time_s=0.02,
        ad_backward_time_s=0.03,
        ad_residual=1e-7,
        measurements=(BenchmarkMeasurement(name="peak_mem_mb", value=12.5, unit="MiB"),),
        tags=("official", "compile"),
    )

    payload = record.to_dict()

    assert payload["benchmark_name"] == "benchmark_matrix_stack_diagnostics.py"
    assert payload["concern"] == "matrix_compile"
    assert payload["cold_time_s"] == 1.2
    assert payload["ad_backward_time_s"] == 0.03
    assert payload["measurements"][0]["name"] == "peak_mem_mb"


def test_benchmark_report_writes_json(tmp_path: Path) -> None:
    report = BenchmarkReport(
        benchmark_name="benchmark_fft_nufft.py",
        concern="transform_speed",
        category="transform",
        records=(
            BenchmarkRecord(
                benchmark_name="benchmark_fft_nufft.py",
                concern="transform_speed",
                category="transform",
                implementation="repo_native",
                operation="nufft",
                device="gpu",
                dtype="complex64",
                cold_time_s=0.9,
                warm_time_s=0.02,
            ),
        ),
        environment={"jax_platform": "cuda"},
    )

    out = write_benchmark_report(tmp_path / "report.json", report)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "v1"
    assert payload["benchmark_name"] == "benchmark_fft_nufft.py"
    assert payload["records"][0]["device"] == "gpu"
    assert payload["records"][0]["warm_time_s"] == 0.02
