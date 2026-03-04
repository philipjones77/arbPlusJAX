from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.benchmark
def test_benchmark_harness_cli_smoke() -> None:
    if os.getenv("ARBPLUSJAX_RUN_BENCHMARKS", "0") != "1":
        pytest.skip("Set ARBPLUSJAX_RUN_BENCHMARKS=1 to run benchmark smoke checks.")
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / "bench_harness.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--c-ref-dir" in result.stdout
