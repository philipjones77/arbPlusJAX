from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import os
import subprocess

import pytest

from benchmarks.taxonomy import BENCHMARK_TAXONOMY
from benchmarks.taxonomy import marker_names_for_script
from benchmarks.taxonomy import smoke_script_names


def _smoke_param(script_name: str):
    return pytest.param(script_name, marks=[getattr(pytest.mark, name) for name in marker_names_for_script(script_name)])


@pytest.mark.parametrize("script_name", [_smoke_param(name) for name in smoke_script_names()])
def test_benchmark_cli_smoke(script_name: str) -> None:
    if os.getenv("ARBPLUSJAX_RUN_BENCHMARKS", "0") != "1":
        pytest.skip("Set ARBPLUSJAX_RUN_BENCHMARKS=1 to run benchmark smoke checks.")
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / script_name
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    meta = BENCHMARK_TAXONOMY[script_name]
    if script_name == "bench_harness.py":
        assert "--c-ref-dir" in result.stdout
    elif meta.category == "api":
        assert "--runs" in result.stdout
    elif meta.category == "backend_matrix":
        assert "--density" in result.stdout
    elif meta.category == "backend_transform":
        assert "--n-points" in result.stdout
