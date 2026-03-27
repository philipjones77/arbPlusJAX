from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_and_load(args: list[str], out_name: str) -> dict:
    out = REPO_ROOT / "outputs" / "test_benchmark_reports" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, *args, "--output", str(out)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_fft_nufft_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(["benchmarks/benchmark_fft_nufft.py", "--repeat", "1", "--dtype", "float32"], "benchmark_fft_nufft.json")
    assert payload["benchmark_name"] == "benchmark_fft_nufft.py"
    assert payload["category"] == "transform"
    assert payload["records"]
    assert all("warm_time_s" in row for row in payload["records"])
    assert {row["dtype"] for row in payload["records"]} == {"float32", "complex64"}
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_sparse_matrix_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_sparse_matrix_surface.py", "--n", "4", "--warmup", "0", "--runs", "1", "--dtype", "float32", "--smoke"],
        "benchmark_sparse_matrix_surface.json",
    )
    assert payload["benchmark_name"] == "benchmark_sparse_matrix_surface.py"
    assert payload["category"] == "matrix_sparse"
    assert payload["records"]
    assert any(row["operation"] == "matvec" for row in payload["records"])
    assert any(row["operation"] == "storage_prepare" for row in payload["records"])
    assert any(row["operation"] == "cached_prepare" for row in payload["records"])
    assert any(row["operation"] == "lu_prepare" for row in payload["records"])
    assert any(row["operation"] == "spd_prepare" for row in payload["records"])
    assert {"float32", "complex64"} == {row["dtype"] for row in payload["records"]}
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_dense_matrix_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_dense_matrix_surface.py", "--n", "4", "--warmup", "0", "--runs", "1", "--dtype", "float32", "--smoke"],
        "benchmark_dense_matrix_surface.json",
    )
    assert payload["benchmark_name"] == "benchmark_dense_matrix_surface.py"
    assert payload["category"] == "matrix_dense"
    assert payload["records"]
    assert any(row["operation"] == "cached_matvec" for row in payload["records"])
    assert any(row["operation"] == "direct_solve" for row in payload["records"])
    assert {"float32", "complex64"} == {row["dtype"] for row in payload["records"]}
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_barnes_double_gamma_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_barnes_double_gamma.py", "--iters", "1", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_barnes_double_gamma.json",
    )
    assert payload["benchmark_name"] == "benchmark_barnes_double_gamma.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert any(row["implementation"] == "ifj" for row in payload["records"])
    assert any(row["operation"] == "barnes_double_gamma_scalar" for row in payload["records"])
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_block_sparse_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        [
            "benchmarks/benchmark_block_sparse_matrix_surface.py",
            "--n-blocks",
            "2",
            "--block-size",
            "2",
            "--warmup",
            "0",
            "--runs",
            "1",
            "--dtype",
            "float32",
            "--smoke",
        ],
        "benchmark_block_sparse_matrix_surface.json",
    )
    assert payload["benchmark_name"] == "benchmark_block_sparse_matrix_surface.py"
    assert payload["category"] == "matrix_block_sparse"
    assert payload["records"]
    assert any(row["operation"] == "matvec" for row in payload["records"])
    assert any(row["operation"] == "storage_prepare" for row in payload["records"])
    assert any(row["operation"] == "cached_prepare" for row in payload["records"])
    assert {"float32", "complex64"} == {row["dtype"] for row in payload["records"]}
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_vblock_sparse_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        [
            "benchmarks/benchmark_vblock_sparse_matrix_surface.py",
            "--n",
            "4",
            "--warmup",
            "0",
            "--runs",
            "1",
            "--dtype",
            "float32",
            "--smoke",
        ],
        "benchmark_vblock_sparse_matrix_surface.json",
    )
    assert payload["benchmark_name"] == "benchmark_vblock_sparse_matrix_surface.py"
    assert payload["category"] == "matrix_vblock_sparse"
    assert payload["records"]
    assert any(row["operation"] == "matvec" for row in payload["records"])
    assert any(row["operation"] == "storage_prepare" for row in payload["records"])
    assert any(row["operation"] == "cached_prepare" for row in payload["records"])
    assert {"float32", "complex64"} == {row["dtype"] for row in payload["records"]}
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_arb_poly_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_arb_poly.py", "--samples", "128", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_arb_poly.json",
    )
    assert payload["benchmark_name"] == "benchmark_arb_poly.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert payload["records"][0]["operation"] == "eval_cubic_batch"
    assert payload["records"][0]["dtype"] == "float32"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_acb_poly_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_acb_poly.py", "--samples", "128", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_acb_poly.json",
    )
    assert payload["benchmark_name"] == "benchmark_acb_poly.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert payload["records"][0]["operation"] == "eval_cubic_batch"
    assert payload["records"][0]["dtype"] == "complex64"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_arb_calc_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_arb_calc.py", "--samples", "128", "--steps", "8", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_arb_calc.json",
    )
    assert payload["benchmark_name"] == "benchmark_arb_calc.py"
    assert payload["category"] == "integration"
    assert payload["records"]
    assert payload["records"][0]["operation"] == "integrate_line_exp"
    assert payload["records"][0]["dtype"] == "float32"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_acb_calc_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_acb_calc.py", "--samples", "128", "--steps", "8", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_acb_calc.json",
    )
    assert payload["benchmark_name"] == "benchmark_acb_calc.py"
    assert payload["category"] == "integration"
    assert payload["records"]
    assert payload["records"][0]["operation"] == "integrate_line_exp"
    assert payload["records"][0]["dtype"] == "complex64"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_dirichlet_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_dirichlet.py", "--samples", "128", "--terms", "8", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_dirichlet.json",
    )
    assert payload["benchmark_name"] == "benchmark_dirichlet.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert payload["records"][0]["operation"] == "zeta_batch"
    assert payload["records"][0]["dtype"] == "float32"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_acb_dirichlet_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_acb_dirichlet.py", "--samples", "128", "--terms", "8", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_acb_dirichlet.json",
    )
    assert payload["benchmark_name"] == "benchmark_acb_dirichlet.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert payload["records"][0]["operation"] == "zeta_batch"
    assert payload["records"][0]["dtype"] == "complex64"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_hypgeom_extra_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_hypgeom_extra.py", "--iters", "2", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_hypgeom_extra.json",
    )
    assert payload["benchmark_name"] == "benchmark_hypgeom_extra.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert payload["records"][0]["dtype"] == "float32"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_incomplete_bessel_benchmark_writes_shared_schema_report() -> None:
    payload = _run_and_load(
        ["benchmarks/benchmark_incomplete_bessel.py", "--iters", "2", "--dtype", "float32", "--jax-mode", "cpu", "--smoke"],
        "benchmark_incomplete_bessel.json",
    )
    assert payload["benchmark_name"] == "benchmark_incomplete_bessel.py"
    assert payload["category"] == "special"
    assert payload["records"]
    assert any(row["implementation"] == "incomplete_bessel_k" for row in payload["records"])
    assert any(row["operation"] == "quadrature_point" for row in payload["records"])
    assert payload["records"][0]["dtype"] == "float32"
    assert payload["environment"]["jax"]["requested_mode"] == "cpu"


def test_normalized_benchmark_help_shows_dtype_portability_controls() -> None:
    for script in (
        "benchmarks/benchmark_fft_nufft.py",
        "benchmarks/benchmark_sparse_matrix_surface.py",
        "benchmarks/benchmark_block_sparse_matrix_surface.py",
        "benchmarks/benchmark_dense_matrix_surface.py",
        "benchmarks/benchmark_vblock_sparse_matrix_surface.py",
        "benchmarks/benchmark_arb_poly.py",
        "benchmarks/benchmark_acb_poly.py",
        "benchmarks/benchmark_arb_calc.py",
        "benchmarks/benchmark_acb_calc.py",
        "benchmarks/benchmark_dirichlet.py",
        "benchmarks/benchmark_acb_dirichlet.py",
        "benchmarks/benchmark_hypgeom_extra.py",
        "benchmarks/benchmark_incomplete_bessel.py",
        "benchmarks/benchmark_barnes_double_gamma.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--dtype" in result.stdout
        if script.endswith("benchmark_barnes_double_gamma.py"):
            assert "--jax-mode" in result.stdout
