from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from nbclient import NotebookClient
import nbformat

from source_tree_bootstrap import ensure_src_on_path
from runtime_manifest import collect_runtime_manifest, write_runtime_manifest


REPO_ROOT = ensure_src_on_path(__file__)
EXAMPLES_DIR = REPO_ROOT / "examples"


def _default_notebooks() -> list[str]:
    return ["example_core_scalar_surface.ipynb", "example_api_surface.ipynb"]


def _example_output_root(notebook_name: str) -> Path:
    example_name = Path(notebook_name).stem
    return EXAMPLES_DIR / "outputs" / example_name


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_notebook(
    notebook_path: Path,
    *,
    kernel_name: str,
    timeout_s: int,
) -> dict[str, object]:
    started = time.perf_counter()
    with notebook_path.open("r", encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name=kernel_name,
        timeout=timeout_s,
        resources={"metadata": {"path": str(REPO_ROOT)}},
        record_timing=True,
    )
    client.execute()
    elapsed_s = time.perf_counter() - started
    return {"notebook": nb, "elapsed_s": elapsed_s}


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the canonical example notebooks and retain CPU/GPU output artifacts.")
    parser.add_argument("--jax-mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--jax-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--python", default=sys.executable, help="Interpreter path recorded into notebook env and runtime manifests.")
    parser.add_argument("--notebooks", nargs="*", default=_default_notebooks())
    args = parser.parse_args()

    os.environ["ARBPLUSJAX_PYTHON"] = args.python
    os.environ["JAX_MODE"] = args.jax_mode
    os.environ["JAX_DTYPE"] = args.jax_dtype
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    if args.jax_mode == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_ENABLE_X64"] = "1" if args.jax_dtype == "float64" else "0"

    run_rows: list[dict[str, object]] = []
    for notebook_name in args.notebooks:
        notebook_path = (EXAMPLES_DIR / notebook_name).resolve()
        if not notebook_path.exists():
            raise FileNotFoundError(f"Missing notebook: {notebook_path}")
        output_root = _example_output_root(notebook_name)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = collect_runtime_manifest(REPO_ROOT, jax_mode=args.jax_mode, python_path=args.python)
        manifest["example_notebook"] = {
            "notebook": notebook_name,
            "jax_dtype": args.jax_dtype,
            "kernel_name": args.kernel_name,
        }
        write_runtime_manifest(output_root, manifest, filename=f"runtime_manifest_{args.jax_mode}.json")

        result = execute_notebook(notebook_path, kernel_name=args.kernel_name, timeout_s=args.timeout)
        executed_path = output_root / f"{Path(notebook_name).stem}_{args.jax_mode}_executed.ipynb"
        with executed_path.open("w", encoding="utf-8") as handle:
            nbformat.write(result["notebook"], handle)
        summary = {
            "notebook": notebook_name,
            "executed_notebook": str(executed_path.relative_to(REPO_ROOT)),
            "output_root": str(output_root.relative_to(REPO_ROOT)),
            "jax_mode": args.jax_mode,
            "jax_dtype": args.jax_dtype,
            "elapsed_s": round(float(result["elapsed_s"]), 6),
        }
        _write_json(output_root / f"execution_summary_{args.jax_mode}.json", summary)
        run_rows.append(summary)
        print(f"[notebook] {notebook_name} elapsed={summary['elapsed_s']:.3f}s output={output_root}")

    suite_root = EXAMPLES_DIR / "outputs" / "example_run_suite" / f"notebooks_{args.jax_mode}"
    suite_root.mkdir(parents=True, exist_ok=True)
    suite_manifest = collect_runtime_manifest(REPO_ROOT, jax_mode=args.jax_mode, python_path=args.python)
    suite_manifest["example_notebooks"] = {"jax_dtype": args.jax_dtype, "notebooks": args.notebooks}
    write_runtime_manifest(suite_root, suite_manifest)
    _write_json(suite_root / "notebook_execution_index.json", run_rows)
    lines = [
        f"# Example Notebook Execution Summary ({args.jax_mode})",
        "",
        f"- python: `{args.python}`",
        f"- jax_dtype: `{args.jax_dtype}`",
        "",
        "## Notebooks",
        "",
    ]
    for row in run_rows:
        lines.append(
            f"- `{row['notebook']}`: elapsed_s={row['elapsed_s']}, executed={row['executed_notebook']}, output_root={row['output_root']}"
        )
    (suite_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
