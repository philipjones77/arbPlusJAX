from __future__ import annotations

import ast
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "entry_script_startup_inventory.md"

ENTRY_SCRIPTS = (
    ("benchmarks/benchmark_api_surface.py", "help"),
    ("benchmarks/benchmark_matrix_service_api.py", "help"),
    ("benchmarks/benchmark_special_function_service_api.py", "help"),
    ("benchmarks/benchmark_matrix_free_krylov.py", "help"),
    ("benchmarks/benchmark_core_scalar_service_api.py", "help"),
    ("benchmarks/benchmark_dense_matrix_surface.py", "help"),
    ("benchmarks/benchmark_sparse_matrix_surface.py", "help"),
    ("benchmarks/benchmark_block_sparse_matrix_surface.py", "help"),
    ("benchmarks/benchmark_vblock_sparse_matrix_surface.py", "help"),
    ("benchmarks/benchmark_matrix_stack_diagnostics.py", "help"),
    ("benchmarks/benchmark_arb_mat.py", "help"),
    ("benchmarks/benchmark_acb_calc.py", "help"),
    ("benchmarks/benchmark_barnes_double_gamma.py", "help"),
    ("benchmarks/benchmark_hypgeom.py", "help"),
    ("benchmarks/compare_arb_mat.py", "help"),
    ("benchmarks/compare_acb_calc.py", "help"),
    ("examples/example_latent_gaussian_laplace.py", "import"),
)


def _run_python(code: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "error": completed.stderr.strip() or completed.stdout.strip() or f"subprocess exited with code {completed.returncode}",
            "returncode": completed.returncode,
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _measure_once(path: str, mode: str) -> dict[str, object]:
    code = f"""
import json
import runpy
import sys
import time
from pathlib import Path

path = {str(REPO_ROOT / path)!r}
mode = {mode!r}
script_dir = str(Path(path).resolve().parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if mode == "help":
    sys.argv = [path, "--help"]
    started = time.perf_counter()
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit:
        pass
else:
    started = time.perf_counter()
    runpy.run_path(path, run_name="__not_main__")
elapsed = time.perf_counter() - started
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({{"seconds": elapsed, "arbplusjax_module_count": len(mods), "arbplusjax_modules": mods}}))
"""
    return _run_python(code)


def _measure_entry(path: str, mode: str, repeats: int = 3) -> dict[str, object]:
    payloads = [_measure_once(path, mode) for _ in range(repeats)]
    errors = [item for item in payloads if "error" in item]
    if errors:
        return {
            "path": path,
            "mode": mode,
            "seconds_mean": None,
            "seconds_min": None,
            "seconds_max": None,
            "arbplusjax_module_count": None,
            "arbplusjax_modules": [],
            "error": str(errors[-1]["error"]),
        }
    times = [float(item["seconds"]) for item in payloads]
    last = payloads[-1]
    return {
        "path": path,
        "mode": mode,
        "seconds_mean": statistics.mean(times),
        "seconds_min": min(times),
        "seconds_max": max(times),
        "arbplusjax_module_count": int(last["arbplusjax_module_count"]),
        "arbplusjax_modules": list(last["arbplusjax_modules"]),
    }


def _top_level_imports(path: Path) -> dict[str, object]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return {
            "arbplusjax_imports": [],
            "jax_imports": [],
        }
    arbplusjax_imports: list[str] = []
    jax_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax" or alias.name.startswith("jax."):
                    jax_imports.append(alias.name)
                if alias.name == "arbplusjax" or alias.name.startswith("arbplusjax."):
                    arbplusjax_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "jax" or module.startswith("jax."):
                jax_imports.append(module)
            if module == "arbplusjax" or module.startswith("arbplusjax."):
                arbplusjax_imports.append(module)
    return {
        "arbplusjax_imports": sorted(set(arbplusjax_imports)),
        "jax_imports": sorted(set(jax_imports)),
    }


def _remaining_top_level_arbplusjax_scripts() -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for folder in ("benchmarks", "examples"):
        for path in sorted((REPO_ROOT / folder).glob("*.py")):
            imports = _top_level_imports(path)
            count = len(imports["arbplusjax_imports"])
            if count:
                rows.append((path.relative_to(REPO_ROOT).as_posix(), count))
    return sorted(rows, key=lambda item: (-item[1], item[0]))


def render() -> str:
    rows: list[dict[str, object]] = []
    for path, mode in ENTRY_SCRIPTS:
        meta = _top_level_imports(REPO_ROOT / path)
        measured = _measure_entry(path, mode)
        measured["top_level_arbplusjax_imports"] = meta["arbplusjax_imports"]
        measured["top_level_jax_imports"] = meta["jax_imports"]
        rows.append(measured)

    remaining = _remaining_top_level_arbplusjax_scripts()
    import_debt = [row for row in rows if row["top_level_arbplusjax_imports"]]
    errors = [row for row in rows if row.get("error")]
    mostly_backend = [
        row
        for row in rows
        if row["seconds_mean"] is not None and not row["top_level_arbplusjax_imports"] and float(row["seconds_mean"]) >= 0.6
    ]

    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# Entry Script Startup Inventory",
        "",
        "This report measures benchmark/example entry-script startup after the lazy-entry refactor and records where top-level `arbplusjax` imports still remain.",
        "",
        "Interpretation:",
        "- `top-level arbplusjax imports` means the script still pulls repo family modules during module import before first real use",
        "- `arbplusjax module count` records how much of the repo was loaded just to reach `--help` or import-only startup",
        "- high startup with zero top-level `arbplusjax` imports usually means the remaining delay is mostly JAX/Python/runtime bootstrap cost, not repo family import debt",
        "",
        "| path | mode | mean startup s | min s | max s | arbplusjax modules | top-level arbplusjax imports | top-level jax imports |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        arb_imports = ", ".join(row["top_level_arbplusjax_imports"]) or "-"
        jax_imports = ", ".join(row["top_level_jax_imports"]) or "-"
        if row.get("error"):
            lines.append(
                f"| [{row['path']}](/{row['path']}) | `{row['mode']}` | `error` | `-` | `-` | `-` | `{arb_imports}` | `{jax_imports}` |"
            )
            continue
        lines.append(
            f"| [{row['path']}](/{row['path']}) | `{row['mode']}` | `{row['seconds_mean']:.3f}` | `{row['seconds_min']:.3f}` | `{row['seconds_max']:.3f}` | `{row['arbplusjax_module_count']}` | `{arb_imports}` | `{jax_imports}` |"
        )

    lines.extend(
        [
            "",
            "## Remaining Top-Level `arbplusjax` Import Debt",
            "",
            f"- scripts still carrying top-level `arbplusjax` imports across `benchmarks/` and `examples/`: `{len(remaining)}`",
        ]
    )
    for path, count in remaining[:20]:
        lines.append(f"- [{path}](/{path}) : `{count}` top-level `arbplusjax` import statements")

    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"- import-boundary debt still exists in `{len(remaining)}` benchmark/example scripts that keep top-level `arbplusjax` imports",
            f"- in this measured entry set, `{len(import_debt)}` scripts still have direct top-level `arbplusjax` imports and should be treated as remaining repo import debt",
            f"- in this measured entry set, `{len(mostly_backend)}` scripts have zero top-level `arbplusjax` imports but still take at least `0.6s` to start; those are now mostly dominated by JAX import/backend/runtime bootstrap cost",
            f"- `{len(errors)}` measured entry scripts currently fail before timing completes because of missing optional dependencies or other local runtime issues",
            "- `--help` paths that still import `jax` at module top level will continue to pay significant cold-start cost even after repo-family lazy-loading is fixed",
        ]
    )
    if errors:
        lines.extend(["", "## Measurement Failures", ""])
        for row in errors:
            lines.append(f"- [{row['path']}](/{row['path']}) : `{row['error'].splitlines()[-1]}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
