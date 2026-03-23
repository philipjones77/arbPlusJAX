from __future__ import annotations

from datetime import datetime
from datetime import timezone
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _link(path: Path, *, from_path: Path) -> str:
    rel = Path(os.path.relpath(path, start=from_path.parent))
    return f"[{path.name}]({rel.as_posix()})"


def _markdown_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.md") if p.name != "README.md")


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def render_root_readme() -> str:
    out_path = REPO_ROOT / "README.md"
    lines = [
        "# arbPlusJAX",
        "",
        "arbPlusJAX is an independent JAX implementation derived from Arb and FLINT.",
        "It is not the official Arb project.",
        "",
        "## Layout",
        "",
        "- `src/arbplusjax/`: primary runtime source tree",
        "- `tests/`: correctness and contract test surface",
        "- `benchmarks/`: benchmark and comparison surface",
        "- `configs/`: checked-in repo-level configuration and profile definitions",
        "- `examples/`: canonical demonstration notebooks",
        "- `experiments/`: larger exploratory work and retained experiment outputs",
        "- `tools/`: harnesses, generators, and repo utilities",
        "- `docs/`: theory, implementation, practical, standards, reports, and status docs",
        "- `contracts/`: binding runtime and API guarantees",
        "",
        "## Primary Run Surfaces",
        "",
        "- tests: `python tools/run_test_harness.py --profile chassis --jax-mode cpu`",
        "- benchmarks: `python benchmarks/run_benchmarks.py --profile quick`",
        "- runtime check: `python tools/check_jax_runtime.py --quick-bench`",
        f"- examples: see {_link(REPO_ROOT / 'examples' / 'README.md', from_path=out_path)}",
        "",
        "Tests, benchmarks, and notebooks are expected to run against the source tree in `src/arbplusjax` by default.",
        "",
        "## Documentation Entry Points",
        "",
        f"- standards: {_link(DOCS_ROOT / 'standards' / 'README.md', from_path=out_path)}",
        f"- reports: {_link(DOCS_ROOT / 'reports' / 'README.md', from_path=out_path)}",
        f"- status: {_link(DOCS_ROOT / 'status' / 'README.md', from_path=out_path)}",
        f"- practical run guidance: {_link(DOCS_ROOT / 'practical' / 'README.md', from_path=out_path)}",
        f"- implementation notes: {_link(DOCS_ROOT / 'implementation' / 'README.md', from_path=out_path)}",
        "",
        "## Install",
        "",
        "Editable install:",
        "",
        "```bash",
        "python -m pip install -e .",
        "```",
        "",
        "Direct source-tree test run:",
        "",
        "```bash",
        "PYTHONPATH=src python -m pytest tests -q -m \"not parity\"",
        "```",
        "",
        "## Notes",
        "",
        "- JAX is the primary implementation surface.",
        "- Reference software and external engines are validation/comparison layers, not the default runtime path.",
        f"- See {_link(REPO_ROOT / 'NOTICE', from_path=out_path)} for acknowledgments and reference links.",
    ]
    return "\n".join(lines)


def render_reports_readme() -> str:
    reports_dir = DOCS_ROOT / "reports"
    out_path = reports_dir / "README.md"
    files = _markdown_files(reports_dir)
    lines = [
        f"Last updated: {_timestamp()}",
        "",
        "# Reports",
        "",
        "This section holds current repo inventories and report-style summaries.",
        "",
        "Standards:",
        f"- {_link(DOCS_ROOT / 'standards' / 'report_standard.md', from_path=out_path)}",
        f"- {_link(DOCS_ROOT / 'standards' / 'documentation.md', from_path=out_path)}",
        "",
        "Current reports:",
    ]
    for path in files:
        lines.append(f"- {_link(path, from_path=out_path)}")
    return "\n".join(lines)


def render_status_readme() -> str:
    status_dir = DOCS_ROOT / "status"
    out_path = status_dir / "README.md"
    files = _markdown_files(status_dir)
    primary_order = [
        "todo.md",
        "audit.md",
        "matrix_free_completion_plan.md",
        "sparse_completion_plan.md",
        "test_coverage_matrix.md",
        "test_gap_checklist.md",
    ]
    file_map = {p.name: p for p in files}
    primary = [file_map[name] for name in primary_order if name in file_map]
    remaining = [p for p in files if p not in primary]

    lines = [
        f"Last updated: {_timestamp()}",
        "",
        "# Status",
        "",
        "This section holds active TODOs, audits, completion plans, and current implementation-state tracking.",
        "",
        "Standards:",
        f"- {_link(DOCS_ROOT / 'standards' / 'status_standard.md', from_path=out_path)}",
        f"- {_link(DOCS_ROOT / 'standards' / 'documentation.md', from_path=out_path)}",
        "",
        "Primary entry points:",
    ]
    for path in primary:
        lines.append(f"- {_link(path, from_path=out_path)}")
    if remaining:
        lines.extend(["", "Additional status files:"])
        for path in remaining:
            lines.append(f"- {_link(path, from_path=out_path)}")
    return "\n".join(lines)


def main() -> int:
    _write(REPO_ROOT / "README.md", render_root_readme())
    _write(DOCS_ROOT / "reports" / "README.md", render_reports_readme())
    _write(DOCS_ROOT / "status" / "README.md", render_status_readme())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
