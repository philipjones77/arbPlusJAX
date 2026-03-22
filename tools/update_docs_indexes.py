from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _link(path: Path) -> str:
    return f"[{path.name}]({path.resolve().as_posix()})"


def _markdown_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.md") if p.name != "README.md")


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def render_reports_readme() -> str:
    reports_dir = DOCS_ROOT / "reports"
    files = _markdown_files(reports_dir)
    lines = [
        f"Last updated: {_timestamp()}",
        "",
        "# Reports",
        "",
        "This section holds current repo inventories and report-style summaries.",
        "",
        "Standards:",
        f"- {_link(DOCS_ROOT / 'standards' / 'report_standard.md')}",
        f"- {_link(DOCS_ROOT / 'standards' / 'documentation.md')}",
        "",
        "Current reports:",
    ]
    for path in files:
        lines.append(f"- {_link(path)}")
    return "\n".join(lines)


def render_status_readme() -> str:
    status_dir = DOCS_ROOT / "status"
    files = _markdown_files(status_dir)
    primary_order = [
        "todo.md",
        "audit.md",
        "function_gap_plan.md",
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
        f"- {_link(DOCS_ROOT / 'standards' / 'status_standard.md')}",
        f"- {_link(DOCS_ROOT / 'standards' / 'documentation.md')}",
        "",
        "Primary entry points:",
    ]
    for path in primary:
        lines.append(f"- {_link(path)}")
    if remaining:
        lines.extend(["", "Additional status files:"])
        for path in remaining:
            lines.append(f"- {_link(path)}")
    return "\n".join(lines)


def main() -> int:
    _write(DOCS_ROOT / "reports" / "README.md", render_reports_readme())
    _write(DOCS_ROOT / "status" / "README.md", render_status_readme())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
