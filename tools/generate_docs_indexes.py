from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _title_from_name(path: Path) -> str:
    if path.name == "README.md":
        return path.parent.name.replace("_", " ").title()
    stem = path.stem.replace("_", " ").replace("-", " ")
    return stem.title()


def _doc_link(path: Path) -> str:
    return f"[{path.name}]({path.resolve()})"


def _sorted_markdown_files(folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in folder.glob("*.md")
            if path.name != "README.md"
        ],
        key=lambda path: path.name,
    )


def render_docs_index() -> str:
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Docs Index",
        "",
        "Use the docs tree by intent.",
        "",
        f"- repo architecture: [{'governance/architecture.md'}]({(DOCS_ROOT / 'governance' / 'architecture.md').resolve()})",
        f"- governance and placement rules: [{'governance/documentation_governance.md'}]({(DOCS_ROOT / 'governance' / 'documentation_governance.md').resolve()})",
        f"- repo overview: [{'project_overview.md'}]({(DOCS_ROOT / 'project_overview.md').resolve()})",
        f"- standards and contracts map: [{'standards/README.md'}]({(DOCS_ROOT / 'standards' / 'README.md').resolve()})",
        f"- current repo inventories and generated reports: [{'reports/README.md'}]({(DOCS_ROOT / 'reports' / 'README.md').resolve()})",
        f"- implementation notes: [{'implementation/README.md'}]({(DOCS_ROOT / 'implementation' / 'README.md').resolve()})",
        f"- theory and methodology notes: [{'theory/README.md'}]({(DOCS_ROOT / 'theory' / 'README.md').resolve()})",
        f"- practical runbooks and numerical guidance: [{'practical/README.md'}]({(DOCS_ROOT / 'practical' / 'README.md').resolve()})",
        f"- current implementation state and TODOs: [{'status/README.md'}]({(DOCS_ROOT / 'status' / 'README.md').resolve()})",
    ]
    return "\n".join(lines) + "\n"


def render_project_overview() -> str:
    top_level = [
        "src",
        "tests",
        "benchmarks",
        "tools",
        "docs",
        "contracts",
        "examples",
        "experiments",
        "outputs",
        "data",
    ]
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Project Overview",
        "",
        "arbPlusJAX is the active hardened JAX numerical-kernel workspace. The repository separates runtime code, conformance tests, benchmarks, examples, tooling, contracts, and documentation into stable top-level folders.",
        "",
        "## Repo Root",
        "",
    ]
    for name in top_level:
        path = REPO_ROOT / name
        if path.exists():
            lines.append(f"- `{name}/`")
    lines.extend(
        [
            "",
            "## Docs Map",
            "",
            f"- governance: [{DOCS_ROOT / 'governance' / 'README.md'}]({(DOCS_ROOT / 'governance' / 'README.md').resolve()})",
            f"- standards: [{DOCS_ROOT / 'standards' / 'README.md'}]({(DOCS_ROOT / 'standards' / 'README.md').resolve()})",
            f"- reports: [{DOCS_ROOT / 'reports' / 'README.md'}]({(DOCS_ROOT / 'reports' / 'README.md').resolve()})",
            f"- status: [{DOCS_ROOT / 'status' / 'README.md'}]({(DOCS_ROOT / 'status' / 'README.md').resolve()})",
            f"- theory: [{DOCS_ROOT / 'theory' / 'README.md'}]({(DOCS_ROOT / 'theory' / 'README.md').resolve()})",
            f"- implementation: [{DOCS_ROOT / 'implementation' / 'README.md'}]({(DOCS_ROOT / 'implementation' / 'README.md').resolve()})",
            f"- practical: [{DOCS_ROOT / 'practical' / 'README.md'}]({(DOCS_ROOT / 'practical' / 'README.md').resolve()})",
            "",
            "## Generation Rule",
            "",
            "Docs landing pages, report indexes, status indexes, and current repo mapping are generated and should be refreshed through `python tools/check_generated_reports.py` before commit/push.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_governance_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "governance")
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Governance",
        "",
        "This section holds structural and process rules for the repository.",
        "",
        "Current documents:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    return "\n".join(lines) + "\n"


def render_standards_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "standards")
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Standards",
        "",
        "This section holds repo-wide standards and governance-linked policy documents.",
        "",
        "Current standards:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    lines.extend(
        [
            "",
            "Generated reports that describe the current repo state belong in `docs/reports/`.",
            "Current implementation progress and active TODOs belong in `docs/status/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_reports_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "reports")
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Reports",
        "",
        "This section holds current repo inventories and report-style summaries.",
        "",
        "Standards:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'report_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'governance' / 'documentation_governance.md')}",
        "",
        "Current reports:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    return "\n".join(lines) + "\n"


def render_status_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "status")
    primary = ["todo.md", "audit.md", "test_coverage_matrix.md", "test_gap_checklist.md"]
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Status",
        "",
        "This section holds active TODOs, audits, completion plans, and current implementation-state tracking.",
        "",
        "Standards:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'status_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'governance' / 'documentation_governance.md')}",
        "",
        "Primary entry points:",
    ]
    for name in primary:
        path = DOCS_ROOT / "status" / name
        if path.exists():
            lines.append(f"- {_doc_link(path)}")
    extras = [path for path in files if path.name not in set(primary)]
    if extras:
        lines.extend(["", "Other status documents:"])
        lines.extend(f"- {_doc_link(path)}" for path in extras)
    return "\n".join(lines) + "\n"


def render_theory_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "theory")
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Theory",
        "",
        "This section collects mathematical background and methodology notes for the interval, matrix, transform, and special-function machinery used in arbPlusJAX.",
        "",
        "Theory notes explain algorithmic meaning and mathematical interpretation. Standards explain how those algorithms should be exposed, benchmarked, diagnosed, and taught through examples.",
        "",
        "Current methodology notes:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    lines.extend(
        [
            "",
            "Current theory status remains `in_progress`: the main interval/core/matrix/transform/gamma foundations are documented, but further methodology notes should continue to be added as new hardened families land.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_current_repo_mapping() -> str:
    root_dirs = sorted(
        [
            path
            for path in REPO_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".") and path.name not in {"__pycache__"}
        ],
        key=lambda path: path.name,
    )
    doc_sections = [
        "governance",
        "standards",
        "reports",
        "status",
        "theory",
        "implementation",
        "practical",
        "specs",
        "objects",
        "notation",
    ]
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Current Repo Mapping",
        "",
        "This report records the current high-level repository and documentation-tree mapping.",
        "",
        "## Repo Root",
        "",
    ]
    lines.extend(f"- `{path.name}/`" for path in root_dirs)
    lines.extend(["", "## Docs Sections", ""])
    for name in doc_sections:
        path = DOCS_ROOT / name
        if path.exists():
            lines.append(f"- `{name}/`")
    lines.extend(
        [
            "",
            "## Generation Rule",
            "",
            "This report is generated from the current filesystem layout and should be refreshed through `python tools/check_generated_reports.py` before commit/push.",
        ]
    )
    return "\n".join(lines) + "\n"


def generated_docs() -> dict[Path, str]:
    return {
        DOCS_ROOT / "index.md": render_docs_index(),
        DOCS_ROOT / "project_overview.md": render_project_overview(),
        DOCS_ROOT / "governance" / "README.md": render_governance_readme(),
        DOCS_ROOT / "standards" / "README.md": render_standards_readme(),
        DOCS_ROOT / "status" / "README.md": render_status_readme(),
        DOCS_ROOT / "theory" / "README.md": render_theory_readme(),
        DOCS_ROOT / "reports" / "current_repo_mapping.md": render_current_repo_mapping(),
        DOCS_ROOT / "reports" / "README.md": render_reports_readme(),
    }


def main() -> None:
    for path, content in generated_docs().items():
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
