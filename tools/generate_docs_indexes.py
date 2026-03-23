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
    repo_path = path.relative_to(REPO_ROOT).as_posix()
    return f"[{path.name}](/{repo_path})"


def _repo_link(path: Path) -> str:
    repo_path = path.relative_to(REPO_ROOT).as_posix()
    return f"[{repo_path}](/{repo_path})"


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
        f"- repo architecture: [governance/architecture.md](/{(DOCS_ROOT / 'governance' / 'architecture.md').relative_to(REPO_ROOT).as_posix()})",
        f"- governance and placement rules: [governance/documentation_governance.md](/{(DOCS_ROOT / 'governance' / 'documentation_governance.md').relative_to(REPO_ROOT).as_posix()})",
        f"- repo overview: [project_overview.md](/{(DOCS_ROOT / 'project_overview.md').relative_to(REPO_ROOT).as_posix()})",
        f"- notation and symbol conventions: [notation/README.md](/{(DOCS_ROOT / 'notation' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- semantic definitions and invariants: [specs/README.md](/{(DOCS_ROOT / 'specs' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- named runtime catalogs and object registries: [objects/README.md](/{(DOCS_ROOT / 'objects' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- standards and contracts map: [standards/README.md](/{(DOCS_ROOT / 'standards' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- current repo inventories and generated reports: [reports/README.md](/{(DOCS_ROOT / 'reports' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- implementation notes: [implementation/README.md](/{(DOCS_ROOT / 'implementation' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- theory and methodology notes: [theory/README.md](/{(DOCS_ROOT / 'theory' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- practical runbooks and numerical guidance: [practical/README.md](/{(DOCS_ROOT / 'practical' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
        f"- current implementation state and TODOs: [status/README.md](/{(DOCS_ROOT / 'status' / 'README.md').relative_to(REPO_ROOT).as_posix()})",
    ]
    return "\n".join(lines) + "\n"


def render_project_overview() -> str:
    top_level = [
        "src",
        "tests",
        "benchmarks",
        "configs",
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
            f"- governance: {_repo_link(DOCS_ROOT / 'governance' / 'README.md')}",
            f"- standards: {_repo_link(DOCS_ROOT / 'standards' / 'README.md')}",
            f"- notation: {_repo_link(DOCS_ROOT / 'notation' / 'README.md')}",
            f"- specs: {_repo_link(DOCS_ROOT / 'specs' / 'README.md')}",
            f"- objects: {_repo_link(DOCS_ROOT / 'objects' / 'README.md')}",
            f"- reports: {_repo_link(DOCS_ROOT / 'reports' / 'README.md')}",
            f"- status: {_repo_link(DOCS_ROOT / 'status' / 'README.md')}",
            f"- theory: {_repo_link(DOCS_ROOT / 'theory' / 'README.md')}",
            f"- implementation: {_repo_link(DOCS_ROOT / 'implementation' / 'README.md')}",
            f"- practical: {_repo_link(DOCS_ROOT / 'practical' / 'README.md')}",
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
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Standards",
        "",
        "This section holds repo-wide standards and governance-linked policy documents.",
        "",
        "The detailed standards overlap in a few places, so the practical reading model is concept-first rather than filename-first. The current standards set consolidates into six primary concept groups.",
        "",
        "## Consolidated Concept Groups",
        "",
        "### 1. Runtime, Numerics, and Production Calling",
        "",
        "Primary owner:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'jax_api_runtime_standard.md')}",
        "",
        "Specialized companion documents:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'engineering_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'jax_surface_policy_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'precision_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'core_scalar_service_calling_standard.md')}",
        "",
        "Consolidation note:",
        "- treat `jax_api_runtime_standard.md` as the canonical runtime/API contract",
        "- treat `engineering_standard.md` as the hardening and status-interpretation overlay",
        "- treat `core_scalar_service_calling_standard.md` as a tranche-specific specialization, not a second general runtime policy",
        "- API calling shape, binder reuse, diagnostics payloads, logging hooks, and the rule that diagnostics/profiling stay outside the mandatory numeric hot path all belong to this runtime concept",
        "",
        "### 2. Validation, Benchmarking, and Executable Examples",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'benchmark_validation_policy_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'example_notebook_standard.md')}",
        "",
        "Specialized companion documents:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'benchmark_grouping_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'pytest_test_naming_standard.md')}",
        "",
        "Consolidation note:",
        "- `benchmark_validation_policy_standard.md` owns measurement and benchmark-contract policy",
        "- `benchmark_grouping_standard.md` is the taxonomy companion for the same benchmark concept",
        "- `example_notebook_standard.md` is the executable-teaching analogue of the same validation/communication layer",
        "",
        "### 3. Portability and Run Layout",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'environment_portability_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'experiment_layout_standard.md')}",
        "",
        "Consolidation note:",
        "- these documents jointly own where things run and where artifacts live",
        "",
        "### 4. Contracts And Provider Boundary",
        "",
        "Primary owner:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'contract_and_provider_boundary_standard.md')}",
        "",
        "Consolidation note:",
        "- this document consolidates the missing contract-placement and provider-boundary policy into one public-surface concept",
        "- downstream-facing API capability contracts, metadata guarantees, and provider-grade surface rules belong here rather than in ad hoc per-family notes",
        "",
        "### 5. Documentation Outputs And Generated Communication Surfaces",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'generated_documentation_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'report_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'status_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'repo_standards.md')}",
        "",
        "Consolidation note:",
        "- `generated_documentation_standard.md` owns the shared generation rule",
        "- `repo_standards.md` owns repo-root communication and placement",
        "- the report/status standards remain specialized audience documents under the same documentation-output concept",
        "",
        "### 6. Theory, Notation, and Naming Semantics",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'theory_notation_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'function_naming_standard.md')}",
        "",
        "Specialized companion documents:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'pytest_test_naming_standard.md')}",
        "",
        "Consolidation note:",
        "- `theory_notation_standard.md` owns methodology-note and notation governance",
        "- function naming and test naming remain the explicit naming-policy layer",
        "",
        "## Detailed Standards",
        ]
    files = _sorted_markdown_files(DOCS_ROOT / "standards")
    lines.extend(f"- {_doc_link(path)}" for path in files)
    lines.extend(
        [
            "",
            "Generated reports that describe the current repo state belong in `docs/reports/`.",
            "Current implementation progress and active TODOs belong in `docs/status/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_notation_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "notation")
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Notation",
        "",
        "This section holds authoritative notation conventions, symbol tables, and naming bridges between code and mathematics.",
        "",
        "Current notation documents:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
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
        DOCS_ROOT / "notation" / "README.md": render_notation_readme(),
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
