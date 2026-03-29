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


def _append_hidden_toctree(lines: list[str], entries: list[str], *, maxdepth: int = 2) -> None:
    if not entries:
        return
    lines.extend(
        [
            "",
            "```{toctree}",
            f":maxdepth: {maxdepth}",
            ":hidden:",
            "",
        ]
    )
    lines.extend(entries)
    lines.append("```")


def render_root_readme() -> str:
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
        "- `papers/`: long-form LaTeX manuscript and publication projects",
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
        "- examples: see [examples/README.md](/examples/README.md)",
        "",
        "Tests, benchmarks, and notebooks are expected to run against the source tree in `src/arbplusjax` by default.",
        "",
        "## Documentation Entry Points",
        "",
        "- docs root: [docs/README.md](/docs/README.md)",
        "- project overview: [docs/project_overview.md](/docs/project_overview.md)",
        "- standards: [docs/standards/README.md](/docs/standards/README.md)",
        "- reports: [docs/reports/README.md](/docs/reports/README.md)",
        "- status: [docs/status/README.md](/docs/status/README.md)",
        "- theory: [docs/theory/README.md](/docs/theory/README.md)",
        "- practical run guidance: [docs/practical/README.md](/docs/practical/README.md)",
        "- implementation notes: [docs/implementation/README.md](/docs/implementation/README.md)",
        "",
        "## Install",
        "",
        "Editable install:",
        "",
        "```bash",
        "python -m pip install -e .",
        "```",
        "",
        "Colab bootstrap:",
        "",
        "```bash",
        "python -m pip install -r requirements-colab.txt",
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
        "- The package root uses lazy public-module loading to keep import-time cost low.",
        "- Reference software and external engines are validation/comparison layers, not the default runtime path.",
        "- Linux, Windows, WSL, GitHub submission, and Colab all use the same source tree; Colab has a CPU-safe bootstrap surface in [requirements-colab.txt](/requirements-colab.txt) and [tools/colab_bootstrap.sh](/tools/colab_bootstrap.sh).",
        "- Optional comparison/reference backends are tracked in [configs/optional_comparison_backends.json](/configs/optional_comparison_backends.json); Mathematica, `c_arb`, `mpmath`, `scipy`, `jax.scipy`, and experimental JAX paths are comparison layers, not mandatory runtime dependencies.",
        "- See [NOTICE](/NOTICE) for acknowledgments and reference links.",
    ]
    return "\n".join(lines) + "\n"


def render_project_overview() -> str:
    top_level = [
        "src",
        "tests",
        "benchmarks",
        "configs",
        "papers",
        "tools",
        "docs",
        "contracts",
        "examples",
        "experiments",
        "outputs",
        "data",
    ]
    lines = [
        "Last updated: 2026-03-29T00:00:00Z",
        "",
        "# Project Overview",
        "",
        "arbPlusJAX is the active hardened JAX numerical-kernel workspace. The repository separates runtime code, conformance tests, benchmarks, examples, tooling, contracts, documentation, and publication manuscripts into stable top-level folders.",
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
            "Use the docs tree by intent.",
            "",
            f"- repo architecture: {_repo_link(DOCS_ROOT / 'governance' / 'architecture.md')}",
            f"- governance and placement rules: {_repo_link(DOCS_ROOT / 'governance' / 'documentation_governance.md')}",
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


def render_docs_readme() -> str:
    section_paths = [
        DOCS_ROOT / "project_overview.md",
        DOCS_ROOT / "governance" / "README.md",
        DOCS_ROOT / "standards" / "README.md",
        DOCS_ROOT / "theory" / "README.md",
        DOCS_ROOT / "practical" / "README.md",
        REPO_ROOT / "examples" / "README.md",
        DOCS_ROOT / "implementation" / "README.md",
        DOCS_ROOT / "reports" / "README.md",
        DOCS_ROOT / "status" / "README.md",
        DOCS_ROOT / "objects" / "README.md",
        DOCS_ROOT / "specs" / "README.md",
        DOCS_ROOT / "notation" / "README.md",
    ]
    lines = [
        "Last updated: 2026-03-29T00:00:00Z",
        "",
        "# Documentation",
        "",
        "This folder is the Markdown-first source of truth for active repo documentation.",
        "It is authored and maintained directly in `docs/`.",
        "",
        "When the publication layer is used, Sphinx + MyST renders this same folder into HTML and PDF output.",
        "That publish layer must adapt to this docs tree rather than requiring a second parallel documentation corpus.",
        "",
        "## Contents",
        "",
        f"- overview: {_doc_link(DOCS_ROOT / 'project_overview.md')}",
        f"- governance: {_doc_link(DOCS_ROOT / 'governance' / 'README.md')}",
        f"- standards: {_doc_link(DOCS_ROOT / 'standards' / 'README.md')}",
        f"- theory: {_doc_link(DOCS_ROOT / 'theory' / 'README.md')}",
        f"- practical: {_doc_link(DOCS_ROOT / 'practical' / 'README.md')}",
        "- examples: [README.md](/examples/README.md)",
        f"- implementation: {_doc_link(DOCS_ROOT / 'implementation' / 'README.md')}",
        f"- reports: {_doc_link(DOCS_ROOT / 'reports' / 'README.md')}",
        f"- status: {_doc_link(DOCS_ROOT / 'status' / 'README.md')}",
        f"- objects: {_doc_link(DOCS_ROOT / 'objects' / 'README.md')}",
        f"- specs: {_doc_link(DOCS_ROOT / 'specs' / 'README.md')}",
        f"- notation: {_doc_link(DOCS_ROOT / 'notation' / 'README.md')}",
        "",
        "## Build Model",
        "",
        "- source of truth: `docs/`",
        "- HTML renderer: Sphinx + MyST",
        "- HTML host: GitHub Pages",
        "- PDF renderer: Sphinx `latexpdf`/LuaLaTeX lane",
        "- publication manuscripts: [papers/README.md](/papers/README.md)",
        "",
        "## Site Navigation",
        "",
        "The publication layer uses this `README.md` as the docs root document.",
        "",
        "```{toctree}",
        ":maxdepth: 2",
        ":hidden:",
        "",
    ]
    for path in section_paths:
        if path.is_relative_to(DOCS_ROOT):
            lines.append(path.relative_to(DOCS_ROOT).as_posix().removesuffix(".md"))
        else:
            lines.append(path.relative_to(REPO_ROOT).as_posix().removesuffix(".md"))
    lines.extend(
        [
            "```",
            "",
            "This file is generated and should be refreshed through `python tools/generate_docs_indexes.py` before commit/push.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_governance_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "governance")
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
        "",
        "# Governance",
        "",
        "This section holds structural and process rules for the repository.",
        "",
        "Current documents:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    _append_hidden_toctree(lines, [path.stem for path in files])
    return "\n".join(lines) + "\n"


def render_standards_readme() -> str:
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
        "",
        "# Standards",
        "",
        "This section holds repo-wide standards and governance-linked policy documents.",
        "",
        "The standards are written with a two-layer reading model:",
        "",
        "- general JAX-library rule:",
        "  the part intended to be reusable across JAX-first numerical libraries",
        "- arbPlusJAX specialization:",
        "  the repo-specific adaptation, naming, or ownership choice for this library",
        "",
        "The detailed standards still overlap in a few places, so the practical reading model is concept-first rather than filename-first. The current standards set consolidates into eight primary concept groups.",
        "",
        "## Reading Rule",
        "",
        "Read the standards in this order:",
        "",
        "1. reusable owner standard",
        "2. companion standard that narrows one aspect of that owner",
        "3. arbPlusJAX specialization standard for a concrete tranche or family",
        "",
        "Do not treat every file here as an equal top-level owner.",
        "",
        "The preferred split is:",
        "",
        "- reusable owner standards:",
        "  standards that should make sense for a general JAX-first numerical library",
        "- arbPlusJAX specialization standards:",
        "  standards that apply the owner rules to this repo's concrete surface kinds, family tranches, naming, or rollout state",
        "",
        "## Consolidated Concept Groups",
        "",
        "### 1. Runtime, Numerics, and Production Calling",
        "",
        "Primary owner:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'jax_api_runtime_standard.md')}",
        "",
        "Specialized companion documents:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'api_surface_kinds_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'engineering_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'backend_realized_performance_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'error_handling_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'caching_recompilation_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'implicit_adjoint_operator_solve_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'jax_surface_policy_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'lazy_loading_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'logging_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'metadata_registry_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'precision_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'configuration_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'core_scalar_service_calling_standard.md')}",
        "",
        "Consolidation note:",
        "- treat `jax_api_runtime_standard.md` as the canonical runtime/API contract",
        "- treat `api_surface_kinds_standard.md` as the canonical taxonomy for direct, light-wrapper, bound-service, compiled-bound, diagnostics-bearing, prepared-plan, and policy-helper public surfaces",
        "- treat `engineering_standard.md` as the hardening and status-interpretation overlay",
        "- treat `backend_realized_performance_standard.md` as the practical backend-performance companion layered on top of structural fast-JAX/runtime readiness",
        "- treat `error_handling_standard.md` as the canonical hard-failure, numerical-status, fallback, and shared status-code companion",
        "- treat `caching_recompilation_standard.md` as the explicit cache, binder-reuse, prepared-plan, and recompilation-discipline companion",
        "- treat `implicit_adjoint_operator_solve_standard.md` as the operator-first solve, transpose-solve, and implicit-adjoint differentiation companion",
        "- treat `lazy_loading_standard.md` as the canonical import-time load and public lazy-boundary companion",
        "- treat `logging_standard.md` as the canonical structured logging, verbosity, and off-hot-path emission companion",
        "- treat `metadata_registry_standard.md` as the canonical public metadata/capability reporting companion",
        "- treat `configuration_standard.md` as the checked-in runtime/optional-backend configuration companion",
        "- treat `core_scalar_service_calling_standard.md` as a tranche-specific specialization, not a second general runtime policy",
        "- API calling shape, binder reuse, diagnostics payloads, error policy, logging hooks, optional backend declaration, fallback visibility, and the rule that diagnostics/profiling stay outside the mandatory numeric hot path all belong to this runtime concept",
        "- `fast_jax_standard.md`, `operational_jax_standard.md`, and `core_scalar_service_calling_standard.md` together replace the older point-specific owner standards",
        "",
        "### 2. Validation, Benchmarking, and Executable Examples",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'benchmark_validation_policy_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'example_notebook_standard.md')}",
        "",
        "Specialized companion documents:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'benchmark_grouping_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'api_usability_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'pytest_test_naming_standard.md')}",
        "",
        "Consolidation note:",
        "- `benchmark_validation_policy_standard.md` owns measurement and benchmark-contract policy",
        "- `benchmark_grouping_standard.md` is the taxonomy companion for the same benchmark concept",
        "- `example_notebook_standard.md` is the executable-teaching analogue of the same validation/communication layer",
        "- `api_usability_standard.md` owns the intended public calling pattern and how notebooks/practical docs should teach it",
        "- `api_usability_standard.md` is broadly reusable; notebook-family or tranche-specific usage documents should be treated as specializations",
        "",
        "### 3. Portability, Startup, and Run Layout",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'environment_portability_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'experiment_layout_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'startup_compile_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'startup_import_boundary_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'startup_probe_standard.md')}",
        "",
        "Consolidation note:",
        "- these documents jointly own where things run, how startup/import/compile behavior is governed, and where artifacts live",
        "- GitHub submission, Windows, Linux/WSL, and Colab portability expectations belong here rather than in ad hoc runbook notes",
        "- the startup standards remain separate because they govern different evidence surfaces: compile budget, import boundary, and startup probes",
        "- repo-root path conventions, retained artifact layout, and checked-in notebook policy are arbPlusJAX specializations of the broader portability/layout layer",
        "",
        "### 4. Contracts And Provider Boundary",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'contract_and_provider_boundary_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'contracts_surface_standard.md')}",
        "",
        "Consolidation note:",
        "- this document consolidates the missing contract-placement and provider-boundary policy into one public-surface concept",
        "- `contracts_surface_standard.md` owns what belongs in the root `contracts/` layer and how binding guarantees differ from implementation notes or status docs",
        "- downstream-facing API capability contracts, metadata guarantees, and provider-grade surface rules belong here rather than in ad hoc per-family notes",
        "- the generic rule is that libraries should expose stable capability contracts; the arbPlusJAX specialization is which provider and capability terms this repo actually uses",
        "",
        "### 5. Documentation, Implementation, and Generated Communication Surfaces",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'code_documentation_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'specs_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'objects_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'generated_documentation_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'implementation_docs_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'report_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'status_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'repo_standards.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'update_standard.md')}",
        "",
        "Consolidation note:",
        "- `code_documentation_standard.md` owns code-local docstrings, inline comments, and code-to-doc linkage",
        "- `specs_standard.md` owns what belongs in `docs/specs/` and how semantic specifications differ from contracts, theory, and status",
        "- `objects_standard.md` owns what belongs in `docs/objects/` and how catalogs/inventories differ from generated reports",
        "- `generated_documentation_standard.md` owns the shared generation rule",
        "- `implementation_docs_standard.md` owns what belongs in `docs/implementation/` and how implementation mapping differs from contracts, reports, and status",
        "- `repo_standards.md` owns repo-root communication and placement",
        "- `update_standard.md` is the operational companion for repo-maintained artifact refresh/update discipline",
        "- the report/status standards remain specialized audience documents under the same documentation-output concept",
        "- repo communication and generated landing pages are mostly arbPlusJAX-specific, but the split between governance, implementation, reports, status, and generated indexes should remain reusable",
        "",
        "### 6. Release, Security, Support, and Capability Governance",
        "",
        "Primary owners:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'release_packaging_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'docs_build_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'docs_publishing_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'release_governance_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'security_supply_chain_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'operational_support_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'capability_maturity_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'production_readiness_standard.md')}",
        "",
        "Consolidation note:",
        "- these documents own the missing production-library layer that sits above runtime correctness and beneath public release claims",
        "- `docs_build_standard.md` owns the Markdown-first source/build/rewrite model for Sphinx/MyST publication",
        "- `production_readiness_standard.md` owns the Markdown-first governance model that ties standards, status, reports, and automation together",
        "- release build verification, docs publishing, changelog/support/security entrypoints, and capability reporting should be governed explicitly rather than left as repo folklore",
        "",
        "### 7. Theory, Notation, and Naming Semantics",
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
        "- mathematical-family naming and point/basic terminology are arbPlusJAX specializations layered on top of more general notation discipline",
        "",
        "### 8. ArbPlusJAX Specialization Standards",
        "",
        "These standards should remain separate from the more reusable owner standards, but they are best read as one specialization layer rather than as many unrelated top-level policies.",
        "",
        "Primary specialization documents:",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'special_function_ad_standard.md')}",
        f"- {_doc_link(DOCS_ROOT / 'standards' / 'core_scalar_service_calling_standard.md')}",
        "",
        "Consolidation note:",
        "- these documents are intentionally not merged into the general owner standards because they encode repo-specific tranche rules and family-level calling guidance",
        "- read them only after the owner standards for runtime, validation, portability/startup, and release governance",
        "",
        "## Detailed Standards",
        ]
    files = _sorted_markdown_files(DOCS_ROOT / "standards")
    lines.extend(f"- {_doc_link(path)}" for path in files)
    _append_hidden_toctree(
        lines,
        [path.stem for path in files],
    )
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
        "Last updated: 2026-03-25T00:00:00Z",
        "",
        "# Notation",
        "",
        "This section holds authoritative notation conventions, symbol tables, and naming bridges between code and mathematics.",
        "",
        "Current notation documents:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    _append_hidden_toctree(lines, [path.stem for path in files])
    return "\n".join(lines) + "\n"


def render_reports_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "reports")
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
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
    _append_hidden_toctree(lines, [path.stem for path in files])
    return "\n".join(lines) + "\n"


def render_status_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "status")
    primary = ["todo.md", "audit.md", "test_coverage_matrix.md", "test_gap_checklist.md"]
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
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
    _append_hidden_toctree(lines, [path.stem for path in files])
    return "\n".join(lines) + "\n"


def render_theory_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "theory")
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
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
    _append_hidden_toctree(lines, [path.stem for path in files])
    lines.extend(
        [
            "",
            "Current theory status remains `in_progress`: the main interval/core/matrix/transform/gamma foundations are documented, but further methodology notes should continue to be added as new hardened families land.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_implementation_readme() -> str:
    files = _sorted_markdown_files(DOCS_ROOT / "implementation")
    module_files = _sorted_markdown_files(DOCS_ROOT / "implementation" / "modules")
    wrapper_files = _sorted_markdown_files(DOCS_ROOT / "implementation" / "wrappers")
    external_files = _sorted_markdown_files(DOCS_ROOT / "implementation" / "external")
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
        "",
        "# Implementation",
        "",
        "This section holds implementation-facing notes: subsystem mapping, wrapper layout, build/bootstrap notes, benchmark/process notes, and external lineage reviews.",
        "",
        "Implementation notes explain how the code is built and organized. Practical runbooks and operator guidance belong under [practical/README.md](/docs/practical/README.md). Theory and methodology belong under [theory/README.md](/docs/theory/README.md).",
        "",
        "## Subtree Indexes",
        "",
        f"- module notes ({len(module_files)} files): {_doc_link(DOCS_ROOT / 'implementation' / 'modules' / 'README.md')}",
        f"- wrapper notes ({len(wrapper_files)} files): {_doc_link(DOCS_ROOT / 'implementation' / 'wrappers' / 'README.md')}",
        f"- external lineage reviews ({len(external_files)} files): {_doc_link(DOCS_ROOT / 'implementation' / 'external' / 'README.md')}",
        "",
        "## Direct Implementation Notes",
        "",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    toctree_entries = [
        "modules/README",
        "wrappers/README",
        "external/README",
        *[path.stem for path in files],
    ]
    _append_hidden_toctree(lines, toctree_entries)
    return "\n".join(lines) + "\n"


def render_implementation_subtree_readme(
    folder: Path,
    *,
    title: str,
    description: str,
) -> str:
    files = _sorted_markdown_files(folder)
    lines = [
        "Last updated: 2026-03-25T00:00:00Z",
        "",
        f"# {title}",
        "",
        description,
        "",
        f"This index is generated from `{folder.relative_to(REPO_ROOT).as_posix()}/` and should be refreshed through `python tools/check_generated_reports.py` before commit/push.",
        "",
        "Current documents:",
    ]
    lines.extend(f"- {_doc_link(path)}" for path in files)
    _append_hidden_toctree(lines, [path.stem for path in files])
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
        "Last updated: 2026-03-25T00:00:00Z",
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
        REPO_ROOT / "README.md": render_root_readme(),
        DOCS_ROOT / "README.md": render_docs_readme(),
        DOCS_ROOT / "project_overview.md": render_project_overview(),
        DOCS_ROOT / "governance" / "README.md": render_governance_readme(),
        DOCS_ROOT / "implementation" / "README.md": render_implementation_readme(),
        DOCS_ROOT / "implementation" / "modules" / "README.md": render_implementation_subtree_readme(
            DOCS_ROOT / "implementation" / "modules",
            title="Implementation Modules",
            description="This section indexes module-level implementation notes for Arb/ACB scalar, matrix, transform, special-function, and supporting numeric subsystems.",
        ),
        DOCS_ROOT / "implementation" / "wrappers" / "README.md": render_implementation_subtree_readme(
            DOCS_ROOT / "implementation" / "wrappers",
            title="Implementation Wrappers",
            description="This section indexes wrapper and mode-dispatch notes for the public calling layer, interval/point routing, and related runtime support surfaces.",
        ),
        DOCS_ROOT / "implementation" / "external" / "README.md": render_implementation_subtree_readme(
            DOCS_ROOT / "implementation" / "external",
            title="Implementation External Reviews",
            description="This section indexes external lineage reviews and compatibility notes for upstream or adjacent systems that inform arbPlusJAX implementation work.",
        ),
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
