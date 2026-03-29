from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SNIPPETS = (
    "/home/phili/projects/arbplusJAX",
    "file:///home/phili/projects/arbplusJAX",
)


def _assert_no_forbidden_repo_paths(paths: list[Path]) -> None:
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                violations.append(str(path.relative_to(REPO_ROOT)))
                break
    assert not violations, f"Machine-specific repo paths found in: {violations}"


def test_markdown_files_use_repo_root_links() -> None:
    markdown_paths = sorted(REPO_ROOT.rglob("*.md"))
    _assert_no_forbidden_repo_paths(markdown_paths)


def test_generators_do_not_embed_machine_specific_repo_paths() -> None:
    generator_paths = [
        REPO_ROOT / "tools" / "generate_docs_indexes.py",
        REPO_ROOT / "tools" / "function_provenance_report.py",
        REPO_ROOT / "tools" / "hypgeom_status_report.py",
        REPO_ROOT / "tools" / "generate_example_notebooks.py",
        REPO_ROOT / "tools" / "generate_tools_readme.py",
        REPO_ROOT / "src" / "arbplusjax" / "function_provenance.py",
        REPO_ROOT / "benchmarks" / "matrix_surface_workbook.py",
        REPO_ROOT / "benchmarks" / "benchmark_matrix_suite.py",
    ]
    _assert_no_forbidden_repo_paths(generator_paths)


def test_portable_example_templates_do_not_embed_machine_specific_repo_paths() -> None:
    template_paths = [
        REPO_ROOT / "examples" / "inputs" / "example_run_suite" / "example_run_template.json",
    ]
    _assert_no_forbidden_repo_paths(template_paths)
