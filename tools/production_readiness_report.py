from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "production_readiness.md"


def _exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def _all_exist(paths: tuple[str, ...]) -> bool:
    return all(_exists(path) for path in paths)


def _any_exist(paths: tuple[str, ...]) -> bool:
    return any(_exists(path) for path in paths)


def _status_for(paths: tuple[str, ...]) -> str:
    if _all_exist(paths):
        return "present"
    if _any_exist(paths):
        return "partial"
    return "missing"


def _todo_category_statuses() -> list[tuple[str, str]]:
    text = (REPO_ROOT / "docs" / "status" / "todo.md").read_text(encoding="utf-8")
    pattern = re.compile(r"##\s+([^\n]+)\n\nStatus:\s+`([^`]+)`")
    rows: list[tuple[str, str]] = []
    for name, status in pattern.findall(text):
        if name[:1].isdigit():
            rows.append((name.strip(), status.strip()))
    return rows


AREAS = (
    {
        "area": "release and packaging",
        "standards": (
            "docs/standards/release_packaging_standard.md",
            "docs/standards/production_readiness_standard.md",
        ),
        "status": ("docs/status/release_packaging_todo.md",),
        "automation": (
            ".github/workflows/build-dist.yml",
            ".github/workflows/publish-release.yml",
            "pyproject.toml",
        ),
    },
    {
        "area": "docs publishing",
        "standards": (
            "docs/standards/docs_publishing_standard.md",
            "docs/standards/production_readiness_standard.md",
        ),
        "status": ("docs/status/docs_publishing_todo.md",),
        "automation": (".github/workflows/docs-publish.yml", "pyproject.toml"),
    },
    {
        "area": "release governance",
        "standards": ("docs/standards/release_governance_standard.md",),
        "status": ("docs/status/production_readiness_todo.md",),
        "automation": ("CHANGELOG.md",),
    },
    {
        "area": "security and supply chain",
        "standards": ("docs/standards/security_supply_chain_standard.md",),
        "status": ("docs/status/security_supply_chain_todo.md",),
        "automation": ("SECURITY.md", ".github/workflows/dependency-audit.yml"),
    },
    {
        "area": "operational support",
        "standards": ("docs/standards/operational_support_standard.md",),
        "status": ("docs/status/operational_support_todo.md",),
        "automation": ("CONTRIBUTING.md", "SUPPORT.md"),
    },
    {
        "area": "capability and maturity reporting",
        "standards": ("docs/standards/capability_maturity_standard.md",),
        "status": ("docs/status/capability_maturity_todo.md",),
        "automation": (),
    },
)


def render() -> str:
    lines = [
        "Last updated: 2026-03-29T00:00:00Z",
        "",
        "# Production Readiness",
        "",
        "This generated report summarizes the repo's production-readiness governance layer.",
        "",
        "It is the current-state companion to:",
        "- [production_readiness_standard.md](/docs/standards/production_readiness_standard.md)",
        "- [release_packaging_standard.md](/docs/standards/release_packaging_standard.md)",
        "- [docs_publishing_standard.md](/docs/standards/docs_publishing_standard.md)",
        "- [release_governance_standard.md](/docs/standards/release_governance_standard.md)",
        "- [security_supply_chain_standard.md](/docs/standards/security_supply_chain_standard.md)",
        "- [operational_support_standard.md](/docs/standards/operational_support_standard.md)",
        "- [capability_maturity_standard.md](/docs/standards/capability_maturity_standard.md)",
        "",
        "Interpretation:",
        "- `present`: the required Markdown and automation surfaces exist",
        "- `partial`: some, but not all, required surfaces exist",
        "- `missing`: the governed surfaces are not yet in place",
        "",
        "| area | standards | status lane | automation / repo surface | readiness |",
        "|---|---|---|---|---|",
    ]
    for row in AREAS:
        readiness = _status_for(row["standards"] + row["status"] + row["automation"])
        standards = ", ".join(f"[{Path(path).name}](/{path})" for path in row["standards"])
        status = ", ".join(f"[{Path(path).name}](/{path})" for path in row["status"])
        automation = (
            ", ".join(f"[{Path(path).name}](/{path})" for path in row["automation"])
            if row["automation"]
            else "`planned`"
        )
        lines.append(f"| `{row['area']}` | {standards} | {status} | {automation} | `{readiness}` |")

    lines.extend(
        [
            "",
            "## Packaging Extras",
            "",
            "Current expected optional dependency groups from `pyproject.toml`:",
            "- `compare`",
            "- `docs`",
            "- `dev`",
            "- `bench`",
            "- `release`",
            "- `colab`",
            "",
            "## Main Function-Category Closeout Snapshot",
            "",
            "| category | current status |",
            "|---|---|",
        ]
    )
    for name, status in _todo_category_statuses():
        lines.append(f"| `{name}` | `{status}` |")

    lines.extend(
        [
            "",
            "## Current Reading",
            "",
            "- This report measures structure and governance presence, not full implementation quality.",
            "- Category statuses above come from [todo.md](/docs/status/todo.md) and remain the canonical implementation-state signal.",
            "- Production claims should rely on this report together with the category-specific reports and standards verification surfaces.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
