from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from arbplusjax import api

try:
    from tools import point_basic_surface_report as family_report
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import point_basic_surface_report as family_report


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "point_basic_function_verification.md"
TARGETS_PATH = REPO_ROOT / "tests" / "targets.csv"


@dataclass(frozen=True)
class FunctionRow:
    name: str
    family: str
    point: str
    basic: str
    diagnostics: str
    direct_tested: str
    benchmark_evidence: str
    notebook_evidence: str
    ad_status: str
    verification_status: str


def _load_targets_tested() -> set[str]:
    tested: set[str] = set()
    with TARGETS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("tested") == "yes":
                tested.add(row["function"])
    return tested


def _public_entries():
    return [
        entry
        for entry in api.list_public_function_metadata()
        if entry.point_support or "basic" in entry.interval_modes
    ]


def _family_evidence_complete(family: str) -> bool:
    if family not in family_report.FAMILY_EVIDENCE:
        return False
    evidence = family_report.FAMILY_EVIDENCE[family]
    return all((REPO_ROOT / rel).exists() for rel in (*evidence["tests"], *evidence["benchmarks"], *evidence["notebooks"]))


def _family_has_benchmark_evidence(family: str) -> bool:
    evidence = family_report.FAMILY_EVIDENCE.get(family)
    return bool(evidence) and all((REPO_ROOT / rel).exists() for rel in evidence["benchmarks"])


def _family_has_notebook_evidence(family: str) -> bool:
    evidence = family_report.FAMILY_EVIDENCE.get(family)
    return bool(evidence) and all((REPO_ROOT / rel).exists() for rel in evidence["notebooks"])


def _family_ad_status(family: str) -> str:
    status = family_report.FAMILY_AD_EVIDENCE.get(family)
    if status is None:
        return "not_mapped"
    if isinstance(status, str):
        return status
    paths = (*status["tests"], *status["benchmarks"], *status["notebooks"])
    return "argument+parameter" if all((REPO_ROOT / rel).exists() for rel in paths) else "partial"


def _has_diagnostics_surface(name: str) -> bool:
    if "diagnostics" in name.lower() or "with_diagnostics" in name.lower():
        return True
    suffixes = (
        f"{name}_with_diagnostics_point",
        f"{name}_with_diagnostics_basic",
        f"{name}_diagnostics",
    )
    public_names = {entry.name for entry in _public_entries()}
    return any(candidate in public_names for candidate in suffixes)


def _verification_status(
    *,
    direct_tested: bool,
    benchmark_evidence: bool,
    notebook_evidence: bool,
    family_evidence_complete: bool,
) -> str:
    if direct_tested and benchmark_evidence and notebook_evidence and family_evidence_complete:
        return "verified"
    if direct_tested:
        return "tested_only"
    if benchmark_evidence or notebook_evidence:
        return "family_evidence_only"
    return "unverified"


def build_rows() -> list[FunctionRow]:
    tested = _load_targets_tested()
    public_names = {entry.name for entry in _public_entries()}
    rows: list[FunctionRow] = []
    for entry in sorted(_public_entries(), key=lambda row: (row.family, row.name)):
        direct_tested = entry.name in tested
        benchmark_evidence = _family_has_benchmark_evidence(entry.family)
        notebook_evidence = _family_has_notebook_evidence(entry.family)
        rows.append(
            FunctionRow(
                name=entry.name,
                family=entry.family,
                point="yes" if entry.point_support else "no",
                basic="yes" if "basic" in entry.interval_modes else "no",
                diagnostics="yes"
                if (
                    "diagnostics" in entry.name.lower()
                    or "with_diagnostics" in entry.name.lower()
                    or any(
                        candidate in public_names
                        for candidate in (
                            f"{entry.name}_with_diagnostics_point",
                            f"{entry.name}_with_diagnostics_basic",
                            f"{entry.name}_diagnostics",
                        )
                    )
                )
                else "no",
                direct_tested="yes" if direct_tested else "no",
                benchmark_evidence="yes" if benchmark_evidence else "no",
                notebook_evidence="yes" if notebook_evidence else "no",
                ad_status=_family_ad_status(entry.family),
                verification_status=_verification_status(
                    direct_tested=direct_tested,
                    benchmark_evidence=benchmark_evidence,
                    notebook_evidence=notebook_evidence,
                    family_evidence_complete=_family_evidence_complete(entry.family),
                ),
            )
        )
    return rows


def render() -> str:
    rows = build_rows()
    total = len(rows)
    verified = sum(row.verification_status == "verified" for row in rows)
    direct_tested = sum(row.direct_tested == "yes" for row in rows)
    point_only = sum(row.point == "yes" and row.basic == "no" for row in rows)
    basic_exposed = sum(row.basic == "yes" for row in rows)
    lines = [
        "Last updated: 2026-03-27T00:00:00Z",
        "",
        "# Point And Basic Function Verification",
        "",
        "This report is the per-function verification ledger for public `point` and `basic` surfaces. It is derived from the public metadata registry, [tests/targets.csv](/tests/targets.csv), and the checked-in family benchmark/notebook evidence maps.",
        "",
        "Policy references:",
        "- [point_surface_standard.md](/docs/standards/point_surface_standard.md)",
        "- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)",
        "- [special_function_ad_standard.md](/docs/standards/special_function_ad_standard.md)",
        "- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)",
        "- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)",
        "",
        f"Total public point/basic functions: `{total}`",
        f"Directly tested by name in targets inventory: `{direct_tested}`",
        f"Per-function verified rows: `{verified}`",
        f"Point-only rows: `{point_only}`",
        f"Rows exposing `basic`: `{basic_exposed}`",
        "",
        "Interpretation:",
        "- `direct_tested=yes` is a conservative direct-name match against [tests/targets.csv](/tests/targets.csv).",
        "- `benchmark_evidence` and `notebook_evidence` are inherited from the current owner-family evidence map.",
        "- `verification_status=verified` requires direct target evidence plus owner benchmark and notebook evidence.",
        "- `ad_status=argument+parameter` means the owning family has checked-in evidence for both argument-direction and parameter-direction AD; non-parameterized families are marked `n/a_non_parameterized`.",
        "",
        "| function | family | point | basic | diagnostics | direct_tested | benchmark_evidence | notebook_evidence | ad_status | verification_status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.name}` | `{row.family}` | `{row.point}` | `{row.basic}` | `{row.diagnostics}` | `{row.direct_tested}` | `{row.benchmark_evidence}` | `{row.notebook_evidence}` | `{row.ad_status}` | `{row.verification_status}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
