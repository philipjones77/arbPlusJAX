from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from arbplusjax import api


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "point_basic_surface_status.md"
TARGETS_PATH = REPO_ROOT / "tests" / "targets.csv"
CAPABILITY_REGISTRY_PATH = REPO_ROOT / "docs" / "reports" / "function_capability_registry.json"


FAMILY_EVIDENCE: dict[str, dict[str, tuple[str, ...]]] = {
    "core": {
        "tests": (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_core_scalar_api_contracts.py",
        ),
        "benchmarks": (
            "benchmarks/benchmark_core_scalar_service_api.py",
            "benchmarks/benchmark_api_surface.py",
        ),
        "notebooks": (
            "examples/example_core_scalar_surface.ipynb",
            "examples/example_api_surface.ipynb",
        ),
    },
    "bessel": {
        "tests": (
            "tests/test_special_function_hardening.py",
            "tests/test_incomplete_bessel_i.py",
        ),
        "benchmarks": (
            "benchmarks/benchmark_incomplete_bessel.py",
            "benchmarks/benchmark_special_function_service_api.py",
        ),
        "notebooks": ("examples/example_bessel_modes_sweep.ipynb",),
    },
    "gamma": {
        "tests": (
            "tests/test_special_function_hardening.py",
            "tests/test_incomplete_gamma.py",
        ),
        "benchmarks": (
            "benchmarks/benchmark_gamma_compare.py",
            "benchmarks/benchmark_special_function_service_api.py",
        ),
        "notebooks": ("examples/example_gamma_family_surface.ipynb",),
    },
    "hypergeometric": {
        "tests": (
            "tests/test_special_function_hardening.py",
            "tests/test_hypgeom_wrappers_contracts.py",
        ),
        "benchmarks": (
            "benchmarks/benchmark_hypgeom.py",
            "benchmarks/benchmark_special_function_service_api.py",
        ),
        "notebooks": ("examples/example_hypgeom_family_surface.ipynb",),
    },
    "barnes": {
        "tests": (
            "tests/test_special_function_hardening.py",
            "tests/test_barnes_double_gamma_ifj_contracts.py",
        ),
        "benchmarks": ("benchmarks/benchmark_barnes_double_gamma.py",),
        "notebooks": ("examples/example_barnes_double_gamma_surface.ipynb",),
    },
    "integration": {
        "tests": (
            "tests/test_api_selection_contracts.py",
            "tests/test_core_scalar_api_contracts.py",
        ),
        "benchmarks": (
            "benchmarks/benchmark_acb_calc.py",
            "benchmarks/benchmark_arb_calc.py",
        ),
        "notebooks": ("examples/example_calc_modes_demo.ipynb",),
    },
    "matrix": {
        "tests": (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_sparse_format_modes.py",
            "tests/test_matrix_free_basic.py",
        ),
        "benchmarks": (
            "benchmarks/benchmark_dense_matrix_surface.py",
            "benchmarks/benchmark_sparse_matrix_surface.py",
            "benchmarks/benchmark_matrix_free_krylov.py",
        ),
        "notebooks": (
            "examples/example_dense_matrix_surface.ipynb",
            "examples/example_sparse_matrix_surface.ipynb",
            "examples/example_matrix_free_operator_surface.ipynb",
        ),
    },
}

CURVATURE_EVIDENCE = {
    "tests": (
        "tests/test_curvature_contracts.py",
        "tests/test_sparse_format_modes.py",
    ),
    "benchmarks": ("benchmarks/benchmark_matrix_free_krylov.py",),
    "notebooks": ("examples/example_matrix_free_operator_surface.ipynb",),
}

FAMILY_AD_EVIDENCE: dict[str, dict[str, tuple[str, ...]] | str] = {
    "core": {
        "tests": ("tests/test_parameterized_family_ad_directions.py", "tests/test_parameterized_public_ad_audit.py"),
        "benchmarks": ("benchmarks/public_surface_ad_benchmark.py",),
        "notebooks": ("examples/example_core_scalar_surface.ipynb", "examples/example_api_surface.ipynb"),
    },
    "bessel": {
        "tests": ("tests/test_special_function_ad_directions.py", "tests/test_parameterized_public_ad_audit.py"),
        "benchmarks": ("benchmarks/special_function_ad_benchmark.py",),
        "notebooks": ("examples/example_bessel_modes_sweep.ipynb",),
    },
    "gamma": {
        "tests": ("tests/test_special_function_ad_directions.py", "tests/test_parameterized_public_ad_audit.py"),
        "benchmarks": ("benchmarks/special_function_ad_benchmark.py",),
        "notebooks": ("examples/example_gamma_family_surface.ipynb",),
    },
    "hypergeometric": {
        "tests": ("tests/test_special_function_ad_directions.py", "tests/test_parameterized_public_ad_audit.py"),
        "benchmarks": ("benchmarks/special_function_ad_benchmark.py",),
        "notebooks": ("examples/example_hypgeom_family_surface.ipynb",),
    },
    "barnes": {
        "tests": ("tests/test_special_function_ad_directions.py", "tests/test_parameterized_public_ad_audit.py"),
        "benchmarks": ("benchmarks/special_function_ad_benchmark.py",),
        "notebooks": ("examples/example_barnes_double_gamma_surface.ipynb",),
    },
    "integration": "n/a_non_parameterized",
    "matrix": {
        "tests": ("tests/test_parameterized_family_ad_directions.py", "tests/test_parameterized_public_ad_audit.py"),
        "benchmarks": ("benchmarks/public_surface_ad_benchmark.py",),
        "notebooks": (
            "examples/example_dense_matrix_surface.ipynb",
            "examples/example_sparse_matrix_surface.ipynb",
            "examples/example_matrix_free_operator_surface.ipynb",
        ),
    },
}


@dataclass(frozen=True)
class FamilyRow:
    family: str
    point_count: int
    basic_count: int
    diagnostics_count: int
    tested_direct_matches: int
    evidence_status: str
    ad_status: str
    representative_surfaces: str
    evidence: str


def _load_targets_tested() -> set[str]:
    tested: set[str] = set()
    with TARGETS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("tested") == "yes":
                tested.add(row["function"])
    return tested


def _load_capability_registry() -> dict[str, dict[str, object]]:
    payload = json.loads(CAPABILITY_REGISTRY_PATH.read_text(encoding="utf-8"))
    return payload["functions"]


def _family_entries() -> dict[str, list]:
    grouped: dict[str, list] = {}
    for entry in api.list_public_function_metadata():
        if not entry.point_support and "basic" not in entry.interval_modes:
            continue
        grouped.setdefault(entry.family, []).append(entry)
    return grouped


def _evidence_status(paths: tuple[str, ...], other: tuple[str, ...], notebooks: tuple[str, ...]) -> str:
    all_paths = (*paths, *other, *notebooks)
    return "complete" if all((REPO_ROOT / p).exists() for p in all_paths) else "incomplete"


def _representative_surfaces(entries: list) -> str:
    point_names = sorted(entry.name for entry in entries if entry.point_support)[:2]
    basic_names = sorted(entry.name for entry in entries if "basic" in entry.interval_modes)[:2]
    chunks: list[str] = []
    if point_names:
        chunks.append("point: " + ", ".join(f"`{name}`" for name in point_names))
    if basic_names:
        chunks.append("basic: " + ", ".join(f"`{name}`" for name in basic_names))
    return "<br>".join(chunks) if chunks else "n/a"


def _evidence_text(family: str) -> str:
    evidence = FAMILY_EVIDENCE[family]
    return "<br>".join(
        [
            "tests: " + ", ".join(f"[{Path(path).name}](/{path})" for path in evidence["tests"]),
            "benchmarks: " + ", ".join(f"[{Path(path).name}](/{path})" for path in evidence["benchmarks"]),
            "notebooks: " + ", ".join(f"[{Path(path).name}](/{path})" for path in evidence["notebooks"]),
        ]
    )


def _ad_status(family: str) -> str:
    record = FAMILY_AD_EVIDENCE[family]
    if isinstance(record, str):
        return record
    all_paths = (*record["tests"], *record["benchmarks"], *record["notebooks"])
    return "argument+parameter" if all((REPO_ROOT / path).exists() for path in all_paths) else "partial"


def _ad_evidence_text(family: str) -> str:
    record = FAMILY_AD_EVIDENCE[family]
    if isinstance(record, str):
        return record
    return "<br>".join(
        [
            "tests: " + ", ".join(f"[{Path(path).name}](/{path})" for path in record["tests"]),
            "benchmarks: " + ", ".join(f"[{Path(path).name}](/{path})" for path in record["benchmarks"]),
            "notebooks: " + ", ".join(f"[{Path(path).name}](/{path})" for path in record["notebooks"]),
        ]
    )


def build_rows() -> list[FamilyRow]:
    tested = _load_targets_tested()
    registry = _load_capability_registry()
    grouped = _family_entries()
    rows: list[FamilyRow] = []
    for family in sorted(grouped):
        entries = grouped[family]
        point_count = sum(entry.point_support for entry in entries)
        basic_count = sum("basic" in entry.interval_modes for entry in entries)
        diagnostics_count = sum(
            1
            for payload in registry.values()
            if payload["family"] == family
            and (
                "diagnostics" in str(payload["name"]).lower()
                or "with_diagnostics" in str(payload["name"]).lower()
            )
            and (payload["point_support"] or "basic" in payload["interval_modes"])
        )
        tested_direct_matches = sum(
            1 for entry in entries if (entry.point_support or "basic" in entry.interval_modes) and entry.name in tested
        )
        evidence = FAMILY_EVIDENCE[family]
        rows.append(
            FamilyRow(
                family=family,
                point_count=point_count,
                basic_count=basic_count,
                diagnostics_count=diagnostics_count,
                tested_direct_matches=tested_direct_matches,
                evidence_status=_evidence_status(evidence["tests"], evidence["benchmarks"], evidence["notebooks"]),
                ad_status=_ad_status(family),
                representative_surfaces=_representative_surfaces(entries),
                evidence=_evidence_text(family) + "<br><br>AD: " + _ad_evidence_text(family),
            )
        )
    return rows


def render() -> str:
    rows = build_rows()
    total_point = sum(row.point_count for row in rows)
    total_basic = sum(row.basic_count for row in rows)
    lines = [
        "Last updated: 2026-03-27T00:00:00Z",
        "",
        "# Point And Basic Surface Status",
        "",
        "This report joins the public metadata registry, the checked-in target inventory, and the current test/benchmark/notebook evidence for the public `point` and `basic` surfaces.",
        "",
        "Policy references:",
        "- [point_surface_standard.md](/docs/standards/point_surface_standard.md)",
        "- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)",
        "- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)",
        "- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)",
        "- [theory_notation_standard.md](/docs/standards/theory_notation_standard.md)",
        "",
        "Interpretation:",
        "- `point_count` is the number of public metadata entries that expose a public point surface.",
        "- `basic_count` is the number of public metadata entries that explicitly advertise `mode=\"basic\"`.",
        "- `diagnostics_count` is the number of public capability-registry entries in the family that expose diagnostics-bearing point/basic helper surfaces.",
        "- `tested_direct_matches` is a conservative direct-name match against [tests/targets.csv](/tests/targets.csv); aliases and routed names may be tested without matching this count.",
        "- `evidence_status=complete` means the mapped owner tests, benchmarks, and notebooks all exist for the family.",
        "- `ad_status=argument+parameter` means the family has explicit argument-direction and parameter-direction AD evidence in tests, benchmarks, and canonical notebooks.",
        "- The per-function verification ledger is published in [point_basic_function_verification.md](/docs/reports/point_basic_function_verification.md).",
        "",
        f"Total public point entries: `{total_point}`",
        f"Total public basic entries: `{total_basic}`",
        "",
        "## Family Audit",
        "",
        "| family | point_count | basic_count | diagnostics_count | tested_direct_matches | evidence_status | ad_status | representative verified surfaces | evidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.family}` | `{row.point_count}` | `{row.basic_count}` | `{row.diagnostics_count}` | `{row.tested_direct_matches}` | `{row.evidence_status}` | `{row.ad_status}` | {row.representative_surfaces} | {row.evidence} |"
        )
    lines.extend(
        [
            "",
            "## Cross Helper Audit",
            "",
            "| helper | point/basic role | evidence_status | ad_status | evidence |",
            "|---|---|---|---|---|",
            "| `curvature` | cross-helper layer that consumes point/basic matrix and matrix-free operator surfaces | "
            + f"`{_evidence_status(CURVATURE_EVIDENCE['tests'], CURVATURE_EVIDENCE['benchmarks'], CURVATURE_EVIDENCE['notebooks'])}`"
            + " | `argument+parameter` | "
            + "<br>".join(
                [
                    "tests: " + ", ".join(f"[{Path(path).name}](/{path})" for path in CURVATURE_EVIDENCE["tests"]),
                    "benchmarks: "
                    + ", ".join(f"[{Path(path).name}](/{path})" for path in CURVATURE_EVIDENCE["benchmarks"]),
                    "notebooks: "
                    + ", ".join(f"[{Path(path).name}](/{path})" for path in CURVATURE_EVIDENCE["notebooks"]),
                ]
            )
            + " |",
            "",
            "## Current Status Note",
            "",
            "- The public point surface is broad and fully registered, while public `basic` support remains concentrated in the enclosure-oriented scalar and special-function families.",
            "- AD evidence is tracked separately as argument-direction plus parameter-direction proof, not as a single generic AD claim.",
            "- Dense/sparse/matrix-free matrix families are currently point-first at the public metadata layer; where `basic` semantics matter there, they are often carried by dedicated helper surfaces and diagnostics-bearing wrappers rather than by universal family-level `mode=\"basic\"` exposure.",
            "- This report is the family-level verification ledger. The machine-derived per-function companion lives in [point_basic_function_verification.md](/docs/reports/point_basic_function_verification.md).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
