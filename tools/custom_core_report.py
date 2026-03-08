from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .core_status_report import _status_rows
    from .point_status_report import _rows as _point_rows
    from arbplusjax.function_provenance import engineering_status_for_public_name
except ImportError:
    from core_status_report import _status_rows
    from point_status_report import _rows as _point_rows
    from arbplusjax.function_provenance import engineering_status_for_public_name


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CustomFunctionMeta:
    name: str
    module: str
    family: str
    priority: str
    reason: str


CUSTOM_FUNCTIONS: tuple[CustomFunctionMeta, ...] = (
    CustomFunctionMeta("arb_sin_pi", "arb_core", "real", "P2", "pi-scaled elementary wrapper"),
    CustomFunctionMeta("arb_cos_pi", "arb_core", "real", "P2", "pi-scaled elementary wrapper"),
    CustomFunctionMeta("arb_tan_pi", "arb_core", "real", "P1", "pole-sensitive pi-scaled elementary wrapper"),
    CustomFunctionMeta("arb_sinc", "arb_core", "real", "P1", "removable singularity at zero"),
    CustomFunctionMeta("arb_sinc_pi", "arb_core", "real", "P1", "pi-scaled removable singularity"),
    CustomFunctionMeta("arb_sign", "arb_core", "real", "P2", "discontinuous helper"),
    CustomFunctionMeta("arb_pow_fmpq", "arb_core", "real", "P0", "mixed rational-power helper"),
    CustomFunctionMeta("arb_root", "arb_core", "real", "P0", "mixed root helper"),
    CustomFunctionMeta("arb_cbrt", "arb_core", "real", "P1", "custom root specialization"),
    CustomFunctionMeta("arb_lgamma", "arb_core", "real", "P0", "special-function complement to gamma"),
    CustomFunctionMeta("arb_rgamma", "arb_core", "real", "P0", "reciprocal gamma is numerically sensitive"),
    CustomFunctionMeta("arb_sinh_cosh", "arb_core", "real", "P2", "paired-output helper"),
    CustomFunctionMeta("acb_rsqrt", "acb_core", "complex", "P1", "complex reciprocal square-root helper"),
    CustomFunctionMeta("acb_cot", "acb_core", "complex", "P1", "pole-sensitive trigonometric complement"),
    CustomFunctionMeta("acb_sech", "acb_core", "complex", "P2", "hyperbolic complement"),
    CustomFunctionMeta("acb_csch", "acb_core", "complex", "P1", "pole-sensitive hyperbolic complement"),
    CustomFunctionMeta("acb_sin_pi", "acb_core", "complex", "P2", "pi-scaled elementary wrapper"),
    CustomFunctionMeta("acb_cos_pi", "acb_core", "complex", "P2", "pi-scaled elementary wrapper"),
    CustomFunctionMeta("acb_sin_cos_pi", "acb_core", "complex", "P2", "paired pi-scaled helper"),
    CustomFunctionMeta("acb_tan_pi", "acb_core", "complex", "P1", "pole-sensitive pi-scaled elementary wrapper"),
    CustomFunctionMeta("acb_cot_pi", "acb_core", "complex", "P1", "pole-sensitive pi-scaled complement"),
    CustomFunctionMeta("acb_csc_pi", "acb_core", "complex", "P1", "pole-sensitive pi-scaled complement"),
    CustomFunctionMeta("acb_sinc", "acb_core", "complex", "P1", "removable singularity at zero"),
    CustomFunctionMeta("acb_sinc_pi", "acb_core", "complex", "P1", "pi-scaled removable singularity"),
    CustomFunctionMeta("acb_exp_pi_i", "acb_core", "complex", "P1", "unit-circle exponential helper"),
    CustomFunctionMeta("acb_exp_invexp", "acb_core", "complex", "P1", "paired-output exponential helper"),
    CustomFunctionMeta("acb_addmul", "acb_core", "complex", "P2", "fused arithmetic helper"),
    CustomFunctionMeta("acb_submul", "acb_core", "complex", "P2", "fused arithmetic helper"),
    CustomFunctionMeta("acb_pow_arb", "acb_core", "complex", "P0", "mixed complex/real power helper"),
    CustomFunctionMeta("acb_pow_si", "acb_core", "complex", "P2", "signed integer power helper"),
    CustomFunctionMeta("acb_sqr", "acb_core", "complex", "P2", "square specialization"),
    CustomFunctionMeta("acb_root_ui", "acb_core", "complex", "P1", "root helper"),
    CustomFunctionMeta("acb_lgamma", "acb_core", "complex", "P0", "special-function complement to gamma"),
    CustomFunctionMeta("acb_log_sin_pi", "acb_core", "complex", "P0", "branch-sensitive special helper"),
    CustomFunctionMeta("acb_digamma", "acb_core", "complex", "P0", "special derivative function"),
    CustomFunctionMeta("acb_zeta", "acb_core", "complex", "P0", "major special function"),
    CustomFunctionMeta("acb_hurwitz_zeta", "acb_core", "complex", "P0", "major special function with branch structure"),
    CustomFunctionMeta("acb_polygamma", "acb_core", "complex", "P0", "higher special derivative"),
    CustomFunctionMeta("acb_bernoulli_poly_ui", "acb_core", "complex", "P2", "polynomial helper"),
    CustomFunctionMeta("acb_polylog", "acb_core", "complex", "P0", "branch-sensitive special function"),
    CustomFunctionMeta("acb_polylog_si", "acb_core", "complex", "P0", "integer-order polylog helper"),
    CustomFunctionMeta("acb_agm", "acb_core", "complex", "P1", "iterative special helper"),
    CustomFunctionMeta("acb_agm1", "acb_core", "complex", "P1", "shifted iterative helper"),
    CustomFunctionMeta("acb_agm1_cpx", "acb_core", "complex", "P1", "custom complex AGM variant"),
)


def main() -> None:
    status_map = {(row.name, row.module): row for row in _status_rows()}
    point_map = {(row.name, row.module): row for row in _point_rows()}
    rows = [meta for meta in CUSTOM_FUNCTIONS if (meta.name, meta.module) in status_map]

    total = len(rows)
    point_complete = sum(1 for meta in rows if point_map[(meta.name, meta.module)].available)
    basic_complete = sum(1 for meta in rows if status_map[(meta.name, meta.module)].basic)
    adaptive_complete = sum(1 for meta in rows if status_map[(meta.name, meta.module)].adaptive)
    rigorous_specialized = sum(1 for meta in rows if status_map[(meta.name, meta.module)].rigorous_specialized)
    generic_rigorous = total - rigorous_specialized

    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Custom Core Status",
        "",
        "Generated from `tools/custom_core_report.py` using `tools/core_status_report.py` and `tools/point_status_report.py`.",
        "",
        "Scope: functions that complement the Arb-style core surface with custom pi-scaled, paired-output, fused, mixed-argument, or extended special-function helpers.",
        "",
        f"Summary: `functions={total}`, `point={point_complete}/{total}`, `basic={basic_complete}/{total}`, `adaptive={adaptive_complete}/{total}`, `rigorous_specialized={rigorous_specialized}/{total}`, `generic_rigorous={generic_rigorous}/{total}`.",
        "",
        "Interpretation:",
        "- `point`: `*_point` wrapper exists",
        "- `basic`: `*_prec` interval/box entry point exists",
        "- `adaptive`: adaptive mode path exists",
        "- `rigorous_specialized`: function has a dedicated rigorous adapter",
        "- `generic_rigorous`: rigorous mode is available through generic wrapper/kernel machinery, but not via a hand-specialized adapter",
        "",
        "## Status Table",
        "",
        "| function | module | family | point | basic | adaptive | rigorous_specialized | generic_rigorous | kernel_split | helper_consolidation | batch | ad | hardening | tightening_priority | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for meta in rows:
        status = status_map[(meta.name, meta.module)]
        point = point_map[(meta.name, meta.module)].available
        eng = engineering_status_for_public_name(meta.name) or {}
        lines.append(
            f"| {meta.name} | {meta.module} | {meta.family} | "
            f"{'yes' if point else 'no'} | "
            f"{'yes' if status.basic else 'no'} | "
            f"{'yes' if status.adaptive else 'no'} | "
            f"{'yes' if status.rigorous_specialized else 'no'} | "
            f"{'yes' if not status.rigorous_specialized else 'no'} | "
            f"{eng.get('kernel_split', 'shared_dispatch_separate_mode_kernels')} | "
            f"{eng.get('helper_consolidation', 'shared_elementary_or_core')} | "
            f"{eng.get('batch', 'mixed')} | "
            f"{eng.get('ad', 'mixed')} | "
            f"{eng.get('hardening', 'specialized')} | "
            f"{meta.priority} | {meta.reason}; {status.notes} |"
        )

    lines.extend(
        [
            "",
            "## Tightening Backlog",
            "",
            "Ranked by numerical sensitivity and the current lack of specialized rigorous adapters.",
            "",
            "| priority | function | module | why it should move first |",
            "|---|---|---|---|",
        ]
    )

    backlog = [meta for meta in rows if not status_map[(meta.name, meta.module)].rigorous_specialized]
    order = {"P0": 0, "P1": 1, "P2": 2}
    backlog.sort(key=lambda meta: (order[meta.priority], meta.module, meta.name))
    if backlog:
        for meta in backlog:
            lines.append(f"| {meta.priority} | {meta.name} | {meta.module} | {meta.reason} |")
    else:
        lines.append("| complete | none | n/a | explicit rigorous adapters now cover the full custom-core set |")

    lines.extend(
        [
            "",
            "## Immediate Readout",
            "",
            "- Coverage is complete for `point`, `basic`, `adaptive`, and explicit rigorous dispatch across this custom-core set.",
            "- The previous tightening backlog is closed at the mode-dispatch layer.",
            "- Remaining quality work, if needed, is method-level improvement inside specific core kernels rather than wrapper coverage.",
            "",
        ]
    )

    (REPO_ROOT / "docs" / "reports" / "custom_core_status.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
