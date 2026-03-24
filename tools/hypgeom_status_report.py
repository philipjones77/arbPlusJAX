from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HypgeomRow:
    public_name: str
    module: str
    lineage: str
    family: str
    point: str
    basic: str
    adaptive: str
    rigorous: str
    tightening: str
    kernel_split: str
    helper_consolidation: str
    pure_jax: str
    batch_recompile: str
    ad: str
    notes: str


ROWS: tuple[HypgeomRow, ...] = (
    HypgeomRow(
        "arb_hypgeom_0f1 / acb_hypgeom_0f1",
        "hypgeom",
        "canonical",
        "0f1",
        "yes",
        "yes",
        "yes",
        "yes",
        "specialized tail-bound rigorous kernels",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous",
        "targeted",
        "Canonical real and complex rigorous kernels exist and are explicitly wired in hypgeom_wrappers; basic/adaptive/rigorous batch paths now all have dedicated fixed-shape entry points.",
    ),
    HypgeomRow(
        "arb_hypgeom_1f1 / acb_hypgeom_1f1",
        "hypgeom",
        "canonical",
        "1f1 / m",
        "yes",
        "yes",
        "yes",
        "yes",
        "shared regime/sample candidate helpers with specialized tail-bound rigorous kernels",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous; direct family mode-batch cores",
        "boundary-audited + complex cut/corner audited",
        "`m` is an alias of `1f1`; canonical rigorous kernels exist, regime candidates are selected through shared helper builders in `hypgeom.py`, and the batch path now dispatches directly to family kernels instead of generic mode lambdas. Targeted compile probe: family events `6 -> 3`, total padded compile count `27 -> 31`.",
    ),
    HypgeomRow(
        "arb_hypgeom_2f1 / acb_hypgeom_2f1",
        "hypgeom",
        "canonical",
        "2f1",
        "yes",
        "yes",
        "yes",
        "yes",
        "shared transform/sample candidate helpers with specialized tail-bound rigorous kernels",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous; direct family mode-batch cores",
        "boundary-audited + complex cut/corner audited",
        "Canonical real and complex rigorous kernels exist and are explicitly wired in hypgeom_wrappers; the scalar path now shares transform/sample helper builders and the batch path dispatches directly to family kernels. Targeted compile probe: family events `6 -> 3`, total padded compile count `27 -> 31`.",
    ),
    HypgeomRow(
        "arb_hypgeom_u / acb_hypgeom_u",
        "hypgeom",
        "canonical",
        "u",
        "yes",
        "yes",
        "yes",
        "yes",
        "shared regime/sample candidate helpers with specialized regime-aware rigorous kernels",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous; direct family mode-batch cores",
        "boundary-audited + complex cut/corner audited",
        "Rigorous path exists and the batch path now dispatches directly to family kernels for all three interval modes; scalar real/complex paths use shared regime/sample helper builders and deeper complex boundary sweeps around the main regime changes. Targeted compile probe: family events `6 -> 3`, total padded compile count `27 -> 31`.",
    ),
    HypgeomRow(
        "arb_hypgeom_pfq / acb_hypgeom_pfq",
        "hypgeom",
        "canonical",
        "pfq",
        "yes",
        "yes",
        "yes",
        "yes",
        "series-driven with explicit sample tightening",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape fixed/padded batch available for basic/adaptive/rigorous; direct family mode-batch cores",
        "boundary-audited + fixed/padded mode-batch audited",
        "Point/basic/adaptive/rigorous are present; this family now has dedicated fixed-shape fixed/padded batch entry points, direct family mode-batch dispatch, explicit sample-based tightening, and real/complex fixed-vs-padded mode-batch proof coverage, but it remains less specialized than the headline families.",
    ),
    HypgeomRow(
        "arb_hypgeom_gamma / acb_hypgeom_gamma",
        "hypgeom",
        "canonical",
        "gamma",
        "yes",
        "yes",
        "yes",
        "yes",
        "ball-wrapper hardened",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded point batch; interval modes remain direct-vmap/no fixed-shape audit",
        "targeted",
        "Gamma family uses direct point kernels with fixed/padded point-batch fastpaths plus dedicated ball wrappers in adaptive and rigorous modes.",
    ),
    HypgeomRow(
        "arb_hypgeom_gamma_lower / arb_hypgeom_gamma_upper",
        "hypgeom",
        "canonical",
        "incomplete gamma",
        "yes",
        "yes",
        "yes",
        "yes",
        "shared direct/complement candidate helpers with specialized adaptive/rigorous complement selection",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous; direct specialized mode-batch cores for incomplete gamma",
        "boundary-audited + complex cut/corner audited",
        "Incomplete-gamma support is present on the real and complex sides with shared direct/complement helper extraction in `hypgeom.py`, specialized adaptive/rigorous complement-aware wrappers, direct mode-batch cores for the family, fixed-shape batch fastpaths, and explicit real/complex AD boundary checks. Broad-tranche compile totals still read `49 -> 47`, while the targeted incomplete-gamma probe shows family compile events halved from `4` to `2`; the family remains less mature than base gamma.",
    ),
    HypgeomRow(
        "arb_hypgeom_erf / erfc / erfi / erfinv / erfcinv",
        "hypgeom",
        "canonical",
        "erf family",
        "yes",
        "yes",
        "yes",
        "yes",
        "ball-wrapper hardened",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded point batch; interval modes remain direct-vmap/no fixed-shape audit",
        "targeted",
        "Real and complex point paths now have direct fixed/padded point-batch kernels for the implemented erf-family members; the real inverse family keeps dedicated ball wrappers while the complex inverse side remains narrower.",
    ),
    HypgeomRow(
        "arb_hypgeom_ei / si / ci / shi / chi / li / dilog / fresnel",
        "hypgeom",
        "canonical",
        "integral / polylog-like",
        "yes",
        "yes",
        "yes",
        "yes",
        "ball-wrapper hardened",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded point batch; interval modes remain direct-vmap/no fixed-shape audit",
        "targeted",
        "Main real and complex point paths now have direct fixed/padded point-batch kernels for the implemented scalar and tuple-returning families; interval hardening still relies on the existing ball-wrapper layer.",
    ),
    HypgeomRow(
        "arb_hypgeom_legendre_p / legendre_q / jacobi_p / gegenbauer_c",
        "hypgeom",
        "canonical",
        "orthogonal/classical",
        "yes",
        "yes",
        "mixed",
        "yes",
        "specialized recurrence rigorous kernels",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous",
        "point batch + real/complex AD audited",
        "These have explicit rigorous kernels, dedicated fixed-shape batch entry points for basic/adaptive/rigorous modes, direct point-batch API coverage, and real/complex point-AD smoke coverage.",
    ),
    HypgeomRow(
        "arb_hypgeom_chebyshev_t / chebyshev_u / laguerre_l / hermite_h",
        "hypgeom",
        "canonical",
        "orthogonal/classical",
        "yes",
        "yes",
        "yes",
        "yes",
        "generic with explicit endpoint/corner tightening",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mostly",
        "fixed-shape padded batch available for basic/adaptive/rigorous; direct family mode-batch cores",
        "point batch + real/complex AD audited",
        "Canonical point/basic/adaptive/rigorous support exists and now has fixed-shape batch paths, direct family mode-batch dispatch, explicit direct point-batch API coverage, and real/complex point-AD smoke coverage, but it is still more generic than Legendre/Jacobi/Gegenbauer.",
    ),
    HypgeomRow(
        "boost_hypergeometric_1f0 / 0f1 / 1f1 / 2f0 / pfq",
        "boost_hypgeom",
        "alternative",
        "boost hypergeometric",
        "yes",
        "yes",
        "yes",
        "yes",
        "mode-aware alternative; not deeply hardened",
        "shared_dispatch_separate_mode_kernels",
        "shared_helper_layer",
        "mixed",
        "fixed-shape fixed/padded point/basic/adaptive/rigorous batch available on the main families",
        "primary-family point AD + pfq fixed/padded mode-batch audited",
        "Four modes exist and the main families now use direct point kernels plus fixed-shape fixed/padded batch fastpaths on the shared substrate; pfq now has explicit fixed-vs-padded and reciprocal/mode-containment proofs, and regularized `0f1`/`1f1` fixed-vs-padded containment is audited, but tightening still delegates more heavily to canonical or wrapper-level behavior than the canonical hypgeom families.",
    ),
    HypgeomRow(
        "boost_hyp1f1_series / asym / hyp2f1_series / cf / pade / rational / hyp1f2_series",
        "boost_hypgeom",
        "alternative",
        "boost helper methods",
        "yes",
        "yes",
        "yes",
        "yes",
        "method-specific alternatives; not deeply hardened",
        "mixed",
        "n/a",
        "mixed",
        "no dedicated fixed-shape audit",
        "helper alias consistency + point AD audited",
        "Useful for comparing method families; helper aliases now have explicit cross-mode consistency checks and point-AD smoke, but they are not yet engineered to the same standard as canonical kernels.",
    ),
    HypgeomRow(
        "cusf_hyp1f1 / cusf_hyp2f1",
        "cusf_compat",
        "alternative",
        "cusf hypergeometric",
        "yes",
        "yes",
        "yes",
        "yes",
        "mode-aware alternative; not deeply hardened",
        "inherited_or_mixed",
        "n/a",
        "mixed",
        "no fixed-shape audit",
        "point AD + mode containment audited on hyp1f1/2f1",
        "Four modes exist with explicit mode-containment and point-AD checks on `hyp1f1`/`hyp2f1`, but this family is still a compatibility layer rather than a deeply tightened canonical path.",
    ),
)


def render() -> str:
    canonical = [row for row in ROWS if row.lineage == "canonical"]
    alternative = [row for row in ROWS if row.lineage != "canonical"]
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Hypgeom Status",
        "",
        "Generated from `tools/hypgeom_status_report.py` using the current `hypgeom`, `hypgeom_wrappers`, `boost_hypgeom`, and `cusf_compat` implementation surface.",
        "",
        "Scope:",
        "- canonical Arb-like hypergeometric and adjacent special-function families requested for the current staged cleanup",
        "- alternative hypergeometric implementations in `boost_hypgeom.py` and `cusf_compat.py`",
        "",
        f"Summary: `canonical_rows={len(canonical)}`, `alternative_rows={len(alternative)}`, `total_rows={len(ROWS)}`.",
        "",
        "Columns:",
        "- `point/basic/adaptive/rigorous`: current public mode availability for the family row",
        "- `tightening`: current hardening level; this is not the same as mere mode availability",
        "- `kernel_split`: whether the family shares dispatch/padding while keeping separate point/basic/tighter kernels",
        "- `helper_consolidation`: status of shared elementary/core/helper extraction",
        "- `pure_jax`: honest current implementation state, not an aspirational claim",
        "- `batch_recompile`: current batching/recompile state, not a future target",
        "- `ad`: current AD status",
        "",
        "## Status Table",
        "",
        "| public_family | module | lineage | family | point | basic | adaptive | rigorous | tightening | kernel_split | helper_consolidation | pure_jax | batch_recompile | ad | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            f"| {row.public_name} | {row.module} | {row.lineage} | {row.family} | {row.point} | {row.basic} | "
            f"{row.adaptive} | {row.rigorous} | {row.tightening} | {row.kernel_split} | {row.helper_consolidation} | "
            f"{row.pure_jax} | {row.batch_recompile} | {row.ad} | {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## Immediate Readout",
            "",
            "- The main canonical hypergeometric families `0f1`, `1f1`, `2f1`, and `u` already have all four modes and specialized rigorous kernels.",
            "- Gamma, erf, and the real Ei/Si/Ci/Shi/Chi/li/dilog/fresnel families already have four modes with ball-wrapper hardening.",
            "- Orthogonal families are mixed: Legendre/Jacobi/Gegenbauer are stronger than Chebyshev/Laguerre/Hermite.",
            "- Alternative Boost and CuSF families expose four modes, but their tightening remains shallower than canonical `hypgeom`.",
            "- Helper consolidation and stricter JAX engineering work are still partial; this report should be used to drive the next staged cleanup rather than claim completion.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    (REPO_ROOT / "docs" / "reports" / "hypgeom_status.md").write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
