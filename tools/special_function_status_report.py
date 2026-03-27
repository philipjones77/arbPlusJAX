from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "special_function_status.md"


def _load_json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _fmt_float(value: float | None, *, places: int = 6) -> str:
    if value is None:
        return "`n/a`"
    return f"`{value:.{places}f}`"


def render() -> str:
    hardening = _load_json(
        "benchmarks/results/special_function_hardening_benchmark/special_function_hardening_benchmark.json"
    )
    hypgeom_probe = _load_json("benchmarks/results/hypgeom_point_startup_probe/hypgeom_point_startup_probe.json")
    double_gamma_probe = _load_json(
        "benchmarks/results/double_gamma_point_startup_probe/double_gamma_point_startup_probe.json"
    )

    hypgeom_point = hypgeom_probe["arb_hypgeom_1f1_point_path"]
    double_gamma_point = double_gamma_probe["bdg_barnesgamma2_point_path"]
    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# Special-Function Status",
        "",
        "Generated from `tools/special_function_status_report.py` using the checked-in hardening benchmark and startup-probe artifacts.",
        "",
        "Scope:",
        "- production-facing special-function families with active hardening work in this repo",
        "- canonical API/binder surfaces, diagnostics surfaces, benchmarks, startup probes, and example notebooks",
        "",
        "## Canonical Production Surfaces",
        "",
        "| family | point surface | tighter/diagnostics surface | canonical notebook | primary tests | benchmark/startup evidence |",
        "|---|---|---|---|---|---|",
        "| incomplete gamma + incomplete Bessel I/K | `api.bind_point_batch()` / `api.bind_point_batch_jit()` on `incomplete_gamma_upper`, `incomplete_gamma_lower`, `incomplete_bessel_i`, `incomplete_bessel_k` | function-returned diagnostics via `return_diagnostics=True` plus `api.bind_interval_batch()` for stable-shape interval service usage | [examples/example_gamma_family_surface.ipynb](/examples/example_gamma_family_surface.ipynb) | [tests/test_incomplete_gamma.py](/tests/test_incomplete_gamma.py), [tests/test_incomplete_bessel_i.py](/tests/test_incomplete_bessel_i.py), [tests/test_special_function_service_contracts.py](/tests/test_special_function_service_contracts.py), [tests/test_special_function_hardening.py](/tests/test_special_function_hardening.py) | [benchmarks/special_function_hardening_benchmark.py](/benchmarks/special_function_hardening_benchmark.py) |",
        "| Barnes / double-gamma | `double_gamma.ifj_barnesdoublegamma(...)` and explicit implementation-owned scalar/vector paths | `double_gamma.ifj_barnesdoublegamma_diagnostics(...)` and the Barnes G aliases on `acb_core` | [examples/example_barnes_double_gamma_surface.ipynb](/examples/example_barnes_double_gamma_surface.ipynb) | [tests/test_barnes_double_gamma_ifj_contracts.py](/tests/test_barnes_double_gamma_ifj_contracts.py), [tests/test_double_gamma_contracts.py](/tests/test_double_gamma_contracts.py), [tests/test_barnes_tier1.py](/tests/test_barnes_tier1.py), [tests/test_special_function_hardening.py](/tests/test_special_function_hardening.py), [tests/test_special_function_ad_directions.py](/tests/test_special_function_ad_directions.py) | [benchmarks/benchmark_barnes_double_gamma.py](/benchmarks/benchmark_barnes_double_gamma.py), [benchmarks/double_gamma_point_startup_probe.py](/benchmarks/double_gamma_point_startup_probe.py), [benchmarks/special_function_ad_benchmark.py](/benchmarks/special_function_ad_benchmark.py) |",
        "| hypergeom / regularized incomplete gamma | `api.bind_point_batch_jit(\"hypgeom.arb_hypgeom_1f1\", ...)` and other family-owned point kernels | `api.bind_interval_batch(...)` for stable-shape interval routing plus `hypgeom_wrappers.*_mode*_padded` for direct family mode ownership | [examples/example_hypgeom_family_surface.ipynb](/examples/example_hypgeom_family_surface.ipynb) | [tests/test_hypgeom_wrappers_contracts.py](/tests/test_hypgeom_wrappers_contracts.py), [tests/test_hypgeom_engineering.py](/tests/test_hypgeom_engineering.py), [tests/test_hypgeom_startup_lazy_loading.py](/tests/test_hypgeom_startup_lazy_loading.py), [tests/test_special_function_hardening.py](/tests/test_special_function_hardening.py), [tests/test_special_function_ad_directions.py](/tests/test_special_function_ad_directions.py) | [docs/reports/hypgeom_status.md](/docs/reports/hypgeom_status.md), [benchmarks/hypgeom_point_startup_probe.py](/benchmarks/hypgeom_point_startup_probe.py), [benchmarks/special_function_hardening_benchmark.py](/benchmarks/special_function_hardening_benchmark.py), [benchmarks/special_function_ad_benchmark.py](/benchmarks/special_function_ad_benchmark.py) |",
        "",
        "## Current Hardening Snapshot",
        "",
        f"- incomplete Bessel I quadrature: {_fmt_float(hardening['incomplete_bessel_i']['quadrature_s'])}",
        f"- incomplete Bessel I high precision refine: {_fmt_float(hardening['incomplete_bessel_i']['high_precision_refine_s'])}",
        f"- incomplete Bessel I auto: {_fmt_float(hardening['incomplete_bessel_i']['auto_s'])}",
        f"- Barnes IFJ vector: {_fmt_float(hardening['barnes_double_gamma']['ifj_vector_s'])}",
        f"- Barnes provider vector: {_fmt_float(hardening['barnes_double_gamma']['provider_vector_s'])}",
        f"- Barnes BDG vector: {_fmt_float(hardening['barnes_double_gamma']['bdg_vector_s'])}",
        f"- Barnes IFJ recurrence residual: `{hardening['barnes_double_gamma']['ifj_shift_recurrence_abs']}`",
        f"- Barnes provider vs IFJ max abs: `{hardening['barnes_double_gamma']['provider_vs_ifj_vector_max_abs']}`",
        f"- Barnes IFJ diagnostics m_used: `{hardening['barnes_double_gamma']['ifj_diag_m_used']}`",
        f"- Barnes IFJ diagnostics n_shift: `{hardening['barnes_double_gamma']['ifj_diag_n_shift']}`",
        f"- Hypergeom 1f1 point batch: {_fmt_float(hardening['hypgeom']['onef1_point_batch_s'])}",
        f"- Hypergeom U point batch: {_fmt_float(hardening['hypgeom']['u_point_batch_s'])}",
        f"- Hypergeom U adaptive mode batch: {_fmt_float(hardening['hypgeom']['u_adaptive_mode_batch_s'])}",
        f"- Hypergeom U point vs scalar family max abs: `{hardening['hypgeom']['u_point_vs_scalar_family_max_abs']}`",
        f"- Hypergeom U mode vs scalar family max abs: `{hardening['hypgeom']['u_mode_mid_vs_scalar_family_max_abs']}`",
        f"- Hypergeom pfq point batch: {_fmt_float(hardening['hypgeom']['pfq_point_batch_s'])}",
        f"- Hypergeom pfq adaptive mode batch: {_fmt_float(hardening['hypgeom']['pfq_adaptive_mode_batch_s'])}",
        f"- Hypergeom pfq point vs scalar family max abs: `{hardening['hypgeom']['pfq_point_vs_scalar_family_max_abs']}`",
        f"- Hypergeom pfq mode vs scalar family max abs: `{hardening['hypgeom']['pfq_mode_mid_vs_scalar_family_max_abs']}`",
        f"- Hypergeom regularized lower batch: {_fmt_float(hardening['hypgeom']['gamma_lower_regularized_batch_s'])}",
        f"- Hypergeom regularized upper batch: {_fmt_float(hardening['hypgeom']['gamma_upper_regularized_batch_s'])}",
        "",
        "## Startup Probe Snapshot",
        "",
        f"- [hypgeom_point_startup_probe.json](/benchmarks/results/hypgeom_point_startup_probe/hypgeom_point_startup_probe.json): import={_fmt_float(hypgeom_probe['import_arbplusjax_api']['seconds'])}, compile+first={_fmt_float(hypgeom_point['compile_plus_first_point_batch_s'])}, steady={_fmt_float(hypgeom_point['steady_point_batch_s'])}",
        f"- [double_gamma_point_startup_probe.json](/benchmarks/results/double_gamma_point_startup_probe/double_gamma_point_startup_probe.json): import={_fmt_float(double_gamma_probe['import_arbplusjax_api']['seconds'])}, compile_error_type=`{double_gamma_point['compile_error_type']}`",
        "",
        "## Notes",
        "",
        "- `special_function_hardening_benchmark.py` is the current cross-family benchmark/diagnostics rollup.",
        "- `special_function_ad_benchmark.py` is the current cross-family argument-vs-parameter AD benchmark rollup.",
        "- `hypgeom_status.md` remains the detailed family-by-family hypergeom engineering inventory.",
        "- The point-surface audit now compares public point wrappers against scalar family-owned exact-input evaluations; current remaining pfq drift is in the batched interval/mode path, not the point wrapper.",
        "- The Barnes startup probe still records a compile failure on the legacy `bdg_barnesgamma2` point path; the IFJ diagnostics-backed path and provider alias are the currently hardened public route.",
        "- Canonical notebook teaching is now split by ownership: gamma/incomplete-tail, Barnes/double-gamma, and hypergeom each have a dedicated example surface.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
