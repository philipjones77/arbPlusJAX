Last updated: 2026-03-28T00:00:00Z

# Cross-Cutting TODO

This file tracks cross-cutting execution, platform, point-fast, and provider
boundary work that does not belong to one mathematical family alone.

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

## Execution And Run-Platform

Status: `in_progress`

- `done`
  - root-level [tests](/tests) and
    [benchmarks](/benchmarks) remain the canonical run surfaces
  - dedicated test orchestration exists in
    [run_test_harness.py](/tools/run_test_harness.py)
  - dedicated benchmark orchestration exists in
    [run_benchmarks.py](/benchmarks/run_benchmarks.py)
    and
    [run_harness_profile.py](/benchmarks/run_harness_profile.py)
  - a shared `runtime_manifest.json` schema now exists across test and
    benchmark outputs
  - canonical notebook execution now exists through
    [run_example_notebooks.py](/tools/run_example_notebooks.py)
  - canonical notebooks now encode production calling patterns:
    binder reuse, optional padding/chunking, cached plan reuse, and benchmark
    extension guidance
  - the official API benchmark
    [benchmark_api_surface.py](/benchmarks/benchmark_api_surface.py)
    now emits the shared benchmark-report JSON schema
  - Windows, Linux, and Colab run instructions are documented
  - a CPU-safe Colab bootstrap surface now exists in
    [requirements-colab.txt](/requirements-colab.txt)
    and
    [colab_bootstrap.sh](/tools/colab_bootstrap.sh)
  - bounded CPU validation profiles were re-run through
    [run_test_harness.py](/tools/run_test_harness.py) for `matrix`,
    `special`, and `bench-smoke`; see
    [cpu_validation_profiles.md](/docs/reports/cpu_validation_profiles.md)
  - a dedicated sparse-matrix harness profile exists in
    [run_test_harness.py](/tools/run_test_harness.py)
  - a single repo-facing execution checklist now exists in
    [release_execution_checklist_standard.md](/docs/standards/release_execution_checklist_standard.md)
    and
    [release_execution_checklist.md](/docs/implementation/release_execution_checklist.md)
- `in_progress`
  - unify long-run benchmark scheduling and report collection behind a single
    environment manifest and execution policy
  - keep benchmark ownership distinct from correctness ownership, but make the
    pass/fail boundaries clearer in status docs
  - keep docs landing pages, report indexes, status indexes, and current repo
    mapping generated automatically so push/commit does not rely on hand-edited
    tree summaries
  - normalize more legacy benchmark scripts onto the shared benchmark-report
    schema instead of stdout-only summaries
  - keep normalized benchmark CLIs explicitly parameterized for CPU/GPU
    portability and `float32`/`float64` execution, even when the current
    validation slice only runs on CPU
- `planned`
  - retain a broader periodic CPU validation slice for the newest matrix-free
    estimator and contour-action tranches once the heavier test/runtime slices
    settle

## Point-Fast JAX Program

Status: `in_progress`

Primary plan:
- [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)

- `done`
  - the repo now has an explicit definition of `fast JAX` for point mode in
    [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
  - all public point functions now have compiled single-call, compiled batch,
    and family-owned direct batch public surfaces; see
    [point_fast_jax_function_inventory.md](/docs/reports/point_fast_jax_function_inventory.md)
  - the joined family-level point/basic verification ledger now exists in
    [point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md)
  - the explicit parameterized public AD proof ledger now exists in
    [parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md)
- `in_progress`
  - widen the now-landed category proof slices into deeper family-by-family
    numerical proof coverage across the remaining large public
    matrix/core/hypergeometric surfaces
  - keep the parameterized-AD audit widened as new public parameterized
    families or helper surfaces are added
  - refactor point kernels so Python control flow, dynamic shapes, Arb objects,
    and precise fallback logic remain outside the hot path
  - build shared point-fast infrastructure for logspace, recurrence,
    approximants, and region routing
- `planned`
  - add machine-readable point-fast capability metadata for downstream routing

## Provider Boundary

Status: `in_progress`

- `done`
  - arbPlusJAX remains the hardened numeric-kernel repo rather than being
    repurposed as another library's orchestration layer
  - matrix, sparse, block/vblock, and matrix-free/operator infrastructure are
    being hardened as repo-owned numeric infrastructure inside arbPlusJAX
- `in_progress`
  - keep hardening public provider-worthy families instead of exposing more ad
    hoc module-internal integration paths
  - prefer stable capability entrypoints and metadata-bearing public surfaces
    over downstream imports of internal module layout
  - strengthen metadata and diagnostics so downstream orchestration can route
    intelligently on method, hardening level, derivative support, and runtime
    strategy
  - keep notebooks, tests, and benchmarks written as downstream-consumer
    documentation and validation surfaces, not only as internal development
    checks
  - keep cross-repo integration thin: downstream libraries should integrate
    through adapter/provider layers on their side rather than by restructuring
    arbPlusJAX around a specific consumer
  - Barnes/double-gamma now has explicit downstream capability aliases through
    the IFJ-compatible public surface; continue tightening diagnostics and
    narrower provider wording around that capability
  - fragile-regime promotion hooks now have explicit downstream capability
    aliases for incomplete gamma upper and incomplete Bessel `I`/`K`
- `planned`
  - document a narrower capability-contract surface specifically for
    downstream-provider use once the Barnes/promotion/incomplete-Bessel tranche
    is hardened enough to freeze terminology
