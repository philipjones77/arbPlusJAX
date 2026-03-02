Last updated: 2026-03-01T00:00:00Z

# TODO

## Recent updates
- Added asymptotic bessel evaluation + tightened bessel bounds (rigorous/adaptive) with denser sampling; basic mode uses midpoint evaluation; ran warmup bessel benchmarks (5000 samples).
- Added JAX batch warmup timing option (`--jax-warmup`) and cached batch JITs for single-compile runs.
- Added loggamma comparison tool with real/complex + branch-cut stress tests; included jax.scipy (real-only).
- Added explicit asymptotic remainder inflation for real bessel rigorous/adaptive interval bounds; added integer-crossing guards for `Y/K` interval APIs (real + complex box wrappers).
- Enforced source zip filename format in `tools/package_repo.py`: `<repo>_source_YYYY-MM-DD.zip` with validation for custom output paths.
- Added `CubesselK` backend with four-mode usage (point/basic/rigorous/adaptive) implemented in pure JAX (no CUDA dependency).
- Added `CubesselK` to benchmark harness and ran characterization against existing `besselk` backends (`results/benchmarks/cubesselk_compare_purejax_20260301/samples_256_seed_7/summary.csv`).
- Added `cusf_compat` module in this workspace with `Cusf_*` prefixed APIs (functions + helpers) and four-mode support (`point|basic|rigorous|adaptive`) for hypergeometric/Bessel/erf pathways.
- Added `boost_hypgeom` module with Boost-prefixed hypergeometric APIs and helper aliases in four modes (`point|basic|rigorous|adaptive`), with docs and tests.

## Open items
- Continue completing missing JAX implementations and tests for remaining Arb modules (see Missing C implementations section below).

## Missing C implementations
- Source: `docs/audit.md` (snapshot `2026-02-25T03:51:38Z`), grouped by function prefix/module.

- Arb Core: 195
- ACB Core: 144
- ARF: 95
- MAG: 78
- ACB Mat: 110
- Arb Mat: 109
- FMPR: 59
- ACB Dirichlet: 87
- Arb Poly: 87
- ACB Poly: 86
- Dirichlet: 38
- ACB DFT: 0
- Bool Mat: 34
- ACB Modular: 27
- ACF: 10
- ACB Elliptic: 17
- Arb Calc: 0
- ACB Calc: 0
