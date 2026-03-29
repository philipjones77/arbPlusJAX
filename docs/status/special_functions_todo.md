Last updated: 2026-03-29T00:00:00Z

# Special Functions TODO

This file tracks the special-function hardening backlog separately from the
general repo TODO.

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

## Current Status

Status: `done`

- `done`
  - canonical example notebooks now exist for top-level gamma and
    Barnes/double-gamma production surfaces, with explicit production-calling
    guidance and benchmark-extension notes
  - canonical example notebooks now also exist for the hypergeometric
    production surface, so the production teaching surface is now split by
    ownership across gamma/incomplete-tail, Barnes/double-gamma, and hypergeom
  - dedicated theory notes now exist for the gamma-family production stack
  - dedicated theory notes now exist for hypergeometric and Barnes/double-gamma
    production methodology
  - practical runbook coverage now exists in
    [special_functions.md](/docs/practical/special_functions.md)
  - retained CPU/GPU operational service benchmark artifacts now exist for the
    non-Barnes backend-closeout set in
    [benchmark_special_function_service_api_cpu_refresh.json](/benchmarks/results/benchmark_special_function_service_api/benchmark_special_function_service_api_cpu_refresh.json)
    and
    [benchmark_special_function_service_api_gpu_refresh.json](/benchmarks/results/benchmark_special_function_service_api/benchmark_special_function_service_api_gpu_refresh.json)
  - non-Barnes CPU/GPU service-contract, AD-direction, and hardening slices are
    now explicitly part of the validated production set
  - the governed production closeout is now explicit for the non-Barnes set:
    incomplete gamma, incomplete Bessel, and hypergeom are treated as closed
    for API, AD, fast-JAX, operational-JAX, CPU/GPU validation, benchmarks,
    and notebook teaching
- `in_progress`
  - keep Barnes IFJ explicit as a correctness/diagnostics-hardened path that is
    still excluded from backend-throughput closeout claims

## Barnes Exception Backlog

Status: `in_progress`

- finish hardening and characterization of the `bdg_*` Barnes and
  double-gamma family
- keep extending Barnes/double-gamma diagnostics and provider contracts beyond
  the current scalar IFJ surface
- lower startup and compiled-batch cost of Barnes IFJ before moving it into the
  same CPU/GPU operational closeout set as gamma, incomplete Bessel, and
  hypergeom
- reduce runtime cost of rigorous/adaptive `bdg_*` samplers in
  [ball_wrappers.py](/src/arbplusjax/ball_wrappers.py)
- direct owner tests now exist for `bessel_kernels` and `barnesg`
- extend benchmark and RF77-facing usage/report coverage where diagnostics
  exist but packaging is still incomplete

## Planned Additions

Status: `planned`

- extend the general incomplete-tail engine to more hypergeometric-tail
  families
- add multivariate Bessel work only after scalar incomplete infrastructure is
  genuinely stable
- add incomplete multivariate-Bessel-type routines only if reductions justify
  them
- resolve what true arbitrary precision means under a strict pure-JAX
  constraint

## Priority Rule For Remaining Breadth

Status: `in_progress`

- do not migrate the missing callable surface breadth-first
- prefer one canonical JAX-native implementation per important function family
- prioritize IFJ and RF77-facing work first:
  - Barnes-family hardening and IFJ-derived double-gamma work
  - gamma-adjacent continuation functions that unblock contour and residue
    workflows
  - selected complex special functions with direct downstream use:
    `Ei`, `Chi`, `Ci`, dilogarithm, Tricomi `U`, and selected `pfq`
- after that, prioritize broad-value parity:
  - dense matrix parity in `arb_mat` / `acb_mat`
  - polynomial parity in `arb_poly` / `acb_poly`
  - selected scalar gaps such as `lambertw`, zeta-adjacent functions, and
    rising/beta families
- defer broad elliptic/modular and full Dirichlet/L-function expansion until
  the Barnes/gamma/integration path is stable enough to justify it
