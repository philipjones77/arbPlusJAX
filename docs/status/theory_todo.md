Last updated: 2026-03-28T00:00:00Z

# Theory TODO

This file is the dedicated status and tranche tracker for repo-wide theory
coverage.

It exists so theory work is tracked as its own program rather than remaining an
implicit side effect of implementation hardening.

Status legend:
- `done`: theory note or theory-status item exists and is linked into the repo
- `in_progress`: accepted and partially documented, but still needs more notes,
  stronger linkage, or better coverage
- `planned`: accepted next-step theory tranche, not yet written

Current theory snapshot:
- repo-wide theory anchor: `done`
- family methodology coverage: `in_progress`
- structured linear algebra theory hierarchy: `in_progress`
- curvature/posterior-summary theory hierarchy: `planned`
- theory-to-practical linkage: `in_progress`

## Current Anchors

Status: `done`

- the general repo-wide conceptual entry point now exists in
  [repo_theory_overview.md](/docs/theory/repo_theory_overview.md)
- the theory landing page now includes that overview in
  [README.md](/docs/theory/README.md)
- notation governance for theory notes remains owned by
  [theory_notation_standard.md](/docs/standards/theory_notation_standard.md)

## Next Theory Hierarchy

Status: `planned`

The next orderly theory buildout should follow this structure:

1. repo-wide foundations
2. evaluation modes and uncertainty semantics
3. structured linear algebra and operator theory
4. special-function methodology
5. curvature and posterior-summary theory

## Tranche Breakdown

### 1. Repo-Wide Foundations

Status: `done`

- [repo_theory_overview.md](/docs/theory/repo_theory_overview.md) now explains:
  - the main runtime object classes
  - why `point`, `basic`, `adaptive`, and `rigorous` all exist
  - why prepared plans, diagnostics, metadata, and AD are first-class
  - why dense, sparse, matrix-free, transform, special-function, and curvature
    layers all belong in one repo

### 2. Evaluation Modes And Uncertainty Semantics

Status: `in_progress`

- existing anchors:
  - [ball_arithmetic_and_modes.md](/docs/theory/ball_arithmetic_and_modes.md)
  - [point_basic_surface_methodology.md](/docs/theory/point_basic_surface_methodology.md)
  - [matrix_interval_and_modes.md](/docs/theory/matrix_interval_and_modes.md)
- remaining work:
  - tighten the cross-note explanation of how scalar, matrix, and matrix-free
    `basic` semantics differ but remain part of one shared mode story
  - make `adaptive` and `rigorous` status more explicit family-by-family where
    those modes exist

### 3. Structured Linear Algebra And Operator Theory

Status: `in_progress`

- existing anchors:
  - [matrix_free_operator_methodology.md](/docs/theory/matrix_free_operator_methodology.md)
  - [matrix_free_operator_methods.md](/docs/theory/matrix_free_operator_methods.md)
  - [sparse_block_vblock_methodology.md](/docs/theory/sparse_block_vblock_methodology.md)
  - [sparse_selected_inversion_domain_decomposition.md](/docs/theory/sparse_selected_inversion_domain_decomposition.md)
  - [sparse_symmetric_leja_hutchpp_logdet.md](/docs/theory/sparse_symmetric_leja_hutchpp_logdet.md)
- remaining work:
  - add a clearer bridge note for dense/sparse/matrix-free operator semantics
  - make plan preparation, repeated apply, diagnostics, and implicit-adjoint
    ownership more explicit at the theory layer
  - keep estimator-family theory separated from implementation review notes

### 4. Special-Function Methodology

Status: `in_progress`

- existing anchors:
  - [core_functions_methodology.md](/docs/theory/core_functions_methodology.md)
  - [elementary_functions_methodology.md](/docs/theory/elementary_functions_methodology.md)
  - [bessel_family_methodology.md](/docs/theory/bessel_family_methodology.md)
  - [gamma_family_methodology.md](/docs/theory/gamma_family_methodology.md)
  - [hypergeometric_family_methodology.md](/docs/theory/hypergeometric_family_methodology.md)
  - [barnes_double_gamma_methodology.md](/docs/theory/barnes_double_gamma_methodology.md)
- remaining work:
  - continue making regime logic, provider boundaries, diagnostics, and
    parameter-direction AD interpretation explicit in the family notes
  - keep point/basic/adaptive/rigorous wording aligned across families

### 5. Curvature And Posterior-Summary Theory

Status: `planned`

- current state:
  - curvature is treated as a shared helper layer in
    [repo_theory_overview.md](/docs/theory/repo_theory_overview.md)
  - implementation status exists in
    [curvature_implementation.md](/docs/implementation/curvature_implementation.md)
- next theory work:
  - add a dedicated curvature theory note covering:
    - posterior precision operators
    - inverse-diagonal and selected-inverse summaries
    - stochastic trace/logdet estimators in curvature use
    - GGN/Fisher/operator-backed approximation semantics

## Theory-To-Practical Linkage

Status: `in_progress`

- theory notes should continue to have practical companions where the main user
  question is "how do I call this efficiently?"
- the current matrix-free example of that split is:
  - theory:
    [matrix_free_operator_methods.md](/docs/theory/matrix_free_operator_methods.md)
  - practical:
    [matrix_free.md](/docs/practical/matrix_free.md)
- the same pattern should be extended where useful for:
  - backend-realized performance guidance
  - sparse prepared-plan usage
  - curvature operator usage

## Completion Rule

Status: `planned`

Theory status should be considered broadly complete only when:

- the repo-wide foundations are explicit
- each top-level function family has a stable methodology note
- structured linear algebra has a clear dense/sparse/matrix-free hierarchy
- curvature has its own theory note rather than only implementation status
- practical runbooks are linked where theory alone is insufficient for correct
  public usage
