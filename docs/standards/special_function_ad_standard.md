Last updated: 2026-03-26T00:00:00Z

# Special-Function AD Standard

## Purpose

Automatic differentiation for parameterized mathematical families must be
treated as a multi-direction contract, not a single binary property.

Special functions are the primary scope of this standard, but the same rule
also applies to other public parameterized families in the repo, including:

- scalar families with named parameters
- matrix and sparse/operator families with continuous controls
- matrix-free/operator families with shift, damping, regularization, or other
  differentiable solve parameters
- index-like controls when they are modeled as continuous quantities rather
  than discrete selectors

## Required Distinction

For a public family with parameters, AD must be tracked separately for:

1. evaluation-variable AD
   Example: differentiate with respect to `x`, `z`, or the main evaluation
   variable.

2. family-parameter AD
   Example: differentiate with respect to `a`, `b`, `c`, `s`, `nu`, `tau`,
   `lambda`, or another defining family parameter.

These are different contracts. A family is not "AD complete" merely because
evaluation-variable differentiation works.

## Required Policy

1. Public parameterized families must state which AD directions are supported.
2. Continuous family parameters should support parameter-direction AD where the
   mathematical formulation is intended to be smooth.
3. Discrete indices or selector arguments are not gradient targets by default.
   Their AD policy must be stated explicitly:
   - unsupported because the control is discrete
   - supported through a continuous relaxation or analytic continuation
4. Tests, benchmarks, and canonical notebooks must distinguish the two AD
   directions explicitly.
5. If reverse-mode is not appropriate for a family, forward-mode is acceptable
   as the public AD contract, but that choice must be reflected in the tests
   and examples.

## Required Evidence

For a production-facing parameterized family, the repo should provide:

- at least one test for evaluation-variable AD
- at least one test for family-parameter AD
- at least one benchmark artifact comparing the two directions
- at least one canonical notebook section illustrating both directions on the
  real production surface

## Examples

- incomplete gamma:
  - evaluation variable: `z`
  - family parameter: `s`
- Bessel:
  - evaluation variable: `z`
  - family parameter: `nu`
- Barnes/double-gamma:
  - evaluation variable: `z`
  - family parameter: `tau`
- hypergeom:
  - evaluation variable: `z`
  - family parameters: `a`, `b`, `c`, and related tuple parameters where
    supported
- operator families:
  - evaluation variable: the input/state variable
  - family parameters: regularization, damping, spectral shift, continuation
    parameter, or other differentiable operator controls

## Banned Shortcuts

- calling a family "AD supported" when only the evaluation variable is tested
- teaching notebooks that only plot the `x`-gradient for a parameterized family
- benchmarking only one AD direction for a parameterized family
- silently treating discrete selector arguments as differentiable parameters

## Current Repo Tranche

The current first special-function tranche should cover:

- incomplete gamma `s` and `z`
- hypergeom `1f1` and `u` parameter-vs-argument AD
- Bessel order `nu` vs argument `z`
- Barnes/double-gamma `tau` vs argument `z`

The current repo also applies the same contract outward to:

- scalar value-parameter families such as `arb_pow`
- dense and sparse matrix/operator-apply surfaces with differentiable matrix
  controls
- matrix-free/operator surfaces with differentiable shift or solve controls
