Last updated: 2026-03-23T00:00:00Z

# Transform FFT NUFFT Methodology

## Purpose

This note records the current mathematical and production interpretation of the
DFT and NUFFT surfaces used in arbPlusJAX.

## Scope

This note covers:

- discrete Fourier transform surfaces
- cached DFT plan/apply interpretation
- type-1 and type-2 NUFFT surfaces
- cached NUFFT plans
- direct versus accelerated transform interpretation

## DFT Interpretation

The canonical complex DFT is interpreted in the usual form

`(F x)_k = sum_j x_j exp(-2 pi i j k / n)`.

The repo exposes multiple algorithmic realizations, but the public production
interpretation is:

- all paths implement the same transform contract
- algorithm choice is an execution strategy, not a separate mathematical
  surface

Current realizations include naive, radix-2, Bluestein, and cached-plan forms.

## Cached DFT Plans

Cached DFT prepare/apply paths should be interpreted as factorization of
compile-time and setup-time work away from repeated execution.

That is a production contract point, not only a performance trick.

The intended use is:

1. prepare once for a fixed transform size and method
2. reuse the cached plan for repeated calls
3. avoid unnecessary recompiles by keeping transform size and dtype stable

## NUFFT Interpretation

The current NUFFT surfaces are interpreted as approximations to nonuniform
Fourier transforms where samples or modes are not on a regular grid.

Type-1 interpretation:

- nonuniform spatial samples to uniform Fourier modes

Type-2 interpretation:

- uniform Fourier modes to nonuniform spatial samples

The accelerated methods use interpolation/spreading structure rather than
forming the dense transform explicitly.

## Method Interpretation

The current production-facing method distinction is mainly:

- `direct`
  - smaller problems or reference-oriented exact dense-style evaluation
- `lanczos`
  - accelerated practical path for larger repeated workloads

This distinction matters operationally because changing the method changes both:

- numerical behavior
- compile/runtime behavior

So the method should be treated as part of the execution strategy contract.

## Production Calling Contract

The transform tranche should prefer:

- fixed dtype policy
- fixed mode count / output grid size
- cached prepare/apply flows for repeated calls
- stable method choice inside service loops

That is how the transform notebooks and benchmarks should teach usage, because
transform workloads can otherwise incur avoidable recompiles when shapes or
static controls drift.

## Diagnostics Interpretation

The current transform surfaces are less diagnostics-heavy than the
matrix-free/special-function stacks.

For this tranche, the main practical diagnostics are benchmark-facing:

- cold execution
- warm execution
- recompile sensitivity through changed shape or changed cached plan

The benchmarks therefore act as the main diagnostics surface for transform
production behavior today.

## Current Limitations

- full GPU validation is deferred to a later tranche
- not every transform benchmark is yet normalized onto the shared JSON schema
- current theory documentation is oriented toward production interpretation
  rather than full approximation-error derivation for every NUFFT kernel choice
