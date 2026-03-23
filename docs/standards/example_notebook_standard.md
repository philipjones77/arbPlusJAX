Last updated: 2026-03-22T00:00:00Z

# Example Notebook Standard

Status: active

## Purpose

This document defines the required structure for `example_*.ipynb` notebooks in
this repo.

This document owns notebook content requirements.

It does not define:

- experiment folder layout
- cross-environment portability policy

Those belong to:

- [experiment_layout_standard.md](/docs/standards/experiment_layout_standard.md)
- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)

Example notebooks are not optional decoration. They are the canonical
demonstration layer for a functionality group.

They must show:

- how to instantiate a representative object or input family
- how to use the functionality on that object
- what the parameter/value sweeps look like
- what the test and benchmark results say
- how the repo compares to available reference or benchmark software

## Scope

Every major functionality group should have at least one `example_` notebook.

Groups include:

- core/scalar
- special functions
- dense matrix
- sparse/block/vblock matrix
- matrix-free/operator
- transforms
- API/runtime routing
- analytic/algebraic families where the surface is large enough to justify a dedicated example

## File Naming

Notebook filenames must use:

- `example_<family>.ipynb`
- `example_<family>_<focus>.ipynb`

Examples:

- `example_dense_matrix_surface.ipynb`
- `example_hypgeom_robust_modes_sweep.ipynb`

## Required Notebook Sections

Every example notebook should include these sections in some form:

1. scope
   - what functionality group or family the notebook covers
2. environment
   - interpreter
   - JAX backend/device
   - dtype mode
   - explicit note that canonical execution in this repo state is CPU, while
     the notebook calling pattern remains CPU/GPU portable
3. object/input construction
   - instantiate a representative object, matrix, operator, or function input
4. direct usage
   - run the core public operations on that object
5. production pattern
   - show the intended production calling style for the family
   - demonstrate binder reuse, cached plan reuse, or both where relevant
   - show stable dtype/mode usage and optional padding/chunking where repeated calls would otherwise trigger recompiles
   - explain how to extend the owning benchmark surface for adjacent functions
6. AD product pattern
   - show how automatic differentiation is intended to be used on the production surface
   - validate AD behavior on a representative scalarized objective
   - graph primal and derivative behavior over a representative sweep
5. parameter/value sweeps
   - show a complete or representative sweep of values/parameters relevant to the family
6. validation summary
   - summarize relevant test outcomes or validation expectations
7. benchmark summary
   - summarize speed, cold/warm/recompile behavior, or other relevant benchmark outputs
8. comparison summary
   - compare against available external or internal reference software when such comparison exists
9. plots
   - include graphs for the benchmark grouping or value sweep where graphical comparison is meaningful
10. optional diagnostics
   - expose richer diagnostics or full artifacts when requested

## Required Content Rules

### Source-path rule

Example notebooks must default to running against the repo source tree through
`/src`, not against an installed package copy.

That means:

- notebooks should resolve the repo root
- notebooks should add `REPO_ROOT / "src"` to `sys.path` when needed
- notebooks should prefer repo-relative execution

Installed-package execution may still be supported as an explicit secondary
path, but it must not be the default notebook assumption.

### Example input/output ownership rule

Example-owned data should live under:

- `examples/inputs/<example_name>/`
- `examples/outputs/<example_name>/`

Do not scatter unrelated input or output files directly under the root of
`examples/inputs/` or `examples/outputs/`.

Each example should own a single named subfolder and reuse that location for:

- templates
- shared example fixtures
- local working summaries
- example-specific generated outputs during notebook development

Avoid duplicating equivalent templates or retained outputs across multiple
example folders.

Semi-permanent retained artifacts do not belong to the canonical example-owned
output tree. Promote those into named subfolders under the repo-root
`outputs/` directory.

### Instantiation rule

The notebook must instantiate and use a concrete object or input family.

Examples:

- a dense SPD or HPD matrix
- a sparse/block/vblock matrix
- a matrix-free operator plan
- a hypergeometric parameter sweep
- a Bessel/Hankel argument/order sweep

### Sweep rule

The notebook must include a complete or intentionally representative sweep over:

- values
- parameters
- shapes
- methods
- modes

depending on the functionality group.

### Validation rule

The notebook must summarize relevant testing status.

That summary may come from:

- direct notebook calculations
- generated result artifacts
- linked test/benchmark outputs

### Benchmark rule

The notebook must summarize relevant benchmark results for its group, including
whichever of these are meaningful:

- cold time
- warm time
- recompile time
- Python overhead
- memory
- accuracy/residual
- AD cost

The notebook should also show how that benchmark surface can be extended for
additional functions in the same family without inventing a separate local
workflow.

### Production calling rule

The notebook must teach the production-quality calling pattern for its surface.

That means showing the appropriate combination of:

- pre-bound service callables through `api.bind_point_batch()` or
  `api.bind_interval_batch()` where the public API supports them
- cached prepare/apply plan reuse for matrix, sparse, transform, or
  operator-plan surfaces
- fixed dtype, mode, and precision policy across repeated calls
- optional `pad_to` and `chunk_size` controls when they are relevant for
  reducing avoidable recompiles or handling variable request sizes

If a family does not use one of these patterns, the notebook should say so
explicitly instead of leaving the calling convention implicit.

The environment and production sections should also make runtime portability
concrete:

- show explicit `JAX_MODE` handling for CPU/GPU selection
- show explicit `JAX_DTYPE` handling for `float32`/`float64`
- state which slice is exercised by the retained canonical outputs in the
  current repo state

### Diagnostics and metadata rule

When a family exposes method selection, execution strategies, or diagnostics
through public metadata or optional diagnostics objects, the notebook should
show that surface directly.

Examples:

- method parameter names from public metadata
- execution strategy names from public metadata
- representative diagnostics objects from production-facing calls
- compile/recompile diagnostics from benchmark or JAX diagnostics artifacts

### AD rule

Canonical notebooks must show AD on the real production entrypoint for the
family, not only on a toy local helper.

That means:

- differentiate the public bound callable, routed API surface, cached plan
  surface, or other real production entrypoint
- validate AD behavior on a representative scalarized objective
- include a plot that shows primal and derivative behavior over a meaningful
  sweep

### Comparison rule

Where comparison software exists, the notebook must summarize it.

Examples:

- Arb/FLINT parity
- Boost reference paths
- SciPy
- PETSc/SLEPc benchmark-only comparisons
- optional NUFFT backends

If no comparison stack exists for that group, the notebook should say so.

### Plot rule

Each notebook should include plots for:

- sweep results
- benchmark comparisons
- accuracy/runtime tradeoffs

unless the group is too small for a meaningful graph.

## Optional Full Diagnostics

Notebooks may expose richer diagnostics behind a flag or separate section, for
example:

- full benchmark artifact tables
- residual histories
- compile traces
- detailed backend comparison tables

These diagnostics should not overwhelm the main notebook flow.

## Reports Rule

Current notebook coverage and gaps belong in `docs/reports`.
Current notebook completion status belongs in `docs/status`.
