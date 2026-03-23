Last updated: 2026-03-17T00:00:00Z

# Project Overview

arbPlusJAX is the active JAX implementation workspace. The repository separates runtime code, conformance tests, benchmarks, tooling, and documentation into stable top-level folders.

## Repo root

- `src/`: executable implementation
- `tests/`: conformance, regression, and chassis coverage
- `benchmarks/`: benchmark runners and smoke benchmarks
- `tools/`: harnesses, reporting, packaging, and maintenance scripts
- `docs/`: documentation authority and explanation
- `contracts/`: binding runtime/API guarantees
- `examples/`: runnable demos and templates
- `experiments/`: exploratory work
- `experiments/benchmarks/outputs/`: benchmark-side generated artifact root
- `benchmarks/results/`: benchmark run root

## Docs map

- root entry docs:
  - [governance/architecture.md](/home/phili/projects/arbplusJAX/docs/governance/architecture.md)
  - [governance/documentation_governance.md](/home/phili/projects/arbplusJAX/docs/governance/documentation_governance.md)
  - [project_overview.md](/home/phili/projects/arbplusJAX/docs/project_overview.md)
- governance:
  - [governance/README.md](/home/phili/projects/arbplusJAX/docs/governance/README.md)
- standards:
  - [standards/engineering_standard.md](/home/phili/projects/arbplusJAX/docs/standards/engineering_standard.md)
  - [standards/function_naming_standard.md](/home/phili/projects/arbplusJAX/docs/standards/function_naming_standard.md)
  - [standards/jax_surface_policy_standard.md](/home/phili/projects/arbplusJAX/docs/standards/jax_surface_policy_standard.md)
  - [standards/precision_standard.md](/home/phili/projects/arbplusJAX/docs/standards/precision_standard.md)
- implementation docs:
  - [implementation/README.md](/home/phili/projects/arbplusJAX/docs/implementation/README.md)
  - [implementation/modules](/home/phili/projects/arbplusJAX/docs/implementation/modules)
  - [implementation/wrappers](/home/phili/projects/arbplusJAX/docs/implementation/wrappers)
  - [implementation/external](/home/phili/projects/arbplusJAX/docs/implementation/external)
- practical docs:
  - [practical/README.md](/home/phili/projects/arbplusJAX/docs/practical/README.md)
  - [practical/running.md](/home/phili/projects/arbplusJAX/docs/practical/running.md)
  - [practical/benchmarking.md](/home/phili/projects/arbplusJAX/docs/practical/benchmarking.md)
  - [practical/numerical_guidance.md](/home/phili/projects/arbplusJAX/docs/practical/numerical_guidance.md)
- specs:
  - [specs/README.md](/home/phili/projects/arbplusJAX/docs/specs/README.md)
  - [specs/structured_matrix_functionality_poa.md](/home/phili/projects/arbplusJAX/docs/specs/structured_matrix_functionality_poa.md)
- status docs:
  - [status/todo.md](/home/phili/projects/arbplusJAX/docs/status/todo.md)
  - [status/audit.md](/home/phili/projects/arbplusJAX/docs/status/audit.md)
  - [reports/README.md](/home/phili/projects/arbplusJAX/docs/reports/README.md)

## Current conventions

- keep `docs/specs/`, `docs/objects/`, `contracts/`, and `docs/implementation/` as the core repository structure for semantics, object catalogs, guarantees, and implementation notes
- dense explicit matrix work lives under `arb_mat` and `acb_mat`
- Jones-labeled `jrb_mat` and `jcb_mat` are the matrix-free/operator JAX layer
- generated reports belong under `docs/reports/`
- semantic definitions should go in `docs/specs/`, not in ad hoc implementation notes
