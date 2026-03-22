Last updated: 2026-03-22T00:00:00Z

# Experiment Layout Standard

Status: active

## Purpose

This document defines how `experiments/` is used in this repo.

This document owns experiment folder structure and source-control policy.

It does not define:

- notebook content requirements
- benchmark grouping rules
- general portability policy

Those belong to:

- [example_notebook_standard.md](/home/phili/projects/arbplusJAX/docs/standards/example_notebook_standard.md)
- [benchmark_grouping_standard.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_grouping_standard.md)
- [environment_portability_standard.md](/home/phili/projects/arbplusJAX/docs/standards/environment_portability_standard.md)

Experiments are for larger-scale implementations and exploratory work, usually
grown out of `example_*.ipynb` notebooks.

They are not the same thing as:

- `examples/`
  - canonical demonstration notebooks
- `benchmarks/`
  - benchmark CLI and harness code
- `src/arbplusjax/`
  - governed runtime code

## Experiment Root Rule

Each experiment must live in its own subfolder under `experiments/`.

That subfolder should use a human-readable name describing the experiment.

Examples:

- `experiments/hypgeom_batch_recompile`
- `experiments/matrix_free_logdet_sweep`
- `experiments/bessel_region_stability`

## Subfolder Layout Rule

Inside each experiment subfolder:

- scripts and notebooks live at the root of that experiment folder
- inputs live in subfolders such as `inputs/`
- outputs live in subfolders such as `outputs/`

Recommended shape:

```text
experiments/<human_name>/
  README.md
  notebook.ipynb
  run.py
  inputs/
  outputs/
```

Allowed additional non-source-controlled subfolders:

- `artifacts/`
- `cache/`
- `figures/`
- `tables/`

## Source Control Rule

Subfolders inside each experiment are not all treated the same.

Inputs and transient artifact folders should not be under source control.

That means folders such as:

- `inputs/`
- `artifacts/`
- `cache/`
- `figures/`
- `tables/`

should be gitignored by default.

`outputs/` is different.

`outputs/` is the retained artifact area for important experiment results that
need to be segregated for backup and correct storage.

Rules for `outputs/`:

- `outputs/` should use human-readable named subfolders
- `outputs/` is intended to be under source control
- retained `outputs/` artifacts should use the repo's lower-level large-artifact
  storage path, currently assumed to mean Git LFS
- transient scratch outputs should go in `artifacts/` or `cache/`, not in the
  retained `outputs/` area

The root experiment folder may still keep:

- `README.md`
- notebooks
- driver scripts
- small checked-in config templates when they are part of the experiment definition

## Relationship To Examples

Typical flow:

1. create or refine an `example_*.ipynb` notebook
2. grow the investigation into a larger experiment under `experiments/<name>/`
3. keep large sweeps, heavy diagnostics, and exploratory outputs in the experiment
4. promote stable conclusions back into docs, tests, benchmarks, or runtime code

## Relationship To Benchmarks

Benchmark code stays in `benchmarks/`.

Experiment folders may:

- call benchmark scripts
- read or summarize benchmark artifacts from `benchmarks/results/`
- produce additional plots or derived analysis

But benchmark CLI entrypoints do not move into experiment folders.

Experiment-generated files must still stay inside the owning experiment tree.

That means:

- experiments may consume data from `benchmarks/results/`
- experiments must not write their own retained outputs into `benchmarks/results/`
- retained experiment outputs belong under `experiments/<name>/outputs/`
- transient experiment scratch files belong under `experiments/<name>/artifacts/` or
  `experiments/<name>/cache/`

## Naming Rule

Experiment folder names should be:

- human-readable
- descriptive of the question being studied
- stable enough to be recognizable later

Avoid opaque timestamp-only folder names for the experiment root.
Timestamps belong inside output or artifact subfolders when needed.
